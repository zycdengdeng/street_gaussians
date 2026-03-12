#!/usr/bin/env python3
"""
Preprocess car-road roadside data for Street Gaussians training.

Converts raw data from /mnt/car_road_data_TianJin/<scene>/ to the format
expected by lib/datasets/carroad_readers.py.

Usage:
    python script/carroad/prepare_carroad_data.py \
        --raw_dir /mnt/car_road_data_TianJin/001_car0325_road0327_t1 \
        --output_dir ./data/carroad/scene_001 \
        --cameras pinhole0 pinhole1 pinhole2 pinhole3 \
        --start_frame 0 \
        --end_frame -1

Output structure:
    output_dir/
    ├── images/               # {frame:06d}_{cam_id}.png (undistorted)
    ├── intrinsics/           # {cam_id}.txt (fx fy cx cy, undistorted)
    ├── extrinsics/           # {cam_id}.txt (4x4 cam-to-world)
    ├── pointcloud.npz        # 'points', 'colors'
    ├── timestamps.json       # frame/camera timestamps
    ├── track/                # tracklet info (if labels available)
    ├── lidar_depth/          # depth maps from LiDAR projection
    └── sky_mask/             # (optional, needs external model)
"""

import argparse
import os
import sys
import json
import glob
import shutil
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path


# ─── Pinhole camera folder → calib.json camera key mapping ─────────────
# This mapping is HARDCODED per the dataset documentation
PINHOLE_TO_CAM_KEY = {
    'pinhole0': 'cam3',
    'pinhole1': 'cam6',
    'pinhole2': 'cam9',
    'pinhole3': 'cam0',
}

# Internal camera ID used by Street Gaussians (0-indexed sequential)
PINHOLE_TO_CAM_ID = {
    'pinhole0': 0,
    'pinhole1': 1,
    'pinhole2': 2,
    'pinhole3': 3,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare car-road data for Street Gaussians')
    parser.add_argument('--raw_dir', type=str, required=True,
                        help='Path to raw scene directory, e.g. /mnt/car_road_data_TianJin/001_xxx')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--cameras', nargs='+', default=['pinhole0', 'pinhole1', 'pinhole2', 'pinhole3'],
                        help='Pinhole cameras to use')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Start frame index')
    parser.add_argument('--end_frame', type=int, default=-1,
                        help='End frame index (-1 for all)')
    parser.add_argument('--frame_step', type=int, default=1,
                        help='Frame sampling step (e.g. 3 = every 3rd frame)')
    parser.add_argument('--undistort_alpha', type=float, default=0.0,
                        help='Alpha for getOptimalNewCameraMatrix (0=crop, 1=keep all)')
    parser.add_argument('--generate_depth', action='store_true', default=True,
                        help='Generate LiDAR depth maps')
    parser.add_argument('--depth_dilation_iters', type=int, default=15,
                        help='Morphological dilation iterations for depth maps')
    parser.add_argument('--filter_visible', action='store_true', default=True,
                        help='Only keep LiDAR points visible from at least one camera')
    parser.add_argument('--calib_override', type=str, default=None,
                        help='Path to calib.json override (if not in standard location)')
    parser.add_argument('--pinhole_cam_mapping', type=str, default=None,
                        help='JSON string for custom pinhole->cam key mapping, e.g. \'{"pinhole0":"cam3"}\'')
    return parser.parse_args()


# ─── Calibration parsing ───────────────────────────────────────────────

def load_calib(calib_path):
    """Load calib.json and return camera dict."""
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    return calib


def rodrigues_to_rotmat(rvec):
    """Convert Rodrigues rotation vector to 3x3 rotation matrix."""
    rvec = np.array(rvec, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R


def parse_camera_params(calib, cam_key):
    """
    Parse camera intrinsics and extrinsics from calib.json.

    Returns:
        K: 3x3 intrinsic matrix (original, with distortion)
        dist: distortion coefficients
        R_w2c: 3x3 world-to-camera rotation
        t_w2c: 3x1 world-to-camera translation
    """
    cam_data = calib['camera'][cam_key]

    # Intrinsics (row-major 3x3)
    intri = np.array(cam_data['intri'], dtype=np.float64).reshape(3, 3)

    # Distortion
    dist = np.array(cam_data['distor'], dtype=np.float64)

    # Extrinsics: virtualLidarToCam (world-to-camera)
    rvec = cam_data['virtualLidarToCam']['rotate']
    tvec = cam_data['virtualLidarToCam']['trans']

    R_w2c = rodrigues_to_rotmat(rvec)
    t_w2c = np.array(tvec, dtype=np.float64)

    return intri, dist, R_w2c, t_w2c


def undistort_camera(K, dist, width, height, alpha=0.0):
    """
    Compute undistortion maps and new camera matrix.

    Returns:
        K_new: undistorted intrinsic matrix
        mapx, mapy: undistortion remap arrays
    """
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), alpha)
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K_new, (width, height), cv2.CV_32FC1)
    return K_new, mapx, mapy


# ─── Point cloud reading ──────────────────────────────────────────────

def read_pcd_ascii(pcd_path):
    """Read ASCII PCD file. Returns points (N,3) and intensities (N,)."""
    with open(pcd_path, 'r') as f:
        lines = f.readlines()

    # Find DATA line
    data_start = 0
    num_points = 0
    for i, line in enumerate(lines):
        if line.startswith('POINTS'):
            num_points = int(line.split()[1])
        if line.startswith('DATA'):
            data_start = i + 1
            break

    points = []
    intensities = []
    for line in lines[data_start:data_start + num_points]:
        parts = line.strip().split()
        if len(parts) >= 3:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            intensity = float(parts[3]) if len(parts) > 3 else 0.0
            points.append([x, y, z])
            intensities.append(intensity)

    return np.array(points, dtype=np.float64), np.array(intensities, dtype=np.float64)


def read_pcd_file(pcd_path):
    """Read PCD file (auto-detect ASCII or binary)."""
    with open(pcd_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line.startswith('DATA'):
                break

        data_format = header_lines[-1].split()[1]
        num_points = 0
        for line in header_lines:
            if line.startswith('POINTS'):
                num_points = int(line.split()[1])

    if data_format == 'ascii':
        return read_pcd_ascii(pcd_path)
    else:
        # Binary PCD - use open3d if available
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
            intensities = np.zeros(len(points))
            return points, intensities
        except ImportError:
            raise RuntimeError("Binary PCD requires open3d. Install with: pip install open3d")


# ─── Label parsing ────────────────────────────────────────────────────

def parse_labels(label_dir, frame_timestamps):
    """
    Parse road_labels to extract object tracking info.

    Expected label format per frame (JSON or txt):
    Each label file contains objects with:
      - track_id, class, bbox3d (cx, cy, cz, l, w, h, heading)

    Returns:
        track_info_lines: list of strings for track_info.txt
        track_camera_vis: dict for track_camera_vis.json
    """
    track_info_lines = ['frame_id track_id class score height width length cx cy cz heading']
    track_camera_vis = {}

    if not os.path.exists(label_dir):
        print(f"Label directory not found: {label_dir}")
        return track_info_lines, track_camera_vis

    # Try to find label files
    label_files = sorted(
        glob.glob(os.path.join(label_dir, '*.json')) +
        glob.glob(os.path.join(label_dir, '*.txt'))
    )

    if not label_files:
        # Check subdirectories
        for subdir in ['interpolation_labels', 'ori_labels']:
            subpath = os.path.join(label_dir, subdir)
            if os.path.exists(subpath):
                label_files = sorted(
                    glob.glob(os.path.join(subpath, '*.json')) +
                    glob.glob(os.path.join(subpath, '*.txt'))
                )
                if label_files:
                    break

    if not label_files:
        print(f"No label files found in {label_dir}")
        return track_info_lines, track_camera_vis

    print(f"Found {len(label_files)} label files in {label_dir}")

    for frame_idx, label_file in enumerate(label_files):
        ext = os.path.splitext(label_file)[1].lower()
        objects = []

        if ext == '.json':
            with open(label_file, 'r') as f:
                data = json.load(f)

            # Handle various JSON formats
            if isinstance(data, list):
                objects = data
            elif isinstance(data, dict):
                if 'objects' in data:
                    objects = data['objects']
                elif 'annotations' in data:
                    objects = data['annotations']
                elif 'labels' in data:
                    objects = data['labels']
                else:
                    # Try to interpret as single object dict
                    objects = [data]

        elif ext == '.txt':
            with open(label_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 10:
                    try:
                        obj = {
                            'track_id': int(parts[0]),
                            'class': parts[1],
                            'height': float(parts[2]),
                            'width': float(parts[3]),
                            'length': float(parts[4]),
                            'cx': float(parts[5]),
                            'cy': float(parts[6]),
                            'cz': float(parts[7]),
                            'heading': float(parts[8]),
                        }
                        if len(parts) > 9:
                            obj['score'] = float(parts[9])
                        objects.append(obj)
                    except (ValueError, IndexError):
                        continue

        for obj in objects:
            track_id = obj.get('track_id', obj.get('id', obj.get('tracking_id', -1)))
            obj_class = obj.get('class', obj.get('type', obj.get('category', 'vehicle')))
            score = obj.get('score', obj.get('confidence', 1.0))

            # 3D bbox
            height = obj.get('height', obj.get('h', 1.5))
            width = obj.get('width', obj.get('w', 1.8))
            length = obj.get('length', obj.get('l', 4.5))
            cx = obj.get('cx', obj.get('x', 0))
            cy = obj.get('cy', obj.get('y', 0))
            cz = obj.get('cz', obj.get('z', 0))
            heading = obj.get('heading', obj.get('yaw', obj.get('rotation_y', 0)))

            if track_id < 0:
                continue

            line = f"{frame_idx} {track_id} {obj_class} {score:.2f} {height:.3f} {width:.3f} {length:.3f} {cx:.3f} {cy:.3f} {cz:.3f} {heading:.4f}"
            track_info_lines.append(line)

            # All cameras can see (roadside, wide coverage)
            tid = str(track_id)
            fid = str(frame_idx)
            if tid not in track_camera_vis:
                track_camera_vis[tid] = {}
            track_camera_vis[tid][fid] = [0, 1, 2, 3]

    return track_info_lines, track_camera_vis


# ─── Depth map generation ─────────────────────────────────────────────

def project_points_to_camera(points_3d, K, R_w2c, t_w2c, width, height, min_depth=0.5):
    """
    Project 3D points to camera image plane.

    Returns:
        u, v: pixel coordinates (float)
        depth: camera-frame depth
        mask: valid projection mask
    """
    # Transform to camera frame
    pts_cam = (R_w2c @ points_3d.T).T + t_w2c  # (N, 3)
    depth = pts_cam[:, 2]

    # Filter behind camera
    mask = depth > min_depth

    # Project
    u = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]

    # Filter out of bounds
    mask &= (u >= 0) & (u < width) & (v >= 0) & (v < height)

    return u, v, depth, mask


def generate_lidar_depth_map(points_3d, K, R_w2c, t_w2c, width, height,
                              dilation_iters=15):
    """
    Generate dense-ish depth map from LiDAR point projection.

    Returns:
        depth_npy: dict with 'mask' and 'value' for sparse depth
    """
    u, v, depth, mask = project_points_to_camera(points_3d, K, R_w2c, t_w2c, width, height)

    u_valid = u[mask].astype(np.int32)
    v_valid = v[mask].astype(np.int32)
    d_valid = depth[mask]

    # Sparse depth map (nearest depth per pixel)
    depth_map = np.zeros((height, width), dtype=np.float32)
    depth_count = np.zeros((height, width), dtype=np.int32)

    for i in range(len(u_valid)):
        px, py = u_valid[i], v_valid[i]
        if depth_count[py, px] == 0 or d_valid[i] < depth_map[py, px]:
            depth_map[py, px] = d_valid[i]
            depth_count[py, px] = 1

    # Return sparse depth as mask + values
    valid_mask = depth_count > 0
    valid_values = depth_map[valid_mask]

    return {'mask': valid_mask, 'value': valid_values}


# ─── Color point cloud ────────────────────────────────────────────────

def color_pointcloud(points_3d, cameras_params, images):
    """
    Color 3D points by projecting onto camera images.
    For overlapping views, use the camera with smallest depth.

    cameras_params: list of (K, R_w2c, t_w2c, width, height)
    images: list of np.array (H, W, 3) in [0, 1]

    Returns colors (N, 3) and mask of colored points.
    """
    N = len(points_3d)
    best_depth = np.full(N, np.inf)
    colors = np.ones((N, 3), dtype=np.float32)
    colored = np.zeros(N, dtype=bool)

    for (K, R, t, w, h), img in zip(cameras_params, images):
        u, v, depth, mask = project_points_to_camera(points_3d, K, R, t, w, h)

        # Only update where this camera is closer
        closer = mask & (depth < best_depth)
        if closer.any():
            ui = u[closer].astype(np.int32)
            vi = v[closer].astype(np.int32)
            colors[closer] = img[vi, ui]
            best_depth[closer] = depth[closer]
            colored[closer] = True

    return colors, colored


# ─── Main processing ──────────────────────────────────────────────────

def main():
    args = parse_args()

    raw_dir = args.raw_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Override pinhole mapping if provided
    pinhole_cam_mapping = PINHOLE_TO_CAM_KEY.copy()
    if args.pinhole_cam_mapping:
        custom = json.loads(args.pinhole_cam_mapping)
        pinhole_cam_mapping.update(custom)

    # ─── Find calib.json ────────────────────────────────────────────
    calib_path = args.calib_override
    if calib_path is None:
        # Search common locations
        for candidate in [
            os.path.join(raw_dir, 'calib.json'),
            os.path.join(raw_dir, 'road', 'calib.json'),
            os.path.join(raw_dir, '..', 'calib.json'),
        ]:
            if os.path.exists(candidate):
                calib_path = candidate
                break

    if calib_path is None or not os.path.exists(calib_path):
        raise FileNotFoundError(
            f"calib.json not found. Searched in {raw_dir}, {raw_dir}/road/, "
            f"and parent. Use --calib_override to specify path."
        )

    print(f"Using calibration: {calib_path}")
    calib = load_calib(calib_path)

    # Image size from calib
    img_size = calib.get('imgSize', {})
    not_fish_size = img_size.get('notFish', [1280, 720])
    img_width, img_height = not_fish_size[0], not_fish_size[1]
    print(f"Image size: {img_width}x{img_height}")

    # ─── Process cameras ────────────────────────────────────────────
    print("\n=== Processing cameras ===")
    os.makedirs(os.path.join(output_dir, 'intrinsics'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'extrinsics'), exist_ok=True)

    camera_params = {}  # cam_name -> {K_new, R_w2c, t_w2c, mapx, mapy, cam_id, c2w}

    for cam_name in args.cameras:
        cam_key = pinhole_cam_mapping.get(cam_name)
        if cam_key is None:
            # Try extracting key number from calib
            print(f"Warning: no mapping for {cam_name}, trying direct lookup")
            cam_key = cam_name

        # Strip 'cam' prefix to get the key in calib.json
        calib_key = cam_key.replace('cam', '') if cam_key.startswith('cam') else cam_key

        if calib_key not in calib['camera']:
            raise ValueError(f"Camera key '{calib_key}' not found in calib.json. "
                           f"Available: {list(calib['camera'].keys())}")

        K, dist, R_w2c, t_w2c = parse_camera_params(calib, calib_key)
        K_new, mapx, mapy = undistort_camera(K, dist, img_width, img_height, args.undistort_alpha)

        cam_id = PINHOLE_TO_CAM_ID.get(cam_name, args.cameras.index(cam_name))

        # Camera-to-world transform
        w2c = np.eye(4)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        c2w = np.linalg.inv(w2c)

        camera_params[cam_name] = {
            'K_orig': K,
            'K_new': K_new,
            'dist': dist,
            'R_w2c': R_w2c,
            't_w2c': t_w2c,
            'mapx': mapx,
            'mapy': mapy,
            'cam_id': cam_id,
            'c2w': c2w,
        }

        # Save intrinsics (fx fy cx cy)
        fx, fy = K_new[0, 0], K_new[1, 1]
        cx, cy = K_new[0, 2], K_new[1, 2]
        np.savetxt(os.path.join(output_dir, 'intrinsics', f'{cam_id}.txt'),
                   [fx, fy, cx, cy], fmt='%.6f')

        # Save extrinsics (4x4 cam-to-world)
        np.savetxt(os.path.join(output_dir, 'extrinsics', f'{cam_id}.txt'),
                   c2w, fmt='%.10f')

        cam_center = c2w[:3, 3]
        print(f"  {cam_name} (cam_key={calib_key}, id={cam_id}): "
              f"fx={fx:.1f}, fy={fy:.1f}, center=[{cam_center[0]:.1f}, {cam_center[1]:.1f}, {cam_center[2]:.1f}]")

    # ─── Discover frames ────────────────────────────────────────────
    print("\n=== Discovering frames ===")

    # Find image directories
    road_cameras_dir = os.path.join(raw_dir, 'road', 'cameras')
    if not os.path.exists(road_cameras_dir):
        # Try alternative paths
        road_cameras_dir = os.path.join(raw_dir, 'road')

    first_cam = args.cameras[0]
    cam_image_dir = os.path.join(road_cameras_dir, first_cam)
    if not os.path.exists(cam_image_dir):
        # Try img/ subdirectory (self_Dataset format)
        cam_image_dir = os.path.join(raw_dir, 'img', first_cam)

    if not os.path.exists(cam_image_dir):
        raise FileNotFoundError(
            f"Image directory not found. Tried:\n"
            f"  {os.path.join(road_cameras_dir, first_cam)}\n"
            f"  {os.path.join(raw_dir, 'img', first_cam)}"
        )

    # Get all timestamps from first camera
    all_image_files = sorted(
        glob.glob(os.path.join(cam_image_dir, '*.png')) +
        glob.glob(os.path.join(cam_image_dir, '*.jpg'))
    )
    all_timestamps = [os.path.splitext(os.path.basename(f))[0] for f in all_image_files]
    print(f"Found {len(all_timestamps)} frames from {first_cam}")

    # Apply frame selection
    if args.end_frame < 0:
        end_idx = len(all_timestamps) - 1
    else:
        end_idx = min(args.end_frame, len(all_timestamps) - 1)

    selected_timestamps = all_timestamps[args.start_frame:end_idx + 1:args.frame_step]
    num_frames = len(selected_timestamps)
    print(f"Selected {num_frames} frames (start={args.start_frame}, end={end_idx}, step={args.frame_step})")

    # ─── Process images ─────────────────────────────────────────────
    print("\n=== Processing images ===")
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    timestamps_data = {'FRAME': {}, 'CAMERAS': {}}
    for cam_name in args.cameras:
        cam_id = camera_params[cam_name]['cam_id']
        timestamps_data['CAMERAS'][str(cam_id)] = {}

    for frame_idx, ts in enumerate(tqdm(selected_timestamps, desc="Processing frames")):
        # Frame timestamp
        try:
            ts_float = float(ts) / 1000.0  # ms -> seconds
        except ValueError:
            ts_float = float(frame_idx)
        timestamps_data['FRAME'][f'{frame_idx:06d}'] = ts_float

        for cam_name in args.cameras:
            params = camera_params[cam_name]
            cam_id = params['cam_id']

            # Find the image
            for img_dir_candidate in [
                os.path.join(road_cameras_dir, cam_name),
                os.path.join(raw_dir, 'img', cam_name),
            ]:
                img_path = None
                for ext in ['.png', '.jpg']:
                    candidate = os.path.join(img_dir_candidate, f'{ts}{ext}')
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                if img_path:
                    break

            if img_path is None:
                print(f"Warning: image not found for {cam_name} frame {ts}")
                continue

            # Read and undistort
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: failed to read {img_path}")
                continue

            img_undist = cv2.remap(img, params['mapx'], params['mapy'], cv2.INTER_LINEAR)

            # Save with Street Gaussians naming convention
            out_name = f'{frame_idx:06d}_{cam_id}.png'
            cv2.imwrite(os.path.join(output_dir, 'images', out_name), img_undist)

            # Camera timestamp
            timestamps_data['CAMERAS'][str(cam_id)][f'{frame_idx:06d}'] = ts_float

    # Save timestamps
    with open(os.path.join(output_dir, 'timestamps.json'), 'w') as f:
        json.dump(timestamps_data, f, indent=2)

    # ─── Process point cloud ────────────────────────────────────────
    print("\n=== Processing point cloud ===")

    # Find merged LiDAR point clouds
    lidar_dirs = [
        os.path.join(raw_dir, 'road', 'lidar', 'merged_pcd'),
        os.path.join(raw_dir, 'road', 'lidar'),
    ]

    all_points = []
    all_intensities = []

    for lidar_dir in lidar_dirs:
        if not os.path.exists(lidar_dir):
            continue

        pcd_files = sorted(
            glob.glob(os.path.join(lidar_dir, '*.pcd'))
        )
        if not pcd_files:
            continue

        print(f"Found {len(pcd_files)} PCD files in {lidar_dir}")

        # Load PCDs matching selected timestamps
        for ts in tqdm(selected_timestamps, desc="Loading PCDs"):
            pcd_path = os.path.join(lidar_dir, f'{ts}.pcd')
            if not os.path.exists(pcd_path):
                # Try finding any PCD in directory (single frame case)
                if len(pcd_files) == 1:
                    pcd_path = pcd_files[0]
                else:
                    continue

            points, intensities = read_pcd_file(pcd_path)
            all_points.append(points)
            all_intensities.append(intensities)

        break  # Found PCDs, stop searching

    if not all_points:
        # Try single PCD file in raw_dir (self_Dataset format)
        single_pcds = glob.glob(os.path.join(raw_dir, '*.pcd'))
        if single_pcds:
            print(f"Using single PCD: {single_pcds[0]}")
            points, intensities = read_pcd_file(single_pcds[0])
            all_points.append(points)
            all_intensities.append(intensities)

    if all_points:
        merged_points = np.concatenate(all_points, axis=0)
        merged_intensities = np.concatenate(all_intensities, axis=0)

        # Remove duplicates via voxel grid
        if len(all_points) > 1:
            try:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(merged_points)
                pcd = pcd.voxel_down_sample(voxel_size=0.05)
                merged_points = np.asarray(pcd.points)
                merged_intensities = np.zeros(len(merged_points))  # intensity lost after dedup
            except ImportError:
                pass

        print(f"Total points: {len(merged_points)}")

        # Color the point cloud
        print("Coloring point cloud from images...")
        cam_params_list = []
        undist_images = []

        for cam_name in args.cameras:
            params = camera_params[cam_name]
            cam_params_list.append((
                params['K_new'],
                params['R_w2c'],
                params['t_w2c'],
                img_width, img_height
            ))

            # Load first frame undistorted image for coloring
            cam_id = params['cam_id']
            img_path = os.path.join(output_dir, 'images', f'000000_{cam_id}.png')
            if os.path.exists(img_path):
                img = cv2.imread(img_path)[..., [2, 1, 0]] / 255.0
                undist_images.append(img)
            else:
                undist_images.append(np.ones((img_height, img_width, 3)))

        colors, colored_mask = color_pointcloud(merged_points, cam_params_list, undist_images)

        if args.filter_visible:
            merged_points = merged_points[colored_mask]
            colors = colors[colored_mask]
            print(f"After visibility filter: {len(merged_points)} points")

        # Save
        np.savez(os.path.join(output_dir, 'pointcloud.npz'),
                 points=merged_points.astype(np.float32),
                 colors=colors.astype(np.float32))
        print(f"Saved pointcloud.npz: {len(merged_points)} points")
    else:
        print("WARNING: No point cloud data found!")

    # ─── Process labels ─────────────────────────────────────────────
    print("\n=== Processing labels ===")
    os.makedirs(os.path.join(output_dir, 'track'), exist_ok=True)

    label_dir = os.path.join(raw_dir, 'road_labels')
    track_lines, track_vis = parse_labels(label_dir, selected_timestamps)

    # Write track_info.txt
    with open(os.path.join(output_dir, 'track', 'track_info.txt'), 'w') as f:
        f.write('\n'.join(track_lines))
    print(f"Wrote {len(track_lines) - 1} tracklet entries")

    # Write track_camera_vis.json
    with open(os.path.join(output_dir, 'track', 'track_camera_vis.json'), 'w') as f:
        json.dump(track_vis, f)

    # ─── Generate depth maps ────────────────────────────────────────
    if args.generate_depth and all_points:
        print("\n=== Generating LiDAR depth maps ===")
        os.makedirs(os.path.join(output_dir, 'lidar_depth'), exist_ok=True)

        for frame_idx, ts in enumerate(tqdm(selected_timestamps, desc="Depth maps")):
            # Use the merged point cloud (or per-frame if available)
            pts = merged_points

            for cam_name in args.cameras:
                params = camera_params[cam_name]
                cam_id = params['cam_id']

                depth_data = generate_lidar_depth_map(
                    pts, params['K_new'], params['R_w2c'], params['t_w2c'],
                    img_width, img_height,
                    dilation_iters=args.depth_dilation_iters,
                )

                out_name = f'{frame_idx:06d}_{cam_id}'
                np.save(
                    os.path.join(output_dir, 'lidar_depth', f'{out_name}.npy'),
                    depth_data,
                    allow_pickle=True,
                )

    # ─── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Output: {output_dir}")
    print(f"Cameras: {len(args.cameras)}")
    print(f"Frames: {num_frames}")
    if all_points:
        print(f"Points: {len(merged_points)}")
    print(f"\nTo train:")
    print(f"  python train.py --config configs/carroad/your_scene.yaml")
    print("=" * 60)


if __name__ == '__main__':
    main()
