"""
Car-Road roadside dataset reader for Street Gaussians.

Reads data preprocessed by script/carroad/prepare_carroad_data.py.
Expected directory structure after preprocessing:

    data/carroad/<scene_name>/
    ├── images/
    │   ├── 000000_0.png       # {frame:06d}_{cam_id}.png
    │   ├── 000000_1.png
    │   └── ...
    ├── intrinsics/
    │   ├── 0.txt              # fx fy cx cy (undistorted)
    │   └── ...
    ├── extrinsics/
    │   ├── 0.txt              # 4x4 cam-to-world (virtualLidar)
    │   └── ...
    ├── pointcloud.npz         # 'points': (N,3), 'colors': (N,3), 'camera_projection': dict
    ├── timestamps.json        # frame/camera timestamps
    ├── track/
    │   ├── track_info.txt     # object tracklets
    │   └── track_camera_vis.json
    ├── lidar_depth/           # optional: {frame:06d}_{cam_id}.npy
    ├── sky_mask/              # optional: {frame:06d}_{cam_id}.png
    └── dynamic_mask/          # optional: {frame:06d}_{cam_id}.png

The world coordinate system is "virtualLidar" from the roadside calibration.
Roadside cameras are STATIC (mounted on poles), so there is no ego motion.
ego_pose is identity for all frames.
"""

import os
import sys
import json
import math
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from PIL import Image

from lib.config import cfg
from lib.datasets.base_readers import (
    CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly,
    get_Sphere_Norm,
)
from lib.utils.graphics_utils import focal2fov, BasicPointCloud
from lib.utils.data_utils import get_val_frames
from lib.utils.general_utils import matrix_to_quaternion, quaternion_to_matrix_numpy
from lib.utils.box_utils import bbox_to_corner3d, inbbox_points, get_bound_2d_mask


# ─── Label class mapping ───────────────────────────────────────────────
carroad_track2label = {
    "vehicle": 0,
    "car": 0,
    "truck": 0,
    "bus": 0,
    "pedestrian": 1,
    "cyclist": 2,
    "bicycle": 2,
    "misc": -1,
}


# ─── Camera helpers ─────────────────────────────────────────────────────

def _load_intrinsics(intrinsics_dir, num_cameras):
    """Load per-camera intrinsics.  Each file has: fx fy cx cy"""
    intrinsics = []
    for i in range(num_cameras):
        vals = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
        fx, fy, cx, cy = vals[0], vals[1], vals[2], vals[3]
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])
        intrinsics.append(K)
    return intrinsics


def _load_extrinsics(extrinsics_dir, num_cameras):
    """Load per-camera extrinsics (4x4 camera-to-world)."""
    extrinsics = []
    for i in range(num_cameras):
        ext = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))
        extrinsics.append(ext.reshape(4, 4))
    return extrinsics


def _image_filename_to_frame(basename):
    """000012_2.png -> 12"""
    return int(basename.split('.')[0].split('_')[0])


def _image_filename_to_cam(basename):
    """000012_2.png -> 2"""
    return int(basename.split('.')[0].split('_')[1])


# ─── Object tracking ───────────────────────────────────────────────────

def _make_obj_pose_world(box_info):
    """
    Create 4x4 object pose in world frame.
    box_info: [cx, cy, cz, heading]  (heading = yaw around Z axis)
    Returns: obj_pose_world (4x4), obj_pose_flat (7,) [x,y,z,qw,qx,qy,qz]
    """
    tx, ty, tz, heading = box_info
    c = math.cos(heading)
    s = math.sin(heading)
    rotz = np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

    pose = np.eye(4)
    pose[:3, :3] = rotz
    pose[:3, 3] = np.array([tx, ty, tz])

    rot_torch = torch_import().from_numpy(pose[:3, :3]).float().unsqueeze(0)
    quat = matrix_to_quaternion(rot_torch).squeeze(0).numpy()
    quat = quat / np.linalg.norm(quat)
    flat = np.concatenate([pose[:3, 3], quat])  # [x, y, z, qw, qx, qy, qz]
    return pose, flat


def torch_import():
    import torch
    return torch


def _load_track_info(datadir, selected_frames, cameras, num_frames_total):
    """
    Load object tracking information.

    track_info.txt format (tab/space separated, first line is header):
        frame_id  track_id  class  score  height  width  length  cx  cy  cz  heading

    track_camera_vis.json:
        { "track_id": { "frame_id": [cam_list] } }

    Objects are in WORLD frame (virtualLidar), since roadside cameras are static.
    """
    track_dir = os.path.join(datadir, 'track')
    track_info_path = os.path.join(track_dir, 'track_info.txt')
    track_vis_path = os.path.join(track_dir, 'track_camera_vis.json')

    if not os.path.exists(track_info_path):
        print("No track_info.txt found, creating dummy tracklets (no dynamic objects)")
        num_frames = selected_frames[1] - selected_frames[0] + 1
        dummy_tracklets = np.ones([num_frames, 1, 8]) * -1.0
        return dummy_tracklets, {}

    with open(track_info_path, 'r') as f:
        lines = f.read().splitlines()
    header = lines[0]
    lines = lines[1:]  # skip header

    # Load visibility if available
    track_vis = {}
    if os.path.exists(track_vis_path):
        with open(track_vis_path, 'r') as f:
            track_vis = json.load(f)

    start_frame, end_frame = selected_frames
    num_frames = end_frame - start_frame + 1

    objects_info = {}
    tracklets_ls = []
    n_obj_in_frame = np.zeros(num_frames_total)

    for line in lines:
        parts = line.split()
        if len(parts) < 11:
            continue
        frame_id = int(parts[0])
        track_id = int(parts[1])
        obj_class = parts[2].lower()

        if obj_class in ['sign', 'misc']:
            continue

        # Filter by visibility in selected cameras (if vis data available)
        if track_vis and str(track_id) in track_vis:
            if str(frame_id) in track_vis[str(track_id)]:
                vis_cams = track_vis[str(track_id)][str(frame_id)]
                if len(set(cameras) & set(vis_cams)) == 0:
                    continue

        if track_id not in objects_info:
            objects_info[track_id] = {
                'track_id': track_id,
                'class': obj_class if obj_class in carroad_track2label else 'vehicle',
                'class_label': carroad_track2label.get(obj_class, 0),
                'height': float(parts[4]),
                'width': float(parts[5]),
                'length': float(parts[6]),
            }
        else:
            objects_info[track_id]['height'] = max(objects_info[track_id]['height'], float(parts[4]))
            objects_info[track_id]['width'] = max(objects_info[track_id]['width'], float(parts[5]))
            objects_info[track_id]['length'] = max(objects_info[track_id]['length'], float(parts[6]))

        tracklets_ls.append(parts)
        if 0 <= frame_id < num_frames_total:
            n_obj_in_frame[frame_id] += 1

    if len(tracklets_ls) == 0 or len(objects_info) == 0:
        print("No valid tracks found, creating dummy tracklets")
        dummy_tracklets = np.ones([num_frames, 1, 8]) * -1.0
        return dummy_tracklets, {}

    max_obj_per_frame = int(n_obj_in_frame[start_frame:end_frame + 1].max())
    if max_obj_per_frame == 0:
        max_obj_per_frame = 1

    visible_objects_ids = np.ones([num_frames, max_obj_per_frame]) * -1.0
    visible_objects_pose = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0

    for parts in tracklets_ls:
        frame_id = int(parts[0])
        track_id = int(parts[1])
        if start_frame <= frame_id <= end_frame:
            # box_info: cx, cy, cz, heading
            box_info = [float(parts[7]), float(parts[8]), float(parts[9]), float(parts[10])]
            _, pose_flat = _make_obj_pose_world(box_info)

            frame_idx = frame_id - start_frame
            free_slots = np.argwhere(visible_objects_ids[frame_idx] < 0)
            if len(free_slots) == 0:
                continue
            obj_column = free_slots.min()
            visible_objects_ids[frame_idx, obj_column] = track_id
            visible_objects_pose[frame_idx, obj_column] = pose_flat

    # Remove static objects (position std < 0.5m AND distance < 2m)
    print("Removing static objects")
    for key in list(objects_info.keys()):
        all_idx = np.where(visible_objects_ids == key)
        if len(all_idx[0]) > 0:
            positions = visible_objects_pose[all_idx][:, :3]
            distance = np.linalg.norm(positions[0] - positions[-1])
            dynamic = np.any(np.std(positions, axis=0) > 0.5) or distance > 2
            if not dynamic:
                visible_objects_ids[all_idx] = -1.
                visible_objects_pose[all_idx] = -1.
                objects_info.pop(key)
        else:
            objects_info.pop(key)

    # Clip columns
    mask = visible_objects_ids >= 0
    max_obj_new = max(int(np.sum(mask, axis=1).max()), 1)
    if max_obj_new < max_obj_per_frame:
        new_ids = np.ones([num_frames, max_obj_new]) * -1.0
        new_pose = np.ones([num_frames, max_obj_new, 7]) * -1.0
        for fi in range(num_frames):
            col = 0
            for c in range(max_obj_per_frame):
                if visible_objects_ids[fi, c] >= 0:
                    new_ids[fi, col] = visible_objects_ids[fi, c]
                    new_pose[fi, col] = visible_objects_pose[fi, c]
                    col += 1
        visible_objects_ids = new_ids
        visible_objects_pose = new_pose

    print(f"Max objects per frame: {max_obj_new}, total tracked objects: {len(objects_info)}")

    # Postprocess obj_info
    box_scale = cfg.data.get('box_scale', 1.0)
    frames_arr = np.arange(start_frame, end_frame + 1).astype(np.int32)
    for key in objects_info:
        obj = objects_info[key]
        obj['deformable'] = obj['class'] in ['pedestrian']
        obj['width'] *= box_scale
        obj['length'] *= box_scale

        obj_frame_idx = np.argwhere(visible_objects_ids == key)[:, 0]
        obj_frames = frames_arr[obj_frame_idx]
        obj['start_frame'] = int(np.min(obj_frames))
        obj['end_frame'] = int(np.max(obj_frames))

    # Build tracklets array: [num_frames, max_obj, 8] = [track_id, x, y, z, qw, qx, qy, qz]
    tracklets = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose], axis=-1
    )
    return tracklets, objects_info


# ─── Point cloud building ──────────────────────────────────────────────

def _build_pointcloud(datadir, selected_frames, cameras, num_frames, intrinsics,
                      cam_to_worlds, image_filenames_all, tracklets, object_info):
    """
    Build background and per-object point clouds from LiDAR data.
    """
    import open3d as o3d

    pointcloud_dir = os.path.join(cfg.model_path, 'input_ply')
    os.makedirs(pointcloud_dir, exist_ok=True)

    start_frame, end_frame = selected_frames
    num_cams = len(cameras)

    points_xyz_dict = {'bkgd': []}
    points_rgb_dict = {'bkgd': []}
    for track_id in object_info:
        points_xyz_dict[f'obj_{track_id:03d}'] = []
        points_rgb_dict[f'obj_{track_id:03d}'] = []

    # Load point cloud
    pcd_path = os.path.join(datadir, 'pointcloud.npz')
    if os.path.exists(pcd_path):
        pcd_data = np.load(pcd_path, allow_pickle=True)
        if 'points' in pcd_data:
            # Simple format: all points + colors
            all_points = pcd_data['points']
            all_colors = pcd_data.get('colors', np.ones_like(all_points))
            if all_colors.max() > 1.0:
                all_colors = all_colors / 255.0

            # For each frame, separate background and object points
            for fi in range(num_frames):
                frame = start_frame + fi
                points_obj_mask = np.zeros(all_points.shape[0], dtype=bool)

                # Check which points fall inside object bboxes
                frame_tracklets = tracklets[fi]
                for t_info in frame_tracklets:
                    track_id = int(t_info[0])
                    if track_id >= 0 and track_id in object_info:
                        obj_pose = np.eye(4)
                        obj_pose[:3, :3] = quaternion_to_matrix_numpy(t_info[4:8])
                        obj_pose[:3, 3] = t_info[1:4]
                        world2local = np.linalg.inv(obj_pose)

                        pts_h = np.concatenate([all_points, np.ones((len(all_points), 1))], axis=-1)
                        pts_local = (pts_h @ world2local.T)[:, :3]

                        length = object_info[track_id]['length']
                        width = object_info[track_id]['width']
                        height = object_info[track_id]['height']
                        bbox = [[-length / 2, -width / 2, -height / 2],
                                [length / 2, width / 2, height / 2]]
                        corners = bbox_to_corner3d(bbox)
                        in_bbox = inbbox_points(pts_local, corners)
                        points_obj_mask |= in_bbox
                        points_xyz_dict[f'obj_{track_id:03d}'].append(pts_local[in_bbox])
                        points_rgb_dict[f'obj_{track_id:03d}'].append(all_colors[in_bbox])

                points_xyz_dict['bkgd'].append(all_points[~points_obj_mask])
                points_rgb_dict['bkgd'].append(all_colors[~points_obj_mask])

        elif 'pointcloud' in pcd_data:
            # Waymo-style format with per-frame data
            pts3d_dict = pcd_data['pointcloud'].item()
            pts2d_dict = pcd_data.get('camera_projection', np.array(None)).item()

            for fi in range(num_frames):
                frame = start_frame + fi
                if frame not in pts3d_dict:
                    continue

                raw_3d = pts3d_dict[frame]
                points_xyz = raw_3d[:, :3] if raw_3d.shape[1] > 3 else raw_3d
                points_rgb = np.ones_like(points_xyz)

                # Color from images if projection data available
                if pts2d_dict and frame in pts2d_dict:
                    raw_2d = pts2d_dict[frame]
                    points_camera = raw_2d[..., 0].astype(int)
                    points_projw = raw_2d[..., 1].astype(int)
                    points_projh = raw_2d[..., 2].astype(int)

                    for cam_idx, cam in enumerate(cameras):
                        idx_range = list(range(fi * num_cams, (fi + 1) * num_cams))
                        if cam_idx < len(idx_range):
                            img_path = image_filenames_all[idx_range[cam_idx]]
                            mask_cam = (points_camera == cam)
                            if mask_cam.any():
                                image = cv2.imread(img_path)[..., [2, 1, 0]] / 255.
                                pw = points_projw[mask_cam]
                                ph = points_projh[mask_cam]
                                valid = (pw >= 0) & (pw < image.shape[1]) & (ph >= 0) & (ph < image.shape[0])
                                points_rgb[mask_cam][valid] = image[ph[valid], pw[valid]]

                # Separate background and objects
                points_obj_mask = np.zeros(points_xyz.shape[0], dtype=bool)
                frame_tracklets = tracklets[fi]
                for t_info in frame_tracklets:
                    track_id = int(t_info[0])
                    if track_id >= 0 and track_id in object_info:
                        obj_pose = np.eye(4)
                        obj_pose[:3, :3] = quaternion_to_matrix_numpy(t_info[4:8])
                        obj_pose[:3, 3] = t_info[1:4]
                        world2local = np.linalg.inv(obj_pose)

                        pts_h = np.concatenate([points_xyz, np.ones((len(points_xyz), 1))], axis=-1)
                        pts_local = (pts_h @ world2local.T)[:, :3]

                        length = object_info[track_id]['length']
                        width = object_info[track_id]['width']
                        height = object_info[track_id]['height']
                        bbox = [[-length / 2, -width / 2, -height / 2],
                                [length / 2, width / 2, height / 2]]
                        corners = bbox_to_corner3d(bbox)
                        in_bbox = inbbox_points(pts_local, corners)
                        points_obj_mask |= in_bbox
                        points_xyz_dict[f'obj_{track_id:03d}'].append(pts_local[in_bbox])
                        points_rgb_dict[f'obj_{track_id:03d}'].append(points_rgb[in_bbox])

                points_xyz_dict['bkgd'].append(points_xyz[~points_obj_mask])
                points_rgb_dict['bkgd'].append(points_rgb[~points_obj_mask])

    # Save point clouds
    initial_num_obj = 20000
    for k, v_list in points_xyz_dict.items():
        if not v_list:
            continue
        xyz = np.concatenate(v_list, axis=0).astype(np.float32)
        rgb = np.concatenate(points_rgb_dict[k], axis=0).astype(np.float32)

        if k == 'bkgd':
            # Voxel downsample + outlier removal
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            pcd = pcd.voxel_down_sample(voxel_size=0.15)
            pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.5)
            xyz = np.asarray(pcd.points).astype(np.float32)
            rgb = np.asarray(pcd.colors).astype(np.float32)
            storePly(os.path.join(pointcloud_dir, 'points3D_lidar.ply'), xyz, rgb)
            storePly(os.path.join(pointcloud_dir, 'points3D_bkgd.ply'), xyz, rgb)
        else:
            if len(xyz) > initial_num_obj:
                idx = np.random.choice(len(xyz), initial_num_obj, replace=False)
                xyz = xyz[idx]
                rgb = rgb[idx]
            storePly(os.path.join(pointcloud_dir, f'points3D_{k}.ply'), xyz, rgb)

    return os.path.join(pointcloud_dir, 'points3D_bkgd.ply')


# ─── Main reader ───────────────────────────────────────────────────────

def readCarRoadInfo(path, images='images', split_train=-1, split_test=-1, **kwargs):
    """
    Read car-road roadside dataset.

    Parameters
    ----------
    path : str
        Path to the preprocessed scene directory.
    """
    # Detect number of cameras
    intrinsics_dir = os.path.join(path, 'intrinsics')
    num_cameras_total = len(glob(os.path.join(intrinsics_dir, '*.txt')))
    cameras = cfg.data.get('cameras', list(range(num_cameras_total)))

    selected_frames = cfg.data.get('selected_frames', None)
    if cfg.debug:
        selected_frames = [0, 0]

    # Load calibration
    intrinsics = _load_intrinsics(intrinsics_dir, num_cameras_total)
    cam_to_worlds = _load_extrinsics(os.path.join(path, 'extrinsics'), num_cameras_total)

    # For static roadside cameras, extrinsic = cam-to-world, ego_pose = identity
    # The camera-to-world is the same for all frames (static cameras)
    identity_pose = np.eye(4)

    # Discover frames from images
    image_dir = os.path.join(path, images)
    image_filenames_all = sorted(glob(os.path.join(image_dir, '*.png')) +
                                  glob(os.path.join(image_dir, '*.jpg')))
    if not image_filenames_all:
        raise FileNotFoundError(f"No images found in {image_dir}")

    all_frames = sorted(set(_image_filename_to_frame(os.path.basename(f)) for f in image_filenames_all))
    num_frames_total = len(all_frames)

    if selected_frames is None:
        start_frame = 0
        end_frame = num_frames_total - 1
        selected_frames = [start_frame, end_frame]
    else:
        start_frame, end_frame = selected_frames[0], selected_frames[1]
    num_frames = end_frame - start_frame + 1

    # Load timestamps
    timestamps_path = os.path.join(path, 'timestamps.json')
    if os.path.exists(timestamps_path):
        with open(timestamps_path, 'r') as f:
            timestamps = json.load(f)
    else:
        # Generate synthetic timestamps if not available
        print("No timestamps.json found, generating synthetic timestamps")
        timestamps = {'FRAME': {}, 'CAMERAS': {}}
        for i, frame in enumerate(all_frames):
            timestamps['FRAME'][f'{frame:06d}'] = float(i)
        for cam in cameras:
            timestamps['CAMERAS'][str(cam)] = {}
            for i, frame in enumerate(all_frames):
                timestamps['CAMERAS'][str(cam)][f'{frame:06d}'] = float(i)

    # Frame timestamps
    frames_timestamps = []
    for frame in range(start_frame, end_frame + 1):
        key = f'{frame:06d}'
        if key in timestamps.get('FRAME', {}):
            frames_timestamps.append(timestamps['FRAME'][key])
        else:
            frames_timestamps.append(float(frame))

    # Train/test split
    train_frames_set, test_frames_set = get_val_frames(
        num_frames,
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    # ─── Load or build point cloud if in train mode ─────────────────
    bkgd_ply_path = os.path.join(cfg.model_path, 'input_ply/points3D_bkgd.ply')

    # Try to load existing pcd from another experiment
    if cfg.data.get('load_pcd_from', False) and cfg.mode == 'train':
        import shutil
        load_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'input_ply')
        save_dir = os.path.join(cfg.model_path, 'input_ply')
        if os.path.exists(load_dir):
            os.system(f'rm -rf {save_dir}')
            shutil.copytree(load_dir, save_dir)

    build_pointcloud = (cfg.mode == 'train') and (
        not os.path.exists(bkgd_ply_path) or cfg.data.get('regenerate_pcd', False)
    )

    # ─── Load tracking info ─────────────────────────────────────────
    tracklets, object_info = _load_track_info(
        path, selected_frames, cameras, num_frames_total
    )

    # Build image list, poses, timestamps
    frames_list = []
    frames_idx_list = []
    cams_list = []
    image_filenames = []
    ixts = []
    exts = []
    poses_list = []
    c2ws_list = []
    cams_timestamps = []

    for img_path in image_filenames_all:
        basename = os.path.basename(img_path)
        frame = _image_filename_to_frame(basename)
        cam = _image_filename_to_cam(basename)

        if frame < start_frame or frame > end_frame:
            continue
        if cam not in cameras:
            continue

        ixt = intrinsics[cam]
        c2w = cam_to_worlds[cam].copy()  # Static camera: same c2w for all frames

        frames_list.append(frame)
        frames_idx_list.append(frame - start_frame)
        cams_list.append(cam)
        image_filenames.append(img_path)

        ixts.append(ixt)
        exts.append(c2w)  # cam-to-world (used as "extrinsic" in Street Gaussians)
        poses_list.append(identity_pose.copy())  # ego_pose = identity
        c2ws_list.append(c2w)

        # Camera timestamp
        cam_key = str(cam)
        frame_key = f'{frame:06d}'
        if 'CAMERAS' in timestamps and cam_key in timestamps['CAMERAS']:
            if frame_key in timestamps['CAMERAS'][cam_key]:
                cams_timestamps.append(timestamps['CAMERAS'][cam_key][frame_key])
            else:
                cams_timestamps.append(float(frame))
        else:
            cams_timestamps.append(float(frame))

    # Normalize timestamps
    timestamp_offset = min(cams_timestamps + frames_timestamps)
    cams_timestamps = np.array(cams_timestamps) - timestamp_offset
    frames_timestamps = np.array(frames_timestamps) - timestamp_offset
    min_ts = min(cams_timestamps.min(), frames_timestamps.min())
    max_ts = max(cams_timestamps.max(), frames_timestamps.max())

    # Add start/end timestamps to object info
    for track_id in object_info:
        obj_start = object_info[track_id]['start_frame']
        obj_end = object_info[track_id]['end_frame']
        obj_start_key = f'{obj_start:06d}'
        obj_end_key = f'{obj_end:06d}'
        start_ts = timestamps.get('FRAME', {}).get(obj_start_key, float(obj_start)) - timestamp_offset - 0.1
        end_ts = timestamps.get('FRAME', {}).get(obj_end_key, float(obj_end)) - timestamp_offset + 0.1
        object_info[track_id]['start_timestamp'] = max(start_ts, min_ts)
        object_info[track_id]['end_timestamp'] = min(end_ts, max_ts)

    # Build point cloud
    if build_pointcloud:
        bkgd_ply_path = _build_pointcloud(
            path, selected_frames, cameras, num_frames, intrinsics,
            cam_to_worlds, image_filenames, tracklets, object_info,
        )

    # ─── Build scene metadata ──────────────────────────────────────
    scene_metadata = {
        'obj_tracklets': tracklets,
        'tracklet_timestamps': frames_timestamps,
        'obj_meta': object_info,
        'num_images': len(image_filenames),
        'num_cams': len(cameras),
        'num_frames': num_frames,
    }

    camera_timestamps = {}
    for cam in cameras:
        camera_timestamps[cam] = {'train_timestamps': [], 'test_timestamps': []}

    # ─── Build CameraInfo list ─────────────────────────────────────
    # Guidance directories
    dynamic_mask_dir = os.path.join(path, 'dynamic_mask')
    sky_mask_dir = os.path.join(path, 'sky_mask')
    lidar_depth_dir = os.path.join(path, 'lidar_depth')
    load_dynamic_mask = os.path.exists(dynamic_mask_dir)
    load_sky_mask = (cfg.mode == 'train') and os.path.exists(sky_mask_dir)
    load_lidar_depth = (cfg.mode == 'train') and os.path.exists(lidar_depth_dir)

    cam_infos = []
    for i in tqdm(range(len(image_filenames)), desc="Loading cameras"):
        image_path = image_filenames[i]
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path)

        width, height = image.size
        ixt = ixts[i]
        fx, fy = ixt[0, 0], ixt[1, 1]
        FovY = focal2fov(fx, height)
        FovX = focal2fov(fy, width)

        c2w = c2ws_list[i]
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T  # Street Gaussians convention: store transposed
        T = w2c[:3, 3]
        K = ixt.copy().astype(np.float32)

        metadata = {
            'frame': frames_list[i],
            'cam': cams_list[i],
            'frame_idx': frames_idx_list[i],
            'ego_pose': poses_list[i],
            'extrinsic': c2w.astype(np.float32),  # cam-to-world
            'timestamp': cams_timestamps[i],
            'is_val': frames_idx_list[i] in test_frames_set,
        }

        if metadata['is_val']:
            camera_timestamps[cams_list[i]]['test_timestamps'].append(cams_timestamps[i])
        else:
            camera_timestamps[cams_list[i]]['train_timestamps'].append(cams_timestamps[i])

        # Load guidance signals
        guidance = {}

        if load_dynamic_mask:
            dm_path = os.path.join(dynamic_mask_dir, f'{image_name}.png')
            if os.path.exists(dm_path):
                obj_bound = (cv2.imread(dm_path)[..., 0]) > 0
                guidance['obj_bound'] = Image.fromarray(obj_bound)
            else:
                # Generate from tracklets
                obj_bound = _generate_obj_bound(
                    tracklets[frames_idx_list[i]], object_info,
                    ixt, c2w, height, width
                )
                guidance['obj_bound'] = Image.fromarray(obj_bound)

        if load_lidar_depth:
            depth_path = os.path.join(lidar_depth_dir, f'{image_name}.npy')
            if os.path.exists(depth_path):
                depth = np.load(depth_path, allow_pickle=True)
                if isinstance(depth, np.ndarray) and depth.dtype == object:
                    depth = dict(depth.item())
                    mask = depth['mask']
                    value = depth['value']
                    depth_arr = np.zeros_like(mask).astype(np.float32)
                    depth_arr[mask] = value
                    guidance['lidar_depth'] = depth_arr
                else:
                    guidance['lidar_depth'] = depth.astype(np.float32)

        if load_sky_mask:
            sky_path = os.path.join(sky_mask_dir, f'{image_name}.png')
            if os.path.exists(sky_path):
                sky_mask = (cv2.imread(sky_path)[..., 0]) > 0
                guidance['sky_mask'] = Image.fromarray(sky_mask)

        cam_info = CameraInfo(
            uid=i, R=R, T=T, FovY=FovY, FovX=FovX, K=K,
            image=image, image_path=image_path, image_name=image_name,
            width=width, height=height,
            metadata=metadata,
            guidance=guidance,
        )
        cam_infos.append(cam_info)

    # Sort timestamps
    for cam in cameras:
        camera_timestamps[cam]['train_timestamps'] = sorted(camera_timestamps[cam]['train_timestamps'])
        camera_timestamps[cam]['test_timestamps'] = sorted(camera_timestamps[cam]['test_timestamps'])
    scene_metadata['camera_timestamps'] = camera_timestamps

    # Split train/test
    train_cam_infos = [c for c in cam_infos if not c.metadata['is_val']]
    test_cam_infos = [c for c in cam_infos if c.metadata['is_val']]

    print(f"Train cameras: {len(train_cam_infos)}, Test cameras: {len(test_cam_infos)}")

    # ─── Compute scene normalization ────────────────────────────────
    if cfg.mode == 'novel_view':
        nerf_normalization = getNerfppNorm(cam_infos)
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos if train_cam_infos else cam_infos)

    nerf_normalization['radius'] = max(nerf_normalization['radius'], 10)
    if cfg.data.get('extent', False):
        nerf_normalization['radius'] = cfg.data.extent
    cfg.data.extent = float(nerf_normalization['radius'])

    scene_metadata['scene_center'] = nerf_normalization['center']
    scene_metadata['scene_radius'] = nerf_normalization['radius']
    print(f"Scene extent: {nerf_normalization['radius']}")

    # Sphere normalization from point cloud
    lidar_ply_path = os.path.join(cfg.model_path, 'input_ply/points3D_lidar.ply')
    if os.path.exists(lidar_ply_path):
        sphere_pcd = fetchPly(lidar_ply_path)
    elif os.path.exists(bkgd_ply_path):
        sphere_pcd = fetchPly(bkgd_ply_path)
    else:
        # Fallback: use camera centers
        cam_centers = []
        for ci in cam_infos:
            from lib.utils.graphics_utils import getWorld2View2
            W2C = getWorld2View2(ci.R, ci.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3])
        cam_centers = np.array(cam_centers)
        sphere_pcd = BasicPointCloud(
            points=cam_centers,
            colors=np.ones_like(cam_centers),
            normals=np.zeros_like(cam_centers)
        )

    sphere_norm = get_Sphere_Norm(sphere_pcd.points)
    scene_metadata['sphere_center'] = sphere_norm['center']
    scene_metadata['sphere_radius'] = sphere_norm['radius']
    print(f"Sphere extent: {sphere_norm['radius']}")

    # Load point cloud
    if cfg.mode == 'train' and os.path.exists(bkgd_ply_path):
        pcd = fetchPly(bkgd_ply_path)
    else:
        pcd = None
        bkgd_ply_path = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=bkgd_ply_path,
        metadata=scene_metadata,
    )

    return scene_info


def _generate_obj_bound(frame_tracklets, object_info, ixt, c2w, height, width):
    """Generate 2D object bounding mask from 3D tracklets."""
    obj_bound = np.zeros((height, width), dtype=np.uint8)

    w2c = np.linalg.inv(c2w)

    for t_info in frame_tracklets:
        track_id = int(t_info[0])
        if track_id < 0 or track_id not in object_info:
            continue

        obj_pose = np.eye(4)
        obj_pose[:3, :3] = quaternion_to_matrix_numpy(t_info[4:8])
        obj_pose[:3, 3] = t_info[1:4]

        length = object_info[track_id]['length']
        width_box = object_info[track_id]['width']
        height_box = object_info[track_id]['height']
        bbox = np.array([[-length, -width_box, -height_box],
                         [length, width_box, height_box]]) * 0.5
        corners_local = bbox_to_corner3d(bbox)
        corners_h = np.concatenate([corners_local, np.ones_like(corners_local[..., :1])], axis=-1)
        corners_world = corners_h @ obj_pose.T

        try:
            mask = get_bound_2d_mask(
                corners_3d=corners_world[..., :3],
                K=ixt,
                pose=w2c,
                H=height, W=width
            )
            obj_bound = np.logical_or(obj_bound, mask)
        except Exception:
            pass

    return obj_bound.astype(bool)
