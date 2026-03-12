#!/usr/bin/env python3
"""
Inspect a car-road scene directory and report its structure.
Run this FIRST before prepare_carroad_data.py to understand the data layout.

Usage:
    python script/carroad/inspect_scene.py /mnt/car_road_data_TianJin/053_car0402_road0402_t31
"""

import os
import sys
import json
import glob


def inspect_dir(path, prefix="", depth=0, max_depth=4, max_files=10):
    """Recursively inspect directory structure."""
    if depth > max_depth:
        return

    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        print(f"{prefix}[permission denied]")
        return

    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(path, e))]

    for d in dirs:
        full = os.path.join(path, d)
        n_children = len(os.listdir(full)) if os.access(full, os.R_OK) else 0
        print(f"{prefix}{d}/  ({n_children} items)")
        inspect_dir(full, prefix + "  ", depth + 1, max_depth, max_files)

    if files:
        shown = files[:max_files]
        for f in shown:
            fpath = os.path.join(path, f)
            size = os.path.getsize(fpath)
            size_str = f"{size/1024:.1f}K" if size < 1024*1024 else f"{size/1024/1024:.1f}M"
            print(f"{prefix}{f}  ({size_str})")
        if len(files) > max_files:
            print(f"{prefix}... and {len(files) - max_files} more files")


def inspect_calib(scene_dir):
    """Find and inspect calib.json."""
    candidates = [
        os.path.join(scene_dir, 'calib.json'),
        os.path.join(scene_dir, 'road', 'calib.json'),
    ]
    # Also search recursively
    found = glob.glob(os.path.join(scene_dir, '**/calib.json'), recursive=True)
    candidates.extend(found)

    for path in candidates:
        if os.path.exists(path):
            print(f"\n{'='*60}")
            print(f"CALIB found: {path}")
            print(f"{'='*60}")
            with open(path, 'r') as f:
                calib = json.load(f)

            print(f"Top-level keys: {list(calib.keys())}")

            if 'imgSize' in calib:
                print(f"Image sizes: {calib['imgSize']}")

            if 'camera' in calib:
                print(f"\nCameras ({len(calib['camera'])} total):")
                for cam_key, cam_data in sorted(calib['camera'].items()):
                    is_fish = cam_data.get('isFish', -1)
                    cam_type = "FISHEYE" if is_fish == 1 else "PINHOLE" if is_fish == 0 else "UNKNOWN"
                    print(f"  {cam_key}: {cam_type}")

            if 'lidar' in calib:
                print(f"\nLiDARs ({len(calib['lidar'])} total):")
                for lid_key in sorted(calib['lidar'].keys()):
                    print(f"  lidar_{lid_key}: {calib['lidar'][lid_key].get('name', 'N/A')}")

            return calib

    print("\nWARNING: calib.json NOT FOUND!")
    return None


def inspect_images(scene_dir):
    """Find and count images."""
    print(f"\n{'='*60}")
    print("IMAGES")
    print(f"{'='*60}")

    # Check road/cameras/
    road_cam_dir = os.path.join(scene_dir, 'road', 'cameras')
    if os.path.exists(road_cam_dir):
        print(f"Road cameras dir: {road_cam_dir}")
        for cam in sorted(os.listdir(road_cam_dir)):
            cam_dir = os.path.join(road_cam_dir, cam)
            if os.path.isdir(cam_dir):
                imgs = glob.glob(os.path.join(cam_dir, '*.png')) + glob.glob(os.path.join(cam_dir, '*.jpg'))
                if imgs:
                    sample = os.path.basename(sorted(imgs)[0])
                    last = os.path.basename(sorted(imgs)[-1])
                    print(f"  {cam}: {len(imgs)} images, first={sample}, last={last}")

    # Check img/ (self_Dataset format)
    img_dir = os.path.join(scene_dir, 'img')
    if os.path.exists(img_dir):
        print(f"\nAlternative img dir: {img_dir}")
        for cam in sorted(os.listdir(img_dir)):
            cam_dir = os.path.join(img_dir, cam)
            if os.path.isdir(cam_dir):
                imgs = glob.glob(os.path.join(cam_dir, '*.png'))
                if imgs:
                    print(f"  {cam}: {len(imgs)} images")


def inspect_lidar(scene_dir):
    """Find LiDAR point clouds."""
    print(f"\n{'='*60}")
    print("LIDAR")
    print(f"{'='*60}")

    lidar_dirs = [
        os.path.join(scene_dir, 'road', 'lidar', 'merged_pcd'),
        os.path.join(scene_dir, 'road', 'lidar'),
    ]

    for d in lidar_dirs:
        if os.path.exists(d):
            pcds = sorted(glob.glob(os.path.join(d, '*.pcd')))
            if pcds:
                # Check first PCD size
                size_mb = os.path.getsize(pcds[0]) / 1024 / 1024
                first = os.path.basename(pcds[0])
                last = os.path.basename(pcds[-1])
                print(f"  {d}: {len(pcds)} files, first={first}, last={last}, size~{size_mb:.1f}MB")

    # Check individual lidar sensors
    lidar_base = os.path.join(scene_dir, 'road', 'lidar')
    if os.path.exists(lidar_base):
        for sub in sorted(os.listdir(lidar_base)):
            sub_path = os.path.join(lidar_base, sub)
            if os.path.isdir(sub_path) and sub != 'merged_pcd':
                pcds = glob.glob(os.path.join(sub_path, '*.pcd'))
                print(f"  {sub}: {len(pcds)} files")

    # Single PCD in root
    root_pcds = glob.glob(os.path.join(scene_dir, '*.pcd'))
    if root_pcds:
        for p in root_pcds:
            size_mb = os.path.getsize(p) / 1024 / 1024
            print(f"  Root PCD: {os.path.basename(p)} ({size_mb:.1f}MB)")


def inspect_labels(scene_dir):
    """Find and inspect labels."""
    print(f"\n{'='*60}")
    print("LABELS")
    print(f"{'='*60}")

    label_dirs = [
        os.path.join(scene_dir, 'road_labels'),
        os.path.join(scene_dir, 'road_labels', 'interpolation_labels'),
        os.path.join(scene_dir, 'road_labels', 'ori_labels'),
    ]

    for d in label_dirs:
        if os.path.exists(d):
            jsons = sorted(glob.glob(os.path.join(d, '*.json')))
            txts = sorted(glob.glob(os.path.join(d, '*.txt')))
            if jsons:
                print(f"  {d}: {len(jsons)} JSON files")
                # Inspect first label
                with open(jsons[0], 'r') as f:
                    sample = json.load(f)
                if isinstance(sample, dict):
                    print(f"    Keys: {list(sample.keys())}")
                    if 'object' in sample:
                        n_obj = len(sample['object'])
                        print(f"    Objects in first frame: {n_obj}")
                        if n_obj > 0:
                            obj = sample['object'][0]
                            print(f"    Sample object keys: {list(obj.keys())}")
                            print(f"    Sample: id={obj.get('id')}, label={obj.get('label')}, "
                                  f"pos=({obj.get('x',0):.1f}, {obj.get('y',0):.1f}, {obj.get('z',0):.1f})")
                    if 'timestamp' in sample:
                        print(f"    First timestamp: {sample['timestamp']}")
                # Check last file
                with open(jsons[-1], 'r') as f:
                    last_sample = json.load(f)
                if isinstance(last_sample, dict) and 'timestamp' in last_sample:
                    print(f"    Last timestamp: {last_sample['timestamp']}")

            if txts:
                print(f"  {d}: {len(txts)} TXT files")


def inspect_sync(scene_dir):
    """Check sync_info.txt."""
    sync_path = os.path.join(scene_dir, 'sync_info.txt')
    if os.path.exists(sync_path):
        print(f"\n{'='*60}")
        print("SYNC INFO")
        print(f"{'='*60}")
        with open(sync_path, 'r') as f:
            lines = f.readlines()
        print(f"  {len(lines)} lines")
        for line in lines[:5]:
            print(f"  {line.strip()}")
        if len(lines) > 5:
            print(f"  ...")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_scene.py <scene_dir>")
        print("Example: python inspect_scene.py /mnt/car_road_data_TianJin/053_car0402_road0402_t31")
        sys.exit(1)

    scene_dir = sys.argv[1]
    print(f"Inspecting: {scene_dir}")
    print(f"Exists: {os.path.exists(scene_dir)}")

    if not os.path.exists(scene_dir):
        print("ERROR: Directory does not exist!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("DIRECTORY STRUCTURE (top 3 levels)")
    print(f"{'='*60}")
    inspect_dir(scene_dir, max_depth=3)

    calib = inspect_calib(scene_dir)
    inspect_images(scene_dir)
    inspect_lidar(scene_dir)
    inspect_labels(scene_dir)
    inspect_sync(scene_dir)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY & RECOMMENDED COMMAND")
    print(f"{'='*60}")

    if calib and 'camera' in calib:
        pinhole_cams = [k for k, v in calib['camera'].items() if v.get('isFish', 1) == 0]
        print(f"Pinhole cameras in calib: {pinhole_cams}")
        print(f"\nRecommended pinhole-to-cam mapping (verify with image folders):")
        print(f"  pinhole0 -> cam{pinhole_cams[0] if len(pinhole_cams) > 0 else '?'}")
        print(f"  pinhole1 -> cam{pinhole_cams[1] if len(pinhole_cams) > 1 else '?'}")
        print(f"  pinhole2 -> cam{pinhole_cams[2] if len(pinhole_cams) > 2 else '?'}")
        print(f"  pinhole3 -> cam{pinhole_cams[3] if len(pinhole_cams) > 3 else '?'}")

    print(f"\nSuggested preprocessing command:")
    print(f"  python script/carroad/prepare_carroad_data.py \\")
    print(f"      --raw_dir {scene_dir} \\")
    print(f"      --output_dir ./data/carroad/scene_053 \\")
    print(f"      --cameras pinhole0 pinhole1 pinhole2 pinhole3 \\")
    print(f"      --frame_step 3")


if __name__ == '__main__':
    main()
