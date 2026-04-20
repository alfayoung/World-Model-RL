#!/usr/bin/env python3
"""Build sparse/0/points3D.txt for a panorama dataset by replaying simulator depth.

This is the missing "consistent sparse init from simulator depth" step referenced by
docs/2DGS_PANORAMA_DOCUMENTATION.md for image_collect.py panorama datasets.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from robosuite.utils.camera_utils import get_real_depth_map
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.image_collect import get_libero_env, set_camera_pose, set_seed_everywhere


MUJOCO_TO_OPENCV = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--camera_name", type=str, default="frontview")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_points", type=int, default=250000)
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--depth_max", type=float, default=1e6)
    parser.add_argument(
        "--workspace_center",
        type=float,
        nargs=3,
        default=[-0.1, 0.0, 0.8],
        metavar=("X", "Y", "Z"),
        help="World-space center for sparse init cropping.",
    )
    parser.add_argument(
        "--workspace_radius_xy",
        type=float,
        default=1.25,
        help="Keep only points within this XY radius of --workspace_center.",
    )
    parser.add_argument(
        "--workspace_z_min",
        type=float,
        default=0.65,
        help="Minimum world Z kept after backprojection.",
    )
    parser.add_argument(
        "--workspace_z_max",
        type=float,
        default=1.60,
        help="Maximum world Z kept after backprojection.",
    )
    parser.add_argument(
        "--disable_workspace_crop",
        action="store_true",
        help="Disable default tabletop workspace filtering.",
    )
    parser.add_argument("--verify_rgb", action="store_true")
    parser.add_argument("--verify_first_n", type=int, default=1)
    return parser.parse_args()


def load_intrinsics(cameras_txt: Path):
    line = None
    for raw in cameras_txt.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw and not raw.startswith("#"):
            line = raw
            break
    if line is None:
        raise RuntimeError(f"No camera entry found in {cameras_txt}")

    parts = line.split()
    if len(parts) < 8 or parts[1] != "PINHOLE":
        raise RuntimeError(f"Unsupported camera line: {line}")

    width = int(parts[2])
    height = int(parts[3])
    fx = float(parts[4])
    fy = float(parts[5])
    cx = float(parts[6])
    cy = float(parts[7])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, width, height


def camera_pose_from_world_to_cam(T_world_to_cam: np.ndarray):
    R_wc = T_world_to_cam[:3, :3]
    t_wc = T_world_to_cam[:3, 3]
    R_cw = R_wc.T
    cam_pos = -R_cw @ t_wc
    cam_mat = R_cw @ MUJOCO_TO_OPENCV
    q_xyzw = Rotation.from_matrix(cam_mat).as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    return cam_pos, q_wxyz


def reservoir_keep_topk(keys, xyz, rgb, max_points):
    if len(keys) <= max_points:
        return keys, xyz, rgb
    idx = np.argpartition(keys, -max_points)[-max_points:]
    return keys[idx], xyz[idx], rgb[idx]


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    input_dir = dataset_dir / "input"
    sparse_dir = dataset_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    poses_path = input_dir / "camera_poses.json"
    cameras_path = input_dir / "cameras.txt"
    images_path = input_dir / "images.txt"
    points3d_path = sparse_dir / "points3D.txt"
    summary_path = dataset_dir / "sparse_init_summary.json"

    if not poses_path.exists():
        raise FileNotFoundError(f"Missing camera poses: {poses_path}")
    if not cameras_path.exists():
        raise FileNotFoundError(f"Missing intrinsics: {cameras_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"Missing images.txt: {images_path}")

    poses = json.loads(poses_path.read_text(encoding="utf-8"))
    K, width, height = load_intrinsics(cameras_path)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Keep sparse/0 camera metadata aligned with the input export.
    (sparse_dir / "cameras.txt").write_text(cameras_path.read_text(encoding="utf-8"), encoding="utf-8")
    (sparse_dir / "images.txt").write_text(images_path.read_text(encoding="utf-8"), encoding="utf-8")

    set_seed_everywhere(args.seed)
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    env, _ = get_libero_env(task, "rt-1", resolution=height)
    env.reset()
    for _ in range(10):
        env.step([0, 0, 0, 0, 0, 0, -1])

    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    rng = np.random.default_rng(args.seed)
    sample_keys = np.empty((0,), dtype=np.float64)
    sample_xyz = np.empty((0, 3), dtype=np.float32)
    sample_rgb = np.empty((0, 3), dtype=np.uint8)
    workspace_center = np.asarray(args.workspace_center, dtype=np.float32)

    total_valid_points = 0
    total_points_after_crop = 0
    verified = 0

    for row in poses:
        image_name = row["filename"]
        image_path = input_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing panorama image: {image_path}")

        T_world_to_cam = np.asarray(row["T_world_to_cam_opencv_4x4"], dtype=np.float64)
        cam_pos, cam_quat = camera_pose_from_world_to_cam(T_world_to_cam)
        set_camera_pose(env, args.camera_name, cam_pos, cam_quat)

        rgb_render, depth_norm = env.sim.render(
            camera_name=args.camera_name, height=height, width=width, depth=True
        )
        # Match the same vertical-only image convention used when Step 1 saves PNGs.
        rgb = rgb_render[::-1]
        depth = get_real_depth_map(env.sim, depth_norm[::-1]).astype(np.float32)

        if args.verify_rgb and verified < args.verify_first_n:
            saved_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if saved_bgr is None:
                raise RuntimeError(f"Failed to read saved panorama image: {image_path}")
            saved_rgb = cv2.cvtColor(saved_bgr, cv2.COLOR_BGR2RGB)
            if not np.array_equal(saved_rgb, rgb):
                diff = np.abs(saved_rgb.astype(np.int16) - rgb.astype(np.int16)).mean()
                raise RuntimeError(f"RGB verification failed for {image_name}: mean abs diff {diff}")
            verified += 1

        valid = np.isfinite(depth)
        valid &= depth > args.depth_min
        valid &= depth < args.depth_max
        if not np.any(valid):
            continue

        z = depth[valid]
        x = (u[valid] - cx) / fx * z
        y = (v[valid] - cy) / fy * z
        pts_cam = np.stack([x, y, z], axis=1).astype(np.float32)

        R_wc = T_world_to_cam[:3, :3]
        t_wc = T_world_to_cam[:3, 3]
        R_cw = R_wc.T
        t_cw = (-R_cw @ t_wc).astype(np.float32)
        pts_world = (pts_cam @ R_cw.T.astype(np.float32)) + t_cw[None, :]
        cols = rgb[valid]
        total_valid_points += len(pts_world)

        if not args.disable_workspace_crop:
            xy_delta = pts_world[:, :2] - workspace_center[:2]
            within_workspace = np.sum(xy_delta * xy_delta, axis=1) <= args.workspace_radius_xy ** 2
            within_workspace &= pts_world[:, 2] >= args.workspace_z_min
            within_workspace &= pts_world[:, 2] <= args.workspace_z_max
            pts_world = pts_world[within_workspace]
            cols = cols[within_workspace]

        if len(pts_world) == 0:
            continue

        keys = rng.random(len(pts_world))
        total_points_after_crop += len(pts_world)

        sample_keys = np.concatenate([sample_keys, keys], axis=0)
        sample_xyz = np.concatenate([sample_xyz, pts_world], axis=0)
        sample_rgb = np.concatenate([sample_rgb, cols], axis=0)
        sample_keys, sample_xyz, sample_rgb = reservoir_keep_topk(
            sample_keys, sample_xyz, sample_rgb, args.max_points
        )

    with points3d_path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for idx, (xyz, rgb) in enumerate(zip(sample_xyz, sample_rgb), start=1):
            f.write(
                f"{idx} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 1.0\n"
            )

    summary = {
        "dataset_dir": str(dataset_dir),
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "camera_name": args.camera_name,
        "image_size": [width, height],
        "num_views": len(poses),
        "total_valid_points_before_crop": int(total_valid_points),
        "total_valid_points_after_crop": int(total_points_after_crop),
        "num_points_written": int(len(sample_xyz)),
        "max_points": int(args.max_points),
        "depth_min": float(args.depth_min),
        "depth_max": float(args.depth_max),
        "workspace_crop_enabled": bool(not args.disable_workspace_crop),
        "workspace_center": workspace_center.tolist(),
        "workspace_radius_xy": float(args.workspace_radius_xy),
        "workspace_z_min": float(args.workspace_z_min),
        "workspace_z_max": float(args.workspace_z_max),
        "verify_rgb": bool(args.verify_rgb),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote sparse init to: {points3d_path}")
    print(
        f"Views: {len(poses)} | sampled points: {len(sample_xyz)} / {total_points_after_crop} valid points after crop"
    )


if __name__ == "__main__":
    main()
