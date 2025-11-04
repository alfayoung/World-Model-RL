"""
Interactive mode:
python image_collect.py --task_suite_name libero_spatial --task_id 0 --mode interactive

Panoramic sweep mode (automated round view from current camera position):
python image_collect.py --task_suite_name libero_spatial --task_id 0 --mode panoramic --camera_name agentview --panoramic_yaw_angles 8 --output_dir ./panoramic_views

Panoramic sweep with height variation (multiple camera heights):
python image_collect.py --task_suite_name libero_spatial --task_id 0 --mode panoramic --camera_name agentview --panoramic_yaw_angles 8 --use_height_variation True --height_variations [-0.1,0.0,0.1] --output_dir ./panoramic_views

Multi-angle capture mode:
python image_collect.py --task_suite_name libero_spatial --task_id 0 --mode capture --num_angles 8 --output_dir ./captured_views
"""

import argparse
import os
import random
import sys
import termios
import tty
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import draccus
import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def euler2quat(euler):
    """
    Converts euler angles to quaternion.

    IMPORTANT: This function expects (roll, pitch, yaw) parameters but maps them
    according to aviation/camera conventions for LOCAL frame rotations:
    - roll parameter  → rotation around Z axis (camera forward/backward axis)
    - pitch parameter → rotation around X axis (camera right/left axis)
    - yaw parameter   → rotation around Y axis (camera up/down axis)

    Args:
        euler (np.array): (roll, pitch, yaw) euler angles in radians
                         roll = rotation around Z axis (aviation: roll around forward)
                         pitch = rotation around X axis (aviation: pitch around right)
                         yaw = rotation around Y axis (aviation: yaw around up/down)

    Returns:
        np.array: (w,x,y,z) quaternion in MuJoCo convention
    """
    roll, pitch, yaw = euler

    # Map parameters to actual rotations:
    # roll → Z, pitch → X, yaw → Y
    angle_x = pitch
    angle_y = yaw
    angle_z = roll

    # Compute quaternion for rotations around X, Y, Z axes
    cx, cy, cz = np.cos(angle_x / 2), np.cos(angle_y / 2), np.cos(angle_z / 2)
    sx, sy, sz = np.sin(angle_x / 2), np.sin(angle_y / 2), np.sin(angle_z / 2)

    # For independent axis rotations combined
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz

    return np.array([w, x, y, z])


def quat_multiply(quat1, quat2):
    """
    Multiplies two quaternions (quat1 * quat2).

    Args:
        quat1 (np.array): (w,x,y,z) quaternion
        quat2 (np.array): (w,x,y,z) quaternion

    Returns:
        np.array: (w,x,y,z) quaternion
    """
    w1, x1, y1, z1 = quat1
    w2, x2, y2, z2 = quat2

    # Compute quaternion multiplication (Hamilton product)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.array([w, x, y, z])


def quat_to_rot_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    # Standard quaternion-to-rotation conversion
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rot = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )
    return rot


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    # Available camera names: ('frontview', 'birdview', 'agentview', 'sideview', 'galleryview', 'robot0_robotview', 'robot0_eye_in_hand')
    camera_names = [
        "agentview",
        "robot0_eye_in_hand",
        "frontview",
        "sideview",
    ]  # add frontview for recording
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": camera_names,
        "render_camera": "agentview",
        "camera_depths": True,  # False
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        0
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def set_seed_everywhere(seed: int):
    """Sets the random seed for numpy and other libraries for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


class CaptureMode(str, Enum):
    INTERACTIVE = "interactive"
    PANORAMIC = "panoramic"


@dataclass
class InteractiveConfig:
    # fmt: off
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL         # Task suite to load environment
    task_id: int = 0                                        # Task ID to load
    env_img_res: int = 512                                  # Resolution for environment images
    camera_name: str = "agentview"                          # Initial camera to control
    output_image_path: str = "camera_view.png"              # Path to save rendered image
    pos_step: float = 0.02                                  # Step size for position control
    rot_step_deg: float = 5.0                               # Step size for rotation control (in degrees)
    seed: int = 7                                           # Random Seed (for reproducibility)

    # Multi-angle capture settings
    mode: str = CaptureMode.INTERACTIVE                     # Mode: 'interactive', 'capture', or 'panoramic'
    center_point: Optional[List[float]] = None              # Center point to orbit around [x, y, z]. If None, uses table center
    output_dir: str = "./captured_views"                    # Directory to save captured images

    # Panoramic sweep settings (rotates camera in place)
    panoramic_yaw_angles: int = 18                           # Number of yaw angles for panoramic sweep (horizontal rotation)
    panoramic_pitch_angles: List[float] = field(default_factory=lambda: [0])  # Pitch angles in degrees for panoramic sweep
    height_variations: List[float] = field(default_factory=lambda: [0.0])  # Height offsets in meters from base camera height
    use_height_variation: bool = True                       # Enable multiple camera heights

    # SfM-friendly capture settings
    radius_variation: float = 0.3                           # Radius variation as fraction (0.3 = ±30% variation)
    multiple_radii: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])  # Multiple orbital radii as multipliers
    look_at_offset_std: float = 0.1                         # Standard deviation for look-at point jitter (meters)
    use_radius_variation: bool = True                       # Enable radius variation for better SfM geometry
    export_camera_poses: bool = True                        # Export camera poses for SfM (COLMAP format)

    # Interactive mode recording settings
    enable_recording: bool = False                          # Enable automatic recording during interactive mode
    record_every_n_moves: int = 5                           # Record camera pose and image every N moves

    # This is populated automatically
    available_cameras: List[str] = field(default_factory=list, init=False)
    # fmt: on


def get_camera_pose(env, camera_name: str):
    """Get the position and quaternion of a camera."""
    camera_id = env.sim.model.camera_name2id(camera_name)
    pos = env.sim.model.cam_pos[camera_id]
    quat_wxyz = env.sim.model.cam_quat[camera_id]
    return pos, quat_wxyz
    # return [0.7200,  0.0000,  1.4200], [0.5608,  0.4306,  0.4306,  0.5608]


def set_camera_pose(env, camera_name: str, pos, quat_wxyz):
    """Set the position and quaternion of a camera."""
    camera_id = env.sim.model.camera_name2id(camera_name)
    env.sim.model.cam_pos[camera_id][:] = pos
    env.sim.model.cam_quat[camera_id][:] = quat_wxyz
    # We need to call forward to update the simulation state with the new camera pose.
    env.sim.forward()


def render_view(env, camera_name: str, resolution: int):
    """Render a view from a specific camera."""
    # robosuite render returns a flipped image, so we flip it back.
    return env.sim.render(camera_name=camera_name, height=resolution, width=resolution)[
        ::-1, ::-1
    ]

def get_key():
    """Read a single keypress without waiting for ENTER."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def print_controls(recording_enabled=False, move_count=0, record_interval=5):
    """Prints the control keys."""
    print("\n" + "---" * 10)
    print("Controls:")
    print("  w/s: forward/backward (+/- local x)")
    print("  a/d: left/right       (+/- local y)")
    print("  q/e: up/down          (+/- local z)")
    print("\n  u/o: roll ccw/cw     (rot around local z - viewing axis)")
    print("  i/k: pitch up/down    (rot around local x - right axis)")
    print("  j/l: yaw left/right   (rot around local y - down axis)")
    print("\n  c: change active camera")
    print("  p: print current pose again")
    if recording_enabled:
        print("  r: toggle recording ON/OFF")
        next_record = record_interval - (move_count % record_interval)
        print(f"  [Recording: {'ON' if recording_enabled else 'OFF'} | Moves until next record: {next_record}]")
    else:
        print("  r: toggle recording ON/OFF")
    print("  x: exit")
    print("---" * 10)
    print("\nPress any key (no ENTER needed)...")

def export_camera_poses_colmap(output_dir: str, camera_data: list, resolution: int):
    """
    Export camera poses in COLMAP format for SfM reconstruction.

    Args:
        output_dir: Directory to save camera poses
        camera_data: List of dicts with 'filename', 'position', 'quaternion'
        resolution: Image resolution (assumes square images)
    """
    import json

    # Create cameras.txt (intrinsics)
    cameras_file = os.path.join(output_dir, "cameras.txt")
    with open(cameras_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        # Fixme: Assuming pinhole camera model with focal length ~0.7*resolution
        focal = 0.7 * resolution
        cx, cy = resolution / 2, resolution / 2
        f.write(f"1 PINHOLE {resolution} {resolution} {focal} {focal} {cx} {cy}\n")

    # Create images.txt (extrinsics)
    images_file = os.path.join(output_dir, "images.txt")
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for idx, cam in enumerate(camera_data, start=1):
            quat = cam['quaternion']  # [w, x, y, z]
            pos = cam['position']
            f.write(f"{idx} {quat[0]} {quat[1]} {quat[2]} {quat[3]} {pos[0]} {pos[1]} {pos[2]} 1 {cam['filename']}\n")
            f.write("\n")  # Empty line for POINTS2D

    # Also save as JSON for easier parsing
    json_file = os.path.join(output_dir, "camera_poses.json")
    with open(json_file, 'w') as f:
        json.dump(camera_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    print(f"Exported camera poses to {cameras_file}, {images_file}, and {json_file}")


def look_at_quaternion(camera_pos, target_pos, up_vector=np.array([0, 0, 1])):
    """
    Calculate quaternion to make camera look at a target point.

    Args:
        camera_pos: Camera position [x, y, z]
        target_pos: Target position to look at [x, y, z]
        up_vector: Up direction (default: [0, 0, 1] for Z-up)

    Returns:
        Quaternion [w, x, y, z]
    """
    # Calculate forward direction (from camera to target)
    forward = target_pos - camera_pos
    forward = forward / np.linalg.norm(forward)

    # Calculate right direction
    right = np.cross(forward, up_vector)
    right = right / np.linalg.norm(right)

    # Recalculate up direction
    up = np.cross(right, forward)

    # Build rotation matrix
    rotation_matrix = np.eye(3)
    rotation_matrix[:, 0] = right
    rotation_matrix[:, 1] = up
    rotation_matrix[:, 2] = -forward  # Camera looks along -Z in MuJoCo

    # Convert rotation matrix to quaternion
    # Using standard conversion algorithm
    trace = np.trace(rotation_matrix)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        x = 0.25 * s
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        y = 0.25 * s
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def capture_panoramic_sweep(env, cfg: InteractiveConfig):
    """
    Captures a panoramic sweep by moving the camera around a center point on the table.
    Now includes radius variation and look-at jitter for better SfM geometry.

    Key improvements for SfM:
    - Multiple orbital radii to increase baseline variation
    - Look-at point jitter to avoid degenerate configurations
    - Varied elevation angles for better triangulation geometry
    """
    print(f"\n{'='*60}")
    print("PANORAMIC SWEEP MODE - SfM-Optimized Capture")
    print(f"{'='*60}")
    print(f"Camera: {cfg.camera_name}")
    print(f"Number of angles: {cfg.panoramic_yaw_angles}")
    print(f"Pitch angles: {cfg.panoramic_pitch_angles}")
    print(f"Radius variation: {cfg.use_radius_variation}")
    if cfg.use_radius_variation:
        print(f"Radius multipliers: {cfg.multiple_radii}")
        print(f"Look-at jitter std: {cfg.look_at_offset_std}m")
    print(f"Height variation: {cfg.use_height_variation}")
    if cfg.use_height_variation:
        print(f"Height offsets: {cfg.height_variations}m")
    print(f"Output directory: {cfg.output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Get the initial camera pose
    initial_pos, initial_quat = get_camera_pose(env, cfg.camera_name)
    print(f"\nInitial camera position: [{initial_pos[0]:.4f}, {initial_pos[1]:.4f}, {initial_pos[2]:.4f}]")

    # Determine center point to look at
    if cfg.center_point is not None:
        center = np.array(cfg.center_point)
        print(f"Using specified center point: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    else:
        # Use table center as default (estimate from initial camera view)
        # Assuming the camera is initially looking roughly at the table
        # We can use the table height from the environment
        center = np.array([-0.1, 0.0, 0.8])  # Default table center position
        print(f"Using default table center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")

    # Calculate the base radius (distance from center to initial camera position)
    # Project to XY plane for horizontal rotation
    initial_pos_xy = initial_pos[:2]
    center_xy = center[:2]
    base_radius_xy = np.linalg.norm(initial_pos_xy - center_xy)

    # Keep the same height as initial camera
    base_camera_height = initial_pos[2]

    print(f"Base orbit radius: {base_radius_xy:.4f}m, Base camera height: {base_camera_height:.4f}m")

    # Calculate azimuth angles (rotation around the center point in XY plane)
    azimuth_angles = np.linspace(0, 360, cfg.panoramic_yaw_angles, endpoint=False)

    # Determine radius multipliers for multi-radius capture
    if cfg.use_radius_variation:
        radius_multipliers = cfg.multiple_radii
    else:
        radius_multipliers = [1.0]  # Single radius (original behavior)

    # Determine height variations
    if cfg.use_height_variation:
        height_offsets = cfg.height_variations
    else:
        height_offsets = [0.0]  # Single height (original behavior)

    capture_count = 0
    total_captures = len(azimuth_angles) * len(cfg.panoramic_pitch_angles) * len(radius_multipliers) * len(height_offsets)

    print(f"\nCapturing {total_captures} views around the table...")
    print(f"SfM optimization: {'ENABLED' if cfg.use_radius_variation else 'DISABLED'}")

    # Store camera data for export
    camera_data = []

    # Set random seed for reproducible jitter
    np.random.seed(42)

    for radius_mult in radius_multipliers:
        radius_xy = base_radius_xy * radius_mult

        for height_offset in height_offsets:
            for pitch_offset_deg in cfg.panoramic_pitch_angles:
                # Adjust camera height based on pitch offset and height variation
                adjusted_height = base_camera_height + height_offset + radius_xy * np.tan(np.deg2rad(pitch_offset_deg))

                for azimuth_deg in azimuth_angles:
                    azimuth_rad = np.deg2rad(azimuth_deg)

                    # Calculate new camera position (rotating around center in XY plane)
                    new_x = center[0] + radius_xy * np.cos(azimuth_rad)
                    new_y = center[1] + radius_xy * np.sin(azimuth_rad)
                    new_z = adjusted_height
                    new_pos = np.array([new_x, new_y, new_z])

                    # Add look-at point jitter for better SfM geometry
                    if cfg.use_radius_variation and cfg.look_at_offset_std > 0:
                        look_at_jitter = np.random.randn(3) * cfg.look_at_offset_std
                        look_at_target = center + look_at_jitter
                    else:
                        look_at_target = center

                    # Calculate quaternion to look at the target point (with jitter)
                    new_quat = look_at_quaternion(new_pos, look_at_target)

                    # Set the new camera pose
                    set_camera_pose(env, cfg.camera_name, new_pos, new_quat)

                    # Render the view
                    img = render_view(env, cfg.camera_name, cfg.env_img_res)

                    # Save the image
                    filename = f"{cfg.camera_name}_r{radius_mult:.2f}_h{height_offset:+06.3f}_az{azimuth_deg:06.2f}_pitch{pitch_offset_deg:+06.2f}.png"
                    filepath = os.path.join(cfg.output_dir, filename)
                    imageio.imwrite(filepath, img)

                    # Store camera data for export
                    camera_data.append({
                        'filename': filename,
                        'position': new_pos.copy(),
                        'quaternion': new_quat.copy(),
                        'radius_multiplier': radius_mult,
                        'height_offset': height_offset,
                        'azimuth_deg': azimuth_deg,
                        'pitch_deg': pitch_offset_deg
                    })

                    capture_count += 1
                    if cfg.use_radius_variation or cfg.use_height_variation:
                        print(f"  [{capture_count}/{total_captures}] Captured: {filename} (r={radius_mult:.2f}, h={height_offset:+.3f}, pos=[{new_x:.3f}, {new_y:.3f}, {new_z:.3f}])")
                    else:
                        print(f"  [{capture_count}/{total_captures}] Captured: {filename} (pos=[{new_x:.3f}, {new_y:.3f}, {new_z:.3f}])")

    # Export camera poses if requested
    if cfg.export_camera_poses:
        export_camera_poses_colmap(cfg.output_dir, camera_data, cfg.env_img_res)

    # Restore initial pose
    set_camera_pose(env, cfg.camera_name, initial_pos, initial_quat)

    print(f"\n{'='*60}")
    print(f"Panoramic sweep complete! Saved {capture_count} images to {cfg.output_dir}")
    if cfg.use_radius_variation or cfg.use_height_variation:
        dims_str = f"{len(radius_multipliers)} radii"
        if cfg.use_height_variation:
            dims_str += f" × {len(height_offsets)} heights"
        dims_str += f" × {len(cfg.panoramic_pitch_angles)} pitches × {len(azimuth_angles)} azimuths"
        print(f"SfM optimization enabled: {dims_str}")
    if cfg.export_camera_poses:
        print("Camera poses exported for SfM/NeRF pipelines")
    print(f"{'='*60}\n")


@draccus.wrap()
def interactive_camera_controller(cfg: InteractiveConfig):
    """Main entry point - runs either interactive mode or multi-angle capture mode."""
    # Initialize LIBERO environment
    set_seed_everywhere(cfg.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.task_id)
    env, task_description = get_libero_env(task, "rt-1", resolution=cfg.env_img_res)
    print(task_description)
    env.reset()
    for _ in range(10):
        obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])  # Step to stabilize

    cfg.available_cameras = list(env.sim.model.camera_names)
    print(f"Initialized environment for task: '{task_description}'")
    print(f"Available cameras: {cfg.available_cameras}")

    # Choose mode
    if cfg.mode == CaptureMode.PANORAMIC:
        # Panoramic sweep mode
        capture_panoramic_sweep(env, cfg)
        return

    # Interactive mode - Main interactive loop
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE - Manual Camera Control")
    print("=" * 60)

    # Recording state
    recording_enabled = cfg.enable_recording
    move_count = 0
    recorded_frames = []

    if recording_enabled:
        os.makedirs(cfg.output_dir, exist_ok=True)
        print(f"Recording enabled: Will save images every {cfg.record_every_n_moves} moves to {cfg.output_dir}")

    while True:
        pos, quat_wxyz = get_camera_pose(env, cfg.camera_name)

        # Print current pose
        print("\n" + "=" * 50)
        print(f"Active Camera: '{cfg.camera_name}'")
        print(f"Position (x,y,z):      {pos[0]: .4f}, {pos[1]: .4f}, {pos[2]: .4f}")
        print(
            f"Quaternion (w,x,y,z):  {quat_wxyz[0]: .4f}, {quat_wxyz[1]: .4f}, {quat_wxyz[2]: .4f}, {quat_wxyz[3]: .4f}"
        )

        # Render and save the current view
        img = render_view(env, cfg.camera_name, cfg.env_img_res)
        imageio.imwrite(cfg.output_image_path, img)
        print(
            f"\nView saved to '{cfg.output_image_path}'. Check this image to see the current perspective."
        )

        print_controls(recording_enabled, move_count, cfg.record_every_n_moves)

        try:
            cmd = get_key()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        # Echo the pressed key
        print(f"Key pressed: '{cmd}'")

        if cmd == "x":
            print("Exiting.")
            # Save recorded frames if any
            if recording_enabled and recorded_frames and cfg.export_camera_poses:
                print(f"\nExporting {len(recorded_frames)} recorded camera poses...")
                export_camera_poses_colmap(cfg.output_dir, recorded_frames, cfg.env_img_res)
            break
        if cmd == "p":
            continue
        if cmd == "r":
            # Toggle recording
            recording_enabled = not recording_enabled
            if recording_enabled:
                os.makedirs(cfg.output_dir, exist_ok=True)
                print(f"\n*** Recording ENABLED - Will save every {cfg.record_every_n_moves} moves to {cfg.output_dir} ***")
            else:
                print("\n*** Recording DISABLED ***")
            continue
        if cmd == "c":
            # For camera change, we need to read a full line
            # Temporarily restore normal input mode
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                new_cam = input(f"\nEnter new camera name ({cfg.available_cameras}): ")
                if new_cam in cfg.available_cameras:
                    cfg.camera_name = new_cam
                else:
                    print(f"Error: Invalid camera name '{new_cam}'.")
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled camera change.")
            continue

        # Update Pose based on command
        # Position (local frame)
        move = np.zeros(3)
        if "w" in cmd:
            move[0] -= cfg.pos_step
        if "s" in cmd:
            move[0] += cfg.pos_step
        if "a" in cmd:
            move[1] += cfg.pos_step
        if "d" in cmd:
            move[1] -= cfg.pos_step
        if "q" in cmd:
            move[2] += cfg.pos_step
        if "e" in cmd:
            move[2] -= cfg.pos_step

        # Rotation (local frame) - using aviation/camera conventions:
        # roll = rotation around viewing axis (local Z - backward axis in MuJoCo)
        # pitch = rotation around right axis (local X)
        # yaw = rotation around up/down axis (local Y - down in MuJoCo)
        rot_step_rad = np.deg2rad(cfg.rot_step_deg)
        roll, pitch, yaw = 0, 0, 0
        if "u" in cmd:
            roll += rot_step_rad  # Roll counter-clockwise (around Z)
        if "o" in cmd:
            roll -= rot_step_rad  # Roll clockwise (around Z)
        if "i" in cmd:
            pitch += rot_step_rad  # Pitch up (around X)
        if "k" in cmd:
            pitch -= rot_step_rad  # Pitch down (around X)
        if "j" in cmd:
            yaw -= rot_step_rad  # Yaw left (around Y)
        if "l" in cmd:
            yaw += rot_step_rad  # Yaw right (around Y)

        # Convert euler angles (roll, pitch, yaw) to a delta quaternion
        delta_quat_wxyz = euler2quat(np.array([roll, pitch, yaw]))
        new_pos = pos + move

        # Apply rotation (local frame) by post-multiplying the delta rotation
        new_quat_wxyz = quat_multiply(quat_wxyz, delta_quat_wxyz)

        set_camera_pose(env, cfg.camera_name, new_pos, new_quat_wxyz)

        # Check if this was an actual move (not just printing or changing camera)
        position_changed = np.any(move != 0)
        rotation_changed = (roll != 0 or pitch != 0 or yaw != 0)

        if position_changed or rotation_changed:
            move_count += 1

            # Record frame if enabled and we've hit the interval
            if recording_enabled and (move_count % cfg.record_every_n_moves == 0):
                # Render the view at the new position
                recorded_img = render_view(env, cfg.camera_name, cfg.env_img_res)

                # Save the image
                filename = f"{cfg.camera_name}_frame_{move_count:04d}.png"
                filepath = os.path.join(cfg.output_dir, filename)
                imageio.imwrite(filepath, recorded_img)

                # Store camera data
                recorded_frames.append({
                    'filename': filename,
                    'position': new_pos.copy(),
                    'quaternion': new_quat_wxyz.copy(),
                    'move_count': move_count
                })

                print(f"\n*** RECORDED frame {len(recorded_frames)}: {filename} (total moves: {move_count}) ***")


if __name__ == "__main__":
    interactive_camera_controller()
