"""
Collects demonstrations from LIBERO simulated environments with the same output format
as real robot data collection (RGB images, depth images, and camera intrinsics).

This script mimics the purpose of real_to_sim/collect_human_demo.py but operates in simulation.

Usage:
    python scripts/collect_sim_demo.py \
        --save_dir <PATH TO SAVE DIR> \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --task_id <TASK ID (0-indexed)> \
        --demo_id <DEMO ID (0-indexed)> \
        --width 960 \
        --height 540 \
        --camera_name agentview

    Example:
        python scripts/collect_sim_demo.py \
            --save_dir ./sim_demo_data \
            --libero_task_suite libero_spatial \
            --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
            --task_id 0 \
            --demo_id 0
"""

import argparse
import json
import os
import shutil

import cv2
import h5py
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def create_directory(directory):
    """Create directory, removing existing one if present."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def construct_camera_intrinsics_sim(env, camera_name):
    """Gets camera intrinsics from the environment."""
    cam_id = env.sim.model.camera_name2id(camera_name)
    fovy = env.sim.model.cam_fovy[cam_id]
    width, height = env.env.camera_widths[0], env.env.camera_heights[0]
    f = 0.5 * height / np.tan(np.deg2rad(fovy) / 2)
    cx = width / 2
    cy = height / 2
    # See https://github.com/openai/mujoco-py/issues/271#issuecomment-750772769
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def get_camera_to_robot_transform(env, camera_name):
    """
    Computes the 4x4 transformation matrix from camera frame to robot base frame.

    The transformation accounts for:
    1. Camera pose in world frame (from MuJoCo)
    2. OpenCV/OpenGL coordinate system conventions
    3. Robot base frame as reference

    Returns:
        4x4 numpy array representing the transformation from camera to robot base
    """
    # Get camera pose in world frame
    cam_id = env.sim.model.camera_name2id(camera_name)
    cam_pos = env.sim.data.cam_xpos[cam_id]  # Camera position in world frame
    cam_mat = env.sim.data.cam_xmat[cam_id].reshape(3, 3)  # Camera rotation matrix

    # MuJoCo camera looks along -Z axis with Y pointing down
    # OpenCV/FoundationPose expects camera looking along +Z with Y pointing down
    # Apply 180-degree rotation around X to convert MuJoCo to OpenCV convention
    mujoco_to_opencv = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    cam_rot_opencv = cam_mat @ mujoco_to_opencv

    # Construct 4x4 transformation matrix from world to camera (OpenCV convention)
    T_world_to_cam = np.eye(4)
    T_world_to_cam[:3, :3] = cam_rot_opencv.T  # Inverse rotation
    T_world_to_cam[:3, 3] = -cam_rot_opencv.T @ cam_pos  # Inverse translation

    # Get robot base pose in world frame
    # The robot base is typically at the origin or can be obtained from the model
    robot_body_id = env.sim.model.body_name2id("robot0_base") # FIXME: confirm body name
    robot_pos = env.sim.data.body_xpos[robot_body_id]
    robot_mat = env.sim.data.body_xmat[robot_body_id].reshape(3, 3)

    # Construct transformation from world to robot base
    T_world_to_robot = np.eye(4)
    T_world_to_robot[:3, :3] = robot_mat.T
    T_world_to_robot[:3, 3] = -robot_mat.T @ robot_pos

    # Camera-to-robot transform: T_cam_to_robot = T_world_to_robot^-1 @ T_world_to_cam^-1
    T_robot_to_world = np.linalg.inv(T_world_to_robot)
    T_cam_to_world = np.linalg.inv(T_world_to_cam)
    T_cam_to_robot = T_robot_to_world @ T_cam_to_world

    return T_cam_to_robot


def get_libero_env(task, resolution=256, camera_name="agentview"):
    """Initializes and returns the LIBERO environment with depth rendering."""
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": [camera_name],
        "render_camera": camera_name,
        "camera_depths": True,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_dummy_action(model_family):
    """Returns a dummy action for the specified model family."""
    if model_family == "llava":
        return [0, 0, 0, 0, 0, 0, -1]
    else:
        raise ValueError(f"Unknown model family: {model_family}")


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.
    """
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return (
        np.linalg.norm(action[:-1]) < threshold
        and gripper_action == prev_gripper_action
    )


def get_libero_rgbd_image(obs, camera_name):
    """Extracts RGBD image from observations and preprocesses it."""
    img = obs[f"{camera_name}_image"]
    depth = obs[f"{camera_name}_depth"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    depth = depth[
        ::-1, ::-1, 0
    ]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img, depth


def main(args):
    print(f"Collecting simulated demo from {args.libero_task_suite}!")

    # Create save directories
    save_directory = args.save_dir
    rgb_directory = os.path.join(save_directory, "rgb")
    depth_directory = os.path.join(save_directory, "depth")

    create_directory(rgb_directory)
    create_directory(depth_directory)

    # Get task suite and specific task
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    task = task_suite.get_task(args.task_id)

    # Initialize environment
    env, task_description = get_libero_env(
        task, resolution=args.resolution, camera_name=args.camera_name
    )

    print(f"Task: {task_description}")
    print(f"Camera: {args.camera_name}")

    # Get original demo data
    orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
    assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}"

    orig_data_file = h5py.File(orig_data_path, "r")
    orig_data = orig_data_file["data"]
    demo_data = orig_data[f"demo_{args.demo_id}"]
    orig_actions = demo_data["actions"][()]
    orig_states = demo_data["states"][()]

    # Construct and save camera intrinsics
    # Extract FOV directly from MuJoCo camera
    K = construct_camera_intrinsics_sim(env=env, camera_name=args.camera_name)

    intrinsics_path = os.path.join(save_directory, "cam_K.txt")
    np.savetxt(intrinsics_path, K)
    print(f"Camera intrinsics saved to {intrinsics_path}")
    print(f"K matrix:\n{K}")

    # Construct and save camera-to-robot transformation
    T_cam_to_robot = get_camera_to_robot_transform(env=env, camera_name=args.camera_name)

    camera_to_robot_path = os.path.join(save_directory, "camera_to_robot.npy")
    np.save(camera_to_robot_path, T_cam_to_robot)
    print(f"\nCamera-to-robot transform saved to {camera_to_robot_path}")
    print(f"Transform matrix:\n{T_cam_to_robot}")

    # Reset environment and set initial state
    env.reset()
    env.set_init_state(orig_states[0])

    # Let environment settle
    for _ in range(10):
        obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

    frame_count = 0
    prev_action = None

    print(f"\nReplaying demo {args.demo_id}...")

    # Replay demo actions and save observations
    for action_idx, action in enumerate(orig_actions):
        # Skip no-op actions if filter is enabled
        if args.filter_noops and is_noop(action, prev_action):
            print(f"  Skipping no-op action at index {action_idx}")
            continue

        # Execute action
        obs, reward, done, info = env.step(action.tolist())

        # Get RGB image
        rgb_image, depth_image = get_libero_rgbd_image(obs, args.camera_name)

        # Convert depth image from [0, 1] to actual depth in meters
        # See https://github.com/openai/mujoco-py/issues/520#issuecomment-1254452252
        # Actually this is necessary because normalization does affect homography
        extent = env.sim.model.stat.extent
        near = env.sim.model.vis.map.znear * extent
        far = env.sim.model.vis.map.zfar * extent
        depth_image = near / (1 - depth_image * (1 - near / far))
        depth_image = (depth_image * 1000.0).astype(np.uint16)  # Convert to mm and uint16

        # Save RGB image (convert from RGB to BGR for OpenCV)
        rgb_filename = os.path.join(rgb_directory, f"frame_{frame_count:04d}.png")
        cv2.imwrite(rgb_filename, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        # Save depth image
        depth_filename = os.path.join(depth_directory, f"frame_{frame_count:04d}.png")
        cv2.imwrite(depth_filename, depth_image)

        print(f"Frame: {frame_count}")
        frame_count += 1
        prev_action = action

        if done:
            print(f"\nDemo completed successfully!")
            break

    # Save metadata
    metadata = {
        "task_suite": args.libero_task_suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "task_description": task_description,
        "demo_id": args.demo_id,
        "camera_name": args.camera_name,
        "resolution": args.resolution,
        "num_frames": frame_count,
        "filter_noops": args.filter_noops,
    }

    metadata_path = os.path.join(save_directory, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nCaptured {frame_count} frames to {save_directory}")
    print(f"Metadata saved to {metadata_path}")

    # Cleanup
    orig_data_file.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save captured frames"
    )
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        required=True,
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ],
        help="LIBERO task suite",
    )
    parser.add_argument(
        "--libero_raw_data_dir",
        type=str,
        required=True,
        help="Path to directory containing raw HDF5 dataset",
    )
    parser.add_argument(
        "--task_id", type=int, required=True, help="Task ID (0-indexed)"
    )
    parser.add_argument(
        "--demo_id", type=int, required=True, help="Demo ID (0-indexed)"
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="agentview",
        choices=[
            "agentview",
            "robot0_eye_in_hand",
            "frontview",
            "birdview",
            "sideview",
        ],
        help="Camera name to use for rendering",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution (width and height)",
    )
    parser.add_argument(
        "--filter_noops",
        action="store_true",
        help="Filter out no-op actions (recommended)",
    )

    args = parser.parse_args()
    main(args)
