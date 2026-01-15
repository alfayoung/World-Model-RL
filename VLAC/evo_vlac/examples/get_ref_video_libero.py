"""
Guidance:
1. First download the groundtruth data
For example: huggingface-cli download yifengzhu-hf/LIBERO-datasets --repo-type dataset --include libero_90/libero_90/LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray_demo.hdf5 --local-dir .
             huggingface-cli download yifengzhu-hf/LIBERO-datasets --repo-type dataset --include libero_90/KITCHEN_SCENE6_close_the_microwave_demo.hdf5 --local-dir .

2. Then run this script to regenerate reference videos

export PYTHONPATH=$PYTHONPATH:./LIBERO
uv run VLAC/evo_vlac/examples/get_ref_video_libero.py \
    --libero_task_suite libero_90 \
    --task_id 33 \
    --libero_raw_data_dir ./ref_img/libero_90 \
    --ref_images_dir ./ref_img/ \
    --max_ref_images 400

"""

import argparse
import json
import os
import time

import h5py
import imageio
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark
from PIL import Image

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

IMAGE_RESOLUTION = 64

def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset!")

    # Create reference images directory if specified
    if args.ref_images_dir is not None:
        os.makedirs(args.ref_images_dir, exist_ok=True)
        print(f"Reference images will be saved to: {args.ref_images_dir}")

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    task_id = args.task_id
    # Get task in suite
    task = task_suite.get_task(task_id)
    env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

    # Get dataset for task
    orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
    assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
    orig_data_file = h5py.File(orig_data_path, "r")
    orig_data = orig_data_file["data"]

    for i in range(len(orig_data.keys())):
        # Get demo data
        demo_data = orig_data[f"demo_{i}"]
        orig_actions = demo_data["actions"][()]
        orig_states = demo_data["states"][()]

        # Reset environment, set initial state, and wait a few steps for environment to settle
        env.reset()
        env.set_init_state(orig_states[0])
        for _ in range(10):
            obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

        # Set up new data lists
        states = []
        actions = []
        ee_states = []
        gripper_states = []
        joint_states = []
        robot_states = []
        agentview_images = []
        eye_in_hand_images = []

        # Replay original demo actions in environment and record observations
        for _, action in enumerate(orig_actions):
            # Skip transitions with no-op actions
            prev_action = actions[-1] if len(actions) > 0 else None
            if is_noop(action, prev_action):
                print(f"\tSkipping no-op action: {action}")
                num_noops += 1
                continue

            if states == []:
                # In the first timestep, since we're using the original initial state to initialize the environment,
                # copy the initial state (first state in episode) over from the original HDF5 to the new one
                states.append(orig_states[0])
                robot_states.append(demo_data["robot_states"][0])
            else:
                # For all other timesteps, get state from environment and record it
                states.append(env.sim.get_state().flatten())
                robot_states.append(
                    np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                )

            # Record original action (from demo)
            actions.append(action)

            # Record data returned by environment
            if "robot0_gripper_qpos" in obs:
                gripper_states.append(obs["robot0_gripper_qpos"])
            joint_states.append(obs["robot0_joint_pos"])
            ee_states.append(
                np.hstack(
                    (
                        obs["robot0_eef_pos"],
                        T.quat2axisangle(obs["robot0_eef_quat"]),
                    )
                )
            )
            agentview_images.append(obs["agentview_image"])
            eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

            # Execute demo action in environment
            obs, reward, done, info = env.step(action.tolist())

        # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
        if done:
            dones = np.zeros(len(actions)).astype(np.uint8)
            dones[-1] = 1
            rewards = np.zeros(len(actions)).astype(np.uint8)
            rewards[-1] = 1
            assert len(actions) == len(agentview_images)

            # Save reference images if directory is specified
            if args.ref_images_dir is not None:
                # Select which camera view to use
                images_to_save = agentview_images if args.ref_images_camera == "agentview" else eye_in_hand_images
                images_to_save = [img[::-1, ::-1] for img in images_to_save]  # Remove alpha channel if present
                # Sample evenly spaced frames (up to max_ref_images)
                num_frames = len(images_to_save)
                num_ref_images = min(args.max_ref_images, num_frames)

                if num_ref_images > 1:
                    # Evenly sample indices including first and last frame
                    sample_indices = np.linspace(0, num_frames - 1, num_ref_images, dtype=int)
                else:
                    # If only 1 image, take the last frame
                    sample_indices = [num_frames - 1]

                # Create task-specific subdirectory
                task_img_dir = os.path.join(args.ref_images_dir, task.name)
                os.makedirs(task_img_dir, exist_ok=True)

                # Save sampled images
                for idx, frame_idx in enumerate(sample_indices):
                    image = images_to_save[frame_idx]
                    # Convert from numpy array to PIL Image
                    img_pil = Image.fromarray(image.astype(np.uint8))
                    # Save with naming: {task_name}-demo_{demo_id}-frame_{frame_idx}.jpg
                    img_filename = f"{task.name}-demo_{i}-frame_{frame_idx}.jpg"
                    img_path = os.path.join(task_img_dir, img_filename)
                    img_pil.save(img_path)

                print(f"  Saved {len(sample_indices)} reference images to {task_img_dir}")

                # Save video of the entire demonstration
                video_filename = f"{task.name}-demo_{i}.mp4"
                video_path = os.path.join(task_img_dir, video_filename)

                # Write video using imageio (all frames, not just sampled ones)
                with imageio.get_writer(video_path, fps=20, codec='libx264') as writer:
                    for frame in images_to_save:
                        writer.append_data(frame.astype(np.uint8))

                print(f"  Saved video to {video_path}")

            exit(0) # Only keep successful replays for now

        # Count total number of successful replays so far
        print(
            f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
        )

        # Report total number of no-op actions filtered out so far
        print(f"  Total # no-op actions filtered out: {num_noops}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--task_id", type=int, help="Task ID to regenerate data for. Example: 45", required=True)
    parser.add_argument("--libero_raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial", required=True)
    parser.add_argument("--ref_images_dir", type=str, default=None,
                        help="Optional: Path to directory for saving reference images (up to 11 evenly sampled frames per demo). Example: ./ref_images")
    parser.add_argument("--ref_images_camera", type=str, default="agentview", choices=["agentview", "eye_in_hand"],
                        help="Camera view to use for reference images (default: agentview)")
    parser.add_argument("--max_ref_images", type=int, default=11,
                        help="Maximum number of reference images to sample per demo (default: 11)")
    args = parser.parse_args()

    # Start data regeneration
    main(args)