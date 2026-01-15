"""
This is an example metadata file for the mug_on_holder task in the tabletop_env environment.
```
env_name: tabletop_env
env_kitchen_to_robot_transform: [0.198  0.7336 0.3398]
env_robot_qpos: [ 0.0109 -0.6258  0.0274 -1.9208  0.0732  1.2061  0.8767  0.04    0.04  ]
env_camera_eye: [0.11364682 0.61086124 1.10074165]
env_camera_target: [0.69425251 0.12618053 0.4465386 ]
env_ground_altitude: -0.7
env_kitchen_mesh_name: tabletop
env_asset_base_path: xsim/tabletop_env
task_robot_uids: panda_ninja_slow
task_obj_names: ['cup' 'mug_holder']
task_manip_idx: 0
task_randomize_objects_list: [0]
task_demo_name: mug_on_holder
task_num_waypoints: 5
task_first_waypoint_idx: 24
task_last_waypoint_idx: 60
task_goal_thresh: 0.05
task_angle_goal_thresh: 0.3
task_rotation_reward: True
task_require_grasp: True
robot_pos: [-0.02  0.    0.  ]
```
"""

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os

def extract_libero_metadata(task, task_suite_name, task_id, resolution=256):
    """Extract metadata from LIBERO environment matching the X-Sim format."""

    # Initialize environment
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    camera_names = ["agentview", "robot0_eye_in_hand"]
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": camera_names,
        "render_camera": "agentview",
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    obs = env.reset()

    # Stabilize simulation
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    # Extract robot information
    robot = env.robots[0]
    robot_name = robot.name  # e.g., "Panda0"
    robot_qpos = env.sim.data.qpos[robot._ref_joint_pos_indexes].copy()  # 9 values (7 arm + 2 gripper)
    robot_base_offset = robot.robot_model.base_offset  # [x, y, z]

    # Extract camera information
    # Find camera ID by name
    cam_names = [env.sim.model.camera(i).name for i in range(env.sim.model.ncam)]
    agentview_id = cam_names.index("agentview")

    camera_pos = env.sim.model.cam_pos[agentview_id].copy()  # [x, y, z]
    camera_quat = env.sim.model.cam_quat[agentview_id].copy()  # [w, x, y, z]

    # Convert quaternion to target point (camera points along local -Z axis)
    # For X-Sim compatibility, compute a target point 1 meter ahead
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat([camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]])  # scipy uses [x,y,z,w]
    forward = R.apply([0, 0, -1])  # camera looks along -Z
    camera_target = camera_pos + forward

    # Extract object information
    obj_names = list(env.obj_of_interest)

    # Identify manipulated object (from obj_of_interest)
    manip_idx = 0 # assuming first object is the manipulated one
    manip_obj_name = obj_names[0]
    
    # Arena information
    arena_type = env.env._arena_type  # e.g., "table", "study_table"
    ground_altitude = 0.0

    # Task information
    demo_name = task.name  # e.g., "pick_up_the_black_bowl..."

    # Check if grasp is required (look for contact or grasp predicates in goal)
    require_grasp = True

    # Build metadata dictionary
    meta_data = {
        # Environment parameters
        "env_name": f"{task_suite_name}_env",
        "env_kitchen_to_robot_transform": robot_base_offset.tolist(),
        "env_robot_qpos": robot_qpos.tolist(),
        "env_camera_eye": camera_pos.tolist(),
        "env_camera_target": camera_target.tolist(),
        "env_ground_altitude": ground_altitude,
        "env_kitchen_mesh_name": arena_type,
        "env_asset_base_path": f"xsim/libero/{task_suite_name}",

        # Task parameters
        "task_robot_uids": robot_name,
        "task_obj_names": obj_names,
        "task_manip_idx": manip_idx,
        "task_randomize_objects_list": list(range(len(obj_names))),  # All objects randomized in LIBERO
        "task_demo_name": demo_name,

        # Waypoint parameters (placeholder - extract from demonstrations)
        "task_num_waypoints": 5,  # Default, override from demo
        "task_first_waypoint_idx": 0,  # Override from demo
        "task_last_waypoint_idx": 0,  # Override from demo

        # Goal parameters (LIBERO uses BDDL predicates)
        "task_goal_thresh": 0.05,  # Standard LIBERO threshold
        "task_angle_goal_thresh": 0.3,  # Standard LIBERO threshold
        "task_rotation_reward": True,  # Most LIBERO tasks require orientation
        "task_require_grasp": require_grasp,

        # Robot position
        "robot_pos": robot_base_offset.tolist(),
    }

    env.close()
    return meta_data


# Example usage
if __name__ == "__main__":
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_spatial"
    task_suite = benchmark_dict[task_suite_name]()
    task_id = 0
    task = task_suite.get_task(task_id)

    metadata = extract_libero_metadata(task, task_suite_name, task_id)

    print(metadata)

    # Save to .npz file
    output_path = f"/local_data/cf3331/X-Sim/simulation/ManiSkill/mani_skill/assets/xsim/libero_spatial_env/{metadata['task_demo_name']}/metadata.npz"
    np.savez(output_path, **metadata)

    print(f"Metadata saved to {output_path}")
    for key, value in metadata.items():
        print(f"{key}: {value}")