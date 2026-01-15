# LIBERO Metadata Parameter Mapping

This document maps the metadata parameters from `scripts/metadata.py` (designed for self-curated environments) to their LIBERO equivalents.

## Reference: Original Metadata Structure

From `scripts/metadata.py`, the metadata for self-curated environments includes:

```python
meta_data = {
    "env_name": "tabletop_env",
    "env_kitchen_to_robot_transform": [0.198, 0.7336, 0.3398],
    "env_robot_qpos": [0.0109, -0.6258, 0.0274, -1.9208, 0.0732, 1.2061, 0.8767, 0.04, 0.04],
    "env_camera_eye": [0.11364682, 0.61086124, 1.10074165],
    "env_camera_target": [0.69425251, 0.12618053, 0.4465386],
    "env_ground_altitude": -0.7,
    "env_kitchen_mesh_name": "tabletop",
    "env_asset_base_path": "xsim/tabletop_env",
    "task_robot_uids": "panda_ninja_slow",
    "task_obj_names": ["cup", "mug_holder"],
    "task_manip_idx": 0,
    "task_randomize_objects_list": [0],
    "task_demo_name": "mug_on_holder",
    "task_num_waypoints": 5,
    "task_first_waypoint_idx": 24,
    "task_last_waypoint_idx": 60,
    "task_goal_thresh": 0.05,
    "task_angle_goal_thresh": 0.3,
    "task_rotation_reward": True,
    "task_require_grasp": True,
    "robot_pos": [-0.02, 0.0, 0.0],
}
```

## LIBERO Metadata Parameter Mapping

### 1. Environment Parameters

| Original Parameter | LIBERO Equivalent | How to Access | Notes |
|-------------------|-------------------|---------------|-------|
| `env_name` | Task name from BDDL | `task.language` or `env.language_instruction` | Natural language description of task |
| `env_kitchen_to_robot_transform` | Robot base position | `env.robots[0].robot_model.base_offset` or `env.sim.data.body_xpos[env.robots[0].robot_model.root_body]` | Offset from world origin to robot base |
| `env_robot_qpos` | Initial robot joint positions | `env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes]` | 7 DOF arm + 2 DOF gripper = 9 values |
| `env_camera_eye` | Camera position | `env.sim.model.cam_pos[cam_id]` | Position in world frame (x, y, z) |
| `env_camera_target` | Camera orientation | Computed from `env.sim.model.cam_quat[cam_id]` | Convert quaternion to target point |
| `env_ground_altitude` | Arena floor height | `env._arena.floor.get("pos")[2]` or typically 0.0 | Z-coordinate of floor |
| `env_kitchen_mesh_name` | Arena type | `env._arena_type` | e.g., "table", "study_table", "kitchen_table" |
| `env_asset_base_path` | BDDL file path | `env.bddl_file_name` | Path to BDDL task definition |

### 2. Task Parameters

| Original Parameter | LIBERO Equivalent | How to Access | Notes |
|-------------------|-------------------|---------------|-------|
| `task_robot_uids` | Robot name | `env.robots[0].name` | Format: "Panda{index}", e.g., "Panda0" |
| `task_obj_names` | Objects of interest | `list(env.objects_dict.keys())` | All movable objects in scene |
| `task_manip_idx` | Manipulated object index | Identify from `env.obj_of_interest` | Index of primary object to manipulate |
| `task_randomize_objects_list` | Randomized objects | Infer from placement initializer | Objects that change position on reset |
| `task_demo_name` | Task name | `task.problem_folder` or `task.name` | Short identifier for task |
| `task_num_waypoints` | Number of waypoints | **Not directly available** | Must be inferred from demonstrations |
| `task_first_waypoint_idx` | First waypoint index | **Not directly available** | Must be extracted from demo data |
| `task_last_waypoint_idx` | Last waypoint index | **Not directly available** | Must be extracted from demo data |
| `task_goal_thresh` | Position threshold | **Hardcoded in predicates** | Typically 0.05m in BDDL evaluations |
| `task_angle_goal_thresh` | Angle threshold | **Hardcoded in predicates** | Typically 0.3 rad in BDDL evaluations |
| `task_rotation_reward` | Use rotation in reward | **Not explicit** | Depends on goal predicates |
| `task_require_grasp` | Grasp required | Check `env.parsed_problem["goal_state"]` | Look for "Grasped" or contact predicates |
| `robot_pos` | Robot base position | `env.robots[0].robot_model.base_offset` | Offset in arena frame |

## Code Example: Extracting LIBERO Metadata

```python
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
    obj_names = list(env.objects_dict.keys())

    # Identify manipulated object (from obj_of_interest)
    manip_idx = 0
    if hasattr(env, 'obj_of_interest') and len(env.obj_of_interest) > 0:
        manip_obj_name = env.obj_of_interest[0]
        if manip_obj_name in obj_names:
            manip_idx = obj_names.index(manip_obj_name)

    # Arena information
    arena_type = env._arena_type  # e.g., "table", "study_table"
    try:
        ground_altitude = env._arena.floor.get("pos")[2] if hasattr(env._arena, 'floor') else 0.0
    except:
        ground_altitude = 0.0

    # Task information
    demo_name = task.problem_folder  # e.g., "pick_up_the_black_bowl..."

    # Check if grasp is required (look for contact or grasp predicates in goal)
    goal_state = env.parsed_problem.get("goal_state", [])
    require_grasp = any(
        pred[0] in ["contact", "grasped"]
        for pred in goal_state
    )

    # Build metadata dictionary
    meta_data = {
        # Environment parameters
        "env_name": task_suite_name,
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

        # Additional LIBERO-specific fields
        "language_instruction": task_description,
        "bddl_file": task_bddl_file,
        "goal_predicates": str(goal_state),
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

    # Save to .npz file
    output_path = f"/path/to/output/{metadata['task_demo_name']}/metadata.npz"
    np.savez(output_path, **metadata)

    print(f"Metadata saved to {output_path}")
    for key, value in metadata.items():
        print(f"{key}: {value}")
```

## Key Differences from Self-Curated Environments

### 1. **No Explicit Waypoints in LIBERO**
- LIBERO uses BDDL goal specifications instead of waypoints
- Waypoint data must come from demonstration trajectories
- Goal achievement is evaluated via predicate satisfaction

### 2. **Camera System**
- LIBERO uses MuJoCo camera system (fixed in XML)
- Camera parameters: `cam_pos` (position) and `cam_quat` (orientation)
- To get "target", convert quaternion to viewing direction

### 3. **Object Randomization**
- LIBERO randomizes object positions on every reset by default
- Uses placement initializers (site-based, object-based, conditional)
- All movable objects are potentially randomized

### 4. **Goal Specification**
- X-Sim uses distance thresholds for success
- LIBERO uses symbolic predicates (On, Contact, Open, etc.)
- Thresholds are embedded in predicate evaluations

### 5. **Robot Configuration**
- LIBERO only supports Franka Panda robot
- Robot type varies by arena (OnTheGroundPanda vs MountedPanda)
- Always 7 DOF arm + 2 DOF gripper

## Camera ID Lookup Helper

```python
def get_camera_id(env, camera_name):
    """Get MuJoCo camera ID from name."""
    for i in range(env.sim.model.ncam):
        if env.sim.model.camera(i).name == camera_name:
            return i
    raise ValueError(f"Camera {camera_name} not found")

def get_camera_params(env, camera_name):
    """Extract camera position and quaternion."""
    cam_id = get_camera_id(env, camera_name)
    return {
        "pos": env.sim.model.cam_pos[cam_id].copy(),
        "quat": env.sim.model.cam_quat[cam_id].copy(),  # [w, x, y, z]
        "fovy": env.sim.model.cam_fovy[cam_id],
    }
```

## Demonstration Data Integration

For waypoint parameters, you'll need to extract from demonstration files:

```python
import h5py

def extract_waypoint_info(demo_path, task_name):
    """Extract waypoint metadata from LIBERO demonstration file."""
    with h5py.File(demo_path, 'r') as f:
        demo_group = f[f"data/{task_name}/demo_0"]

        # Get trajectory length
        actions = demo_group["actions"][:]
        states = demo_group["states"][:]

        # Detect significant state changes for waypoints
        # (This is heuristic - adjust based on your needs)
        num_waypoints = detect_waypoints(states, actions)

        return {
            "task_num_waypoints": num_waypoints,
            "task_first_waypoint_idx": 0,
            "task_last_waypoint_idx": len(actions) - 1,
        }

def detect_waypoints(states, actions, threshold=0.1):
    """Heuristic to detect waypoints from trajectory."""
    # Simple approach: count action direction changes
    # Sophisticated approach: segment by gripper state changes
    gripper_actions = actions[:, -1]  # Last dimension is gripper
    gripper_changes = np.abs(np.diff(gripper_actions)) > threshold
    waypoints = 2 + np.sum(gripper_changes)  # Start + end + gripper changes
    return int(waypoints)
```

## Summary

Most metadata parameters have direct LIBERO equivalents accessible through the environment API. The main exceptions are:

1. **Waypoint data**: Must be extracted from demonstrations
2. **Goal thresholds**: Embedded in BDDL predicate evaluations
3. **Camera target**: Must be computed from quaternion

All other parameters can be directly queried from the LIBERO environment after initialization.
