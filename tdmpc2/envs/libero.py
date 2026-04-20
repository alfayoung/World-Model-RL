"""LIBERO environment wrapper for TD-MPC2.

Follows the same interface as envs/dmcontrol.py:
  - observation_space: Box (flat state vector or stacked RGB pixels)
  - action_space:      Box (7-dim, scaled to [-1, 1])
  - reset()  -> np.ndarray
  - step(action) -> (obs, reward, done, info)
  - info must contain 'success' (float) and 'terminated' (bool)

Usage (Hydra cfg):
  task: libero-90-57   # libero-<suite>-<task_id>
  obs:  state          # or 'rgb'
"""
import pathlib
from collections import deque

import gymnasium as gym
import numpy as np

from envs.wrappers.timeout import Timeout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_task(task: str):
    """Parse 'libero-90-57' -> (suite='libero_90', task_id=57)."""
    parts = task.split('-')
    # task format: libero-<suite_suffix>-<id>
    # e.g. libero-90-57, libero-10-3, libero-spatial-0
    task_id = int(parts[-1])
    suite_suffix = '-'.join(parts[1:-1])         # '90', '10', 'spatial', etc.
    suite_name = 'libero_' + suite_suffix        # 'libero_90', 'libero_10', …
    return suite_name, task_id


def _get_env(suite_name: str, task_id: int, resolution: int, seed: int):
    """Construct OffScreenRenderEnv for the given LIBERO task."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)

    bddl_file = (
        pathlib.Path(get_libero_path('bddl_files'))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


# ---------------------------------------------------------------------------
# State wrapper
# ---------------------------------------------------------------------------

_STATE_KEYS = [
    'robot0_eef_pos',        # 3
    'robot0_eef_quat',       # 4
    'robot0_gripper_qpos',   # 2
]
_STATE_DIM = 9


class LiberoWrapper:
    """Adapts LIBERO OffScreenRenderEnv to the TD-MPC2 env interface."""

    def __init__(self, env, action_low, action_high):
        self._env = env
        self._action_low = action_low
        self._action_high = action_high

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(_STATE_DIM,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(len(action_low),), dtype=np.float32,
        )

    @property
    def unwrapped(self):
        return self._env

    def _obs(self, raw: dict) -> np.ndarray:
        return np.concatenate([
            raw[k].astype(np.float32) for k in _STATE_KEYS
        ])

    def reset(self):
        raw = self._env.reset()
        return self._obs(raw)

    def step(self, action: np.ndarray):
        # action is in [-1, 1]; LIBERO expects actions in the original space
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        raw, reward, done, _ = self._env.step(action)
        obs = self._obs(raw)
        success = float(self._env.check_success())
        info = {
            'success': success,
            'terminated': bool(success),   # treat task success as termination
        }
        return obs, float(reward), done, info

    def render(self, width=64, height=64, camera_id=0):
        return self._env.env.sim.render(height, width, camera_name='agentview')[::-1]


# ---------------------------------------------------------------------------
# RGB (pixel) wrapper
# ---------------------------------------------------------------------------

class LiberoPixels(gym.Wrapper):
    """Stacks num_frames RGB frames from the agentview camera."""

    def __init__(self, env, num_frames=3, size=64):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(num_frames * 3, size, size), dtype=np.uint8,
        )
        self._frames = deque([], maxlen=num_frames)
        self._size = size

    def _get_frame(self) -> np.ndarray:
        frame = self.env.render(width=self._size, height=self._size)  # (H,W,3)
        return frame.transpose(2, 0, 1)                               # (3,H,W)

    def _get_obs(self, is_reset=False):
        frame = self._get_frame()
        n = self._frames.maxlen if is_reset else 1
        for _ in range(n):
            self._frames.append(frame)
        return np.concatenate(list(self._frames), axis=0)             # (3*n,H,W)

    def reset(self):
        self.env.reset()
        return self._get_obs(is_reset=True)

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        return self._get_obs(), reward, done, info


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_env(cfg):
    """
    Make a LIBERO environment for TD-MPC2.

    Task naming convention:  libero-<suite>-<task_id>
      e.g. libero-90-57, libero-10-3, libero-spatial-0
    """
    task: str = cfg.task
    if not task.startswith('libero-'):
        raise ValueError(f'Not a LIBERO task: {task}')

    assert cfg.obs in {'state', 'rgb'}, \
        'LIBERO wrapper only supports obs=state or obs=rgb.'

    suite_name, task_id = _parse_task(task)
    resolution = getattr(cfg, 'resolution', 64)
    seed = getattr(cfg, 'seed', 0)
    max_episode_steps = getattr(cfg, 'max_episode_steps', 600)

    raw_env, task_description = _get_env(suite_name, task_id, resolution, seed)
    print(f'[LIBERO] Task: {task_description}')

    # Get action bounds from the inner robosuite env
    action_low, action_high = raw_env.env.action_spec
    env = LiberoWrapper(raw_env, action_low, action_high)

    if cfg.obs == 'rgb':
        env = LiberoPixels(env, num_frames=3, size=resolution)

    env = Timeout(env, max_episode_steps=max_episode_steps)
    return env
