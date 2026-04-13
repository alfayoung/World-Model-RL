"""LIBERO + π₀ environment adapter for Dreamer-v3.

Wraps the LIBERO simulator so that:
  * observations  → ``image`` (H,W,3 uint8) and ``vector`` (8, float32)
  * actions       → 32-dim continuous noise fed into frozen π₀ to produce
                    the 7-dim robot action at each timestep
  * reward        → sparse: −1 every step, 0 at the success step
  * is_terminal   → True when the episode ends (success or timeout)

This keeps the action space identical to the DSRL / PixelSAC agent on main
so the two methods can be compared on equal footing.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np

# Vendored embodied must be importable before this module is used.
import dreamer_v3  # noqa: F401 — triggers sys.path setup
import elements  # from vendored dreamer_v3/embodied/…
import embodied


# ---------------------------------------------------------------------------
# Quaternion → axis-angle (copied from train_utils_sim.py)
# ---------------------------------------------------------------------------

def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


# ---------------------------------------------------------------------------
# LIBERO + π₀ embodied.Env
# ---------------------------------------------------------------------------

class LiberoPI0Env(embodied.Env):
    """Single LIBERO task driven by a frozen π₀ diffusion policy.

    Parameters
    ----------
    env_args:
        Dict passed to ``OffScreenRenderEnv`` (must contain ``bddl_file_name``,
        ``camera_heights``, ``camera_widths``).
    pi0_policy:
        OpenPI policy object (``openpi.policies.policy_config.create_trained_policy``).
    task_description:
        Language description of the task, used as the prompt for π₀.
    image_size:
        Pixel resolution to expose to Dreamer (image is cropped from LIBERO's
        raw resolution to ``(image_size, image_size, 3)``).
    seed:
        RNG seed for the LIBERO env.
    max_timesteps:
        Episode length limit (LIBERO default is 400).
    env_max_reward:
        Reward value returned by LIBERO on success (typically 1.0).
    """

    # Space constants fixed at construction time
    _STATE_DIM = 8   # eef_pos(3) + eef_rot_axisangle(3) + gripper(2)

    def __init__(
        self,
        env_args: dict,
        pi0_policy,
        task_description: str,
        *,
        image_size: int = 64,
        seed: int = 0,
        max_timesteps: int = 400,
        env_max_reward: float = 1.0,
    ):
        from libero.libero.envs import OffScreenRenderEnv
        from openpi_client import image_tools

        self._image_tools = image_tools
        self._pi0 = pi0_policy
        self._task_description = task_description
        self._image_size = image_size
        self._max_timesteps = max_timesteps
        self._env_max_reward = env_max_reward

        self._env = OffScreenRenderEnv(**env_args)
        self._env.seed(seed)

        self._done = True
        self._step_count = 0
        self._last_raw_obs = None

    # ------------------------------------------------------------------
    # embodied.Env interface
    # ------------------------------------------------------------------

    @property
    def obs_space(self) -> dict:
        H = W = self._image_size
        return {
            "image":       elements.Space(np.uint8,   (H, W, 3)),
            "vector":      elements.Space(np.float32, (self._STATE_DIM,)),
            "reward":      elements.Space(np.float32),
            "is_first":    elements.Space(bool),
            "is_last":     elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }

    @property
    def act_space(self) -> dict:
        return {
            "action": elements.Space(np.float32, (32,), -1.0, 1.0),
            "reset":  elements.Space(bool),
        }

    def step(self, action: dict) -> dict:
        if action["reset"] or self._done:
            return self._reset()
        noise_32 = np.array(action["action"], dtype=np.float32)  # (32,)
        robot_action = self._pi0_action(noise_32)
        raw_obs, reward, done, info = self._env.step(robot_action)
        self._step_count += 1
        self._last_raw_obs = raw_obs
        timeout = self._step_count >= self._max_timesteps
        self._done = bool(done) or timeout
        # Sparse reward: 0 on success, -1 otherwise
        is_success = (reward >= self._env_max_reward)
        sparse_reward = 0.0 if is_success else -1.0
        return self._make_obs(
            raw_obs,
            reward=sparse_reward,
            is_first=False,
            is_last=self._done,
            is_terminal=self._done,
        )

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset(self) -> dict:
        raw_obs = self._env.reset()
        self._done = False
        self._step_count = 0
        self._last_raw_obs = raw_obs
        return self._make_obs(raw_obs, reward=0.0, is_first=True)

    def _pi0_action(self, noise_32: np.ndarray) -> np.ndarray:
        """Convert 32-dim Dreamer noise action → 7-dim robot action via π₀."""
        raw_obs = self._last_raw_obs
        obs_pi0 = self._build_pi0_obs(raw_obs)
        # π₀ expects noise shaped (B=1, T_noise=50, D=32); we use the single
        # noise vector for t=0 and repeat a zero-noise tail (as in DSRL).
        noise_1 = noise_32[None, None, :]                          # (1, 1, 32)
        noise_tail = np.zeros((1, 49, 32), dtype=np.float32)
        noise = np.concatenate([noise_1, noise_tail], axis=1)      # (1, 50, 32)
        actions = self._pi0.infer(obs_pi0, noise=noise)["actions"] # (N, 7)
        return actions[0]  # first timestep

    def _build_pi0_obs(self, raw_obs: dict) -> dict:
        """Build the observation dict expected by π₀ from raw LIBERO obs."""
        img = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(
            raw_obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img   = self._image_tools.convert_to_uint8(
            self._image_tools.resize_with_pad(img,   224, 224))
        wrist = self._image_tools.convert_to_uint8(
            self._image_tools.resize_with_pad(wrist, 224, 224))
        state = np.concatenate([
            raw_obs["robot0_eef_pos"],
            _quat2axisangle(raw_obs["robot0_eef_quat"]),
            raw_obs["robot0_gripper_qpos"],
        ]).astype(np.float32)
        return {
            "observation/image":       img,
            "observation/wrist_image": wrist,
            "observation/state":       state,
            "prompt":                  self._task_description,
        }

    def _extract_dreamer_obs(self, raw_obs: dict) -> dict:
        """Extract image + proprioceptive state for Dreamer's world model."""
        import PIL.Image
        img = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
        if img.shape[0] != self._image_size:
            img = np.array(
                PIL.Image.fromarray(img).resize(
                    (self._image_size, self._image_size)))
        vector = np.concatenate([
            raw_obs["robot0_eef_pos"],
            _quat2axisangle(raw_obs["robot0_eef_quat"]),
            raw_obs["robot0_gripper_qpos"],
        ]).astype(np.float32)
        return {"image": img.astype(np.uint8), "vector": vector}

    def _make_obs(
        self,
        raw_obs: dict,
        reward: float,
        is_first: bool = False,
        is_last: bool = False,
        is_terminal: bool = False,
    ) -> dict:
        dreamer_obs = self._extract_dreamer_obs(raw_obs)
        return {
            **dreamer_obs,
            "reward":      np.float32(reward),
            "is_first":    is_first,
            "is_last":     is_last,
            "is_terminal": is_terminal,
        }
