"""Simple numpy replay buffer compatible with pixel + state observations."""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional


class ReplayBuffer:
    """
    Stores transitions with dict observations (pixels, state).
    Supports uniform random sampling.
    """

    def __init__(
        self,
        capacity: int,
        obs_space: dict,          # {'pixels': shape, 'state': shape}
        action_dim: int,
        horizon: int = 1,
    ):
        self.capacity = capacity
        self.horizon = horizon
        self._ptr = 0
        self._size = 0

        # Allocate storage
        self._obs: Dict[str, np.ndarray] = {
            k: np.zeros((capacity, *v), dtype=np.uint8 if k == 'pixels' else np.float32)
            for k, v in obs_space.items()
        }
        self._next_obs: Dict[str, np.ndarray] = {
            k: np.zeros((capacity, *v), dtype=np.uint8 if k == 'pixels' else np.float32)
            for k, v in obs_space.items()
        }
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    # ------------------------------------------------------------------
    def insert(
        self,
        obs: dict,
        action: np.ndarray,
        reward: float,
        next_obs: dict,
        done: bool,
    ):
        i = self._ptr
        for k in self._obs:
            self._obs[k][i] = obs[k]
            self._next_obs[k][i] = next_obs[k]
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = float(done)
        self._ptr = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idxs = np.random.randint(0, self._size, size=batch_size)
        return {
            'observations': {k: self._obs[k][idxs] for k in self._obs},
            'next_observations': {k: self._next_obs[k][idxs] for k in self._next_obs},
            'actions': self._actions[idxs],
            'rewards': self._rewards[idxs],
            'dones': self._dones[idxs],
        }

    def __len__(self) -> int:
        return self._size
