"""Episode-based replay buffer for Dreamer-v3.

Dreamer-v3 trains on contiguous sequence chunks sampled from stored episodes,
rather than the independent transitions used by SAC/TD-MPC2.  This buffer
stores complete episodes and samples (B, T) chunks for RSSM training.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np


class EpisodeReplayBuffer:
    """Circular buffer of episodes; samples contiguous (B, T) batches.

    Parameters
    ----------
    capacity:
        Maximum number of *transitions* to keep across all episodes.
    seq_len:
        Length T of each sampled sequence chunk (batch_length in the paper).
    image_size:
        (H, W) of pixel observations stored in uint8.
    state_dim:
        Dimension of the proprioceptive state vector.
    action_dim:
        Dimension of the noise action (32 for DSRL π₀ steering).
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        seq_len: int = 64,
        image_size: int = 64,
        state_dim: int = 8,
        action_dim: int = 32,
    ):
        self.capacity = capacity
        self.seq_len = seq_len
        self.image_size = image_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        self._episodes: deque[dict] = deque()
        self._total_steps: int = 0

    def __len__(self) -> int:
        return self._total_steps

    def add_episode(
        self,
        images:   list[np.ndarray],  # list of (H, W, 3) uint8, length T+1
        states:   list[np.ndarray],  # list of (state_dim,) float32, length T+1
        actions:  list[np.ndarray],  # list of (action_dim,) float32, length T
        rewards:  np.ndarray,        # (T,) float32
        dones:    np.ndarray,        # (T,) bool
    ):
        """Store a complete episode.

        Images and states have length T+1 (including the terminal observation).
        Actions/rewards/dones have length T.
        """
        T = len(actions)
        if T < self.seq_len:
            return  # too short to sample from — skip

        ep = {
            'image':   np.stack(images).astype(np.uint8),    # (T+1, H, W, 3)
            'state':   np.stack(states).astype(np.float32),  # (T+1, D)
            'action':  np.stack(actions).astype(np.float32), # (T, A)
            'reward':  np.array(rewards, dtype=np.float32),  # (T,)
            'done':    np.array(dones,   dtype=np.float32),  # (T,)
            'length':  T,
        }
        self._episodes.append(ep)
        self._total_steps += T

        # Evict oldest episodes if over capacity
        while self._total_steps > self.capacity and self._episodes:
            evicted = self._episodes.popleft()
            self._total_steps -= evicted['length']

    def sample(self, batch_size: int) -> Optional[dict]:
        """Sample a (B, T) batch of contiguous sequence chunks.

        Returns None if the buffer doesn't have enough data yet.
        """
        if self._total_steps < batch_size * self.seq_len:
            return None

        imgs, states, acts, rews, dones, is_firsts = [], [], [], [], [], []
        # Weight episodes by length (longer eps give more training signal)
        eps = list(self._episodes)
        weights = np.array([max(0, ep['length'] - self.seq_len + 1) for ep in eps],
                           dtype=np.float64)
        if weights.sum() == 0:
            return None
        weights /= weights.sum()

        chosen = random.choices(eps, weights=weights.tolist(), k=batch_size)
        for ep in chosen:
            T = ep['length']
            max_start = T - self.seq_len
            t0 = random.randint(0, max(0, max_start))
            t1 = t0 + self.seq_len

            # obs at t0..t1 (seq_len steps), actions t0..t1-1 (seq_len)
            img_chunk   = ep['image'][t0: t1]    # (T, H, W, 3)
            state_chunk = ep['state'][t0: t1]    # (T, D)
            act_chunk   = ep['action'][t0: t1]   # (T, A)
            rew_chunk   = ep['reward'][t0: t1]   # (T,)
            done_chunk  = ep['done'][t0: t1]     # (T,)

            # is_first: True only at t0 if that corresponds to episode start
            is_first = np.zeros(self.seq_len, dtype=bool)
            if t0 == 0:
                is_first[0] = True

            imgs.append(img_chunk)
            states.append(state_chunk)
            acts.append(act_chunk)
            rews.append(rew_chunk)
            dones.append(done_chunk)
            is_firsts.append(is_first)

        return {
            'image':    np.stack(imgs,      axis=0).astype(np.uint8),    # (B, T, H, W, 3)
            'state':    np.stack(states,    axis=0).astype(np.float32),  # (B, T, D)
            'action':   np.stack(acts,      axis=0).astype(np.float32),  # (B, T, A)
            'reward':   np.stack(rews,      axis=0).astype(np.float32),  # (B, T)
            'done':     np.stack(dones,     axis=0).astype(np.float32),  # (B, T)
            'is_first': np.stack(is_firsts, axis=0),                     # (B, T) bool
        }
