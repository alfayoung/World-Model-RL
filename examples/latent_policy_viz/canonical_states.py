"""Canonical State Manager for tracking latent policy evolution.

This module handles the selection and storage of a fixed set of representative
states that will be used throughout training to track how the latent policy evolves.
"""

import numpy as np
import jax.numpy as jnp
from typing import Optional, List, Dict, Any
from flax.core import FrozenDict


class CanonicalStateManager:
    """Manages a fixed set of canonical states for tracking policy evolution.

    The canonical states are sampled once from the replay buffer and remain fixed
    throughout training. This allows us to track how the policy's outputs change
    for the same inputs over time.
    """

    def __init__(self, num_states: int = 15):
        """Initialize the canonical state manager.

        Args:
            num_states: Number of canonical states to maintain (default: 15)
        """
        self.num_states = num_states
        self.states = None  # Will hold FrozenDict of observations
        self.state_metadata = []  # Track source information
        self.is_initialized = False

    def initialize(
        self,
        replay_buffer,
        sample_method: str = 'random',
        seed: int = 42
    ):
        """Sample canonical states from the replay buffer.

        Args:
            replay_buffer: The replay buffer to sample from
            sample_method: Method for sampling ('random' or 'diverse')
            seed: Random seed for reproducibility
        """
        if self.is_initialized:
            print("Warning: Canonical states already initialized. Skipping.")
            return

        rng = np.random.RandomState(seed)

        if sample_method == 'random':
            # Sample uniformly from replay buffer
            batch = replay_buffer.sample(self.num_states)

            # Handle different batch formats (FrozenDict vs namedtuple)
            if isinstance(batch, (dict, FrozenDict)):
                # Real replay buffer returns FrozenDict
                self.states = batch['observations']
            elif hasattr(batch, 'observations'):
                # Mock/test replay buffer might return namedtuple
                self.states = batch.observations
            else:
                raise TypeError(f"Unexpected batch type: {type(batch)}")

            # Store metadata for debugging
            for i in range(self.num_states):
                self.state_metadata.append({
                    'state_idx': i,
                    'sample_method': 'random',
                })

        elif sample_method == 'diverse':
            # For future implementation: use clustering for diverse sampling
            raise NotImplementedError(
                "Diverse sampling not yet implemented. Use 'random' method."
            )
        else:
            raise ValueError(f"Unknown sample method: {sample_method}")

        self.is_initialized = True
        print(f"Initialized {self.num_states} canonical states using '{sample_method}' method")

        # Print observation structure for debugging
        if isinstance(self.states, (dict, FrozenDict)):
            for key, value in self.states.items():
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}")

    def get_states(self):
        """Return the canonical states.

        Returns:
            FrozenDict or dict containing the canonical state observations

        Raises:
            RuntimeError: If canonical states haven't been initialized
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Canonical states not initialized. Call initialize() first."
            )
        return self.states

    def get_num_states(self) -> int:
        """Return the number of canonical states."""
        return self.num_states

    def get_metadata(self) -> List[Dict[str, Any]]:
        """Return metadata about the canonical states."""
        return self.state_metadata

    def reset(self):
        """Reset the canonical state manager (useful for testing)."""
        self.states = None
        self.state_metadata = []
        self.is_initialized = False
