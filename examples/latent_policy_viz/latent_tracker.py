"""Latent Policy Output Tracker.

This module records the latent policy outputs (w vectors) at multiple training
checkpoints for a fixed set of canonical states.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class LatentRecord:
    """Single record of latent policy output."""
    step: int
    state_idx: int
    w: np.ndarray  # Latent action vector


class LatentPolicyTracker:
    """Tracks latent policy outputs across training checkpoints.

    Records π^w(s) for each canonical state at multiple training steps,
    enabling visualization of how the policy evolves over time.
    """

    def __init__(self):
        """Initialize the latent policy tracker."""
        self.records: List[LatentRecord] = []
        self.checkpoints: List[int] = []  # Training steps where we recorded

    def record_checkpoint(
        self,
        step: int,
        agent,
        canonical_states,
        temperature: float = 1.0,
        seed: int = None
    ):
        """Record latent outputs for all canonical states at current checkpoint.

        Args:
            step: Current training step
            agent: The SAC agent with the latent policy
            canonical_states: Batch of canonical state observations
            temperature: Sampling temperature (default: 1.0 for standard sampling)
            seed: Random seed for reproducible sampling (optional)
        """
        # Sample latent actions from the policy
        # agent.sample_actions returns actions sampled from π^w(·|s)
        if seed is not None:
            # For reproducible sampling (useful for testing)
            # Note: JAX agents may need a different approach for seeding
            w_vectors = agent.eval_actions(canonical_states)
        else:
            w_vectors = agent.sample_actions(canonical_states)

        # Convert to numpy if needed
        if isinstance(w_vectors, jnp.ndarray):
            w_vectors = np.array(w_vectors)

        # Store each (step, state_idx, w) record
        num_states = w_vectors.shape[0]
        for state_idx in range(num_states):
            record = LatentRecord(
                step=step,
                state_idx=state_idx,
                w=w_vectors[state_idx].copy()
            )
            self.records.append(record)

        # Track this checkpoint
        if step not in self.checkpoints:
            self.checkpoints.append(step)

        print(f"Recorded latent outputs at step {step} for {num_states} canonical states")

    def get_trajectory_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Convert records to arrays for visualization.

        Returns:
            w_matrix: Array of shape (num_records, action_dim) containing all w vectors
            metadata: Dict with 'step' and 'state_idx' arrays aligned with w_matrix
        """
        if len(self.records) == 0:
            raise RuntimeError("No records available. Call record_checkpoint() first.")

        # Extract data from records
        steps = []
        state_indices = []
        w_vectors = []

        for record in self.records:
            steps.append(record.step)
            state_indices.append(record.state_idx)
            w_vectors.append(record.w)

        # Convert to numpy arrays
        w_matrix = np.array(w_vectors)  # Shape: (num_records, action_dim)
        metadata = {
            'step': np.array(steps),
            'state_idx': np.array(state_indices),
        }

        return w_matrix, metadata

    def get_state_trajectory(self, state_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the trajectory for a specific canonical state.

        Args:
            state_idx: Index of the canonical state

        Returns:
            steps: Array of training steps
            w_vectors: Array of w vectors for this state over time
        """
        steps = []
        w_vectors = []

        for record in self.records:
            if record.state_idx == state_idx:
                steps.append(record.step)
                w_vectors.append(record.w)

        return np.array(steps), np.array(w_vectors)

    def get_num_checkpoints(self) -> int:
        """Return the number of checkpoints recorded."""
        return len(self.checkpoints)

    def get_num_records(self) -> int:
        """Return the total number of records."""
        return len(self.records)

    def get_checkpoints(self) -> List[int]:
        """Return the list of checkpoint steps."""
        return sorted(self.checkpoints)

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the tracked data."""
        if len(self.records) == 0:
            return {
                'num_checkpoints': 0,
                'num_records': 0,
                'num_states': 0,
            }

        num_states = len(set(r.state_idx for r in self.records))

        return {
            'num_checkpoints': len(self.checkpoints),
            'num_records': len(self.records),
            'num_states': num_states,
            'checkpoint_steps': self.get_checkpoints(),
            'action_dim': self.records[0].w.shape[0],
        }

    def reset(self):
        """Clear all recorded data."""
        self.records = []
        self.checkpoints = []
