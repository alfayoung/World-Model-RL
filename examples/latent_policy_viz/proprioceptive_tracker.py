"""Proprioceptive State Trajectory Tracker.

This module records the proprioceptive state of the robot during policy rollouts,
enabling visualization of the robot's movement trajectories in 3D space.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ProprioceptiveTrajectory:
    """Single trajectory of proprioceptive states."""
    episode_id: int
    timesteps: List[int]
    states: np.ndarray  # Shape: (T, state_dim)
    success: bool
    episode_return: float
    training_step: int  # When this trajectory was collected


class ProprioceptiveTracker:
    """Tracks proprioceptive state trajectories across training.

    Records the proprioceptive state (joint positions, end-effector pose, gripper state)
    at each timestep during policy rollouts, enabling 3D visualization of robot motion.
    """

    def __init__(self, max_trajectories: int = 100):
        """Initialize the proprioceptive tracker.

        Args:
            max_trajectories: Maximum number of trajectories to store
        """
        self.max_trajectories = max_trajectories
        self.trajectories: List[ProprioceptiveTrajectory] = []
        self.episode_counter = 0

    def start_trajectory(self, training_step: int):
        """Start tracking a new trajectory.

        Args:
            training_step: Current training step
        """
        self._current_traj_states = []
        self._current_traj_timesteps = []
        self._current_training_step = training_step

    def record_state(self, state: np.ndarray, timestep: int):
        """Record a proprioceptive state at current timestep.

        Args:
            state: Proprioceptive state vector (joint positions, EEF pose, gripper)
            timestep: Current timestep in the episode
        """
        if not hasattr(self, '_current_traj_states'):
            raise RuntimeError("Call start_trajectory() before recording states")

        self._current_traj_states.append(state.copy())
        self._current_traj_timesteps.append(timestep)

    def end_trajectory(self, success: bool, episode_return: float):
        """Finalize the current trajectory.

        Args:
            success: Whether the episode was successful
            episode_return: Total return for the episode
        """
        if not hasattr(self, '_current_traj_states'):
            raise RuntimeError("No active trajectory to end")

        if len(self._current_traj_states) == 0:
            print("Warning: Ending trajectory with no recorded states")
            return

        # Create trajectory record
        trajectory = ProprioceptiveTrajectory(
            episode_id=self.episode_counter,
            timesteps=self._current_traj_timesteps,
            states=np.array(self._current_traj_states),
            success=success,
            episode_return=episode_return,
            training_step=self._current_training_step
        )

        # Store trajectory (FIFO if exceeding max)
        self.trajectories.append(trajectory)
        if len(self.trajectories) > self.max_trajectories:
            self.trajectories.pop(0)

        self.episode_counter += 1

        # Clean up temporary storage
        del self._current_traj_states
        del self._current_traj_timesteps

        print(f"Recorded trajectory {trajectory.episode_id}: "
              f"{len(trajectory.timesteps)} timesteps, "
              f"success={success}, return={episode_return:.2f}")

    def get_recent_trajectories(self, n: int = 10, only_successful: bool = False) -> List[ProprioceptiveTrajectory]:
        """Get the most recent trajectories.

        Args:
            n: Number of recent trajectories to return
            only_successful: If True, only return successful trajectories

        Returns:
            List of recent trajectories
        """
        trajs = self.trajectories
        if only_successful:
            trajs = [t for t in trajs if t.success]
        return trajs[-n:]

    def get_trajectories_at_step(self, training_step: int, window: int = 0) -> List[ProprioceptiveTrajectory]:
        """Get trajectories collected at a specific training step.

        Args:
            training_step: Training step to query
            window: Include trajectories within ±window steps

        Returns:
            List of trajectories from that training step
        """
        return [
            t for t in self.trajectories
            if abs(t.training_step - training_step) <= window
        ]

    def get_stats(self) -> Dict:
        """Return statistics about tracked trajectories."""
        if len(self.trajectories) == 0:
            return {
                'num_trajectories': 0,
                'num_successful': 0,
                'success_rate': 0.0,
            }

        num_successful = sum(t.success for t in self.trajectories)
        avg_return = np.mean([t.episode_return for t in self.trajectories])
        avg_length = np.mean([len(t.timesteps) for t in self.trajectories])

        return {
            'num_trajectories': len(self.trajectories),
            'num_successful': num_successful,
            'success_rate': num_successful / len(self.trajectories),
            'avg_return': avg_return,
            'avg_length': avg_length,
            'state_dim': self.trajectories[0].states.shape[1] if len(self.trajectories) > 0 else 0,
        }

    def reset(self):
        """Clear all recorded trajectories."""
        self.trajectories = []
        self.episode_counter = 0
