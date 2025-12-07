"""3D Proprioceptive Trajectory Visualization.

This module creates 3D plots showing the robot's proprioceptive state trajectories
during policy rollouts, similar to the evolution visualization but for actual robot motion.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from typing import List, Optional, Tuple, Literal
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class ProprioceptivePlotter:
    """Creates 3D visualizations of proprioceptive state trajectories.

    Can either directly plot 3D positions (e.g., end-effector position) or use
    dimensionality reduction to project high-dimensional proprioceptive states to 3D.
    """

    def __init__(
        self,
        mode: Literal['eef_pos', 'pca', 'umap'] = 'eef_pos',
        eef_indices: Optional[List[int]] = None,
        **reducer_kwargs
    ):
        """Initialize the proprioceptive plotter.

        Args:
            mode: Visualization mode:
                - 'eef_pos': Plot end-effector position directly (requires 3D position in state)
                - 'pca': Use PCA to reduce state to 3D
                - 'umap': Use UMAP to reduce state to 3D
            eef_indices: Indices of end-effector position in state vector (e.g., [0, 1, 2])
                        Required for 'eef_pos' mode. Default: [0, 1, 2]
            **reducer_kwargs: Additional arguments for PCA or UMAP
        """
        if mode not in ['eef_pos', 'pca', 'umap']:
            raise ValueError(f"Unknown mode: {mode}. Use 'eef_pos', 'pca', or 'umap'.")

        if mode == 'umap' and not UMAP_AVAILABLE:
            raise ImportError(
                "UMAP not available. Install with: pip install umap-learn"
            )

        self.mode = mode
        self.eef_indices = eef_indices if eef_indices is not None else [0, 1, 2]
        self.reducer_kwargs = reducer_kwargs
        self.reducer = None

    def plot_trajectories(
        self,
        trajectories,
        figsize: Tuple[int, int] = (12, 10),
        colormap: str = 'viridis',
        show_start_end: bool = True,
        show_grid: bool = True,
        elev: int = 20,
        azim: int = 45,
        alpha: float = 0.7,
        linewidth: float = 2.0,
        title: Optional[str] = None,
        label_trajectories: bool = True
    ) -> Figure:
        """Plot multiple proprioceptive trajectories in 3D.

        Args:
            trajectories: List of ProprioceptiveTrajectory objects
            figsize: Figure size (width, height)
            colormap: Matplotlib colormap for coloring trajectories
            show_start_end: Whether to mark start (circle) and end (star) points
            show_grid: Whether to show grid
            elev: Elevation angle for 3D view
            azim: Azimuth angle for 3D view
            alpha: Transparency of trajectory lines
            linewidth: Width of trajectory lines
            title: Custom title for the plot
            label_trajectories: Whether to label each trajectory

        Returns:
            matplotlib Figure object
        """
        if len(trajectories) == 0:
            raise ValueError("No trajectories to plot")

        # Collect all states from all trajectories
        all_states = []
        traj_lengths = []
        for traj in trajectories:
            all_states.append(traj.states)
            traj_lengths.append(len(traj.states))

        # Convert to 3D coordinates based on mode
        if self.mode == 'eef_pos':
            # Directly extract end-effector positions
            traj_3d_list = []
            for states in all_states:
                if states.shape[1] < max(self.eef_indices) + 1:
                    raise ValueError(
                        f"State dimension {states.shape[1]} too small for eef_indices {self.eef_indices}"
                    )
                traj_3d_list.append(states[:, self.eef_indices])

        elif self.mode == 'pca':
            # Concatenate all states and fit PCA
            all_states_concat = np.vstack(all_states)
            self.reducer = PCA(n_components=3, **self.reducer_kwargs)
            all_3d = self.reducer.fit_transform(all_states_concat)

            # Split back into trajectories
            traj_3d_list = []
            start_idx = 0
            for length in traj_lengths:
                traj_3d_list.append(all_3d[start_idx:start_idx + length])
                start_idx += length

            explained_var = self.reducer.explained_variance_ratio_
            print(f"PCA explained variance: {explained_var}")

        elif self.mode == 'umap':
            # Concatenate all states and fit UMAP
            all_states_concat = np.vstack(all_states)
            umap_params = {
                'n_components': 3,
                'n_neighbors': min(15, len(all_states_concat) - 1),
                'min_dist': 0.1,
                'metric': 'euclidean',
                **self.reducer_kwargs
            }
            self.reducer = umap.UMAP(**umap_params)
            all_3d = self.reducer.fit_transform(all_states_concat)

            # Split back into trajectories
            traj_3d_list = []
            start_idx = 0
            for length in traj_lengths:
                traj_3d_list.append(all_3d[start_idx:start_idx + length])
                start_idx += length

        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Color mapping
        cmap = plt.get_cmap(colormap)
        n_trajs = len(trajectories)

        # Plot each trajectory
        for i, (traj, traj_3d) in enumerate(zip(trajectories, traj_3d_list)):
            # Assign unique color to each trajectory
            color = cmap(i / max(1, n_trajs - 1))

            # Use linestyle to distinguish success/failure
            if traj.success:
                linestyle = '-'
                marker_color = 'green'
            else:
                linestyle = '--'
                marker_color = 'red'

            # Plot trajectory line
            ax.plot(
                traj_3d[:, 0],
                traj_3d[:, 1],
                traj_3d[:, 2],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                linestyle=linestyle,
                label=f"Traj {traj.episode_id}" if label_trajectories and i < 10 else None
            )

            # Mark start and end points
            if show_start_end:
                # Start point (circle)
                ax.scatter(
                    traj_3d[0, 0],
                    traj_3d[0, 1],
                    traj_3d[0, 2],
                    c='blue',
                    marker='o',
                    s=100,
                    edgecolors='white',
                    linewidths=2,
                    zorder=10
                )
                # End point (star)
                ax.scatter(
                    traj_3d[-1, 0],
                    traj_3d[-1, 1],
                    traj_3d[-1, 2],
                    c=marker_color,
                    marker='*',
                    s=200,
                    edgecolors='white',
                    linewidths=2,
                    zorder=10
                )

        # Set labels based on mode
        if self.mode == 'eef_pos':
            ax.set_xlabel('X Position (m)', fontsize=11)
            ax.set_ylabel('Y Position (m)', fontsize=11)
            ax.set_zlabel('Z Position (m)', fontsize=11)
            default_title = '3D End-Effector Trajectories'
        elif self.mode == 'pca':
            ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})', fontsize=11)
            ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})', fontsize=11)
            ax.set_zlabel(f'PC3 ({explained_var[2]:.1%})', fontsize=11)
            default_title = '3D Proprioceptive Trajectories (PCA)'
        else:  # umap
            ax.set_xlabel('UMAP Dim 1', fontsize=11)
            ax.set_ylabel('UMAP Dim 2', fontsize=11)
            ax.set_zlabel('UMAP Dim 3', fontsize=11)
            default_title = '3D Proprioceptive Trajectories (UMAP)'

        # Set title
        ax.set_title(title or default_title, fontsize=14, fontweight='bold', pad=20)

        # View angle
        ax.view_init(elev=elev, azim=azim)

        # Grid
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')

        # Legend (only if labeling and not too many trajectories)
        if label_trajectories and n_trajs <= 10:
            ax.legend(loc='upper left', fontsize=9)

        # Stats text
        num_successful = sum(t.success for t in trajectories)
        stats_text = (
            f'Trajectories: {n_trajs} | '
            f'Successful: {num_successful} ({100*num_successful/n_trajs:.1f}%) | '
            f'Mode: {self.mode}'
        )
        fig.text(
            0.5, 0.02, stats_text,
            ha='center', fontsize=10, style='italic', color='gray'
        )

        plt.tight_layout()

        return fig

    def plot_single_trajectory(
        self,
        trajectory,
        figsize: Tuple[int, int] = (10, 8),
        colormap: str = 'plasma',
        show_time_gradient: bool = True,
        title: Optional[str] = None,
        **kwargs
    ) -> Figure:
        """Plot a single trajectory with time-based color gradient.

        Args:
            trajectory: ProprioceptiveTrajectory object
            figsize: Figure size (width, height)
            colormap: Matplotlib colormap for time gradient
            show_time_gradient: Color trajectory by time progression
            title: Custom title
            **kwargs: Additional arguments passed to plot_trajectories

        Returns:
            matplotlib Figure object
        """
        # Get 3D representation
        states = trajectory.states

        if self.mode == 'eef_pos':
            traj_3d = states[:, self.eef_indices]
        elif self.mode == 'pca':
            self.reducer = PCA(n_components=3, **self.reducer_kwargs)
            traj_3d = self.reducer.fit_transform(states)
        elif self.mode == 'umap':
            umap_params = {
                'n_components': 3,
                'n_neighbors': min(15, len(states) - 1),
                'min_dist': 0.1,
                **self.reducer_kwargs
            }
            self.reducer = umap.UMAP(**umap_params)
            traj_3d = self.reducer.fit_transform(states)

        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot with time-based coloring
        if show_time_gradient:
            cmap = plt.get_cmap(colormap)
            n_points = len(traj_3d)

            # Plot segments with color gradient
            for i in range(n_points - 1):
                color = cmap(i / max(1, n_points - 1))
                ax.plot(
                    traj_3d[i:i+2, 0],
                    traj_3d[i:i+2, 1],
                    traj_3d[i:i+2, 2],
                    color=color,
                    linewidth=2.5,
                    alpha=0.8
                )

            # Add colorbar for time
            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=mcolors.Normalize(vmin=0, vmax=n_points-1)
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Timestep', shrink=0.8, pad=0.1)
        else:
            # Single color
            color = 'green' if trajectory.success else 'red'
            ax.plot(
                traj_3d[:, 0],
                traj_3d[:, 1],
                traj_3d[:, 2],
                color=color,
                linewidth=2.5,
                alpha=0.8
            )

        # Mark start and end
        ax.scatter(
            traj_3d[0, 0], traj_3d[0, 1], traj_3d[0, 2],
            c='blue', marker='o', s=150, edgecolors='white', linewidths=2, label='Start'
        )
        ax.scatter(
            traj_3d[-1, 0], traj_3d[-1, 1], traj_3d[-1, 2],
            c='green' if trajectory.success else 'red',
            marker='*', s=250, edgecolors='white', linewidths=2, label='End'
        )

        # Labels
        if self.mode == 'eef_pos':
            ax.set_xlabel('X Position (m)', fontsize=11)
            ax.set_ylabel('Y Position (m)', fontsize=11)
            ax.set_zlabel('Z Position (m)', fontsize=11)
        else:
            ax.set_xlabel(f'{self.mode.upper()} Dim 1', fontsize=11)
            ax.set_ylabel(f'{self.mode.upper()} Dim 2', fontsize=11)
            ax.set_zlabel(f'{self.mode.upper()} Dim 3', fontsize=11)

        # Title
        if title is None:
            success_str = "Successful" if trajectory.success else "Failed"
            title = f'Trajectory {trajectory.episode_id} ({success_str})'
        ax.set_title(title, fontsize=13, fontweight='bold')

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=kwargs.get('elev', 20), azim=kwargs.get('azim', 45))

        plt.tight_layout()

        return fig

    def plot_comparison(
        self,
        trajectories_before: List,
        trajectories_after: List,
        figsize: Tuple[int, int] = (16, 7),
        title_before: str = "Before Training",
        title_after: str = "After Training",
        **kwargs
    ) -> Figure:
        """Plot side-by-side comparison of trajectories at different training stages.

        Args:
            trajectories_before: Trajectories from early training
            trajectories_after: Trajectories from later training
            figsize: Figure size (width, height)
            title_before: Title for left subplot
            title_after: Title for right subplot
            **kwargs: Additional arguments for plotting

        Returns:
            matplotlib Figure with two subplots
        """
        fig = plt.figure(figsize=figsize)

        # Left subplot: before
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_on_axis(ax1, trajectories_before, title_before, **kwargs)

        # Right subplot: after
        ax2 = fig.add_subplot(122, projection='3d')
        self._plot_on_axis(ax2, trajectories_after, title_after, **kwargs)

        plt.tight_layout()

        return fig

    def _plot_on_axis(self, ax, trajectories, title, **kwargs):
        """Helper to plot trajectories on a given 3D axis."""
        # Process trajectories to 3D
        all_states = [traj.states for traj in trajectories]

        if self.mode == 'eef_pos':
            traj_3d_list = [states[:, self.eef_indices] for states in all_states]
        elif self.mode in ['pca', 'umap']:
            all_states_concat = np.vstack(all_states)
            if self.mode == 'pca':
                reducer = PCA(n_components=3, **self.reducer_kwargs)
            else:
                reducer = umap.UMAP(n_components=3, **self.reducer_kwargs)
            all_3d = reducer.fit_transform(all_states_concat)

            traj_3d_list = []
            start_idx = 0
            for states in all_states:
                length = len(states)
                traj_3d_list.append(all_3d[start_idx:start_idx + length])
                start_idx += length

        # Plot
        cmap = plt.get_cmap('viridis')
        for i, (traj, traj_3d) in enumerate(zip(trajectories, traj_3d_list)):
            color = 'green' if traj.success else 'gray'
            ax.plot(
                traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2],
                color=color, alpha=0.7, linewidth=2
            )

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=kwargs.get('elev', 20), azim=kwargs.get('azim', 45))
