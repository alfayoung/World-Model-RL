"""Latent Policy Evolution Visualization.

This module creates 2D trajectory plots showing how the latent policy outputs
evolve over training using dimensionality reduction (PCA or UMAP).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from typing import Optional, Literal, Tuple
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class LatentEvolutionPlotter:
    """Creates visualizations of latent policy evolution over training.

    Uses dimensionality reduction (PCA or UMAP) to project high-dimensional
    latent actions to 2D and plots trajectories showing how they evolve.
    """

    def __init__(
        self,
        method: Literal['pca', 'umap'] = 'pca',
        n_components: int = 2,
        **reducer_kwargs
    ):
        """Initialize the evolution plotter.

        Args:
            method: Dimensionality reduction method ('pca' or 'umap')
            n_components: Number of components (should be 2 for visualization)
            **reducer_kwargs: Additional arguments for the reducer
        """
        if method not in ['pca', 'umap']:
            raise ValueError(f"Unknown method: {method}. Use 'pca' or 'umap'.")

        if method == 'umap' and not UMAP_AVAILABLE:
            raise ImportError(
                "UMAP not available. Install with: pip install umap-learn"
            )

        self.method = method
        self.n_components = n_components
        self.reducer_kwargs = reducer_kwargs
        self.reducer = None  # Will be fitted when plotting

    def fit_and_plot(
        self,
        tracker,
        current_step: int,
        figsize: Tuple[int, int] = (12, 10),
        show_arrows: bool = True,
        show_labels: bool = True,
        colormap: str = 'viridis'
    ) -> Figure:
        """Generate evolution visualization from tracker data.

        Args:
            tracker: LatentPolicyTracker instance with recorded data
            current_step: Current training step (for title)
            figsize: Figure size (width, height)
            show_arrows: Whether to show arrows indicating trajectory direction
            show_labels: Whether to label start points with state indices
            colormap: Matplotlib colormap for time progression

        Returns:
            matplotlib Figure object
        """
        # Get all recorded w vectors and metadata
        w_matrix, metadata = tracker.get_trajectory_data()
        steps = metadata['step']
        state_indices = metadata['state_idx']

        print(f"Fitting {self.method.upper()} on {w_matrix.shape[0]} vectors "
              f"of dimension {w_matrix.shape[1]}")

        # Fit dimensionality reduction on ALL data
        if self.method == 'pca':
            self.reducer = PCA(n_components=self.n_components, **self.reducer_kwargs)
            w_2d = self.reducer.fit_transform(w_matrix)
            explained_var = self.reducer.explained_variance_ratio_
            print(f"PCA explained variance: {explained_var[:2]}")

        elif self.method == 'umap':
            # UMAP hyperparameters for good trajectory visualization
            umap_params = {
                'n_components': self.n_components,
                'n_neighbors': min(15, len(w_matrix) - 1),
                'min_dist': 0.1,
                'metric': 'euclidean',
                **self.reducer_kwargs
            }
            self.reducer = umap.UMAP(**umap_params)
            w_2d = self.reducer.fit_transform(w_matrix)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get unique states and checkpoints
        unique_states = np.unique(state_indices)
        unique_steps = np.unique(steps)
        num_checkpoints = len(unique_steps)

        # Create color map for time progression
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i / max(1, num_checkpoints - 1)) for i in range(num_checkpoints)]

        # Create step-to-color mapping
        step_to_color_idx = {step: idx for idx, step in enumerate(sorted(unique_steps))}

        # Plot trajectory for each canonical state
        for state_idx in unique_states:
            # Get trajectory for this state
            mask = state_indices == state_idx
            trajectory = w_2d[mask]
            state_steps = steps[mask]

            # Sort by step to ensure proper trajectory ordering
            sort_idx = np.argsort(state_steps)
            trajectory = trajectory[sort_idx]
            state_steps = state_steps[sort_idx]

            # Plot trajectory with color gradient
            for i in range(len(trajectory) - 1):
                step_idx = step_to_color_idx[state_steps[i]]
                ax.plot(
                    trajectory[i:i+2, 0],
                    trajectory[i:i+2, 1],
                    'o-',
                    color=colors[step_idx],
                    alpha=0.7,
                    linewidth=2,
                    markersize=6,
                    markeredgecolor='white',
                    markeredgewidth=0.5
                )

                # Add arrow to show direction
                if show_arrows and i < len(trajectory) - 1:
                    dx = trajectory[i+1, 0] - trajectory[i, 0]
                    dy = trajectory[i+1, 1] - trajectory[i, 1]
                    ax.annotate(
                        '',
                        xy=(trajectory[i+1, 0], trajectory[i+1, 1]),
                        xytext=(trajectory[i, 0], trajectory[i, 1]),
                        arrowprops=dict(
                            arrowstyle='->',
                            color=colors[step_idx],
                            alpha=0.5,
                            lw=1.5
                        )
                    )

            # Plot final point
            final_step_idx = step_to_color_idx[state_steps[-1]]
            ax.plot(
                trajectory[-1, 0],
                trajectory[-1, 1],
                'o',
                color=colors[final_step_idx],
                markersize=6,
                markeredgecolor='white',
                markeredgewidth=0.5
            )

            # Label start point
            if show_labels:
                ax.text(
                    trajectory[0, 0],
                    trajectory[0, 1],
                    f'S{state_idx}',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
                    ha='center',
                    va='bottom'
                )

        # Create colorbar for time progression
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=mcolors.Normalize(vmin=min(unique_steps), vmax=max(unique_steps))
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Training Step')

        # Labels and title
        if self.method == 'pca':
            ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12)
            title = f'Latent Policy Evolution (PCA) - Step {current_step}'
        else:
            ax.set_xlabel('UMAP Dimension 1', fontsize=12)
            ax.set_ylabel('UMAP Dimension 2', fontsize=12)
            title = f'Latent Policy Evolution (UMAP) - Step {current_step}'

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add stats text
        stats_text = (
            f'States: {len(unique_states)} | '
            f'Checkpoints: {num_checkpoints} | '
            f'Latent dim: {w_matrix.shape[1]}'
        )
        fig.text(
            0.5, 0.02, stats_text,
            ha='center', fontsize=10, style='italic', color='gray'
        )

        plt.tight_layout()

        return fig

    def plot_per_state_evolution(
        self,
        tracker,
        current_step: int,
        figsize: Optional[Tuple[int, int]] = None,
        colormap: str = 'viridis',
        return_individual: bool = False
    ):
        """Create separate subplot for each canonical state's evolution.

        Args:
            tracker: LatentPolicyTracker instance
            current_step: Current training step
            figsize: Figure size (auto-calculated if None)
            colormap: Colormap for time progression
            return_individual: If True, return dict of individual figures per state.
                             If False, return single figure with all states as subplots.

        Returns:
            If return_individual=False: matplotlib Figure object with subplots
            If return_individual=True: dict mapping state_idx to individual Figure objects
        """
        w_matrix, metadata = tracker.get_trajectory_data()

        # Fit dimensionality reduction
        if self.method == 'pca':
            self.reducer = PCA(n_components=self.n_components)
            w_2d = self.reducer.fit_transform(w_matrix)
            explained_var = self.reducer.explained_variance_ratio_
        else:
            umap_params = {
                'n_components': self.n_components,
                'n_neighbors': min(15, len(w_matrix) - 1),
                'min_dist': 0.1,
            }
            self.reducer = umap.UMAP(**umap_params)
            w_2d = self.reducer.fit_transform(w_matrix)
            explained_var = None

        unique_states = np.unique(metadata['state_idx'])
        num_states = len(unique_states)

        cmap = plt.get_cmap(colormap)
        unique_steps = np.unique(metadata['step'])
        step_to_color = {step: cmap(i / max(1, len(unique_steps) - 1))
                        for i, step in enumerate(sorted(unique_steps))}

        # Return individual figures for each state
        if return_individual:
            state_figures = {}

            for state_idx in unique_states:
                # Create individual figure for this state
                fig, ax = plt.subplots(figsize=figsize or (6, 6))

                mask = metadata['state_idx'] == state_idx
                trajectory = w_2d[mask]
                steps = metadata['step'][mask]

                # Sort by step
                sort_idx = np.argsort(steps)
                trajectory = trajectory[sort_idx]
                steps = steps[sort_idx]

                # Plot trajectory
                for i in range(len(trajectory) - 1):
                    color = step_to_color[steps[i]]
                    ax.plot(
                        trajectory[i:i+2, 0],
                        trajectory[i:i+2, 1],
                        'o-',
                        color=color,
                        alpha=0.7,
                        linewidth=2,
                        markersize=5
                    )

                # Add labels
                if self.method == 'pca' and explained_var is not None:
                    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=11)
                    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=11)
                else:
                    ax.set_xlabel(f'{self.method.upper()} Dimension 1', fontsize=11)
                    ax.set_ylabel(f'{self.method.upper()} Dimension 2', fontsize=11)

                ax.set_title(
                    f'State {state_idx} - Latent Evolution ({self.method.upper()}) - Step {current_step}',
                    fontweight='bold',
                    fontsize=12
                )
                ax.grid(True, alpha=0.3)

                # Add colorbar for time progression
                sm = plt.cm.ScalarMappable(
                    cmap=cmap,
                    norm=mcolors.Normalize(vmin=min(unique_steps), vmax=max(unique_steps))
                )
                sm.set_array([])
                plt.colorbar(sm, ax=ax, label='Training Step')

                plt.tight_layout()
                state_figures[int(state_idx)] = fig

            return state_figures

        # Original behavior: return single figure with subplots
        else:
            # Create subplot grid
            ncols = min(4, num_states)
            nrows = (num_states + ncols - 1) // ncols

            if figsize is None:
                figsize = (4 * ncols, 4 * nrows)

            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            if num_states == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            # Plot each state
            for idx, state_idx in enumerate(unique_states):
                ax = axes[idx]

                mask = metadata['state_idx'] == state_idx
                trajectory = w_2d[mask]
                steps = metadata['step'][mask]

                # Sort by step
                sort_idx = np.argsort(steps)
                trajectory = trajectory[sort_idx]
                steps = steps[sort_idx]

                # Plot trajectory
                for i in range(len(trajectory) - 1):
                    color = step_to_color[steps[i]]
                    ax.plot(
                        trajectory[i:i+2, 0],
                        trajectory[i:i+2, 1],
                        'o-',
                        color=color,
                        alpha=0.7,
                        linewidth=2,
                        markersize=5
                    )

                ax.set_title(f'State {state_idx}', fontweight='bold')
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for idx in range(num_states, len(axes)):
                axes[idx].axis('off')

            fig.suptitle(
                f'Per-State Latent Evolution ({self.method.upper()}) - Step {current_step}',
                fontsize=14,
                fontweight='bold'
            )
            plt.tight_layout()

            return fig
