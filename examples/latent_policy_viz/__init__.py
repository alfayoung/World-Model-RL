"""Latent Policy Evolution Visualization for Vanilla DSRL.

This package provides tools to visualize how the latent policy π^w evolves
during training by tracking its outputs on canonical states and projecting
them to 2D using PCA or UMAP.

It also provides tools to visualize proprioceptive state trajectories during
diffusion policy rollouts in 3D space.
"""

from .canonical_states import CanonicalStateManager
from .latent_tracker import LatentPolicyTracker
from .evolution_plotter import LatentEvolutionPlotter
from .proprioceptive_tracker import ProprioceptiveTracker
from .proprioceptive_plotter import ProprioceptivePlotter

__all__ = [
    'CanonicalStateManager',
    'LatentPolicyTracker',
    'LatentEvolutionPlotter',
    'ProprioceptiveTracker',
    'ProprioceptivePlotter',
]
