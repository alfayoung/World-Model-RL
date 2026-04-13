"""Dreamer-v3 baseline package (implemented in Flax / optax / distrax).

The upstream danijar/dreamerv3 source is vendored under ``dreamerv3/`` and
``embodied/`` for reference only.  We do NOT import it at runtime because its
``elements`` dependency requires jax>=0.9.2 which conflicts with the project's
pinned jax==0.5.0.

Public surface for the training scripts::

    from dreamer_v3.agent import DreamerV3Learner
    from dreamer_v3.replay import EpisodeReplayBuffer
"""
