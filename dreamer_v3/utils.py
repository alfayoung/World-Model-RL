"""Math utilities for Dreamer-v3.

References
----------
Hafner et al. (2023) "Mastering Diverse Domains through World Models"
https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# symlog / symexp
# ---------------------------------------------------------------------------

def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric log transform (identity around 0, log scaling for large values)."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse of symlog."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


# ---------------------------------------------------------------------------
# Two-hot encoding (Dreamer-v3 §3.1)
# ---------------------------------------------------------------------------

def _twohot_bins(num_bins: int = 255) -> jnp.ndarray:
    """Symexp-spaced bins from -20 to 20 (paper default)."""
    lo, hi = -20.0, 20.0
    bins = jnp.linspace(lo, hi, num_bins)
    return bins


def twohot_encode(values: jnp.ndarray, bins: jnp.ndarray) -> jnp.ndarray:
    """Two-hot encode a batch of scalar values into bin logits.

    Args:
        values: (B,) or (B, T) float32 — the target values.
        bins:   (K,) bin centers (e.g. from _twohot_bins).

    Returns:
        (B, K) or (B, T, K) float32 — soft two-hot targets summing to 1.
    """
    below = jnp.sum(bins <= values[..., None], axis=-1) - 1
    below = jnp.clip(below, 0, len(bins) - 2)
    above = below + 1
    equal = (bins[above] == bins[below])
    dist_to_above = jnp.where(equal, 0.5, jnp.abs(values - bins[above]))
    dist_to_below = jnp.where(equal, 0.5, jnp.abs(values - bins[below]))
    total = dist_to_above + dist_to_below
    weight_above = dist_to_below / total
    weight_below = dist_to_above / total
    target = (
        jax.nn.one_hot(below, len(bins)) * weight_below[..., None]
        + jax.nn.one_hot(above, len(bins)) * weight_above[..., None]
    )
    return target


def twohot_decode(logits: jnp.ndarray, bins: jnp.ndarray) -> jnp.ndarray:
    """Expected value from two-hot logits.

    Args:
        logits: (..., K) — raw logits.
        bins:   (K,) bin centers.

    Returns:
        (...,) float32 — expected value.
    """
    probs = jax.nn.softmax(logits, axis=-1)
    return (probs * bins).sum(axis=-1)


# ---------------------------------------------------------------------------
# Return normalization (percentile, Dreamer-v3 §3.3)
# ---------------------------------------------------------------------------

class PercentileNorm:
    """Running percentile normalization for advantage/return scaling.

    Keeps an exponential moving average of the lo/hi percentile values and
    exposes a ``scale`` that can be applied to normalize tensors.

    Not used during JIT-compiled updates — called in Python to update state.
    """

    def __init__(
        self,
        low: float = 5.0,
        high: float = 95.0,
        rate: float = 0.01,
        min_scale: float = 1.0,
    ):
        self.low = low / 100.0
        self.high = high / 100.0
        self.rate = rate
        self.min_scale = min_scale
        self._lo: float = 0.0
        self._hi: float = 1.0

    def update(self, values: np.ndarray):
        lo = float(np.percentile(values, self.low * 100))
        hi = float(np.percentile(values, self.high * 100))
        self._lo = (1 - self.rate) * self._lo + self.rate * lo
        self._hi = (1 - self.rate) * self._hi + self.rate * hi

    @property
    def scale(self) -> float:
        return max(self._hi - self._lo, self.min_scale)

    @property
    def offset(self) -> float:
        return self._lo


# ---------------------------------------------------------------------------
# KL divergence for categorical distributions
# ---------------------------------------------------------------------------

def categorical_kl(
    logit_p: jnp.ndarray,
    logit_q: jnp.ndarray,
    free_bits: float = 1.0,
) -> jnp.ndarray:
    """KL(p || q) for factored categorical, averaged over stoch dims.

    Args:
        logit_p: (..., stoch, classes)
        logit_q: (..., stoch, classes)
        free_bits: minimum KL per dimension (paper: 1 nat).

    Returns:
        (...,) mean KL per sample.
    """
    p = jax.nn.softmax(logit_p, axis=-1)
    q = jax.nn.softmax(logit_q, axis=-1)
    kl = (p * (jnp.log(p + 1e-8) - jnp.log(q + 1e-8))).sum(axis=-1)  # (..., stoch)
    kl = jnp.maximum(kl, free_bits)
    return kl.mean(axis=-1)  # average over stoch dims → (...)


# ---------------------------------------------------------------------------
# Lambda return
# ---------------------------------------------------------------------------

def lambda_return(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    continues: jnp.ndarray,
    horizon: int,
    lam: float = 0.95,
    discount: float = 0.997,
) -> jnp.ndarray:
    """Compute λ-returns over an imagined H-step trajectory.

    All inputs have shape (B, H+1) where index H is the bootstrap value.

    Returns:
        (B, H) λ-returns (without the bootstrap step).
    """
    # Terminal discount: continues[t] ∈ {0, 1}
    disc = discount * continues  # (B, H+1)
    # Bootstrap: V(s_H)
    ret = values[:, -1]
    rets = []
    for t in reversed(range(horizon)):
        ret = rewards[:, t] + disc[:, t] * ((1 - lam) * values[:, t + 1] + lam * ret)
        rets.insert(0, ret)
    return jnp.stack(rets, axis=1)  # (B, H)
