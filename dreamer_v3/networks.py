"""Flax neural-network components for Dreamer-v3.

All modules are standalone Flax Linen modules. They are instantiated as plain
Python attributes of DreamerV3Learner (NOT as Flax sub-modules of a container),
so their parameters are initialised and applied independently.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# CNN Encoder: (B, H, W, C) → (B, embed_dim)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Four strided-conv layers; channels double each layer."""
    depth: int = 32
    embed_dim: int = 1024

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, C) uint8 in [0, 255]
        x = x.astype(jnp.float32) / 255.0 - 0.5
        for mult in [1, 2, 4, 8]:
            x = nn.Conv(self.depth * mult, (4, 4), strides=(2, 2), padding="VALID")(x)
            x = jax.nn.silu(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.embed_dim)(x)
        x = jax.nn.silu(x)
        return x


# ---------------------------------------------------------------------------
# CNN Decoder: (B, feat_dim) → (B, H, W, 3)
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    depth: int = 32
    image_channels: int = 3

    @nn.compact
    def __call__(self, feat):
        x = nn.Dense(self.depth * 8 * 4 * 4)(feat)
        x = x.reshape(x.shape[0], 4, 4, self.depth * 8)
        for mult in reversed([1, 2, 4]):
            x = nn.ConvTranspose(
                self.depth * mult, (4, 4), strides=(2, 2), padding="SAME")(x)
            x = jax.nn.silu(x)
        x = nn.ConvTranspose(
            self.image_channels, (4, 4), strides=(2, 2), padding="SAME")(x)
        return x  # (B, H, W, 3) — pixel predictions in [0,1] after sigmoid


# ---------------------------------------------------------------------------
# RSSM Cell: single-step recurrent world model
# ---------------------------------------------------------------------------

class RSSMCell(nn.Module):
    """Recurrent State Space Model (single step).

    carry = {'h': (B, deter), 'z': (B, stoch*classes)}

    observe():  GRU + posterior q(z|h,e) + prior p(z|h)
    imagine():  GRU + prior p(z|h) only

    Uses setup() (not @nn.compact on sub-methods) so each layer gets a unique,
    stable parameter name regardless of which public method is called.
    """
    deter:      int = 1024
    stoch:      int = 32
    classes:    int = 32
    hidden:     int = 512
    embed_dim:  int = 1024

    def setup(self):
        # GRU components
        self.gru_inp_proj = nn.Dense(self.deter)
        self.gru_cell     = nn.GRUCell(self.deter)
        # Prior MLP (no observation)
        self.prior_fc1    = nn.Dense(self.hidden)
        self.prior_ln     = nn.LayerNorm()
        self.prior_fc2    = nn.Dense(self.stoch * self.classes)
        # Posterior MLP (uses observation embedding)
        self.post_fc1     = nn.Dense(self.hidden)
        self.post_ln      = nn.LayerNorm()
        self.post_fc2     = nn.Dense(self.stoch * self.classes)

    # ---- GRU ---------------------------------------------------------------

    def _gru(self, h: jnp.ndarray, inp: jnp.ndarray) -> jnp.ndarray:
        inp_proj = jax.nn.silu(self.gru_inp_proj(inp))
        new_h, _ = self.gru_cell(h, inp_proj)
        return new_h

    # ---- Prior MLP ---------------------------------------------------------

    def _prior_mlp(self, h: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.silu(self.prior_ln(self.prior_fc1(h)))
        x = self.prior_fc2(x)
        return x.reshape(x.shape[0], self.stoch, self.classes)

    # ---- Posterior MLP -----------------------------------------------------

    def _post_mlp(self, h: jnp.ndarray, embed: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.silu(self.post_ln(self.post_fc1(jnp.concatenate([h, embed], -1))))
        x = self.post_fc2(x)
        return x.reshape(x.shape[0], self.stoch, self.classes)

    # ---- Public API --------------------------------------------------------

    def observe(self, carry, embed, action, is_first, rng):
        """Update carry using a real observation (returns prior AND posterior).

        Returns
        -------
        new_carry : dict with 'h', 'z'
        out       : dict with 'post_logit', 'prior_logit', 'stoch'
        """
        B = embed.shape[0]
        h, z = carry['h'], carry['z']

        # Reset at episode boundaries
        h = jnp.where(is_first[:, None], jnp.zeros_like(h), h)
        z = jnp.where(is_first[:, None], jnp.zeros_like(z), z)

        # GRU: advance deterministic state using previous z + action
        inp = jnp.concatenate([z, action], axis=-1)
        h_new = self._gru(h, inp)

        # Prior from h_new (no observation)
        prior_logit = self._prior_mlp(h_new)          # (B, stoch, classes)

        # Posterior from h_new + obs embedding
        post_logit  = self._post_mlp(h_new, embed)    # (B, stoch, classes)

        # Straight-through categorical sample from posterior
        z_new = _st_sample(post_logit, rng)            # (B, stoch*classes)

        new_carry = {'h': h_new, 'z': z_new}
        out = {
            'post_logit':  post_logit,
            'prior_logit': prior_logit,
            'stoch':       z_new,
        }
        return new_carry, out

    def imagine(self, carry, action, rng):
        """Advance carry without observation (prior only)."""
        B = action.shape[0]
        h, z = carry['h'], carry['z']
        inp = jnp.concatenate([z, action], axis=-1)
        h_new = self._gru(h, inp)
        prior_logit = self._prior_mlp(h_new)
        z_new = _st_sample(prior_logit, rng)
        new_carry = {'h': h_new, 'z': z_new}
        out = {'prior_logit': prior_logit, 'stoch': z_new}
        return new_carry, out

    def feat(self, carry) -> jnp.ndarray:
        return jnp.concatenate([carry['h'], carry['z']], axis=-1)

    def initial(self, batch_size: int) -> dict:
        return {
            'h': jnp.zeros((batch_size, self.deter)),
            'z': jnp.zeros((batch_size, self.stoch * self.classes)),
        }


def _st_sample(logit: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
    """Straight-through categorical sample.

    logit: (B, stoch, classes) → (B, stoch * classes) float one-hot.
    """
    probs = jax.nn.softmax(logit, axis=-1)                 # (B, S, C)
    idx   = jax.random.categorical(rng, logit, axis=-1)    # (B, S)
    onehot = jax.nn.one_hot(idx, logit.shape[-1])          # (B, S, C)
    # Straight-through: detach sample, keep gradient through probs
    st = jax.lax.stop_gradient(onehot - probs) + probs
    return st.reshape(st.shape[0], -1)                     # (B, S*C)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    out_dim: int
    units:   int = 512
    layers:  int = 3

    @nn.compact
    def __call__(self, x):
        for _ in range(self.layers):
            x = nn.Dense(self.units)(x)
            x = nn.LayerNorm()(x)
            x = jax.nn.silu(x)
        return nn.Dense(self.out_dim)(x)


# ---------------------------------------------------------------------------
# Policy / Value / Reward / Continue heads
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    action_dim: int = 32
    units:      int = 512
    layers:     int = 3
    min_std:    float = 0.1
    max_std:    float = 1.0

    @nn.compact
    def __call__(self, feat):
        x = feat
        for _ in range(self.layers):
            x = nn.Dense(self.units)(x)
            x = nn.LayerNorm()(x)
            x = jax.nn.silu(x)
        mean = nn.Dense(self.action_dim,
                        kernel_init=nn.initializers.orthogonal(0.01))(x)
        mean = jnp.tanh(mean)
        std_raw = nn.Dense(self.action_dim,
                           kernel_init=nn.initializers.orthogonal(0.01))(x)
        std = jax.nn.softplus(std_raw) + self.min_std
        std = jnp.clip(std, self.min_std, self.max_std)
        return mean, std


class TwoHotHead(nn.Module):
    """MLP → two-hot logits (used for reward and value)."""
    num_bins: int = 255
    units:    int = 512
    layers:   int = 2

    @nn.compact
    def __call__(self, feat):
        x = feat
        for _ in range(self.layers):
            x = nn.Dense(self.units)(x)
            x = nn.LayerNorm()(x)
            x = jax.nn.silu(x)
        return nn.Dense(self.num_bins)(x)


class BinaryHead(nn.Module):
    """MLP → scalar Bernoulli logit (continue predictor)."""
    units:  int = 512
    layers: int = 2

    @nn.compact
    def __call__(self, feat):
        x = feat
        for _ in range(self.layers):
            x = nn.Dense(self.units)(x)
            x = nn.LayerNorm()(x)
            x = jax.nn.silu(x)
        return nn.Dense(1)(x).squeeze(-1)  # (B,)
