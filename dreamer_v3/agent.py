"""DreamerV3Learner — Dreamer-v3 in Flax/optax, DSRL-compatible interface.

The action space is the *same* 32-dim noise space used by DSRL's PixelSAC:
the learner outputs a noise vector that is fed into frozen π₀ to produce the
7-dim LIBERO robot action.  This makes the comparison apples-to-apples.

Reference: Hafner et al. (2023) "Mastering Diverse Domains through World Models"
           https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations

import copy
import pathlib
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from dreamer_v3.networks import (
    Actor, BinaryHead, Decoder, Encoder, RSSMCell, TwoHotHead,
)
from dreamer_v3.utils import (
    PercentileNorm,
    _twohot_bins,
    categorical_kl,
    lambda_return,
    symlog,
    symexp,
    twohot_decode,
    twohot_encode,
)


def _feat(carry: dict) -> jnp.ndarray:
    """Concatenate deterministic and stochastic state."""
    return jnp.concatenate([carry['h'], carry['z']], axis=-1)


# ---------------------------------------------------------------------------
# DreamerV3Learner
# ---------------------------------------------------------------------------

class DreamerV3Learner:
    """Full Dreamer-v3 agent with DSRL-compatible interface.

    Each Flax module is a standalone Python attribute — NOT sub-modules of a
    shared container.  Parameters are stored in a single flat dict keyed by
    module name.

    Public methods
    --------------
    reset_state()               — call at episode boundaries
    sample_actions(obs_dict)    → (1, action_dim) np.ndarray
    eval_actions(obs_dict)      → (1, action_dim) np.ndarray  (deterministic)
    update(batch)               → dict[str, float]  metrics
    save_checkpoint(path, step)
    restore_checkpoint(path)
    """

    def __init__(
        self,
        seed: int = 0,
        # observation / action geometry
        image_size: int = 64,
        state_dim: int = 8,
        action_dim: int = 32,
        # architecture
        encoder_depth: int = 32,
        encoder_embed: int = 1024,
        deter: int = 1024,
        stoch: int = 32,
        classes: int = 32,
        mlp_units: int = 512,
        mlp_layers: int = 3,
        num_bins: int = 255,
        # optimisation
        lr: float = 3e-4,
        grad_clip: float = 100.0,
        # algorithm
        discount: float = 0.997,
        imag_horizon: int = 15,
        lam: float = 0.95,
        kl_free: float = 1.0,
        kl_scale_dyn: float = 0.5,
        kl_scale_rep: float = 0.1,
        rec_scale: float = 1.0,
        rew_scale: float = 1.0,
        con_scale: float = 1.0,
        actor_ent: float = 3e-4,
        value_ema: float = 0.02,
        add_states: bool = True,
    ):
        self.image_size   = image_size
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.deter        = deter
        self.stoch        = stoch
        self.classes      = classes
        self.discount     = discount
        self.imag_horizon = imag_horizon
        self.lam          = lam
        self.kl_free      = kl_free
        self.kl_scale_dyn = kl_scale_dyn
        self.kl_scale_rep = kl_scale_rep
        self.rec_scale    = rec_scale
        self.rew_scale    = rew_scale
        self.con_scale    = con_scale
        self.actor_ent    = actor_ent
        self.value_ema    = value_ema
        self.add_states   = add_states
        self._feat_dim    = deter + stoch * classes

        self._rng  = jax.random.PRNGKey(seed)
        self._bins = _twohot_bins(num_bins)
        self._return_norm = PercentileNorm()

        # ---- Standalone Flax modules ----------------------------------------
        self.encoder   = Encoder(depth=encoder_depth, embed_dim=encoder_embed)
        self.decoder   = Decoder(depth=encoder_depth)
        self.rssm      = RSSMCell(
            deter=deter, stoch=stoch, classes=classes,
            hidden=mlp_units, embed_dim=encoder_embed)
        self.reward    = TwoHotHead(num_bins=num_bins, units=mlp_units, layers=2)
        self.continue_ = BinaryHead(units=mlp_units, layers=2)
        self.actor     = Actor(action_dim=action_dim,
                               units=mlp_units, layers=mlp_layers)
        self.value     = TwoHotHead(num_bins=num_bins,
                                    units=mlp_units, layers=mlp_layers)
        self.value_slow = TwoHotHead(num_bins=num_bins,
                                     units=mlp_units, layers=mlp_layers)

        # ---- Initialize each module independently ---------------------------
        feat_dim       = self._feat_dim
        dummy_img      = jnp.zeros((1, image_size, image_size, 3), jnp.uint8)
        dummy_feat     = jnp.zeros((1, feat_dim))
        dummy_carry    = {'h': jnp.zeros((1, deter)),
                          'z': jnp.zeros((1, stoch * classes))}
        dummy_embed    = jnp.zeros((1, encoder_embed))
        dummy_action   = jnp.zeros((1, action_dim))
        dummy_is_first = jnp.zeros((1,), bool)

        (self._rng, k_enc, k_dec, k_rssm, k_rssm_rng,
         k_rew, k_con, k_act, k_val) = jax.random.split(self._rng, 9)

        enc_vars  = self.encoder.init(k_enc, dummy_img)
        dec_vars  = self.decoder.init(k_dec, dummy_feat)
        rssm_vars = self.rssm.init(
            k_rssm,
            dummy_carry, dummy_embed, dummy_action, dummy_is_first, k_rssm_rng,
            method=self.rssm.observe)
        rew_vars  = self.reward.init(k_rew, dummy_feat)
        con_vars  = self.continue_.init(k_con, dummy_feat)
        act_vars  = self.actor.init(k_act, dummy_feat)
        val_vars  = self.value.init(k_val, dummy_feat)

        self.params: dict = {
            'encoder':   enc_vars['params'],
            'decoder':   dec_vars['params'],
            'rssm':      rssm_vars['params'],
            'reward':    rew_vars['params'],
            'continue_': con_vars['params'],
            'actor':     act_vars['params'],
            'value':     val_vars['params'],
        }
        # Slow value target — initialised equal to fast value
        self.slow_value_params = copy.deepcopy(val_vars['params'])

        # ---- Optimiser (world-model + actor-critic jointly) -----------------
        tx = optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adam(lr, eps=1e-8),
        )
        self.opt_state = tx.init(self.params)
        self._tx       = tx

        # ---- Online RSSM state (maintained across env steps) ----------------
        self._carry: Optional[dict] = None

        # ---- JIT-compiled kernels -------------------------------------------
        self._jit_update  = jax.jit(self._update_step)
        self._jit_encode  = jax.jit(self._encode_fn)
        self._jit_observe = jax.jit(self._observe_fn)
        self._jit_act     = jax.jit(self._act_fn)
        self._jit_act_det = jax.jit(self._act_det_fn)

    # ------------------------------------------------------------------
    # Pure JIT-able kernels (no Python state mutations)
    # ------------------------------------------------------------------

    def _encode_fn(self, params: dict, image: jnp.ndarray) -> jnp.ndarray:
        return self.encoder.apply({'params': params['encoder']}, image)

    def _observe_fn(self, params, carry, embed, action, is_first, rng):
        return self.rssm.apply(
            {'params': params['rssm']},
            carry, embed, action, is_first, rng,
            method=self.rssm.observe)

    def _act_fn(self, params, rng, feat):
        mean, std = self.actor.apply({'params': params['actor']}, feat)
        noise  = jax.random.normal(rng, mean.shape)
        return jnp.tanh(mean + std * noise)

    def _act_det_fn(self, params, feat):
        mean, _ = self.actor.apply({'params': params['actor']}, feat)
        return jnp.tanh(mean)

    # ------------------------------------------------------------------
    # Online carry management
    # ------------------------------------------------------------------

    def reset_state(self):
        """Reset RSSM carry; call at every episode boundary."""
        self._carry = {
            'h': jnp.zeros((1, self.deter)),
            'z': jnp.zeros((1, self.stoch * self.classes)),
        }

    def _ensure_carry(self):
        if self._carry is None:
            self.reset_state()

    # ------------------------------------------------------------------
    # Observation preprocessing
    # ------------------------------------------------------------------

    def _extract_image(self, obs_dict: dict) -> jnp.ndarray:
        """Extract (1, H, W, 3) uint8 image from DSRL obs_dict."""
        pix = obs_dict['pixels']                # (1, H, W, C, 1)
        img = np.array(pix[0, :, :, :3, 0])    # (H, W, 3) — first 3 channels, frame 0
        if img.shape[0] != self.image_size:
            import PIL.Image
            img = np.array(
                PIL.Image.fromarray(img).resize(
                    (self.image_size, self.image_size)))
        return jnp.array(img[np.newaxis])       # (1, H, W, 3)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def sample_actions(self, obs_dict: dict) -> np.ndarray:
        """Stochastic action for exploration.

        Returns (1, action_dim) numpy array.
        """
        self._ensure_carry()
        img    = self._extract_image(obs_dict)
        embed  = self._jit_encode(self.params, img)

        self._rng, rng_obs, rng_act = jax.random.split(self._rng, 3)
        is_first   = jnp.zeros((1,), bool)
        prev_action = jnp.zeros((1, self.action_dim))

        new_carry, _ = self._jit_observe(
            self.params, self._carry, embed, prev_action, is_first, rng_obs)
        self._carry = new_carry

        feat   = _feat(new_carry)
        action = self._jit_act(self.params, rng_act, feat)  # (1, A)
        return np.array(action)

    def eval_actions(self, obs_dict: dict) -> np.ndarray:
        """Deterministic action for evaluation."""
        self._ensure_carry()
        img    = self._extract_image(obs_dict)
        embed  = self._jit_encode(self.params, img)

        self._rng, rng_obs = jax.random.split(self._rng)
        is_first    = jnp.zeros((1,), bool)
        prev_action = jnp.zeros((1, self.action_dim))

        new_carry, _ = self._jit_observe(
            self.params, self._carry, embed, prev_action, is_first, rng_obs)
        self._carry = new_carry

        feat   = _feat(new_carry)
        action = self._jit_act_det(self.params, feat)
        return np.array(action)

    # ------------------------------------------------------------------
    # Offline update
    # ------------------------------------------------------------------

    def update(self, batch: dict) -> dict:
        """One gradient step.

        batch keys: image(B,T,H,W,3) uint8, state(B,T,D), action(B,T,A),
                    reward(B,T), done(B,T), is_first(B,T) bool.
        Returns dict of float scalars.
        """
        self._rng, rng = jax.random.split(self._rng)
        self.params, self.opt_state, metrics = self._jit_update(
            self.params, self.opt_state, self.slow_value_params, rng, batch)

        # EMA update for slow value target (Python-side, not inside JIT)
        self.slow_value_params = jax.tree_util.tree_map(
            lambda slow, fast: (1.0 - self.value_ema) * slow + self.value_ema * fast,
            self.slow_value_params,
            self.params['value'],
        )
        return {k: float(v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # JIT-compiled training step
    # ------------------------------------------------------------------

    def _update_step(self, params, opt_state, slow_val_params, rng, batch):

        def loss_fn(params):
            return self._compute_loss(params, slow_val_params, rng, batch)

        (total_loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, new_opt_state = self._tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, metrics

    # ------------------------------------------------------------------
    # Full Dreamer-v3 loss
    # ------------------------------------------------------------------

    def _compute_loss(self, params, slow_val_params, rng, batch):
        """Compute world-model + actor + value losses on a (B, T) batch."""
        B, T = batch['image'].shape[:2]
        bins = self._bins

        # ---- 1. Encode all observations (B*T, embed_dim) -------------------
        imgs   = batch['image'].reshape(B * T, self.image_size, self.image_size, 3)
        embeds = self.encoder.apply({'params': params['encoder']}, imgs)
        embeds = embeds.reshape(B, T, -1)

        # ---- 2. RSSM observe loop: get post + prior for every step ----------
        carry = {
            'h': jnp.zeros((B, self.deter)),
            'z': jnp.zeros((B, self.stoch * self.classes)),
        }
        post_logits  = []   # (B, stoch, classes) per step
        prior_logits = []
        feats        = []   # (B, feat_dim) per step

        for t in range(T):
            prev_action = (batch['action'][:, t - 1] if t > 0
                           else jnp.zeros((B, self.action_dim)))
            rng, rng_t = jax.random.split(rng)
            new_carry, out = self.rssm.apply(
                {'params': params['rssm']},
                carry, embeds[:, t], prev_action, batch['is_first'][:, t], rng_t,
                method=self.rssm.observe)
            carry = new_carry
            post_logits.append(out['post_logit'])   # (B, stoch, classes)
            prior_logits.append(out['prior_logit'])
            feats.append(_feat(new_carry))           # (B, feat_dim)

        post_logits  = jnp.stack(post_logits,  axis=1)  # (B, T, stoch, classes)
        prior_logits = jnp.stack(prior_logits, axis=1)
        feats        = jnp.stack(feats,        axis=1)  # (B, T, feat_dim)

        # ---- 3. World-model losses -----------------------------------------
        feat_flat = feats.reshape(B * T, -1)

        # 3a. Image reconstruction (MSE in [0, 1])
        recons = self.decoder.apply(
            {'params': params['decoder']}, feat_flat)              # (B*T, H, W, 3)
        recons = recons.reshape(B, T, self.image_size, self.image_size, 3)
        target_img = batch['image'].astype(jnp.float32) / 255.0   # (B, T, H, W, 3)
        rec_loss = jnp.mean((recons - target_img) ** 2)

        # 3b. Reward prediction (symlog two-hot cross-entropy)
        rew_logits = self.reward.apply(
            {'params': params['reward']}, feat_flat).reshape(B, T, -1)
        rew_target = twohot_encode(symlog(batch['reward']), bins)  # (B, T, K)
        rew_loss   = -jnp.mean(jnp.sum(
            rew_target * jax.nn.log_softmax(rew_logits, axis=-1), axis=-1))

        # 3c. Continue prediction (BCE)
        con_logits = self.continue_.apply(
            {'params': params['continue_']}, feat_flat).reshape(B, T)  # (B, T)
        con_target = 1.0 - batch['done']          # 1 = keep going, 0 = terminal
        con_loss   = -jnp.mean(
            con_target * jax.nn.log_sigmoid(con_logits)
            + (1.0 - con_target) * jax.nn.log_sigmoid(-con_logits))

        # 3d. KL (dynamics: prior ← sg(post); representation: sg(prior) → post)
        kl_dyn = categorical_kl(
            jax.lax.stop_gradient(post_logits), prior_logits,
            self.kl_free).mean()
        kl_rep = categorical_kl(
            post_logits, jax.lax.stop_gradient(prior_logits),
            self.kl_free).mean()

        wm_loss = (
            self.rec_scale * rec_loss
            + self.rew_scale * rew_loss
            + self.con_scale * con_loss
            + self.kl_scale_dyn * kl_dyn
            + self.kl_scale_rep * kl_rep
        )

        # ---- 4. Imagination rollout (actor-critic) --------------------------
        H = self.imag_horizon

        # Start from the final observed carry (detached from WM graph)
        cur_carry = jax.lax.stop_gradient(carry)

        img_feats   = []   # length H+1
        img_actions = []   # length H

        for h in range(H):
            feat_h = _feat(cur_carry)
            img_feats.append(feat_h)

            mean_h, std_h = self.actor.apply(
                {'params': params['actor']}, feat_h)
            rng, rng_noise, rng_img = jax.random.split(rng, 3)
            act_h = jnp.tanh(mean_h + std_h * jax.random.normal(rng_noise, mean_h.shape))
            img_actions.append(act_h)

            cur_carry, _ = self.rssm.apply(
                {'params': params['rssm']}, cur_carry, act_h, rng_img,
                method=self.rssm.imagine)

        img_feats.append(_feat(cur_carry))  # bootstrap value at H

        img_feats   = jnp.stack(img_feats,   axis=1)  # (B, H+1, feat_dim)
        img_actions = jnp.stack(img_actions, axis=1)  # (B, H,   action_dim)

        img_feat_flat = img_feats.reshape(B * (H + 1), -1)

        # Predicted rewards and continues over imagination
        img_rew_logits = self.reward.apply(
            {'params': params['reward']}, img_feat_flat).reshape(B, H + 1, -1)
        img_rews = symexp(twohot_decode(img_rew_logits, bins))     # (B, H+1)

        img_con_logits = self.continue_.apply(
            {'params': params['continue_']}, img_feat_flat).reshape(B, H + 1)
        img_cons = jax.nn.sigmoid(img_con_logits)                  # (B, H+1)

        # Slow value target for bootstrapping returns
        img_val_logits_slow = self.value_slow.apply(
            {'params': slow_val_params}, img_feat_flat).reshape(B, H + 1, -1)
        img_vals_slow = symexp(twohot_decode(img_val_logits_slow, bins))  # (B, H+1)

        # λ-returns
        returns = lambda_return(
            img_rews[:, :-1],   # (B, H) — rewards at steps 0..H-1
            img_vals_slow,      # (B, H+1) — values at steps 0..H
            img_cons[:, :-1],   # (B, H) — continues at steps 0..H-1
            horizon=H,
            lam=self.lam,
            discount=self.discount,
        )  # (B, H)

        # ---- 5. Actor loss -------------------------------------------------
        actor_feats = img_feats[:, :-1].reshape(B * H, -1)
        mean_a, std_a = self.actor.apply({'params': params['actor']}, actor_feats)
        log_std = jnp.log(std_a + 1e-8)
        ent = (0.5 * (1.0 + jnp.log(2.0 * jnp.pi)) + log_std).sum(-1)  # (B*H,)

        # Return normalisation (percentile — pure JAX, no Python state)
        returns_flat = returns.reshape(-1)
        ret_scale = jnp.maximum(
            jnp.percentile(returns_flat, 95) - jnp.percentile(returns_flat, 5),
            1.0)
        adv_flat = (returns.reshape(-1) - jnp.mean(returns_flat)) / ret_scale

        img_acts_flat = img_actions.reshape(B * H, -1)
        log_prob = (
            -0.5 * ((img_acts_flat - mean_a) / (std_a + 1e-8)) ** 2
            - jnp.log(std_a + 1e-8)
        ).sum(-1)  # (B*H,)
        actor_loss = -(log_prob * jax.lax.stop_gradient(adv_flat)
                       + self.actor_ent * ent).mean()

        # ---- 6. Value loss -------------------------------------------------
        val_logits = self.value.apply(
            {'params': params['value']},
            jax.lax.stop_gradient(actor_feats),
        ).reshape(B, H, -1)
        val_target = twohot_encode(
            jax.lax.stop_gradient(symlog(returns)), bins)         # (B, H, K)
        value_loss = -jnp.mean(jnp.sum(
            val_target * jax.nn.log_softmax(val_logits, axis=-1), axis=-1))

        total_loss = wm_loss + actor_loss + value_loss

        metrics = {
            'loss/total':    total_loss,
            'loss/rec':      rec_loss,
            'loss/reward':   rew_loss,
            'loss/continue': con_loss,
            'loss/kl_dyn':   kl_dyn,
            'loss/kl_rep':   kl_rep,
            'loss/actor':    actor_loss,
            'loss/value':    value_loss,
            'imag_return':   jnp.mean(returns),
        }
        return total_loss, metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, step: int, interval: int = -1):
        """Save params + opt_state to ``<path>/dreamerv3_ckpt_<step>``."""
        ckpt_dir = pathlib.Path(path) / f"dreamerv3_ckpt_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(str(ckpt_dir), {
            'params':            self.params,
            'opt_state':         self.opt_state,
            'slow_value_params': self.slow_value_params,
        })

    def restore_checkpoint(self, path: str):
        """Restore from a checkpoint directory."""
        checkpointer = ocp.StandardCheckpointer()
        state = checkpointer.restore(path, target={
            'params':            self.params,
            'opt_state':         self.opt_state,
            'slow_value_params': self.slow_value_params,
        })
        self.params            = state['params']
        self.opt_state         = state['opt_state']
        self.slow_value_params = state['slow_value_params']
