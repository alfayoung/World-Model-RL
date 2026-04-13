"""TD-MPC2 agent.

Reference: Hansen et al., "TD-MPC2: Scalable, Robust World Models for
Continuous Control", ICLR 2024.  https://arxiv.org/abs/2310.16828

Key algorithmic components
--------------------------
1. Latent representation via a pixel encoder (CNN).
2. World model: latent dynamics, reward, Q-ensemble, policy.
3. MPPI planning at inference time to select actions.
4. Joint training loss:
   - Consistency loss (latent dynamics matches re-encoded next obs)
   - Reward prediction loss
   - Value (TD) loss
   - Policy loss (maximize Q)
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from tdmpc2.networks.encoder import PixelEncoder
from tdmpc2.networks.world_model import WorldModel
from tdmpc2.utils.math import soft_update, symlog


class TDMPC2Agent:
    """Full TD-MPC2 agent wrapping encoder + world model."""

    def __init__(
        self,
        obs_shape: tuple,          # (C, H, W) pixels
        action_dim: int,
        state_dim: int = 0,        # optional low-dim state appended to latent
        latent_dim: int = 256,
        hidden_dim: int = 256,
        num_q: int = 5,
        mlp_depth: int = 2,
        encoder_feature_dim: int = 256,
        lr: float = 3e-4,
        tau: float = 0.005,
        discount: float = 0.99,
        horizon: int = 5,          # MPC rollout horizon
        num_samples: int = 512,    # MPPI samples
        num_elites: int = 64,
        num_pi_trajs: int = 24,    # policy-guided trajectories in MPPI
        temperature: float = 0.5,  # MPPI temperature
        std_schedule: str = 'linear',  # 'linear' or 'constant'
        std_max: float = 2.0,
        std_min: float = 0.05,
        consistency_coef: float = 20.0,
        reward_coef: float = 0.5,
        value_coef: float = 0.1,
        seed: int = 0,
        device: str = 'cuda',
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Dimensions
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_pi_trajs = num_pi_trajs
        self.temperature = temperature
        self.discount = discount
        self.tau = tau
        self.consistency_coef = consistency_coef
        self.reward_coef = reward_coef
        self.value_coef = value_coef
        self.std_max = std_max
        self.std_min = std_min
        self.std_schedule = std_schedule

        wm_input_dim = encoder_feature_dim + state_dim

        # Networks
        self.encoder = PixelEncoder(
            obs_shape, feature_dim=encoder_feature_dim
        ).to(self.device)
        self.world_model = WorldModel(
            latent_dim=wm_input_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_q=num_q,
            mlp_depth=mlp_depth,
        ).to(self.device)

        # Target networks (EMA)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_world_model = copy.deepcopy(self.world_model)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        for p in self.target_world_model.parameters():
            p.requires_grad_(False)

        # Optimizer (joint)
        self.optim = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.world_model.parameters()),
            lr=lr,
        )

        self._step = 0

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode(self, obs: dict) -> torch.Tensor:
        """Encodes a dict observation to a latent vector."""
        pixels = torch.tensor(obs['pixels'], dtype=torch.float32, device=self.device)
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
        # pixels expected as (B, C, H, W); LIBERO stores (H, W, C)
        if pixels.shape[-1] in (3, 6, 9, 12):
            pixels = pixels.permute(0, 3, 1, 2)
        z = self.encoder(pixels)
        if self.state_dim > 0:
            state = torch.tensor(obs['state'], dtype=torch.float32, device=self.device)
            if state.ndim == 1:
                state = state.unsqueeze(0)
            z = torch.cat([z, state], dim=-1)
        return z

    def _action_std(self) -> float:
        """Linearly decays exploration std over training."""
        if self.std_schedule == 'constant':
            return self.std_min
        frac = min(self._step / 500_000, 1.0)
        return self.std_max + frac * (self.std_min - self.std_max)

    @torch.no_grad()
    def select_action(
        self,
        obs: dict,
        eval_mode: bool = False,
        prev_mean: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """MPPI planning over the latent world model.

        Returns (action_np, new_mean) where new_mean can be passed back as
        prev_mean to warm-start the next call.
        """
        z0 = self._encode(obs)                         # (1, latent_dim)
        std = 0.0 if eval_mode else self._action_std()

        # ---- Build initial sample population --------------------------------
        # num_pi_trajs from the policy; rest from random / warm-start Gaussian
        if prev_mean is None:
            prev_mean = torch.zeros(self.horizon, self.action_dim, device=self.device)

        num_random = self.num_samples - self.num_pi_trajs
        # (num_samples, horizon, action_dim)
        mean_expand = prev_mean.unsqueeze(0).expand(num_random, -1, -1)
        noise = torch.randn_like(mean_expand) * std
        random_actions = (mean_expand + noise).clamp(-1, 1)

        # Policy-guided trajectories
        z_pi = z0.expand(self.num_pi_trajs, -1)
        pi_actions_list = []
        for _ in range(self.horizon):
            a = self.world_model.sample_action(z_pi, std=std)
            pi_actions_list.append(a.unsqueeze(1))
            z_pi = self.world_model.forward_dynamics(z_pi, a)
        pi_actions = torch.cat(pi_actions_list, dim=1)  # (num_pi, H, A)

        actions = torch.cat([random_actions, pi_actions], dim=0)  # (N, H, A)
        N = actions.shape[0]

        # ---- Rollout and estimate returns -----------------------------------
        z = z0.expand(N, -1)                            # (N, latent_dim)
        returns = torch.zeros(N, device=self.device)
        discount = 1.0
        for h in range(self.horizon):
            a_h = actions[:, h]                         # (N, A)
            r_h = self.world_model.forward_reward(z, a_h)
            z = self.world_model.forward_dynamics(z, a_h)
            returns += discount * r_h
            discount *= self.discount

        # Terminal value: min-Q over the ensemble
        a_final = self.world_model.sample_action(z)
        q_terminal = self.world_model.forward_q(z, a_final).min(0).values
        returns += discount * q_terminal

        # ---- MPPI update ----------------------------------------------------
        returns = (returns - returns.max()) / (returns.std() + 1e-6)
        weights = F.softmax(returns / max(self.temperature, 1e-6), dim=0)

        # Update mean
        new_mean = (weights.view(N, 1, 1) * actions).sum(0)  # (H, A)

        # Shift mean for next call
        shifted_mean = torch.cat([new_mean[1:], new_mean[-1:]], dim=0)

        # Select first action (with optional noise at train time)
        if eval_mode:
            action = new_mean[0].clamp(-1, 1)
        else:
            action = (new_mean[0] + torch.randn_like(new_mean[0]) * std).clamp(-1, 1)

        return action.cpu().numpy(), shifted_mean

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, batch: dict) -> dict:
        """One gradient step on a sampled batch.

        batch keys: observations, next_observations, actions, rewards, dones
        Each observation is a dict with 'pixels' and optionally 'state'.
        """
        obs = batch['observations']
        next_obs = batch['next_observations']
        actions = torch.tensor(batch['actions'], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32, device=self.device)

        # Encode current and next observations
        pixels = torch.tensor(obs['pixels'], dtype=torch.float32, device=self.device)
        next_pixels = torch.tensor(next_obs['pixels'], dtype=torch.float32, device=self.device)

        # Handle HWC -> CHW
        if pixels.shape[-1] in (3, 6, 9, 12):
            pixels = pixels.permute(0, 3, 1, 2)
            next_pixels = next_pixels.permute(0, 3, 1, 2)

        z = self.encoder(pixels)
        with torch.no_grad():
            z_next_target = self.target_encoder(next_pixels)

        if self.state_dim > 0:
            state = torch.tensor(obs['state'], dtype=torch.float32, device=self.device)
            next_state = torch.tensor(next_obs['state'], dtype=torch.float32, device=self.device)
            z = torch.cat([z, state], dim=-1)
            z_next_target = torch.cat([z_next_target, next_state], dim=-1)

        # --- Predicted next latent (consistency) ---
        z_pred_next = self.world_model.forward_dynamics(z, actions)
        consistency_loss = F.mse_loss(z_pred_next, z_next_target.detach())

        # --- Reward prediction ---
        r_pred = self.world_model.forward_reward(z, actions)
        reward_loss = F.mse_loss(r_pred, symlog(rewards))

        # --- Value (TD) loss ---
        with torch.no_grad():
            a_next = self.target_world_model.sample_action(z_next_target)
            q_next = self.target_world_model.forward_q(z_next_target, a_next)
            # Two-hot or plain: use minimum over ensemble
            q_next_min = q_next.min(0).values
            q_target = symlog(rewards) + self.discount * (1 - dones) * q_next_min

        q_pred = self.world_model.forward_q(z, actions)   # (num_q, B)
        value_loss = sum(
            F.mse_loss(q_pred[i], q_target) for i in range(q_pred.shape[0])
        ) / q_pred.shape[0]

        # --- Policy loss: maximize min-Q ---
        a_pol = self.world_model.sample_action(z.detach())
        q_pol = self.world_model.forward_q(z.detach(), a_pol).min(0).values
        policy_loss = -q_pol.mean()

        # --- Total loss ---
        loss = (
            self.consistency_coef * consistency_loss
            + self.reward_coef * reward_loss
            + self.value_coef * value_loss
            + policy_loss
        )

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.world_model.parameters()),
            max_norm=20.0,
        )
        self.optim.step()

        # EMA target update
        soft_update(self.target_encoder, self.encoder, self.tau)
        soft_update(self.target_world_model, self.world_model, self.tau)

        self._step += 1

        return {
            'loss/total': loss.item(),
            'loss/consistency': consistency_loss.item(),
            'loss/reward': reward_loss.item(),
            'loss/value': value_loss.item(),
            'loss/policy': policy_loss.item(),
            'q_mean': q_pred.mean().item(),
        }
