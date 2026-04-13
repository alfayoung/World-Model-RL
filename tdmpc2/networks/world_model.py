"""TD-MPC2 world model: dynamics, reward, Q-value, and policy heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def _mlp(dims: list[int], act=nn.SiLU, output_act=None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act())
        elif output_act is not None:
            layers.append(output_act())
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    """
    Latent world model (Dreamer / TD-MPC2 style).

    Components
    ----------
    dynamics   : (z, a) -> z'        (latent transition)
    reward_fn  : (z, a) -> r          (1-step reward)
    value_fn   : (z, a) -> Q  x num_q (ensemble of Q-functions)
    policy_fn  : z -> (mu, log_std)   (squashed Gaussian policy)
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_q: int = 5,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_q = num_q

        h = hidden_dim
        za_dim = latent_dim + action_dim

        # Latent dynamics: z + a -> z'
        dyn_dims = [za_dim] + [h] * mlp_depth + [latent_dim]
        self.dynamics = _mlp(dyn_dims)

        # Reward: z + a -> scalar
        rew_dims = [za_dim] + [h] * mlp_depth + [1]
        self.reward_fn = _mlp(rew_dims)

        # Q-ensemble: z + a -> scalar  (num_q independent heads)
        self.q_heads = nn.ModuleList([
            _mlp([za_dim] + [h] * mlp_depth + [1])
            for _ in range(num_q)
        ])

        # Policy head: z -> (mu, log_std)
        pol_dims = [latent_dim] + [h] * mlp_depth
        self.policy_trunk = _mlp(pol_dims, output_act=nn.SiLU)
        self.policy_mu = nn.Linear(h, action_dim)
        self.policy_log_std = nn.Linear(h, action_dim)

    # ------------------------------------------------------------------
    def forward_dynamics(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        za = torch.cat([z, a], dim=-1)
        return self.dynamics(za)

    def forward_reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        za = torch.cat([z, a], dim=-1)
        return self.reward_fn(za).squeeze(-1)

    def forward_q(
        self, z: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        """Returns (num_q, B) tensor of Q-values."""
        za = torch.cat([z, a], dim=-1)
        return torch.stack([head(za).squeeze(-1) for head in self.q_heads], dim=0)

    def forward_policy(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_std) in pre-squash space."""
        trunk = self.policy_trunk(z)
        mu = self.policy_mu(trunk)
        log_std = self.policy_log_std(trunk).clamp(-5, 2)
        return mu, log_std

    def sample_action(
        self, z: torch.Tensor, std: float = 0.0
    ) -> torch.Tensor:
        """Sample or take mean action, clipped to [-1, 1]."""
        mu, log_std = self.forward_policy(z)
        if std > 0:
            noise = torch.randn_like(mu) * std
            return torch.tanh(mu + noise)
        return torch.tanh(mu)
