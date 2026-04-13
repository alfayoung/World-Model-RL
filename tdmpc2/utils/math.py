"""Small numerical utilities shared across TD-MPC2 components."""
import torch
import torch.nn.functional as F


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform from TD-MPC2 paper."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (x.abs().exp() - 1)


def two_hot(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Two-hot encoding over a fixed set of bins (for distributional RL)."""
    x = x.clamp(bins[0], bins[-1])
    k = torch.searchsorted(bins, x.unsqueeze(-1)).squeeze(-1)
    k = k.clamp(0, len(bins) - 2)
    lower = bins[k]
    upper = bins[k + 1]
    upper_weight = (x - lower) / (upper - lower + 1e-8)
    lower_weight = 1.0 - upper_weight
    hot = torch.zeros(*x.shape, len(bins), device=x.device)
    hot.scatter_(-1, k.unsqueeze(-1), lower_weight.unsqueeze(-1))
    hot.scatter_(-1, (k + 1).unsqueeze(-1), upper_weight.unsqueeze(-1))
    return hot
