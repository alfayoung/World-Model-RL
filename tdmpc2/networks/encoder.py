"""Pixel encoder for TD-MPC2 (convolutional + MLP trunk)."""
import torch
import torch.nn as nn


def _conv_out_size(h: int, w: int, layers: nn.Sequential) -> int:
    with torch.no_grad():
        dummy = torch.zeros(1, *([3, h, w]))
        return layers(dummy).flatten(1).shape[-1]


class PixelEncoder(nn.Module):
    """CNN encoder that maps stacked RGB frames to a flat feature vector."""

    def __init__(
        self,
        obs_shape: tuple,          # (C, H, W)
        feature_dim: int = 256,
        num_layers: int = 4,
        num_filters: int = 32,
    ):
        super().__init__()
        c, h, w = obs_shape
        layers: list[nn.Module] = []
        in_ch = c
        for i in range(num_layers):
            out_ch = num_filters
            stride = 2 if i == 0 else 1
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.LayerNorm([out_ch, h // (2 if i == 0 else 1), w // (2 if i == 0 else 1)]),
                nn.SiLU(),
            ]
            in_ch = out_ch
            if i == 0:
                h = h // 2
                w = w // 2
        self.conv = nn.Sequential(*layers)
        flat_dim = num_filters * h * w
        self.trunk = nn.Sequential(
            nn.Linear(flat_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, C, H, W) uint8 or float in [0,255]."""
        x = obs.float() / 255.0
        x = self.conv(x)
        x = x.flatten(1)
        return self.trunk(x)
