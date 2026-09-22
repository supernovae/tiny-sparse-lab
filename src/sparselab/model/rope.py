"""Real-valued adjacent-pair rotary position embedding."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        inverse_frequency = base ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

    def _cache(self, length: int, device: torch.device) -> tuple[Tensor, Tensor]:
        if self.cos.shape[0] < length or self.cos.device != device:
            positions = torch.arange(length, device=device, dtype=torch.float32)
            angles = torch.outer(positions, self.inverse_frequency.to(device))
            self.cos = angles.cos()[None, None, :, :]
            self.sin = angles.sin()[None, None, :, :]
        return self.cos[:, :, :length], self.sin[:, :, :length]

    def forward(self, x: Tensor) -> Tensor:
        cos, sin = self._cache(x.shape[-2], x.device)
        even, odd = x[..., 0::2], x[..., 1::2]
        output = torch.empty_like(x)
        output[..., 0::2] = even * cos + -odd * sin
        output[..., 1::2] = even * sin + odd * cos
        return output
