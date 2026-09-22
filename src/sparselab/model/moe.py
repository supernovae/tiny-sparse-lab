"""Local top-one routed SwiGLU experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sparselab.model.ffn import SwiGLU


@dataclass(frozen=True)
class RoutingDiagnostics:
    counts: Tensor
    fractions: Tensor
    entropy: Tensor
    maximum_fraction: Tensor


class Top1MoE(nn.Module):
    """Dispatch flattened tokens to one local SwiGLU expert each."""

    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(hidden_dim, ffn_dim) for _ in range(num_experts)
        )
        self.last_diagnostics: RoutingDiagnostics | None = None

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        tokens = x.reshape(-1, original_shape[-1])
        probabilities = torch.softmax(self.router(tokens), dim=-1)
        selected = probabilities.argmax(dim=-1)
        output = torch.empty_like(tokens)
        for expert_index, expert in enumerate(self.experts):
            positions = (selected == expert_index).nonzero(as_tuple=True)[0]
            if positions.numel():
                output.index_copy_(
                    0, positions, expert(tokens.index_select(0, positions))
                )
        counts = torch.bincount(selected, minlength=self.num_experts).detach()
        fractions = counts.float() / max(1, tokens.shape[0])
        entropy = (
            -(
                probabilities
                * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
            )
            .sum(dim=-1)
            .mean()
        )
        self.last_diagnostics = RoutingDiagnostics(
            counts, fractions, entropy.detach(), fractions.max()
        )
        return output.reshape(original_shape)
