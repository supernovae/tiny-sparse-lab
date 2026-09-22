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
    mean_topk_probability: Tensor
    auxiliary_loss: Tensor
    router_logits: Tensor
    selected_experts: Tensor
    selected_weights: Tensor


class TopKMoE(nn.Module):
    """Readable local top-k SwiGLU dispatch with normalized router weights.

    Tokens are flattened from ``[B, T, D]`` to ``[B*T, D]``.  Each selected
    expert receives only its selected rows; weighted ``index_add_`` restores
    the original token order without materializing an expert-by-token tensor.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        experts_per_token: int = 1,
        shared_expert: bool = False,
        auxiliary_loss_coefficient: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.auxiliary_loss_coefficient = auxiliary_loss_coefficient
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(hidden_dim, ffn_dim) for _ in range(num_experts)
        )
        self.shared_expert = SwiGLU(hidden_dim, ffn_dim) if shared_expert else None
        self.last_diagnostics: RoutingDiagnostics | None = None
        self.auxiliary_loss = torch.zeros(())

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        tokens = x.reshape(-1, original_shape[-1])
        router_logits = self.router(tokens)
        probabilities = torch.softmax(router_logits, dim=-1)
        weights, selected = probabilities.topk(self.experts_per_token, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(tokens)
        for expert_index, expert in enumerate(self.experts):
            token_indices, choice_indices = (selected == expert_index).nonzero(
                as_tuple=True
            )
            if token_indices.numel():
                contribution = expert(tokens.index_select(0, token_indices))
                contribution = contribution * weights[
                    token_indices, choice_indices
                ].unsqueeze(-1)
                output.index_add_(0, token_indices, contribution)
        if self.shared_expert is not None:
            output = output + self.shared_expert(tokens)
        counts = torch.bincount(selected.flatten(), minlength=self.num_experts).detach()
        fractions = counts.float() / max(1, selected.numel())
        entropy = (
            -(
                probabilities
                * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
            )
            .sum(dim=-1)
            .mean()
        )
        importance = probabilities.mean(dim=0)
        load = fractions.detach()
        auxiliary_loss = (
            self.auxiliary_loss_coefficient
            * self.num_experts
            * (importance * load).sum()
        )
        self.auxiliary_loss = auxiliary_loss
        self.last_diagnostics = RoutingDiagnostics(
            counts=counts,
            fractions=fractions,
            entropy=entropy.detach(),
            maximum_fraction=fractions.max(),
            mean_topk_probability=weights.mean().detach(),
            auxiliary_loss=auxiliary_loss.detach(),
            router_logits=router_logits.detach(),
            selected_experts=selected.detach(),
            selected_weights=weights.detach(),
        )
        return output.reshape(original_shape)
