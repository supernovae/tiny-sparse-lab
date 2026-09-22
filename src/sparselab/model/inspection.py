"""Honest parameter and storage accounting for the dense baseline."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from sparselab.model.moe import Top1MoE
from sparselab.model.transformer import DenseLM


def _unique_numel(parameters: Iterable[nn.Parameter]) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            total += parameter.numel()
    return total


def inspect_model(model: nn.Module) -> dict[str, int | str]:
    """Count unique storage and one-token direct parameter use, not FLOPs."""
    total = _unique_numel(model.parameters())
    trainable = _unique_numel(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not isinstance(model, DenseLM):
        raise TypeError("inspection currently supports DenseLM")
    embedding = model.embedding.weight.numel()
    tied = model.output.weight is model.embedding.weight
    attention = sum(
        _unique_numel(block.attention.parameters()) for block in model.blocks
    )
    dense_ffn = 0
    expert = 0
    active_expert = 0
    for block in model.blocks:
        if isinstance(block.ffn, Top1MoE):
            expert_parameters = _unique_numel(block.ffn.experts.parameters())
            expert += expert_parameters
            active_expert += _unique_numel(block.ffn.experts[0].parameters())
        else:
            dense_ffn += _unique_numel(block.ffn.parameters())
    norm = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "norm" in name
    )
    output_head = 0 if tied else model.output.weight.numel()
    active = total if tied else total - embedding + model.config.hidden_dim
    if model.config.ffn == "moe":
        active = active - expert + active_expert
    weight_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    optimizer_state_bytes = trainable * 8
    return {
        "total": total,
        "trainable": trainable,
        "active_per_token": active,
        "embedding": embedding,
        "attention": attention,
        "ffn": dense_ffn,
        "norm": norm,
        "output_head": output_head,
        "expert": expert,
        "engram": 0,
        "expert_note": (
            "not present in dense model"
            if model.config.ffn == "dense"
            else "all local expert storage"
        ),
        "engram_note": "not present in dense model",
        "model_weight_bytes": weight_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "estimated_checkpoint_bytes": weight_bytes + optimizer_state_bytes,
    }
