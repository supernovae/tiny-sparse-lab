"""Honest parameter and storage accounting for the dense baseline."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from sparselab.model.moe import TopKMoE
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
        if isinstance(block.ffn, TopKMoE):
            expert_parameters = _unique_numel(block.ffn.experts.parameters())
            active_parameters = _unique_numel(
                parameter
                for expert_module in block.ffn.experts[: block.ffn.experts_per_token]
                for parameter in expert_module.parameters()
            )
            if block.ffn.shared_expert is not None:
                shared = _unique_numel(block.ffn.shared_expert.parameters())
                expert_parameters += shared
                active_parameters += shared
            expert += expert_parameters
            active_expert += active_parameters
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
    memory_parameters = 0
    memory_tables = 0
    active_memory_rows = 0
    if model.memory is not None:
        memory_parameters = _unique_numel(model.memory.parameters())
        for module in model.memory.modules():
            if isinstance(module, nn.Embedding):
                memory_tables += module.weight.numel()
                active_memory_rows += module.embedding_dim
        active = active - memory_tables + active_memory_rows
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
        "engram": memory_parameters,
        "engram_table": memory_tables,
        "engram_adapter": memory_parameters - memory_tables,
        "expert_note": (
            "not present in dense model"
            if model.config.ffn == "dense"
            else "all local expert storage"
        ),
        "engram_note": (
            "not present"
            if model.memory is None
            else "all tables and adapters counted in total; one row per table in active_per_token"
        ),
        "model_weight_bytes": weight_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "estimated_checkpoint_bytes": weight_bytes + optimizer_state_bytes,
    }


def architecture_metrics(model: DenseLM) -> dict[str, float]:
    """Scalar diagnostics from the last training microbatch, not update averages."""
    result: dict[str, float] = {}
    if model.memory is not None and model.memory.last_diagnostics is not None:
        diagnostic = model.memory.last_diagnostics
        for name in (
            "lookup_count",
            "unique_addresses",
            "collision_count",
            "bucket_reuse_rate",
            "table_utilization",
            "maximum_address_fraction",
            "gate_mean",
            "value_norm",
            "hidden_norm",
        ):
            result[f"engram/{name}"] = float(getattr(diagnostic, name).detach())
    for index, block in enumerate(model.blocks):
        if isinstance(block.ffn, TopKMoE) and block.ffn.last_diagnostics is not None:
            diagnostic = block.ffn.last_diagnostics
            for metric, field in (
                ("router_entropy", "entropy"),
                ("maximum_expert_fraction", "maximum_fraction"),
                ("mean_topk_probability", "mean_topk_probability"),
            ):
                result[f"moe/layer_{index}/{metric}"] = float(
                    getattr(diagnostic, field).detach()
                )
        diagnostic = getattr(block.attention, "last_diagnostics", None)
        if diagnostic is not None:
            for metric, field in (
                ("available_tokens", "available_tokens"),
                ("selected_tokens", "selected_tokens"),
                ("selection_ratio", "selection_ratio"),
                ("estimated_flops", "estimated_attention_flops"),
                ("dense_teacher_mass", "dense_teacher_mass"),
                ("dense_teacher_topk_recall", "dense_teacher_topk_recall"),
            ):
                result[f"attention/layer_{index}/{metric}"] = float(
                    getattr(diagnostic, field).detach()
                )
    return result
