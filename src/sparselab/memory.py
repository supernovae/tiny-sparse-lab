"""Conservative shape-only memory accounting and explicit proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sparselab.config.models import RunConfig
from sparselab.model.inspection import ParameterInventory, parameter_inventory
from sparselab.runtime import RuntimeInfo, allocated_memory_bytes, process_rss_bytes


@dataclass(frozen=True)
class MemoryEstimate:
    version: str
    result: Literal["LIKELY_TO_FIT", "LIKELY_TO_EXCEED", "UNKNOWN"]
    peak_bytes: int
    capacity_ceiling_bytes: int | None
    resident_weights_bytes: int
    gradients_bytes: int
    optimizer_bytes: int
    activations_bytes: int
    attention_working_bytes: int
    workspace_bytes: int
    headroom_bytes: int
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class ResourceProposal:
    config: dict[str, object]
    decisions: tuple[dict[str, object], ...]
    estimate: MemoryEstimate


def _capacity_ceiling(
    config: RunConfig, runtime: RuntimeInfo
) -> tuple[int | None, str | None]:
    """Return a physical capacity ceiling, never treating an artificial cap as RAM."""
    fraction = config.runtime.memory.max_device_memory_fraction
    backend = runtime.backend
    if backend == "cpu":
        if runtime.system_total_bytes is None or runtime.system_available_bytes is None:
            return None, "CPU total or available RAM is unavailable"
        return max(
            0,
            min(
                int(fraction * runtime.system_total_bytes),
                runtime.system_available_bytes,
            ),
        ), None
    if backend in {"cuda", "rocm", "xpu"}:
        if runtime.device_total_bytes is None or runtime.device_free_bytes is None:
            return None, "discrete device total or free memory is unavailable"
        return max(
            0,
            min(int(fraction * runtime.device_total_bytes), runtime.device_free_bytes),
        ), None
    if backend in {"mps", "metal"}:
        if (
            runtime.device_recommended_bytes is None
            or runtime.system_available_bytes is None
            or runtime.device_driver_allocated_bytes is None
        ):
            return (
                None,
                "unified-memory recommendation, driver allocation or available system RAM is unavailable",
            )
        return max(
            0,
            min(
                int(fraction * runtime.device_recommended_bytes)
                - runtime.device_driver_allocated_bytes,
                runtime.system_available_bytes,
            ),
        ), None
    return None, f"unknown backend {backend!r} has no capacity policy"


def estimate_memory(
    config: RunConfig,
    runtime_info: RuntimeInfo,
    inventory: ParameterInventory,
    *,
    calibration: float | None = None,
) -> MemoryEstimate:
    """Estimate disjoint training-memory categories without materializing tensors."""
    b, t, d, f, l, h, v = (
        config.training.micro_batch_size,
        config.training.seq_len,
        config.model.hidden_dim,
        config.model.ffn_dim,
        config.model.num_layers,
        config.model.num_heads,
        config.model.vocab_size,
    )
    m, attention = config.model, config.attention
    weights = inventory.total * 4
    gradients = inventory.trainable * 4
    optimizer = inventory.trainable * (8 if config.optimizer.name == "adamw" else 4)

    # Retained non-attention values are separate from attention's projections
    # and score/probability working tensors.  All bytes are conservative FP32.
    if m.ffn == "dense":
        per_block_nonattention = b * t * (4 * 6 * d + 4 * 3 * f)
    else:
        direct_experts = m.experts_per_token + int(m.shared_expert)
        per_block_nonattention = (
            b * t * (4 * 6 * d + 4 * (3 * f * direct_experts + 2 * m.num_experts))
        )
    projected_width = (
        2 * d + (attention.latent_dim or d) if attention.kind == "mla" else 3 * d
    )
    per_block_attention = 4 * b * t * projected_width + 8 * b * h * t * t
    streams = (
        (len(m.memory_ngram_orders) or 1) * m.memory_hash_heads
        if m.memory == "ngram"
        else int(m.memory != "none")
    )
    logits_and_memory = 8 * b * t * v + 8 * b * t * streams * d
    if config.runtime.memory.activation_checkpointing.enabled:
        activations = (
            (l + 1) * 4 * b * t * d + per_block_nonattention + logits_and_memory
        )
        attention_working = per_block_attention
    else:
        activations = l * per_block_nonattention + logits_and_memory
        attention_working = l * per_block_attention
    workspace = max(
        64 * 1024 * 1024,
        int(0.1 * (weights + gradients + optimizer)),
    )
    subtotal = (
        weights + gradients + optimizer + activations + attention_working + workspace
    )
    headroom = int(0.15 * subtotal)
    peak = subtotal + headroom
    if calibration is not None:
        peak = int(peak * max(calibration, 1.0))

    physical_ceiling, missing = _capacity_ceiling(config, runtime_info)
    budget = config.runtime.memory.budget_bytes
    ceiling = (
        min(physical_ceiling, budget)
        if physical_ceiling is not None and budget is not None
        else (physical_ceiling if physical_ceiling is not None else budget)
    )
    if physical_ceiling is None:
        # An artificial budget can disprove fit, but cannot certify physical fit.
        result: Literal["LIKELY_TO_FIT", "LIKELY_TO_EXCEED", "UNKNOWN"] = (
            "LIKELY_TO_EXCEED" if budget is not None and peak > budget else "UNKNOWN"
        )
    else:
        result = "LIKELY_TO_FIT" if peak <= ceiling else "LIKELY_TO_EXCEED"
    assumptions = [
        "FP32 master parameters and gradients; autocast does not reduce resident model or AdamW state",
        "categories are disjoint: logits are retained activations, not workspace",
        "dense score/probability allowance remains conservative for sliding, MLA, and block-sparse reference attention",
    ]
    if config.runtime.memory.activation_offload.enabled:
        assumptions.append(
            "activation offload is unsupported here and receives no estimated saving"
        )
    if missing is not None:
        assumptions.append(missing)
    if budget is not None and physical_ceiling is None:
        assumptions.append(
            "artificial budget can only demonstrate an exceedance; it cannot certify a physical fit"
        )
    return MemoryEstimate(
        "reference-v1",
        result,
        peak,
        ceiling,
        weights,
        gradients,
        optimizer,
        activations,
        attention_working,
        workspace,
        headroom,
        tuple(assumptions),
    )


def _decision(
    name: str,
    before: MemoryEstimate,
    after: MemoryEstimate,
    reason: str,
    **extra: object,
) -> dict[str, object]:
    return {
        "requested": name,
        "reason": reason,
        "before_peak_bytes": before.peak_bytes,
        "after_peak_bytes": after.peak_bytes,
        **extra,
    }


def plan_memory(
    config: RunConfig, runtime_info: RuntimeInfo, estimate: MemoryEstimate
) -> ResourceProposal:
    """Produce a complete immutable candidate; never alter the supplied config."""
    candidate = config
    # Proposals are serialized complete configurations, so their comparison
    # must be derived again from that configuration rather than trusting a
    # caller's potentially stale estimate object.
    current = estimate_memory(config, runtime_info, parameter_inventory(config))
    decisions: list[dict[str, object]] = []
    policy = config.runtime.memory.policy
    effective = config.training.micro_batch_size * config.training.gradient_accumulation
    if current.result == "LIKELY_TO_EXCEED" and policy in {
        "balanced",
        "low_memory",
        "max_fit",
    }:
        target_micro = (
            1
            if policy in {"low_memory", "max_fit"}
            else max(1, config.training.micro_batch_size // 2)
        )
        if (
            target_micro < config.training.micro_batch_size
            and effective % target_micro == 0
        ):
            candidate = candidate.model_copy(
                update={
                    "training": candidate.training.model_copy(
                        update={
                            "micro_batch_size": target_micro,
                            "gradient_accumulation": effective // target_micro,
                        }
                    )
                }
            )
            after = estimate_memory(
                candidate, runtime_info, parameter_inventory(candidate)
            )
            decisions.append(
                _decision(
                    "micro_batch_size",
                    current,
                    after,
                    "preserves effective batch while reducing activation dimensions",
                    effective_batch_size=effective,
                    effective=target_micro,
                )
            )
            current = after
        if (
            not candidate.runtime.memory.activation_checkpointing.enabled
            and candidate.runtime.engine == "pytorch"
        ):
            runtime = candidate.runtime.model_copy(
                update={
                    "memory": candidate.runtime.memory.model_copy(
                        update={
                            "activation_checkpointing": candidate.runtime.memory.activation_checkpointing.model_copy(
                                update={"enabled": True}
                            )
                        }
                    )
                }
            )
            proposed = candidate.model_copy(update={"runtime": runtime})
            after = estimate_memory(
                proposed, runtime_info, parameter_inventory(proposed)
            )
            decisions.append(
                _decision(
                    "activation_checkpointing",
                    current,
                    after,
                    "supported PyTorch transformer-block recomputation proposal",
                    effective=True,
                )
            )
            candidate, current = proposed, after
        elif not candidate.runtime.memory.activation_checkpointing.enabled:
            decisions.append(
                {
                    "requested": "activation_checkpointing",
                    "effective": False,
                    "reason": "unsupported engine; no saving was assumed",
                }
            )
        if candidate.runtime.memory.activation_offload.enabled:
            decisions.append(
                {
                    "requested": "activation_offload",
                    "effective": True,
                    "reason": "unsupported by this planner; no saving was assumed",
                }
            )
        if current.result == "LIKELY_TO_EXCEED" and policy == "max_fit":
            lengths = sorted(
                {
                    length
                    for length in candidate.runtime.memory.allowed_sequence_lengths
                    if length < candidate.training.seq_len
                },
                reverse=True,
            )
            if lengths:
                length = lengths[0]
                proposed = candidate.model_copy(
                    update={
                        "training": candidate.training.model_copy(
                            update={"seq_len": length}
                        )
                    }
                )
                after = estimate_memory(
                    proposed, runtime_info, parameter_inventory(proposed)
                )
                decisions.append(
                    _decision(
                        "sequence_length",
                        current,
                        after,
                        "explicit allowed_sequence_lengths candidate; scientifically significant",
                        effective=length,
                        scientifically_significant=True,
                    )
                )
                candidate, current = proposed, after
            if candidate.runtime.precision != "fp32":
                decisions.append(
                    {
                        "requested": "precision",
                        "effective": candidate.runtime.precision,
                        "reason": "no validated precision saving is advertised",
                    }
                )
    return ResourceProposal(
        candidate.model_dump(mode="json"), tuple(decisions), current
    )


class MemoryMonitor:
    def __init__(self, device: object) -> None:
        self.device = device
        self.samples: list[dict[str, int | None]] = []

    def sample(self, phase: str) -> dict[str, int | None]:
        allocated = allocated_memory_bytes(self.device)  # type: ignore[arg-type]
        sample = {
            "phase": phase,
            "memory/device_allocated_bytes": allocated,
            "memory/process_rss_bytes": process_rss_bytes(),
        }
        self.samples.append(sample)
        return sample
