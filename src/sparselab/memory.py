"""Conservative shape-only memory accounting and explicit proposals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sparselab.config.models import RunConfig
from sparselab.runtime import RuntimeInfo, allocated_memory_bytes, process_rss_bytes


@dataclass(frozen=True)
class ParameterInventory:
    total: int
    trainable: int
    embedding: int
    output_head: int
    attention: int
    dense_ffn: int
    routed_expert: int
    shared_expert: int
    router: int
    norm: int
    memory_table: int
    memory_adapter: int
    frozen: int


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


def parameter_inventory(config: RunConfig) -> ParameterInventory:
    m = config.model
    d, f, l, v = m.hidden_dim, m.ffn_dim, m.num_layers, m.vocab_size
    embedding = v * d
    output = 0 if m.tie_embeddings else v * d
    attention = l * 4 * d * d
    norm = l * 2 * d + d
    dense = routed = shared = router = 0
    if m.ffn == "dense":
        dense = l * 3 * d * f
    else:
        routed = l * m.num_experts * 3 * d * f
        shared = l * 3 * d * f if m.shared_expert else 0
        router = l * d * m.num_experts
    streams = len(m.memory_ngram_orders) or (1 if m.memory != "none" else 0)
    table = streams * m.memory_hash_heads * m.memory_table_size * m.memory_dim
    adapter = streams * (m.memory_dim * d + d) if streams else 0
    frozen = table if m.memory == "portable" else 0
    total = embedding + output + attention + norm + dense + routed + shared + router + table + adapter
    return ParameterInventory(total, total - frozen, embedding, output, attention, dense, routed, shared, router, norm, table, adapter, frozen)


def estimate_memory(config: RunConfig, runtime_info: RuntimeInfo, inventory: ParameterInventory, *, calibration: float | None = None) -> MemoryEstimate:
    b, t, d, f, l, h, v = (config.training.micro_batch_size, config.training.seq_len, config.model.hidden_dim, config.model.ffn_dim, config.model.num_layers, config.model.num_heads, config.model.vocab_size)
    weights = inventory.total * 4
    gradients = inventory.trainable * 4
    optimizer = inventory.trainable * 8 if config.optimizer.name == "adamw" else inventory.trainable * 4
    nonattention = l * b * t * (4 * 6 * d + 4 * 3 * f)
    attention = l * (3 * 4 * b * t * d + 8 * b * h * t * t)
    activations = ((l + 1) * 4 * b * t * d + (nonattention + attention) // max(l, 1)) if config.runtime.memory.activation_checkpointing.enabled else nonattention + attention
    logits = 8 * b * t * v
    workspace = max(64 * 1024 * 1024, int(0.1 * (weights + gradients + optimizer)) + logits)
    subtotal = weights + gradients + optimizer + activations + workspace
    headroom = int(0.15 * subtotal)
    peak = int((subtotal + headroom) * max(calibration or 1.0, 1.0))
    total = runtime_info.device_total_bytes or runtime_info.system_total_bytes
    free = runtime_info.device_free_bytes or runtime_info.system_available_bytes
    ceiling = min(int(total * config.runtime.memory.max_device_memory_fraction), free) if total is not None and free is not None else None
    if config.runtime.memory.budget_bytes is not None:
        ceiling = min(ceiling, config.runtime.memory.budget_bytes) if ceiling is not None else config.runtime.memory.budget_bytes
    result: Literal["LIKELY_TO_FIT", "LIKELY_TO_EXCEED", "UNKNOWN"] = "UNKNOWN" if ceiling is None else ("LIKELY_TO_FIT" if peak <= ceiling else "LIKELY_TO_EXCEED")
    return MemoryEstimate("reference-v1", result, peak, ceiling, weights, gradients, optimizer, activations, attention, workspace, headroom, ("FP32 master parameters, gradients, and AdamW moments", "uncalibrated conservative dense attention allowance"))


def plan_memory(config: RunConfig, runtime_info: RuntimeInfo, estimate: MemoryEstimate) -> ResourceProposal:
    candidate = config
    decisions: list[dict[str, object]] = []
    if estimate.result == "LIKELY_TO_EXCEED" and config.runtime.memory.policy in {"balanced", "low_memory", "max_fit"}:
        effective = config.training.micro_batch_size * config.training.gradient_accumulation
        micro = 1 if config.runtime.memory.policy == "low_memory" else max(1, config.training.micro_batch_size // 2)
        accumulation = effective // micro if effective % micro == 0 else config.training.gradient_accumulation
        runtime = config.runtime.model_copy(update={"memory": config.runtime.memory.model_copy(update={"activation_checkpointing": config.runtime.memory.activation_checkpointing.model_copy(update={"enabled": True})})})
        candidate = config.model_copy(update={"runtime": runtime, "training": config.training.model_copy(update={"micro_batch_size": micro, "gradient_accumulation": accumulation})})
        decisions.append({"requested": "micro_batch_size", "effective": micro, "reason": "preserve effective batch while reducing retained activations"})
    return ResourceProposal(candidate.model_dump(mode="json"), tuple(decisions), estimate)


class MemoryMonitor:
    def __init__(self, device: object) -> None:
        self.device = device
        self.samples: list[dict[str, int | None]] = []

    def sample(self, phase: str) -> dict[str, int | None]:
        allocated = allocated_memory_bytes(self.device)  # type: ignore[arg-type]
        sample = {"phase": phase, "memory/device_allocated_bytes": allocated, "memory/process_rss_bytes": process_rss_bytes()}
        self.samples.append(sample)
        return sample
