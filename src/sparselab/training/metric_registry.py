"""Canonical metric metadata shared by training and dashboard views."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    summary: str
    help_slug: str
    producer: str


METRICS = {
    spec.name: spec
    for spec in (
        MetricSpec(
            "train/loss",
            "nats/target",
            "Mean next-token negative log-probability.",
            "loss",
            "trainer",
        ),
        MetricSpec(
            "validation/loss",
            "nats/target",
            "Held-out next-token negative log-probability.",
            "loss",
            "evaluation",
        ),
        MetricSpec(
            "validation/perplexity",
            "ratio",
            "exp(validation/loss).",
            "perplexity",
            "evaluation",
        ),
        MetricSpec(
            "optimizer/learning_rate",
            "ratio",
            "AdamW update learning rate.",
            "learning-rate",
            "optimizer",
        ),
        MetricSpec(
            "optimizer/grad_norm",
            "L2",
            "Global norm before clipping.",
            "gradient-norm",
            "trainer",
        ),
        MetricSpec(
            "performance/tokens_per_second",
            "targets/s",
            "Valid targets processed per elapsed second.",
            "throughput",
            "trainer",
        ),
        MetricSpec(
            "moe/router_auxiliary_loss",
            "loss",
            "Auxiliary router load-balance term.",
            "moe",
            "trainer",
        ),
        MetricSpec(
            "memory/device_allocated_bytes",
            "bytes",
            "Native device allocation at an update boundary.",
            "memory",
            "memory-monitor",
        ),
        MetricSpec(
            "memory/process_rss_bytes",
            "bytes",
            "Resident host process memory at an update boundary.",
            "memory",
            "memory-monitor",
        ),
        MetricSpec(
            "batch/effective_tokens_per_update",
            "targets",
            "Committed valid prediction targets in one optimizer update.",
            "gradient-accumulation",
            "trainer",
        ),
    )
}


def metric_spec(name: str) -> MetricSpec | None:
    """Return canonical metadata for an exact persisted metric name."""
    return METRICS.get(name)
