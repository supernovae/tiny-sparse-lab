"""Deterministic v1-to-v2 run-configuration migration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def migrate_v1(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a v1 run mapping without changing scientific settings."""
    if raw.get("schema_version") != 1:
        raise ValueError("only schema_version: 1 run configs can be migrated")
    result = deepcopy(raw)
    result["schema_version"] = 2
    device = result.pop("device", "auto")
    result["runtime"] = {
        "engine": "pytorch",
        "backend": device,
        "device_index": 0,
        "precision": "fp32",
        "memory": {
            "policy": "balanced",
            "max_device_memory_fraction": 0.90,
            "budget_bytes": None,
            "activation_checkpointing": {"enabled": False, "strategy": "transformer_block"},
            "activation_offload": {"enabled": False},
            "allowed_sequence_lengths": [],
            "allowed_optimizers": [],
        },
    }
    training = result["training"]
    training["micro_batch_size"] = training.pop("batch_size")
    training["gradient_accumulation"] = 1
    optimizer = result.get("optimizer", {})
    optimizer["name"] = "adamw"
    optimizer["peak"] = optimizer.pop("learning_rate", 3e-4)
    optimizer["floor"] = optimizer.pop("min_learning_rate", 3e-5)
    optimizer.setdefault("warmup_steps", 10)
    optimizer.setdefault("weight_decay", 0.1)
    optimizer.setdefault("betas", [0.9, 0.95])
    optimizer.setdefault("eps", 1e-8)
    optimizer["state_offload"] = False
    result["optimizer"] = optimizer
    logging = result["logging"]
    cadence = logging.pop("checkpoint_every_steps", 50)
    result["checkpoint"] = {"every_steps": cadence, "every_tokens": None, "every_minutes": None, "keep_periodic": True}
    result["staging"] = {"smoke_steps": 2, "warmup_steps": 5}
    return result


def migrate_file(source: Path, output: Path) -> list[str]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("config must be a YAML mapping")
    migrated = migrate_v1(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(migrated, sort_keys=False), encoding="utf-8")
    return ["schema_version: 1 -> 2", "device -> runtime.backend", "training.batch_size -> training.micro_batch_size", "logging.checkpoint_every_steps -> checkpoint.every_steps"]
