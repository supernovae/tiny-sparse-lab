"""Immutable preflight staging for a concrete local run configuration."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from sparselab.config.models import RunConfig
from sparselab.memory import estimate_memory, parameter_inventory
from sparselab.runtime import validate_runtime
from sparselab.training.stages import ExperimentStage, StageHistory


def stage(config: RunConfig, output: Path, through: str = "smoke") -> Path:
    levels = {"inspect": 1, "validate": 2, "smoke": 3, "warmup": 4}
    if through not in levels:
        raise ValueError("through must be inspect, validate, smoke, or warmup")
    if output.exists():
        raise FileExistsError(f"stage output exists: {output}")
    output.mkdir(parents=True)
    history = StageHistory()
    history.start(ExperimentStage.CONFIGURED)
    history.finish()
    history.start(ExperimentStage.INSPECTED)
    inventory = parameter_inventory(config)
    history.finish()
    runtime = None
    estimate = None
    if levels[through] >= 2:
        history.start(ExperimentStage.VALIDATED)
        runtime = validate_runtime(config)
        estimate = estimate_memory(config, runtime, inventory)
        history.finish()
    if levels[through] >= 3:
        history.start(ExperimentStage.SMOKE_TEST)
        history.finish("evidence_required", reason="run via sparselab run")
    if levels[through] >= 4:
        history.start(ExperimentStage.WARMUP)
        history.finish("evidence_required", reason="run via sparselab run")
    (output / "stage.json").write_text(
        json.dumps(
            {
                "config": config.model_dump(mode="json"),
                "inventory": asdict(inventory),
                "runtime": runtime.as_dict() if runtime else None,
                "estimate": asdict(estimate) if estimate else None,
                "stages": [asdict(x) for x in history.records],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    return output
