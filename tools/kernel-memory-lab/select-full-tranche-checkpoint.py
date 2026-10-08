"""Verify all full-tranche boundaries and freeze the predeclared best checkpoint."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from sparselab.config.loading import load_config
from sparselab.evaluation.evidence import experiment_evidence
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import canonical_json, read_manifest, sha256_file

STEPS = (0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4883)
RUN_ID = "kml-card05-full-tranche-v1"


def select(root: Path, checkout: Path) -> dict[str, object]:
    config = load_config(
        checkout / "experiments/research/kernel-memory-lab/card05-full-tranche-v1.yaml"
    )
    run = config.logging.root_dir / RUN_ID
    progress = json.loads((run / "progress.json").read_text())
    if (
        progress.get("status") != "completed"
        or progress.get("step") != 4883
        or progress.get("tokens_seen") != 5_000_000
        or progress.get("parent_run_id") is not None
    ):
        raise ValueError("full tranche did not complete fresh at the exact target")
    evidence = experiment_evidence(run)
    checkpoints = evidence["checkpoints"]
    observations = evidence["quality_observations"]
    if (
        evidence["run_id"] != RUN_ID
        or not evidence["verified_checkpoints"]
        or evidence["missing_reports"]
        or evidence["rejected_reports"]
        or len(checkpoints) != len(STEPS)
        or len(observations) != len(STEPS)
    ):
        raise ValueError("full-tranche evidence incomplete or rejected")
    by_step = {row["step"]: row for row in checkpoints}
    by_observation = {row["step"]: row for row in observations}
    if set(by_step) != set(by_observation) or set(by_step) != set(STEPS):
        raise ValueError("validation/checkpoint cadence differs from declaration")
    for step in STEPS:
        checkpoint, observation = by_step[step], by_observation[step]
        if (
            not checkpoint["verified"]
            or checkpoint["errors"]
            or observation["checkpoint"] != checkpoint["path"]
            or observation["checkpoint_sha256"] != checkpoint["digest"]
            or observation["batches"] != 1
            or observation["max_batches"] != 1
            or not isinstance(observation["loss"], (int, float))
            or not math.isfinite(observation["loss"])
        ):
            raise ValueError(f"invalid validation/checkpoint event at {step}")
    chosen_step = min(STEPS, key=lambda step: (by_observation[step]["loss"], step))
    chosen = by_step[chosen_step]
    best = json.loads((run / "checkpoints/best.json").read_text())
    if (
        best["step"] != chosen_step
        or best["relative_path"] != chosen["path"]
        or best["manifest_sha256"] != chosen["digest"]
    ):
        raise ValueError("native best pointer differs from predeclared selection")
    manifest = read_manifest(run / "manifest.json")
    import hashlib

    manifest_sha = hashlib.sha256(canonical_json(manifest)).hexdigest()
    manager = CheckpointManager(run, manifest_sha256=manifest_sha)
    for step in {0, chosen_step, 4883}:
        report = manager.verify(
            run / "checkpoints" / by_step[step]["path"],
            expected_manifest=manifest_sha,
            require_training_state=True,
            expected_config=config,
        )
        if not report.valid or report.resume_level != "full":
            raise ValueError(f"full checkpoint verification failed at {step}")
    result: dict[str, object] = {
        "format": "kml-card05-selected-checkpoint-v1",
        "run_id": RUN_ID,
        "selected_step": chosen_step,
        "checkpoint": chosen["path"],
        "checkpoint_sha256": chosen["digest"],
        "loss": by_observation[chosen_step]["loss"],
        "validation_steps": list(STEPS),
        "run_manifest_file_sha256": sha256_file(run / "manifest.json"),
        "progress_file_sha256": sha256_file(run / "progress.json"),
        "selection": "earliest verified checkpoint attaining minimum finite one-batch held-out loss",
    }
    output = root / "selected-checkpoint.json"
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


if __name__ == "__main__":
    select(Path(os.environ["KML_PROFILE_ROOT"]), Path(os.environ["KML_CHECKOUT"]))
