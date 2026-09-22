"""Summarize verified local artifacts without turning observations into benchmark claims."""
from __future__ import annotations

import json
from pathlib import Path

from sparselab.training.checkpoints import CheckpointManager


def experiment_evidence(run: Path) -> dict[str, object]:
    """Return checkpoint-integrity and held-out quality observations for one run."""
    manifest = json.loads((run / "manifest.json").read_text())
    manager = CheckpointManager(run)
    checkpoints = []
    for path in sorted((run / "checkpoints").glob("step_*_gen_*")):
        report = manager.verify(path)
        checkpoint_manifest = json.loads((path / "manifest.json").read_text())
        checkpoints.append(
            {
                "path": path.name,
                "step": checkpoint_manifest["step"],
                "tokens_seen": checkpoint_manifest["tokens_seen"],
                "validation_loss": checkpoint_manifest["validation_loss"],
                "verified": report.valid,
                "errors": list(report.errors),
            }
        )
    observations = []
    for path in sorted((run / "evaluations").glob("validation_step_*_gen_*.json")):
        payload = json.loads(path.read_text())
        observations.append(
            {
                "checkpoint": payload["checkpoint"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "step": payload["step"],
                "tokens_seen": payload["tokens_seen"],
                "loss": payload["loss"],
                "perplexity": payload["perplexity"],
            }
        )
    verified = bool(checkpoints) and all(item["verified"] for item in checkpoints)
    level = "checkpointed_held_out" if verified and observations else "artifact_only"
    return {
        "format": "experiment_evidence_v1",
        "run_id": manifest["run_id"],
        "source_identity_sha256": manifest["source_identity"]["sha256"],
        "verified_checkpoints": verified,
        "checkpoint_count": len(checkpoints),
        "quality_observations": observations,
        "evidence_level": level,
        "interpretation": (
            "Verified checkpoints are paired with held-out loss observations. "
            "This validates the configured local experiment path, not a general model-quality claim."
            if level == "checkpointed_held_out"
            else "No verified checkpoint and held-out observation pair is available."
        ),
        "checkpoints": checkpoints,
    }
