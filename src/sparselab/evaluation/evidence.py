"""Summarize only integrity-bound local evaluation observations."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import canonical_json, read_manifest, sha256_file


def _inside(root: Path, path: Path) -> bool:
    try:
        return root.resolve() in path.resolve().parents
    except OSError:
        return False


def _validated_artifacts(run: Path, manifest: dict[str, object]) -> dict[str, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("run manifest lacks artifact inventory")
    verified: dict[str, str] = {}
    for item in artifacts:
        if not isinstance(item, dict):
            raise TypeError("invalid artifact inventory")
        name, digest = item.get("relative_path"), item.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str):
            raise TypeError("invalid artifact identity")
        path = run / name
        if (
            not _inside(run, path)
            or path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            raise ValueError(f"artifact integrity failure: {name}")
        verified[name] = digest
    return verified


def _report_observation(
    path: Path,
    checkpoints: dict[str, dict[str, object]],
    artifacts: dict[str, str],
    manifest: dict[str, object],
    validation_tokens: int,
) -> tuple[dict[str, object] | None, str | None]:
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return None, "report is not an object"
        digest = payload.pop("sha256", None)
        if (
            not isinstance(digest, str)
            or hashlib.sha256(canonical_json(payload)).hexdigest() != digest
        ):
            return None, "report hash mismatch"
        checkpoint = payload.get("checkpoint")
        bound = checkpoints.get(checkpoint) if isinstance(checkpoint, str) else None
        if bound is None or bound.get("digest") != payload.get("checkpoint_sha256"):
            return None, "checkpoint binding mismatch"
        if payload.get("kind") != "held_out_validation_v2":
            return None, "unknown evaluation protocol"
        identities = payload.get("identities")
        if (
            not isinstance(identities, dict)
            or identities.get("protocol") != "next-token-cross-entropy-v1"
        ):
            return None, "evaluation identity mismatch"
        for required in ("data/validation.npy", "tokenizer.json"):
            if identities.get(required) != artifacts.get(required):
                return None, f"{required} identity mismatch"
        if identities.get("source_identity_sha256") != manifest.get(
            "source_identity", {}
        ).get("sha256"):
            return None, "source identity mismatch"
        if payload.get("step") != bound.get("step") or payload.get(
            "tokens_seen"
        ) != bound.get("tokens_seen"):
            return None, "checkpoint counter mismatch"
        loss = payload.get("loss")
        valid_targets = payload.get("valid_targets")
        if (
            not isinstance(loss, (int, float))
            or not math.isfinite(loss)
            or not isinstance(valid_targets, int)
            or valid_targets <= 0
        ):
            return None, "invalid loss or target count"
        checkpoint_loss = bound.get("validation_loss")
        if not isinstance(checkpoint_loss, (int, float)) or loss != checkpoint_loss:
            return None, "checkpoint loss mismatch"
        perplexity = payload.get("perplexity")
        reason = payload.get("perplexity_unavailable_reason")
        if perplexity is None:
            if not isinstance(reason, str) or loss <= math.log(
                float.fromhex("0x1.fffffffffffffp+1023")
            ):
                return None, "invalid perplexity unavailability reason"
        elif (
            not isinstance(perplexity, (int, float))
            or not math.isfinite(perplexity)
            or reason is not None
            or loss > math.log(float.fromhex("0x1.fffffffffffffp+1023"))
            or not math.isclose(perplexity, math.exp(loss), rel_tol=1e-12)
        ):
            return None, "invalid perplexity"
        for field in ("batch_size", "max_batches", "seq_len"):
            if not isinstance(payload.get(field), int) or payload[field] <= 0:
                return None, f"missing {field}"
        config = manifest["effective_config"]
        expected_protocol = {
            "batch_size": config["training"]["micro_batch_size"],
            "seq_len": config["training"]["seq_len"],
            "max_batches": config["evaluation"]["max_batches"],
        }
        if any(payload[key] != value for key, value in expected_protocol.items()):
            return None, "evaluation protocol differs from run config"
        blocks = min(
            (validation_tokens - 1) // payload["seq_len"],
            payload["batch_size"] * payload["max_batches"],
        )
        if valid_targets != blocks * payload["seq_len"] or payload.get(
            "batches"
        ) != math.ceil(blocks / payload["batch_size"]):
            return None, "validation target or batch count mismatch"
        return payload, None
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return None, str(error)


def experiment_evidence(run: Path) -> dict[str, object]:
    """Return observations only when every referenced immutable artifact verifies."""
    manifest = read_manifest(run / "manifest.json")
    artifacts = _validated_artifacts(run, manifest)
    validation_tokens = len(
        np.load(run / "data/validation.npy", mmap_mode="r", allow_pickle=False)
    )
    manifest_digest = hashlib.sha256(canonical_json(manifest)).hexdigest()
    manager = CheckpointManager(run, manifest_sha256=manifest_digest)
    checkpoints: list[dict[str, object]] = []
    checkpoint_lookup: dict[str, dict[str, object]] = {}
    for path in sorted((run / "checkpoints").glob("step_*_gen_*")):
        report = manager.verify(path, manifest_digest, require_training_state=False)
        item: dict[str, object] = {
            "path": path.name,
            "verified": report.valid,
            "errors": list(report.errors),
            "verification_scope": report.resume_level,
        }
        if report.valid:
            raw = json.loads((path / "manifest.json").read_text())
            item.update(
                {
                    "step": raw["step"],
                    "tokens_seen": raw["tokens_seen"],
                    "validation_loss": raw["validation_loss"],
                    "digest": raw["sha256"],
                }
            )
            checkpoint_lookup[path.name] = item
        checkpoints.append(item)
    observations: list[dict[str, object]] = []
    rejected_reports: list[dict[str, str]] = []
    for path in sorted((run / "evaluations").glob("validation_step_*_gen_*.json")):
        result, reason = _report_observation(
            path, checkpoint_lookup, artifacts, manifest, validation_tokens
        )
        if result is None:
            rejected_reports.append(
                {"path": path.name, "reason": reason or "invalid report"}
            )
        else:
            observations.append(result)
    verified = bool(checkpoints) and all(item["verified"] for item in checkpoints)
    observed_generations = {item["checkpoint"] for item in observations}
    missing_reports = sorted(
        name
        for name, item in checkpoint_lookup.items()
        if item["validation_loss"] is not None and name not in observed_generations
    )
    level = "artifact_only"
    if verified and observations:
        level = (
            "partial_held_out"
            if rejected_reports or missing_reports
            else "checkpointed_held_out"
        )
    return {
        "format": "experiment_evidence_v2",
        "run_id": manifest["run_id"],
        "source_identity_sha256": manifest["source_identity"]["sha256"],
        "verified_checkpoints": verified,
        "checkpoint_count": len(checkpoints),
        "quality_observations": observations,
        "rejected_reports": rejected_reports,
        "missing_reports": missing_reports,
        "evidence_level": level,
        "interpretation": "Verified checkpoints are paired with integrity-bound validation observations. Hashes bind artifacts but do not by themselves prove split separation or general model quality."
        if level == "checkpointed_held_out"
        else "Some observations verify, but reports are missing or rejected; inspect details before claiming complete evidence."
        if level == "partial_held_out"
        else "No complete integrity-bound checkpoint and validation evidence is available.",
        "checkpoints": checkpoints,
    }
