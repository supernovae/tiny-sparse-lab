"""Select and bind one verified checkpoint generation before child dispatch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import canonical_json


def select_generation(
    run: Path, *, selector: str, at_step: int | None, full_state: bool
) -> dict[str, Any]:
    """No mutable latest/best alias participates in a child identity."""
    if selector not in {"terminal", "best_validation"}:
        raise ValueError(f"unsupported parent selector: {selector}")
    manager = CheckpointManager(run)
    records, rejected = manager._verified_records(require_training_state=full_state)
    if rejected:
        raise ValueError(f"parent run contains invalid generations: {rejected}")
    if selector == "terminal":
        if at_step is None:
            raise ValueError("terminal parent needs a prespecified update")
        candidates = [item for item in records if item.step == at_step]
        selected = max(candidates, key=lambda item: item.generation_id, default=None)
    else:
        candidates = [item for item in records if item.validation_loss is not None]
        selected = min(
            candidates,
            key=lambda item: (float(item.validation_loss), item.generation_id),
            default=None,
        )
    if selected is None:
        raise ValueError(
            f"no verified {selector} checkpoint generation at step {at_step}"
        )
    path = run / "checkpoints" / selected.relative_path
    report = manager.verify(path, require_training_state=full_state)
    if not report.valid or path.is_symlink():
        raise ValueError(f"selected checkpoint generation is unverified: {path}")
    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": selected.manifest_sha256,
        "step": selected.step,
        "validation_loss": selected.validation_loss,
        "resume_level": report.resume_level,
        "run_id": run.name,
    }


def bind_generation(
    workspace: Path,
    *,
    plan_sha256: str,
    cell_id: str,
    parent_cell_id: str,
    parent_run: Path,
    selector: str,
    at_step: int | None,
    full_state: bool,
) -> tuple[Path, dict[str, Any]]:
    """Exclusive immutable execution binding; replay verifies the original bytes."""
    selected = select_generation(
        parent_run, selector=selector, at_step=at_step, full_state=full_state
    )
    payload = {
        "format": "sparselab-plan-execution-binding-v1",
        "plan_sha256": plan_sha256,
        "cell_id": cell_id,
        "parent_cell_id": parent_cell_id,
        "selector": selector,
        "at_step": at_step,
        **selected,
    }
    encoded = canonical_json(payload) + b"\n"
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    binding = workspace / "bindings" / f"{digest}.json"
    binding.parent.mkdir(parents=True, exist_ok=True)
    if binding.exists():
        if binding.read_bytes() != encoded:
            raise ValueError(f"published execution binding changed: {binding}")
    else:
        with binding.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
    return binding, payload


def read_binding(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "sparselab-plan-execution-binding-v1":
        raise ValueError(f"invalid phase binding: {path}")
    path = Path(payload["checkpoint_path"])
    if path.name in {"latest.json", "best.json"} or path.is_symlink():
        raise ValueError("execution binding must pin an immutable generation directory")
    report = CheckpointManager(path.parent.parent).verify(
        path, require_training_state=payload["resume_level"] == "full"
    )
    if not report.valid:
        raise ValueError("bound checkpoint generation no longer verifies")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("sha256") != payload["checkpoint_sha256"]
        or manifest.get("step") != payload["step"]
    ):
        raise ValueError("checkpoint digest or update differs from execution binding")
    return payload
