"""One immutable development panel per invocation, using native loss evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import torch
from pydantic import Field, model_validator

from sparselab.config.models import StrictModel
from sparselab.data.packing import (
    TokenBlockDataset,
    _tokenizer_sha256,
    load_prepared_data,
)
from sparselab.evaluation.inference import load_run
from sparselab.evaluation.language_model import evaluate
from sparselab.training.attempt_budget import AttemptBudget
from sparselab.training.manifest import canonical_json

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Positive = Annotated[int, Field(strict=True, gt=0)]


class HeldoutPanel(StrictModel):
    schema_version: Literal[1]
    name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
    purpose: Literal["development"]
    prepared_path: Path
    prepared_manifest_sha256: Digest
    tokenizer_file_sha256: Digest
    tokenizer_canonical_sha256: Digest
    checkpoint_sha256: Digest
    split: Literal["validation"]
    sampling: Literal["first_eligible_blocks"]
    sequence_length: Positive
    blocks: Positive
    supervised_targets: Positive
    batch_size: Positive
    max_forward_positions: Positive

    @model_validator(mode="after")
    def budget(self) -> HeldoutPanel:
        positions = self.blocks * self.sequence_length
        if (
            positions > self.max_forward_positions
            or self.supervised_targets > positions
        ):
            raise ValueError("panel selection exceeds its forward or target budget")
        return self


def _dataset(panel: HeldoutPanel, root: Path) -> tuple[TokenBlockDataset, dict]:
    prepared = load_prepared_data(root, byte_enabled=False)
    manifest = prepared.manifest
    if manifest["manifest_sha256"] != panel.prepared_manifest_sha256:
        raise ValueError("panel prepared manifest identity mismatch")
    if manifest["tokenizer_sha256"] != panel.tokenizer_canonical_sha256:
        raise ValueError("panel prepared tokenizer identity mismatch")
    if manifest.get("allocation") is not None:
        raise ValueError("named panels do not support allocation sidecars")
    dataset = TokenBlockDataset(
        prepared.validation,
        panel.sequence_length,
        supervision=prepared.validation_supervision,
    )
    if len(dataset) < panel.blocks:
        raise ValueError("panel has insufficient eligible blocks (possibly empty)")
    dataset.block_indices = dataset.block_indices[: panel.blocks].copy()
    targets = sum(
        int(np.count_nonzero(dataset.numpy_block(i)[1] != -100))
        for i in range(len(dataset))
    )
    if targets != panel.supervised_targets:
        raise ValueError("panel supervised-target coverage differs from declaration")
    return dataset, {
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "validation_sha256": manifest["validation"]["sha256"],
        "supervision": manifest["supervision"],
        "packing_version": manifest["packing_version"],
        "source_identity_sha256": manifest.get("source_identity_sha256"),
    }


def run_panel(
    run_id: str,
    runs_dir: Path,
    checkpoint: str,
    declaration: Path,
    output: Path,
    *,
    backend: str | None = None,
    authorization: Any = None,
) -> dict[str, Any]:
    """Preserve incomplete evidence; output must be a new file outside input roots."""
    panel = HeldoutPanel.model_validate_json(declaration.read_text(encoding="utf-8"))
    prepared_root = panel.prepared_path
    if not prepared_root.is_absolute():
        prepared_root = declaration.resolve().parent / prepared_root
    for root in ((runs_dir / run_id).resolve(), prepared_root.resolve()):
        if output.resolve().is_relative_to(root):
            raise ValueError("panel output must be outside the immutable input roots")
    protocol = panel.model_dump(
        mode="json", exclude={"prepared_path", "checkpoint_sha256"}
    )
    protocol_sha = hashlib.sha256(canonical_json(protocol)).hexdigest()
    result: dict[str, Any] = {
        "source": "heldout_panel",
        "version": 1,
        "panel": panel.name,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "requested_checkpoint_sha256": panel.checkpoint_sha256,
        "identity": None,
        "data_identity": None,
        "status": "incomplete",
        "evidence_complete": False,
        "accounting_complete": False,
        "reason": "evaluation did not complete",
        "loss": None,
        "nll_sum": 0.0,
        "valid_targets": 0,
        "batches": 0,
        "attempted_forward_positions": 0,
        "evaluation_seconds": 0.0,
    }
    started = time.monotonic()
    evaluation_started = None
    with output.open("x", encoding="utf-8") as handle:

        def persist() -> None:
            handle.seek(0)
            handle.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())

        persist()
        try:
            dataset, result["data_identity"] = _dataset(panel, prepared_root)
            AttemptBudget.require_fixed_evaluation_allocation_from_environment(
                fixed_positions=panel.blocks * panel.sequence_length,
                total_positions=panel.blocks * panel.sequence_length,
                calls=0,
                tokens=0,
            )
            loaded = load_run(
                run_id,
                runs_dir,
                checkpoint,
                backend,
                authorization=authorization,
                expected_checkpoint_sha256=panel.checkpoint_sha256,
            )
            result["identity"] = loaded.identity
            result["comparison_group"] = hashlib.sha256(
                canonical_json(
                    {
                        "protocol_sha256": protocol_sha,
                        "runtime": loaded.identity["runtime"],
                    }
                )
            ).hexdigest()
            if (
                loaded.engine is not None
                or loaded.config.model.memory != "none"
                or loaded.config.model.ffn != "dense"
                or loaded.config.dataset.allocation_manifest_path is not None
                or loaded.config.model.semantic_memory_dim is not None
                or not isinstance(loaded.device, torch.device)
            ):
                raise ValueError(
                    "named panels require a dense PyTorch run without memory"
                )
            if (
                loaded.identity["tokenizer_sha256"] != panel.tokenizer_file_sha256
                or _tokenizer_sha256(loaded.tokenizer)
                != panel.tokenizer_canonical_sha256
            ):
                raise ValueError("panel checkpoint tokenizer identity mismatch")
            if panel.sequence_length > loaded.config.model.max_seq_len:
                raise ValueError("panel sequence length exceeds checkpoint context")

            def before_batch(positions: int) -> None:
                if (
                    result["attempted_forward_positions"] + positions
                    > panel.max_forward_positions
                ):
                    raise ValueError("panel forward budget exhausted")
                if AttemptBudget.forward_allocation_active_from_environment():
                    AttemptBudget.reserve_forward_from_environment(
                        f"panel:{panel.checkpoint_sha256}:{protocol_sha}:{result['batches']}",
                        kind="fixed_profile",
                        positions=positions,
                    )
                result["attempted_forward_positions"] += positions

            def progress(total: float, targets: int, batches: int) -> None:
                result.update(nll_sum=total, valid_targets=targets, batches=batches)

            evaluation_started = time.monotonic()
            metrics = evaluate(
                loaded.model,
                dataset,
                batch_size=panel.batch_size,
                max_batches=math.ceil(panel.blocks / panel.batch_size),
                device=loaded.device,
                before_batch=before_batch,
                progress=progress,
            )
            if metrics["valid_targets"] != panel.supervised_targets:
                raise ValueError("native evaluation did not complete panel coverage")
            result.update(metrics)
            result.update(status="complete", evidence_complete=True, reason=None)
        except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - retain failed panel evidence
            result.update(
                status="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                reason=f"{type(error).__name__}: {error}",
            )
        finally:
            if evaluation_started is not None:
                result["evaluation_seconds"] = time.monotonic() - evaluation_started
            result["accounting_complete"] = True
            result["elapsed_seconds"] = time.monotonic() - started
            persist()
    return result
