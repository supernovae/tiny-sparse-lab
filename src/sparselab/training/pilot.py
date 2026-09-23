"""Fixed subprocess endpoint for staging pilots; never execute bundle-supplied code."""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

import torch

from sparselab.config.models import RunConfig
from sparselab.data.packing import TokenBlockDataset, load_prepared_data
from sparselab.model.transformer import DenseLM
from sparselab.staging import _read_sealed, _seal, pilot_config
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import canonical_json, read_manifest, source_identity
from sparselab.training.trainer import _train_impl


def run_pilot(root: Path, purpose: str, *, cancel_path: Path | None = None) -> Path:
    inputs = _read_sealed(root / "inputs.json")
    config = pilot_config(
        RunConfig.model_validate(inputs["requested_config"]), purpose, root
    )
    run_id = f"{purpose}-{uuid.uuid4().hex}"
    _train_impl(
        config,
        stage_bundle=root,
        run_id=run_id,
        purpose=purpose,
        cancel_path=cancel_path,
    )
    if cancel_path is not None and cancel_path.exists():
        raise InterruptedError("pilot cancelled at committed update boundary")
    gc.collect()
    run = config.logging.root_dir / run_id
    manifest = read_manifest(run / "manifest.json")
    manifest_digest = hashlib.sha256(canonical_json(manifest)).hexdigest()
    manager = CheckpointManager(run, manifest_sha256=manifest_digest)
    checkpoint = manager.root / "latest.json"
    verification = manager.verify(checkpoint)
    if not verification.valid or verification.resume_level != "full":
        raise ValueError(f"pilot checkpoint failed verification: {verification.errors}")
    snapshot = manager.load(checkpoint)
    if (
        snapshot.step != config.training.max_steps
        or not 0 < snapshot.tokens_seen <= config.training.max_tokens
    ):
        raise ValueError("pilot did not complete its declared horizon")
    model_config = config.model
    if model_config.memory_package_path is not None:
        model_config = model_config.model_copy(
            update={"memory_package_path": run / "portable_package"}
        )
    reloaded = DenseLM(model_config, config.attention).eval()
    reloaded.load_state_dict(snapshot.model, strict=True)
    data = load_prepared_data(
        run / "data", byte_enabled=config.model.memory in {"byte", "portable"}
    )
    example = TokenBlockDataset(
        data.validation, config.training.seq_len, data.validation_byte_addresses
    )[0]
    with torch.inference_mode():
        logits = reloaded(
            example[0].unsqueeze(0),
            byte_addresses=example[2].unsqueeze(0) if len(example) == 3 else None,
        )
        if not torch.isfinite(logits).all():
            raise FloatingPointError("reloaded pilot produced nonfinite logits")
    store_path = config.logging.root_dir / "experiments.sqlite3"
    with closing(sqlite3.connect(store_path.as_uri() + "?mode=ro", uri=True)) as con:
        timings = con.execute(
            "SELECT step,tokens_seen,value FROM metrics WHERE run_id=? AND name='performance/step_seconds' ORDER BY step",
            (run_id,),
        ).fetchall()
        gradients = con.execute(
            "SELECT value FROM metrics WHERE run_id=? AND name='optimizer/grad_norm' ORDER BY step",
            (run_id,),
        ).fetchall()
        peaks = dict(
            con.execute(
                "SELECT name,MAX(value) FROM metrics WHERE run_id=? AND name LIKE 'memory/%' GROUP BY name",
                (run_id,),
            ).fetchall()
        )
    if len(timings) != snapshot.step or len(gradients) != snapshot.step:
        raise ValueError("pilot lacks committed update timing or gradient evidence")
    if any(not math.isfinite(row[0]) for row in gradients) or any(
        row[2] <= 0 for row in timings
    ):
        raise ValueError("pilot has invalid gradient or timing evidence")
    native = max(
        (
            peaks[name]
            for name in (
                "memory/device_peak_allocated_bytes",
                "memory/device_peak_reserved_bytes",
            )
            if name in peaks
        ),
        default=None,
    )
    device_peak = (
        native
        if native is not None
        else max(
            (
                peaks[name]
                for name in (
                    "memory/device_sampled_peak_bytes",
                    "memory/driver_allocated_bytes",
                )
                if name in peaks
            ),
            default=None,
        )
    )
    host_peak = peaks.get("memory/process_peak_rss_bytes")
    backend = manifest["runtime"]["backend"]
    scope = (
        "host"
        if backend == "cpu"
        else "unified"
        if backend in {"mps", "metal"}
        else "device"
    )
    observed = (
        max(
            (value for value in (host_peak, device_peak) if value is not None),
            default=None,
        )
        if scope == "unified"
        else host_peak
        if scope == "host"
        else device_peak
    )
    elapsed = sum(row[2] for row in timings)
    report = {
        "format_version": 1,
        "status": "complete",
        "purpose": purpose,
        "run_id": run_id,
        "derived_config": config.model_dump(mode="json"),
        "step": snapshot.step,
        "tokens_seen": snapshot.tokens_seen,
        "timed_updates": len(timings),
        "update_seconds": elapsed,
        "tokens_per_second": snapshot.tokens_seen / elapsed,
        "runtime": manifest["runtime"],
        "source_identity_sha256": source_identity()["sha256"],
        "manifest_sha256": manifest_digest,
        "checkpoint_sha256": snapshot.checkpoint_sha256,
        "checkpoint_reload": "verified_full_state_and_finite_forward",
        "observed_peak_bytes": observed,
        "observed_host_peak_bytes": host_peak,
        "observed_device_peak_bytes": device_peak,
        "peak_scope": scope,
        "peak_method": "native" if native is not None else "sampled_lower_bound",
        "certifies_fit": False,
        "memory_metrics": peaks,
        "limitation": "pilot measurements cover these updates only; sampled peaks cannot certify an allocator high-water mark",
    }
    # A portable immutable pilot database must not depend on mutable WAL/SHM files.
    with closing(sqlite3.connect(store_path)) as con:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if con.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
            raise RuntimeError("pilot database did not finalize its WAL")
    path = config.logging.root_dir / "report.json"
    _seal(path, report)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("purpose", choices=("smoke", "warmup"))
    parser.add_argument("--cancel-path", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    try:
        print(run_pilot(root, args.purpose, cancel_path=args.cancel_path))
    except Exception as error:
        out_of_memory = isinstance(error, (MemoryError, torch.OutOfMemoryError)) or (
            isinstance(error, RuntimeError)
            and any(
                message in str(error)
                for message in (
                    "MPS backend out of memory",
                    "DefaultCPUAllocator: can't allocate memory",
                )
            )
        )
        _seal(
            root / "pilots" / args.purpose / "failure.json",
            {
                "format_version": 1,
                "kind": "out_of_memory" if out_of_memory else "pilot_failed",
                "exception_type": type(error).__name__,
                "message": str(error)[:16384],
            },
        )
        raise


if __name__ == "__main__":
    main()
