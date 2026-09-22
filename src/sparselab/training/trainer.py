"""Single-process PyTorch training with explicit update boundaries."""

from __future__ import annotations

import json
import shutil
import signal
import socket
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.nn import functional

from sparselab.config.models import RunConfig
from sparselab.data.packing import (
    BatchCursor,
    PreparedData,
    TokenBlockDataset,
    epoch_order,
    prepare_data,
)
from sparselab.data.tokenizer import load_tokenizer
from sparselab.evaluation.language_model import evaluate
from sparselab.model.inspection import architecture_metrics, inspect_model
from sparselab.model.transformer import DenseLM
from sparselab.runtime import (
    seed_everything,
    synchronize,
    torch_device_for,
    validate_runtime,
)
from sparselab.training.checkpoints import (
    CheckpointManager,
    CheckpointRecord,
    TrainingSnapshot,
)
from sparselab.training.manifest import (
    ArtifactIdentity,
    RunManifest,
    canonical_json,
    read_manifest,
    sha256_file,
    source_identity,
    write_manifest,
)
from sparselab.training.metrics import ExperimentStore
from sparselab.training.offload import ActivationOffload
from sparselab.training.optimizer import (
    learning_rate_for_step,
    make_adafactor,
    make_optimizer,
)


def _safe_rng_state() -> dict[str, object]:
    python_state = __import__("random").getstate()
    numpy_state = np.random.get_state()
    return {
        "python": python_state,
        "numpy_kind": numpy_state[0],
        "numpy_keys": torch.from_numpy(numpy_state[1].copy()),
        "numpy_pos": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gaussian": float(numpy_state[4]),
        "torch": torch.get_rng_state(),
    }


def _restore_safe_rng(state: dict[str, object]) -> None:
    __import__("random").setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(
        (
            str(state["numpy_kind"]),
            state["numpy_keys"].numpy(),  # type: ignore[union-attr]
            int(state["numpy_pos"]),
            int(state["numpy_has_gauss"]),
            float(state["numpy_cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]


def _load_run_data(run: Path, config: RunConfig) -> PreparedData:
    root = run / "data"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"resume run lacks immutable prepared data: {root}")
    byte_enabled = config.model.memory in {"byte", "portable"}
    paths = (root / "train.npy", root / "validation.npy")
    if not all(path.is_file() for path in paths):
        raise ValueError("resume run lacks prepared arrays")
    if byte_enabled and not all(
        (root / name).is_file()
        for name in ("train_byte_addresses.npy", "validation_byte_addresses.npy")
    ):
        raise ValueError("resume run lacks byte-address artifacts")
    manifest = json.loads(manifest_path.read_text())
    expected = (
        (
            "train.npy",
            manifest.get("train", {}).get("sha256")
            if isinstance(manifest.get("train"), dict)
            else None,
        ),
        (
            "validation.npy",
            manifest.get("validation", {}).get("sha256")
            if isinstance(manifest.get("validation"), dict)
            else None,
        ),
    )
    for name, digest in expected:
        if not isinstance(digest, str) or sha256_file(root / name) != digest:
            raise ValueError(f"resume data artifact hash mismatch: {name}")
    if byte_enabled:
        byte = manifest.get("byte_addressing")
        if not isinstance(byte, dict):
            raise ValueError("resume data manifest lacks byte-address identity")
        for name, key in (
            ("train_byte_addresses.npy", "train_sha256"),
            ("validation_byte_addresses.npy", "validation_sha256"),
        ):
            if (
                not isinstance(byte.get(key), str)
                or sha256_file(root / name) != byte[key]
            ):
                raise ValueError(f"resume data artifact hash mismatch: {name}")
    return PreparedData(
        root,
        np.load(paths[0], mmap_mode="r"),
        np.load(paths[1], mmap_mode="r"),
        np.load(root / "train_byte_addresses.npy", mmap_mode="r")
        if byte_enabled
        else None,
        np.load(root / "validation_byte_addresses.npy", mmap_mode="r")
        if byte_enabled
        else None,
        manifest,
    )


def _copy_artifacts(
    run: Path, config: RunConfig, data: PreparedData, source_run: Path | None = None
) -> tuple[ArtifactIdentity, ...]:
    artifacts: list[ArtifactIdentity] = []
    tokenizer_source = (
        source_run / "tokenizer.json"
        if source_run is not None
        else config.tokenizer.path
    )
    tokenizer = run / "tokenizer.json"
    shutil.copy2(tokenizer_source, tokenizer)
    if source_run is None:
        for source in (config.tokenizer.path.with_name("tokenizer_manifest.json"),):
            if source.is_file():
                shutil.copy2(source, run / source.name)
    elif (source_run / "tokenizer_manifest.json").is_file():
        shutil.copy2(
            source_run / "tokenizer_manifest.json", run / "tokenizer_manifest.json"
        )
    shutil.copytree(data.root, run / "data")
    package_source = (
        source_run / "portable_package"
        if source_run is not None
        else config.model.memory_package_path
    )
    if package_source is not None and package_source.exists():
        destination = run / "portable_package"
        if package_source.is_dir():
            shutil.copytree(package_source, destination)
        else:
            shutil.copy2(package_source, destination)
    for path in sorted(
        item
        for item in run.rglob("*")
        if item.is_file() and item.name != "manifest.json"
    ):
        artifacts.append(
            ArtifactIdentity(
                str(path.relative_to(run)), sha256_file(path), path.stat().st_size
            )
        )
    return tuple(artifacts)


def _checkpoint_due(
    config: RunConfig,
    step: int,
    tokens: int,
    elapsed: float,
    watermarks: dict[str, float],
) -> bool:
    checks = (
        (config.checkpoint.every_steps, step - watermarks.get("step", 0)),
        (config.checkpoint.every_tokens, tokens - watermarks.get("tokens", 0)),
        (
            config.checkpoint.every_minutes,
            (elapsed - watermarks.get("minutes", 0)) / 60,
        ),
    )
    return any(interval is not None and value >= interval for interval, value in checks)


def _save(
    manager: CheckpointManager,
    model: DenseLM,
    optimizer: torch.optim.Optimizer,
    config: RunConfig,
    run_id: str,
    cursor: BatchCursor,
    step: int,
    tokens: int,
    watermarks: dict[str, float],
    validation_loss: float | None = None,
) -> CheckpointRecord:
    snapshot = TrainingSnapshot(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        optimizer.state_dict(),
        {
            "kind": "warmup_cosine_v1",
            "completed_updates": step,
            "max_steps": config.training.max_steps,
            "warmup_steps": config.optimizer.warmup_steps,
            "peak": config.optimizer.peak,
            "floor": config.optimizer.floor,
        },
        step,
        tokens,
        (cursor.epoch, cursor.next_block),
        config.model_dump(mode="json"),
        run_id,
        _safe_rng_state(),
        None,
        validation_loss,
        watermarks.copy(),
        "pytorch",
        config.runtime.backend,
    )
    return manager.save(snapshot, validation_loss)


def _write_validation_report(
    run: Path, record: CheckpointRecord, result: dict[str, object], config: RunConfig
) -> Path:
    evaluations = run / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    report = (
        evaluations
        / f"validation_step_{record.step:08d}_gen_{record.generation_id:06d}.json"
    )
    payload: dict[str, object] = {
        "kind": "held_out_validation_v2",
        "checkpoint": record.relative_path,
        "checkpoint_sha256": record.manifest_sha256,
        "step": record.step,
        "tokens_seen": record.tokens_seen,
        "identities": {
            "data/validation.npy": sha256_file(run / "data" / "validation.npy"),
            "tokenizer.json": sha256_file(run / "tokenizer.json"),
            "source_identity_sha256": source_identity()["sha256"],
            "protocol": "next-token-cross-entropy-v1",
        },
        "batch_size": config.training.micro_batch_size,
        "max_batches": config.evaluation.max_batches,
        "seq_len": config.training.seq_len,
        **result,
    }
    payload["sha256"] = (
        __import__("hashlib").sha256(canonical_json(payload)).hexdigest()
    )
    content = canonical_json(payload) + b"\n"
    if report.exists():
        if report.read_bytes() != content:
            raise FileExistsError(
                f"immutable validation report already exists: {report}"
            )
        return report
    temporary = report.with_name(report.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        __import__("os").fsync(handle.fileno())
    temporary.replace(report)
    return report


def _restore_model_state(
    model: DenseLM, optimizer: torch.optim.Optimizer, snapshot: TrainingSnapshot
) -> None:
    model.load_state_dict(snapshot.model)
    optimizer.load_state_dict(snapshot.optimizer)
    if snapshot.rng is not None:
        _restore_safe_rng(snapshot.rng)


def train(
    config: RunConfig,
    *,
    resume: Path | None = None,
    promote: Path | None = None,
    recover: Path | None = None,
    run_id: str | None = None,
    stop_after_step: int | None = None,
    worker_id: str | None = None,
    allow_runtime_drift: bool = False,
    stage_bundle: Path | None = None,
    cancel_path: Path | None = None,
) -> str:
    choices = [path for path in (resume, promote, recover) if path is not None]
    if len(choices) > 1:
        raise ValueError("resume, promote, and recover are mutually exclusive")
    if config.runtime.engine == "mlx":
        from sparselab.training.mlx_trainer import train_mlx

        if promote is not None or recover is not None:
            raise ValueError("MLX promotion/recovery is not yet supported")
        return train_mlx(
            config,
            resume=resume,
            run_id=run_id,
            stop_after_step=stop_after_step,
        )
    runtime = validate_runtime(config)
    device = torch_device_for(runtime.backend, config.runtime.device_index)
    source_run: Path | None = None
    snapshot: TrainingSnapshot | None = None
    continuation: Literal["FRESH", "RESUMED", "PROMOTED"] = "FRESH"
    if resume is not None or promote is not None:
        selected = resume or promote
        assert selected is not None
        source_run = selected.parent.parent
        source_manager = CheckpointManager(source_run)
        snapshot = source_manager.load(selected, "resume" if resume else "promote")
        continuation = "RESUMED" if resume else "PROMOTED"
        if promote:
            source_manifest = read_manifest(source_run / "manifest.json")
            requested = source_manifest.get("effective_config")
            if not isinstance(requested, dict):
                raise ValueError("promotion source manifest lacks effective config")
            if requested.get("model") != config.model.model_dump(
                mode="json"
            ) or requested.get("attention") != config.attention.model_dump(mode="json"):
                raise ValueError(
                    "promotion requires matching source architecture semantics"
                )
            if sha256_file(source_run / "tokenizer.json") != sha256_file(
                config.tokenizer.path
            ):
                raise ValueError("promotion requires matching tokenizer identity")
        if resume:
            expected = RunConfig.model_validate(snapshot.config)
            current = config.model_dump(mode="json")
            saved = expected.model_dump(mode="json")
            for key in ("logging", "checkpoint"):
                current.pop(key, None)
                saved.pop(key, None)
            if current != saved:
                raise ValueError(
                    "resume configuration differs from checkpoint; use promotion for changed scientific settings"
                )
            if (
                config.runtime.engine != snapshot.engine
                or runtime.backend != snapshot.backend
            ):
                raise ValueError("full resume requires the same engine and backend")
    if recover is not None:
        source_run = recover
        recovery = CheckpointManager(source_run).latest_valid()
        if recovery.record is None:
            raise ValueError("no valid checkpoint generation for recovery")
        snapshot = CheckpointManager(source_run).load(
            source_run / "checkpoints" / recovery.record.relative_path
        )
        continuation = "RESUMED"
        expected = RunConfig.model_validate(snapshot.config)
        current = config.model_dump(mode="json")
        saved = expected.model_dump(mode="json")
        for key in ("logging", "checkpoint"):
            current.pop(key, None)
            saved.pop(key, None)
        if (
            current != saved
            or config.runtime.engine != snapshot.engine
            or runtime.backend != snapshot.backend
        ):
            raise ValueError(
                "recovery requires matching resume configuration and backend"
            )
    if continuation == "RESUMED":
        assert source_run is not None
        source_manifest = read_manifest(source_run / "manifest.json")
        inventory = {
            item["relative_path"]: item["sha256"]
            for item in source_manifest.get("artifacts", [])
            if isinstance(item, dict)
            and isinstance(item.get("relative_path"), str)
            and isinstance(item.get("sha256"), str)
        }
        for relative in ("tokenizer.json", "data/train.npy", "data/validation.npy"):
            path = source_run / relative
            if inventory.get(relative) != sha256_file(path):
                raise ValueError(f"resume artifact hash mismatch: {relative}")
        if config.model.memory_package_path is not None:
            for path in (source_run / "portable_package").rglob("*"):
                if path.is_file():
                    relative = str(path.relative_to(source_run))
                    if inventory.get(relative) != sha256_file(path):
                        raise ValueError(f"resume artifact hash mismatch: {relative}")
    seed_everything(config.seed, deterministic_cpu=config.training.deterministic)
    if continuation == "RESUMED":
        assert source_run is not None
        data = _load_run_data(source_run, config)
    else:
        tokenizer = load_tokenizer(config.tokenizer.path)
        data = prepare_data(config, tokenizer)
    dataset = TokenBlockDataset(
        data.train, config.training.seq_len, data.train_byte_addresses
    )
    validation_dataset = TokenBlockDataset(
        data.validation, config.training.seq_len, data.validation_byte_addresses
    )
    model_config = config.model
    if continuation == "RESUMED" and config.model.memory_package_path is not None:
        assert source_run is not None
        owned_package = source_run / "portable_package"
        if not owned_package.exists():
            raise ValueError("resume run lacks immutable portable package")
        model_config = config.model.model_copy(
            update={"memory_package_path": owned_package}
        )
    model = DenseLM(model_config, config.attention).to(device)
    offload = (
        ActivationOffload(device)
        if config.runtime.memory.activation_offload.enabled
        else None
    )
    optimizer = (
        make_optimizer(
            model,
            config.optimizer.peak,
            config.optimizer.weight_decay,
            config.optimizer.betas,
            config.optimizer.eps,
        )
        if config.optimizer.name == "adamw"
        else make_adafactor(
            model,
            config.optimizer.peak,
            config.optimizer.weight_decay,
            config.optimizer.beta2_decay,
            config.optimizer.eps,
            config.optimizer.d,
        )
    )
    step = tokens = 0
    cursor = BatchCursor()
    parent_run_id = None
    watermarks: dict[str, float] = {}
    if snapshot is not None and continuation == "RESUMED":
        _restore_model_state(model, optimizer, snapshot)
        step, tokens = snapshot.step, snapshot.tokens_seen
        cursor = BatchCursor(*snapshot.cursor)
        watermarks = snapshot.cadence or {}
        parent_run_id = snapshot.run_id
    elif snapshot is not None:
        model.load_state_dict(snapshot.model)
    if continuation == "RESUMED" and (
        step >= config.training.max_steps or tokens >= config.training.max_tokens
    ):
        raise ValueError("cannot resume a completed training budget")
    if stop_after_step is not None and stop_after_step <= step:
        raise ValueError("stop-after-step must exceed current step")
    run_id = run_id or uuid.uuid4().hex
    run = config.logging.root_dir / run_id
    if run.exists():
        raise FileExistsError(f"run exists: {run_id}")
    run.mkdir(parents=True)
    artifacts = _copy_artifacts(
        run, config, data, source_run if continuation == "RESUMED" else None
    )
    (run / "resolved_config.yaml").write_text(
        json.dumps(config.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
    )
    architecture = {
        "model": config.model.model_dump(mode="json"),
        "attention": config.attention.model_dump(mode="json"),
    }
    manifest = RunManifest(
        run_id,
        config.name,
        runtime,
        config.model_dump(mode="json"),
        config.model_dump(mode="json"),
        sha256_file(run / "resolved_config.yaml") if architecture else "",
        source_identity(),
        worker_id or socket.gethostname(),
        continuation,
        parent_run_id=parent_run_id,
        checkpoint_sha256=None,
        artifacts=artifacts,
    )
    manifest_digest = write_manifest(run / "manifest.json", manifest)
    manager = CheckpointManager(run, manifest_sha256=manifest_digest)
    store = ExperimentStore(config.logging.root_dir)
    store.create_run(
        run_id,
        config.model_dump(mode="json"),
        {
            "inspection": inspect_model(model),
            "runtime": runtime.as_dict(),
            "manifest_sha256": manifest_digest,
        },
        parent_run_id,
    )
    started = time.perf_counter()
    interrupted = False

    def evaluate_and_record(elapsed: float) -> dict[str, object]:
        result = evaluate(
            model,
            validation_dataset,
            batch_size=config.training.micro_batch_size,
            max_batches=config.evaluation.max_batches,
            device=device,
        )
        metrics: dict[str, float] = {"validation/loss": float(result["loss"])}
        if isinstance(result.get("perplexity"), (int, float)):
            metrics["validation/perplexity"] = float(result["perplexity"])
        store.log_metrics(run_id, step, tokens, elapsed, metrics)
        store.log_event(run_id, step, tokens, elapsed, "validation_completed", result)
        return result

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    old_int, old_term = (
        signal.signal(signal.SIGINT, request_stop),
        signal.signal(signal.SIGTERM, request_stop),
    )
    try:
        initial_validation = evaluate_and_record(0.0)
        record = _save(
            manager,
            model,
            optimizer,
            config,
            run_id,
            cursor,
            step,
            tokens,
            watermarks,
            initial_validation["loss"],
        )
        _write_validation_report(run, record, initial_validation, config)
        store.log_event(
            run_id,
            step,
            tokens,
            0.0,
            "checkpoint_saved",
            {"validation_loss": initial_validation["loss"]},
        )
        while step < config.training.max_steps and tokens < config.training.max_tokens:
            if cancel_path is not None and cancel_path.exists():
                interrupted = True
            order = epoch_order(len(dataset), config.seed, cursor.epoch)
            count = (
                config.training.micro_batch_size * config.training.gradient_accumulation
            )
            indices = order[cursor.next_block : cursor.next_block + count].tolist()
            if not indices:
                cursor = BatchCursor(cursor.epoch + 1, 0)
                continue
            records = [dataset[index] for index in indices]
            remaining = config.training.max_tokens - tokens
            labels: list[torch.Tensor] = []
            for record in records:
                label = record[1].clone()
                allowed = max(0, min(remaining, label.numel()))
                if allowed < label.numel():
                    label.flatten()[allowed:] = -100
                remaining -= allowed
                labels.append(label)
            valid_targets = sum(int((label != -100).sum()) for label in labels)
            if valid_targets == 0:
                break
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            update_started = time.perf_counter()
            language_sum = 0.0
            aux_sum = 0.0
            for offset in range(0, len(records), config.training.micro_batch_size):
                chunk = records[offset : offset + config.training.micro_batch_size]
                chunk_labels = labels[
                    offset : offset + config.training.micro_batch_size
                ]
                chunk_valid = sum(int((label != -100).sum()) for label in chunk_labels)
                if chunk_valid == 0:
                    continue
                x = torch.stack([record[0] for record in chunk]).to(device)
                y = torch.stack(chunk_labels).to(device)
                addresses = (
                    torch.stack([record[2] for record in chunk]).to(device)
                    if len(chunk[0]) == 3
                    else None
                )
                context = offload.hooks() if offload is not None else nullcontext()
                with context:
                    logits, auxiliary = model.forward_with_aux(
                        x,
                        byte_addresses=addresses,
                        valid_target_mask=y != -100,
                        activation_checkpointing=config.runtime.memory.activation_checkpointing.enabled,
                    )
                ce_sum = functional.cross_entropy(
                    logits.flatten(0, 1),
                    y.flatten(),
                    ignore_index=-100,
                    reduction="sum",
                )
                if not torch.isfinite(ce_sum) or not torch.isfinite(auxiliary):
                    raise FloatingPointError("nonfinite training loss")
                ((ce_sum + auxiliary * chunk_valid) / valid_targets).backward()
                language_sum += float(ce_sum.detach())
                aux_sum += float(auxiliary.detach()) * chunk_valid
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.grad_clip_norm,
                error_if_nonfinite=True,
                foreach=False,
            )
            next_step = step + 1
            lr = learning_rate_for_step(
                next_step,
                config.training.max_steps,
                config.optimizer.warmup_steps,
                config.optimizer.peak,
                config.optimizer.floor,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            synchronize(device)
            update_seconds = time.perf_counter() - update_started
            step, tokens = next_step, tokens + valid_targets
            cursor = BatchCursor(cursor.epoch, cursor.next_block + len(indices))
            elapsed = time.perf_counter() - started
            metric_values = {
                "train/loss": language_sum / valid_targets,
                "moe/router_auxiliary_loss": aux_sum / valid_targets,
                "optimizer/learning_rate": lr,
                "optimizer/grad_norm": float(norm),
                "performance/step_seconds": update_seconds,
                "performance/tokens_per_second": valid_targets
                / max(update_seconds, 1e-9),
                "batch/micro_batch_size": float(config.training.micro_batch_size),
                "batch/accumulation_steps": float(
                    config.training.gradient_accumulation
                ),
                "batch/effective_batch_size": float(len(indices)),
                "batch/effective_tokens_per_update": float(valid_targets),
            }
            metric_values.update(architecture_metrics(model))
            if offload is not None:
                metrics = offload.metrics
                metric_values.update(
                    {
                        "offload/bytes_to_cpu": float(metrics.bytes_to_cpu),
                        "offload/bytes_to_device": float(metrics.bytes_to_device),
                        "offload/peak_host_bytes": float(metrics.peak_host_bytes),
                    }
                )
            store.log_metrics(run_id, step, tokens, elapsed, metric_values)
            terminal = (
                step >= config.training.max_steps
                or tokens >= config.training.max_tokens
                or step == stop_after_step
                or interrupted
            )
            evaluation_due = terminal or step % config.evaluation.every_steps == 0
            validation = evaluate_and_record(elapsed) if evaluation_due else None
            if (
                _checkpoint_due(config, step, tokens, elapsed, watermarks)
                or terminal
                or evaluation_due
            ):
                watermarks = {
                    "step": float(step),
                    "tokens": float(tokens),
                    "minutes": elapsed,
                }
                record = _save(
                    manager,
                    model,
                    optimizer,
                    config,
                    run_id,
                    cursor,
                    step,
                    tokens,
                    watermarks,
                    validation["loss"] if validation is not None else None,
                )
                if validation is not None:
                    _write_validation_report(run, record, validation, config)
                store.log_event(
                    run_id,
                    step,
                    tokens,
                    elapsed,
                    "checkpoint_saved",
                    {"validation_loss": validation["loss"] if validation else None},
                )
            if terminal:
                status = (
                    "completed"
                    if step >= config.training.max_steps
                    or tokens >= config.training.max_tokens
                    else "interrupted"
                )
                store.finish_run(run_id, status, str(manager.root / "latest.json"))
                store.log_event(run_id, step, tokens, elapsed, f"run_{status}", {})
                return run_id
        return run_id
    except BaseException as error:
        elapsed = time.perf_counter() - started
        store.finish_run(run_id, "failed", str(manager.root / "latest.json"))
        store.log_event(
            run_id, step, tokens, elapsed, "run_failed", {"reason": str(error)}
        )
        raise
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
