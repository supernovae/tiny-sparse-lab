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
from sparselab.model.inspection import inspect_model
from sparselab.model.transformer import DenseLM
from sparselab.runtime import seed_everything, torch_device_for, validate_runtime
from sparselab.training.checkpoints import CheckpointManager, TrainingSnapshot
from sparselab.training.manifest import (
    ArtifactIdentity,
    RunManifest,
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
    np.random.set_state((
        str(state["numpy_kind"]),
        state["numpy_keys"].numpy(),  # type: ignore[union-attr]
        int(state["numpy_pos"]),
        int(state["numpy_has_gauss"]),
        float(state["numpy_cached_gaussian"]),
    ))
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
    if byte_enabled and not all((root / name).is_file() for name in ("train_byte_addresses.npy", "validation_byte_addresses.npy")):
        raise ValueError("resume run lacks byte-address artifacts")
    return PreparedData(root, np.load(paths[0], mmap_mode="r"), np.load(paths[1], mmap_mode="r"), np.load(root / "train_byte_addresses.npy", mmap_mode="r") if byte_enabled else None, np.load(root / "validation_byte_addresses.npy", mmap_mode="r") if byte_enabled else None, json.loads(manifest_path.read_text()))


def _copy_artifacts(run: Path, config: RunConfig, data: PreparedData) -> tuple[ArtifactIdentity, ...]:
    artifacts: list[ArtifactIdentity] = []
    tokenizer = run / "tokenizer.json"
    shutil.copy2(config.tokenizer.path, tokenizer)
    for source in (config.tokenizer.path.with_name("tokenizer_manifest.json"),):
        if source.is_file():
            shutil.copy2(source, run / source.name)
    shutil.copytree(data.root, run / "data")
    if config.model.memory_package_path is not None:
        destination = run / "portable_package"
        if config.model.memory_package_path.is_dir():
            shutil.copytree(config.model.memory_package_path, destination)
        else:
            shutil.copy2(config.model.memory_package_path, destination)
    for path in sorted(item for item in run.rglob("*") if item.is_file() and item.name != "manifest.json"):
        artifacts.append(ArtifactIdentity(str(path.relative_to(run)), sha256_file(path), path.stat().st_size))
    return tuple(artifacts)


def _checkpoint_due(config: RunConfig, step: int, tokens: int, elapsed: float, watermarks: dict[str, float]) -> bool:
    checks = (
        (config.checkpoint.every_steps, step - watermarks.get("step", 0)),
        (config.checkpoint.every_tokens, tokens - watermarks.get("tokens", 0)),
        (config.checkpoint.every_minutes, (elapsed - watermarks.get("minutes", 0)) / 60),
    )
    return any(interval is not None and value >= interval for interval, value in checks)

def _save(manager: CheckpointManager, model: DenseLM, optimizer: torch.optim.Optimizer, config: RunConfig, run_id: str, cursor: BatchCursor, step: int, tokens: int, watermarks: dict[str, float], validation_loss: float | None = None) -> None:
    snapshot = TrainingSnapshot(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        optimizer.state_dict(),
        {"kind": "warmup_cosine_v1", "completed_updates": step, "max_steps": config.training.max_steps, "warmup_steps": config.optimizer.warmup_steps, "peak": config.optimizer.peak, "floor": config.optimizer.floor},
        step, tokens, (cursor.epoch, cursor.next_block), config.model_dump(mode="json"),
        run_id, _safe_rng_state(), None, validation_loss, watermarks.copy(), "pytorch",
        config.runtime.backend,
    )
    manager.save(snapshot, validation_loss)


def _restore_model_state(model: DenseLM, optimizer: torch.optim.Optimizer, snapshot: TrainingSnapshot) -> None:
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
        if resume:
            expected = RunConfig.model_validate(snapshot.config)
            current = config.model_dump(mode="json")
            saved = expected.model_dump(mode="json")
            for key in ("logging", "checkpoint"):
                current.pop(key, None)
                saved.pop(key, None)
            if current != saved and not allow_runtime_drift:
                raise ValueError("resume configuration differs from checkpoint; use promotion for changed scientific settings")
    if recover is not None:
        source_run = recover
        recovery = CheckpointManager(source_run).latest_valid()
        if recovery.record is None:
            raise ValueError("no valid checkpoint generation for recovery")
        snapshot = CheckpointManager(source_run).load(source_run / "checkpoints" / recovery.record.relative_path)
        continuation = "RESUMED"
    seed_everything(config.seed, deterministic_cpu=config.training.deterministic)
    if source_run is None:
        tokenizer = load_tokenizer(config.tokenizer.path)
        data = prepare_data(config, tokenizer)
    else:
        data = _load_run_data(source_run, config)
    dataset = TokenBlockDataset(data.train, config.training.seq_len, data.train_byte_addresses)
    model = DenseLM(config.model, config.attention).to(device)
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
    if stop_after_step is not None and stop_after_step <= step:
        raise ValueError("stop-after-step must exceed current step")
    run_id = run_id or uuid.uuid4().hex
    run = config.logging.root_dir / run_id
    if run.exists():
        raise FileExistsError(f"run exists: {run_id}")
    run.mkdir(parents=True)
    artifacts = _copy_artifacts(run, config, data)
    (run / "resolved_config.yaml").write_text(json.dumps(config.model_dump(mode="json"), sort_keys=True, indent=2) + "\n")
    architecture = {"model": config.model.model_dump(mode="json"), "attention": config.attention.model_dump(mode="json")}
    manifest = RunManifest(run_id, config.name, runtime, config.model_dump(mode="json"), config.model_dump(mode="json"), sha256_file(run / "resolved_config.yaml") if architecture else "", source_identity(), worker_id or socket.gethostname(), continuation, parent_run_id=parent_run_id, checkpoint_sha256=None, artifacts=artifacts)
    manifest_digest = write_manifest(run / "manifest.json", manifest)
    manager = CheckpointManager(run, manifest_sha256=manifest_digest)
    store = ExperimentStore(config.logging.root_dir)
    store.create_run(run_id, config.model_dump(mode="json"), {"inspection": inspect_model(model), "runtime": runtime.as_dict(), "manifest_sha256": manifest_digest}, parent_run_id)
    started = time.perf_counter()
    interrupted = False
    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
    old_int, old_term = signal.signal(signal.SIGINT, request_stop), signal.signal(signal.SIGTERM, request_stop)
    try:
        _save(manager, model, optimizer, config, run_id, cursor, step, tokens, watermarks)
        while step < config.training.max_steps and tokens < config.training.max_tokens:
            if cancel_path is not None and cancel_path.exists():
                interrupted = True
            order = epoch_order(len(dataset), config.seed, cursor.epoch)
            count = config.training.micro_batch_size * config.training.gradient_accumulation
            indices = order[cursor.next_block : cursor.next_block + count].tolist()
            if not indices:
                cursor = BatchCursor(cursor.epoch + 1, 0)
                continue
            records = [dataset[index] for index in indices]
            remaining = config.training.max_tokens - tokens
            labels: list[torch.Tensor] = []
            for record in records:
                label = record[1].clone()
                if remaining < label.numel():
                    label.flatten()[remaining:] = -100
                remaining -= int((label != -100).sum())
                labels.append(label)
            valid_targets = sum(int((label != -100).sum()) for label in labels)
            if valid_targets == 0:
                break
            optimizer.zero_grad(set_to_none=True)
            language_sum = 0.0
            aux_sum = 0.0
            for offset in range(0, len(records), config.training.micro_batch_size):
                chunk = records[offset : offset + config.training.micro_batch_size]
                chunk_labels = labels[offset : offset + config.training.micro_batch_size]
                x = torch.stack([record[0] for record in chunk]).to(device)
                y = torch.stack(chunk_labels).to(device)
                addresses = torch.stack([record[2] for record in chunk]).to(device) if len(chunk[0]) == 3 else None
                context = offload.hooks() if offload is not None else nullcontext()
                with context:
                    logits, auxiliary = model.forward_with_aux(
                        x,
                        byte_addresses=addresses,
                        valid_target_mask=y != -100,
                        activation_checkpointing=config.runtime.memory.activation_checkpointing.enabled,
                    )
                ce_sum = functional.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum")
                chunk_valid = int((y != -100).sum())
                loss = (ce_sum + auxiliary * chunk_valid) / valid_targets
                loss.backward()
                language_sum += float(ce_sum.detach())
                aux_sum += float(auxiliary.detach()) * chunk_valid
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm, error_if_nonfinite=True, foreach=False)
            next_step = step + 1
            lr = learning_rate_for_step(next_step, config.training.max_steps, config.optimizer.warmup_steps, config.optimizer.peak, config.optimizer.floor)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            step, tokens = next_step, tokens + valid_targets
            cursor = BatchCursor(cursor.epoch, cursor.next_block + len(indices))
            elapsed = time.perf_counter() - started
            store.log_metrics(run_id, step, tokens, elapsed, {"train/loss": language_sum / valid_targets, "moe/router_auxiliary_loss": aux_sum / valid_targets, "optimizer/learning_rate": lr, "optimizer/grad_norm": float(norm), "performance/tokens_per_second": valid_targets / max(elapsed, 1e-9), "batch/micro_batch_size": float(config.training.micro_batch_size), "batch/accumulation_steps": float(config.training.gradient_accumulation), "batch/effective_batch_size": float(len(indices)), "batch/effective_tokens_per_update": float(valid_targets)})
            terminal = step >= config.training.max_steps or tokens >= config.training.max_tokens or step == stop_after_step or interrupted
            if _checkpoint_due(config, step, tokens, elapsed, watermarks) or terminal:
                watermarks = {"step": float(step), "tokens": float(tokens), "minutes": elapsed}
                _save(manager, model, optimizer, config, run_id, cursor, step, tokens, watermarks)
                store.log_event(run_id, step, tokens, elapsed, "checkpoint_saved", {})
            if terminal:
                status = "completed" if step >= config.training.max_steps or tokens >= config.training.max_tokens else "interrupted"
                store.finish_run(run_id, status, str(manager.root / "latest.json"))
                store.log_event(run_id, step, tokens, elapsed, f"run_{status}", {})
                return run_id
        return run_id
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
