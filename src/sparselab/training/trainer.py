from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from sparselab.config.models import RunConfig
from sparselab.data.packing import PreparedData, TokenBlockDataset, prepare_data
from sparselab.data.tokenizer import load_tokenizer
from sparselab.model.inspection import inspect_model
from sparselab.model.moe import TopKMoE
from sparselab.model.transformer import DenseLM
from sparselab.runtime import seed_everything, select_device
from sparselab.training.checkpoints import load_checkpoint, save_checkpoint
from sparselab.training.metrics import ExperimentStore
from sparselab.training.optimizer import learning_rate_for_step, make_optimizer


def _load_run_data(run: Path, config: RunConfig) -> PreparedData:
    root = run / "data"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"resume run lacks immutable prepared data: {root}")
    byte_enabled = config.model.memory == "byte"
    train_byte = root / "train_byte_addresses.npy"
    validation_byte = root / "validation_byte_addresses.npy"
    if byte_enabled and (not train_byte.is_file() or not validation_byte.is_file()):
        raise ValueError("resume run lacks byte-address artifacts")
    return PreparedData(
        root=root,
        train=np.load(root / "train.npy", mmap_mode="r"),
        validation=np.load(root / "validation.npy", mmap_mode="r"),
        train_byte_addresses=np.load(train_byte, mmap_mode="r")
        if byte_enabled
        else None,
        validation_byte_addresses=(
            np.load(validation_byte, mmap_mode="r") if byte_enabled else None
        ),
        manifest=json.loads(manifest_path.read_text()),
    )


def train(
    config: RunConfig,
    *,
    resume: Path | None = None,
    run_id: str | None = None,
    stop_after_step: int | None = None,
) -> str:
    state: dict[str, object] | None = load_checkpoint(resume) if resume else None
    if state is not None:
        checkpoint_config = RunConfig.model_validate(state["config"])
        comparable = config.model_dump(mode="json")
        checkpoint_comparable = checkpoint_config.model_dump(mode="json")
        comparable["device"] = checkpoint_comparable["device"]
        comparable["logging"] = checkpoint_comparable["logging"]
        if comparable != checkpoint_comparable:
            raise ValueError(
                "resume configuration differs from the checkpoint outside device/logging"
            )
    device = select_device(config.device)
    seed_everything(config.seed, deterministic_cpu=config.training.deterministic)
    if state is None:
        tokenizer = load_tokenizer(config.tokenizer.path)
        data = prepare_data(config, tokenizer)
    else:
        source_run = resume.parent.parent
        tokenizer = load_tokenizer(source_run / "tokenizer.json")
        data = _load_run_data(source_run, config)
    dataset = TokenBlockDataset(
        data.train, config.training.seq_len, data.train_byte_addresses
    )
    model = DenseLM(config.model, config.attention).to(device)
    optimizer = make_optimizer(
        model,
        config.optimizer.learning_rate,
        config.optimizer.weight_decay,
        config.optimizer.betas,
        config.optimizer.eps,
    )
    step = tokens = epoch = next_block = 0
    parent = None
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        tokens = int(state["tokens_seen"])
        epoch, next_block = state["cursor"]
        parent = str(state.get("run_id"))
    if stop_after_step is not None and stop_after_step <= step:
        raise ValueError("stop-after-step must exceed current step")
    run_id = run_id or uuid.uuid4().hex
    root = config.logging.root_dir
    run = root / run_id
    if run.exists():
        raise FileExistsError(f"run exists: {run_id}")
    run.mkdir(parents=True)
    (run / "checkpoints").mkdir()
    shutil.copy2(config.tokenizer.path, run / "tokenizer.json")
    tokenizer_manifest = config.tokenizer.path.with_name("tokenizer_manifest.json")
    if tokenizer_manifest.is_file():
        shutil.copy2(tokenizer_manifest, run / "tokenizer_manifest.json")
    shutil.copytree(data.root, run / "data")
    (run / "resolved_config.yaml").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, default=str)
    )
    store = ExperimentStore(root)
    store.create_run(
        run_id,
        config.model_dump(mode="json"),
        {"inspection": inspect_model(model), "device": str(device)},
        parent,
    )
    store.log_event(
        run_id, step, tokens, 0, "run_resumed" if parent else "run_started", {}
    )
    started = time.perf_counter()
    while step < config.training.max_steps and tokens < config.training.max_tokens:
        order = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(config.seed + epoch)
        )
        indices = order[next_block : next_block + config.training.batch_size].tolist()
        if not indices:
            epoch, next_block = epoch + 1, 0
            continue
        batch = [dataset[i] for i in indices]
        x = torch.stack([item[0] for item in batch]).to(device)
        y = torch.stack([item[1] for item in batch]).to(device)
        byte_addresses = (
            torch.stack([item[2] for item in batch]).to(device)
            if len(batch[0]) == 3
            else None
        )
        remaining = config.training.max_tokens - tokens
        if remaining < y.numel():
            y = y.clone()
            y.flatten()[remaining:] = -100
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, byte_addresses=byte_addresses)
        language_loss = functional.cross_entropy(
            logits.flatten(0, 1), y.flatten(), ignore_index=-100
        )
        auxiliary_loss = model.auxiliary_loss
        loss = language_loss + auxiliary_loss
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.grad_clip_norm,
            error_if_nonfinite=True,
            foreach=False,
        )
        step += 1
        lr = learning_rate_for_step(
            step,
            config.training.max_steps,
            config.optimizer.warmup_steps,
            config.optimizer.learning_rate,
            config.optimizer.min_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        valid = int((y != -100).sum())
        tokens += valid
        next_block += len(indices)
        elapsed = time.perf_counter() - started
        metric_values = {
            "train/loss": float(language_loss.detach()),
            "moe/router_auxiliary_loss": float(auxiliary_loss.detach()),
            "optimizer/learning_rate": lr,
            "optimizer/grad_norm": float(norm),
            "performance/tokens_per_second": tokens / max(elapsed, 1e-9),
        }
        for layer_index, block in enumerate(model.blocks):
            if isinstance(block.ffn, TopKMoE) and block.ffn.last_diagnostics:
                diagnostics = block.ffn.last_diagnostics
                prefix = f"moe/layer_{layer_index}"
                metric_values.update(
                    {
                        f"{prefix}/router_entropy": float(diagnostics.entropy),
                        f"{prefix}/maximum_expert_fraction": float(
                            diagnostics.maximum_fraction
                        ),
                        f"{prefix}/mean_topk_probability": float(
                            diagnostics.mean_topk_probability
                        ),
                    }
                )
        if getattr(model.memory, "last_diagnostics", None) is not None:
            diagnostics = model.memory.last_diagnostics
            metric_values.update(
                {
                    "engram/lookups": float(diagnostics.lookup_count),
                    "engram/unique_buckets": float(diagnostics.unique_addresses),
                    "engram/collisions": float(diagnostics.collision_count),
                    "engram/bucket_reuse_rate": float(diagnostics.bucket_reuse_rate),
                    "engram/table_utilization": float(diagnostics.table_utilization),
                    "engram/gate_mean": float(diagnostics.gate_mean),
                }
            )
        store.log_metrics(run_id, step, tokens, elapsed, metric_values)
        terminal = (
            step >= config.training.max_steps
            or tokens >= config.training.max_tokens
            or step == stop_after_step
        )
        if step % config.logging.checkpoint_every_steps == 0 or terminal:
            checkpoint = run / "checkpoints" / f"step_{step:08d}.pt"
            save_checkpoint(
                checkpoint,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "tokens_seen": tokens,
                    "cursor": (epoch, next_block),
                    "config": config.model_dump(mode="json"),
                    "run_id": run_id,
                },
            )
            store.log_event(run_id, step, tokens, elapsed, "checkpoint_saved", {})
        if terminal:
            status = (
                "completed"
                if step >= config.training.max_steps
                or tokens >= config.training.max_tokens
                else "interrupted"
            )
            store.finish_run(run_id, status, str(checkpoint))
            store.log_event(run_id, step, tokens, elapsed, f"run_{status}", {})
            break
    return run_id
