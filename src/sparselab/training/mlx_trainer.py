"""Independent dense MLX training lifecycle with resumable native state."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import mlx.core as mx

from sparselab.config.models import RunConfig
from sparselab.data.packing import TokenBlockDataset, epoch_order, prepare_data
from sparselab.data.tokenizer import load_tokenizer
from sparselab.engines.mlx import MLXEngine


def train_mlx(
    config: RunConfig,
    *,
    resume: Path | None = None,
    run_id: str | None = None,
    stop_after_step: int | None = None,
) -> str:
    if resume is not None and not resume.is_dir():
        raise ValueError("MLX resume must name a native checkpoint directory")
    run_id = run_id or uuid.uuid4().hex
    run = config.logging.root_dir / run_id
    if run.exists():
        raise FileExistsError(f"run exists: {run_id}")
    tokenizer = load_tokenizer(config.tokenizer.path)
    data = prepare_data(config, tokenizer)
    dataset = TokenBlockDataset(
        data.train, config.training.seq_len, data.train_byte_addresses
    )
    mx.random.seed(config.seed)
    engine = MLXEngine(config)
    step = tokens = epoch = next_block = 0
    if resume is not None:
        engine.load_state(resume)
        metadata = json.loads((resume / "state.json").read_text())
        step, tokens, epoch, next_block = (
            metadata["step"],
            metadata["tokens"],
            metadata["epoch"],
            metadata["next_block"],
        )
    run.mkdir(parents=True)
    shutil.copy2(config.tokenizer.path, run / "tokenizer.json")
    shutil.copytree(data.root, run / "data")
    (run / "resolved_config.yaml").write_text(
        json.dumps(config.model_dump(mode="json"), sort_keys=True)
    )
    while step < config.training.max_steps and tokens < config.training.max_tokens:
        order = epoch_order(len(dataset), config.seed, epoch)
        indices = order[
            next_block : next_block + config.training.micro_batch_size
        ].tolist()
        if not indices:
            epoch, next_block = epoch + 1, 0
            continue
        batch = [dataset[index] for index in indices]
        inputs = mx.array(
            __import__("numpy").stack([item[0].numpy() for item in batch])
        )
        targets = mx.array(
            __import__("numpy").stack([item[1].numpy() for item in batch])
        )
        engine.train_update(inputs, targets)
        step += 1
        tokens += len(batch) * config.training.seq_len
        next_block += len(batch)
        terminal = (
            step >= config.training.max_steps
            or tokens >= config.training.max_tokens
            or step == stop_after_step
        )
        if step % (config.checkpoint.every_steps or 1) == 0 or terminal:
            checkpoint = run / "mlx_checkpoints" / f"step_{step:08d}"
            engine.save_state(checkpoint)
            (checkpoint / "state.json").write_text(
                json.dumps(
                    {
                        "codec": "mlx_native",
                        "version": 1,
                        "step": step,
                        "tokens": tokens,
                        "epoch": epoch,
                        "next_block": next_block,
                    }
                )
            )
        if terminal:
            break
    return run_id
