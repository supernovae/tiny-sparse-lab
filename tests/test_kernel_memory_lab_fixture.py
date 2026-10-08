"""Fresh, bounded Card 02 plumbing controls for Kernel Memory Lab."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from test_semantic import _retriever
from test_training import equal

from sparselab.config.migrate import migrate_v1
from sparselab.config.models import (
    AttentionConfig,
    ModelConfig,
    RunConfig,
    TokenizerTrainConfig,
)
from sparselab.data.tokenizer import train_tokenizer
from sparselab.engram.semantic import SemanticQueryBatch
from sparselab.engram.semantic_probe import run_semantic_probe
from sparselab.model.transformer import DenseLM
from sparselab.training.attempt_budget import AttemptBudget, AttemptBudgetError
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import sha256_file
from sparselab.training.trainer import train


def _budget() -> AttemptBudget:
    path = os.environ.get("SPARSELAB_ATTEMPT_BUDGET_LEDGER")
    if not path:
        raise AttemptBudgetError("Card 02 tests require a shared attempt budget ledger")
    budget = AttemptBudget(Path(path))
    budget.remaining_seconds()
    return budget


def test_fresh_fixed_batch_overfit_and_gradient_membership(tmp_path: Path) -> None:
    """A hand-checked one-symbol transition must be learnable from random weights."""
    budget = _budget()
    budget.reserve("Card 02 fixed-batch overfit", 80)
    torch.manual_seed(20261007)
    model = DenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
            max_seq_len=8,
        ),
        AttentionConfig(),
    )
    # Fixture tokenizer: <pad>=0, <unk>=1, <bos>=2, <eos>=3,
    # amber=4, blue=5. This is separate from the main tokenizer.
    tokens = torch.tensor([[4, 5, 4, 5, 4, 5, 4, 5]], dtype=torch.long)
    targets = torch.tensor([[5, 4, 5, 4, 5, 4, 5, 4]], dtype=torch.long)
    assert targets.tolist() == [[5, 4, 5, 4, 5, 4, 5, 4]]
    frozen = next(model.blocks[0].parameters())
    frozen.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.02,
        weight_decay=0,
    )
    members = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert id(frozen) not in members
    assert all(
        id(parameter) in members
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    model.train()
    with torch.no_grad():
        initial = F.cross_entropy(
            model(tokens).reshape(-1, 260), targets.reshape(-1)
        ).item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(tokens).reshape(-1, 260), targets.reshape(-1))
        assert math.isfinite(loss.item())
        loss.backward()
        assert frozen.grad is None
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        budget.remaining_seconds()
        optimizer.step()
    with torch.no_grad():
        final = F.cross_entropy(
            model(tokens).reshape(-1, 260), targets.reshape(-1)
        ).item()
    assert final <= initial * 0.2
    (tmp_path / "overfit-report.json").write_text(
        json.dumps(
            {
                "fixture_seed": 20261007,
                "fixture_tokenizer": {"amber": 4, "blue": 5},
                "targets": targets.tolist(),
                "optimizer_steps": 80,
                "initial_loss": initial,
                "final_loss": final,
                "loss_reduction_fraction": 1 - final / initial,
                "frozen_gradient_absent": frozen.grad is None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_native_strict_resume_with_fresh_project_text(tmp_path: Path) -> None:
    budget = _budget()
    text = "amber blue amber blue amber blue amber blue " * 40
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
    validation_path.write_text(
        json.dumps({"text": "blue amber " * 40}) + "\n", encoding="utf-8"
    )
    assert train_path.stat().st_size + validation_path.stat().st_size < 1_048_576
    dataset = {
        "source": "local_text",
        "cache_dir": tmp_path / "cache",
        "train_path": train_path,
        "validation_path": validation_path,
        "license": "CC0-1.0 fixture authored for Kernel Memory Lab",
        "train_max_documents": 1,
        "validation_max_documents": 1,
        "train_max_tokens": 4096,
        "validation_max_tokens": 4096,
    }
    tokenizer = train_tokenizer(
        TokenizerTrainConfig.model_validate(
            {
                "schema_version": 1,
                "vocab_size": 260,
                "min_frequency": 1,
                "max_documents": 1,
                "output_dir": tmp_path / "tokenizer",
                "dataset": dataset,
            }
        )
    )
    run = RunConfig.model_validate(
        migrate_v1(
            {
                "schema_version": 1,
                "name": "kernel-memory-lab-fixture-v1",
                "seed": 20261007,
                "device": "cpu",
                "model": {
                    "vocab_size": 260,
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 2,
                    "ffn_dim": 32,
                    "max_seq_len": 16,
                    "memory": "none",
                },
                "tokenizer": {"path": tokenizer},
                "dataset": dataset,
                "training": {
                    "batch_size": 1,
                    "seq_len": 8,
                    "max_steps": 4,
                    "max_tokens": 32,
                    "deterministic": True,
                },
                "optimizer": {
                    "learning_rate": 0.003,
                    "min_learning_rate": 0.0003,
                    "warmup_steps": 1,
                },
                "logging": {
                    "root_dir": tmp_path / "runs",
                    "checkpoint_every_steps": 2,
                },
            }
        )
    )
    budget.reserve("Card 02 native full", 4)
    train(run, run_id="full", max_wall_seconds=budget.remaining_seconds())
    budget.reserve("Card 02 native parent", 2)
    train(
        run,
        run_id="part",
        stop_after_step=2,
        max_wall_seconds=budget.remaining_seconds(),
    )
    budget.reserve("Card 02 native resumed child", 2)
    train(
        run,
        run_id="resumed",
        resume=run.logging.root_dir / "part/checkpoints/latest.json",
        max_wall_seconds=budget.remaining_seconds(),
    )
    full = CheckpointManager(run.logging.root_dir / "full").load(
        run.logging.root_dir / "full/checkpoints/latest.json"
    )
    resumed = CheckpointManager(run.logging.root_dir / "resumed").load(
        run.logging.root_dir / "resumed/checkpoints/latest.json"
    )
    assert full.step == resumed.step == 4
    assert full.tokens_seen == resumed.tokens_seen == 32
    assert full.cursor == resumed.cursor
    equal(full.schedule, resumed.schedule)
    equal(full.optimizer, resumed.optimizer)
    equal(full.optimizer_parameter_names, resumed.optimizer_parameter_names)
    equal(full.rng, resumed.rng)
    equal(full.model, resumed.model)
    (tmp_path / "resume-report.json").write_text(
        json.dumps(
            {
                "source_sha256": {
                    "train": sha256_file(train_path),
                    "validation": sha256_file(validation_path),
                },
                "tokenizer_sha256": sha256_file(tokenizer),
                "run_ids": ["full", "part", "resumed"],
                "full_step": full.step,
                "resumed_step": resumed.step,
                "full_tokens_seen": full.tokens_seen,
                "resumed_tokens_seen": resumed.tokens_seen,
                "model_exact": True,
                "optimizer_exact": True,
                "scheduler_equal": True,
                "cursor_equal": True,
                "rng_exact": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_native_semantic_probe_and_disabled_path(tmp_path: Path) -> None:
    budget = _budget()
    budget.reserve("Card 02 supplied-vector probe", 0)
    retriever = _retriever(
        tmp_path / "semantic",
        name="kernel-memory-fixture",
        record_ids=("oracle", "wrong"),
        keys=[[1, 0], [0, 1]],
        values=[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
    )
    probe = {
        "format": "sparselab-semantic-probe-v1",
        "model": {
            "kind": "initialized",
            "model": {
                "vocab_size": 260,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 2,
                "ffn_dim": 32,
                "max_seq_len": 8,
            },
            "attention": {},
            "seed": 20261007,
        },
        "input_tokens": [[4, 5, 4]],
        "queries": [
            {
                "id": "fixture-keys",
                "encoder": retriever.key_encoder.model_dump(mode="json"),
                "vectors": [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
                "mask": [[True, True, False]],
            }
        ],
        "attachments": [
            {
                "name": "fixture",
                "pack": "semantic/pack.enpack",
                "expected_pack_id": retriever.pack_id,
                "query": "fixture-keys",
                "site": "after_block",
                "block_index": 0,
                "min_score": 0.9,
                "weights": {"kind": "initialized", "seed": 20261008},
            }
        ],
    }
    declaration = tmp_path / "probe.yaml"
    declaration.write_text(yaml.safe_dump(probe), encoding="utf-8")
    report = run_semantic_probe(declaration)
    traces = report["attachments"][0]["traces"]
    assert [
        item["trace"]["status"] if item["trace"] else "masked" for item in traces
    ] == [
        "hit",
        "hit",
        "masked",
    ]
    assert [item["trace"]["best_record_id"] for item in traces[:2]] == [
        "oracle",
        "wrong",
    ]
    (tmp_path / "probe-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    torch.manual_seed(20261007)
    model = DenseLM(
        ModelConfig(
            vocab_size=260,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=8,
        ),
        AttentionConfig(),
    )
    model.eval()
    tokens = torch.tensor([[4, 5, 4]])
    with torch.inference_mode():
        dense = model(tokens)
    adapter = model.add_semantic_memory(
        "fixture", retriever, site="after_block", block_index=0, min_score=0.9
    )
    with torch.no_grad():
        adapter.gate.weight.zero_()
    masked_queries = SemanticQueryBatch(
        retriever.key_encoder,
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]),
        mask=torch.zeros((1, 3), dtype=torch.bool),
    )
    with torch.inference_mode():
        disabled = model(tokens, semantic_queries=masked_queries)
    torch.testing.assert_close(disabled, dense, rtol=0, atol=0)
