from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sparselab.config.migrate import migrate_v1
from sparselab.config.models import RunConfig, TokenizerTrainConfig
from sparselab.data.tokenizer import train_tokenizer
from sparselab.evaluation.evidence import experiment_evidence
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import read_manifest
from sparselab.training.trainer import train


def equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            equal(first, second)
    else:
        assert left == right


def config(root: Path) -> RunConfig:
    tokenizer = train_tokenizer(
        TokenizerTrainConfig.model_validate(
            {
                "schema_version": 1,
                "vocab_size": 512,
                "min_frequency": 1,
                "max_documents": 200,
                "output_dir": root / "tokenizer",
                "dataset": {
                    "source": "synthetic",
                    "cache_dir": root / "cache",
                    "train_max_documents": 200,
                    "validation_max_documents": 40,
                    "train_max_tokens": 8192,
                    "validation_max_tokens": 2048,
                    "synthetic_seed": 7,
                },
            }
        )
    )
    return RunConfig.model_validate(
        migrate_v1(
            {
                "schema_version": 1,
                "name": "test",
                "seed": 7,
                "device": "cpu",
                "model": {
                    "vocab_size": 512,
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 2,
                    "ffn_dim": 32,
                    "max_seq_len": 32,
                },
                "tokenizer": {"path": tokenizer},
                "dataset": {
                    "source": "synthetic",
                    "cache_dir": root / "cache",
                    "train_max_documents": 200,
                    "validation_max_documents": 40,
                    "train_max_tokens": 8192,
                    "validation_max_tokens": 2048,
                    "synthetic_seed": 7,
                },
                "training": {
                    "batch_size": 2,
                    "seq_len": 16,
                    "max_steps": 12,
                    "max_tokens": 384,
                    "deterministic": True,
                },
                "optimizer": {
                    "learning_rate": 0.003,
                    "min_learning_rate": 0.0003,
                    "warmup_steps": 2,
                },
                "logging": {"root_dir": root / "runs", "checkpoint_every_steps": 6},
            }
        )
    )


@pytest.mark.parametrize(
    ("memory", "optimizer_name"),
    [("none", "adamw"), ("byte", "adamw"), ("none", "adafactor")],
)
def test_interrupted_resume_matches_uninterrupted(
    tmp_path: Path, memory: str, optimizer_name: str
) -> None:
    memory_settings = (
        {
            "memory": "byte",
            "memory_table_size": 64,
            "memory_dim": 8,
            "memory_ngram_size": 5,
        }
        if memory == "byte"
        else {"memory": "none"}
    )
    full = config(tmp_path / "full")
    full = full.model_copy(
        update={"model": full.model.model_copy(update=memory_settings)}
    )
    if optimizer_name == "adafactor":
        payload = full.model_dump(mode="json")
        payload["optimizer"] = {
            "name": "adafactor",
            "peak": full.optimizer.peak,
            "floor": full.optimizer.floor,
            "warmup_steps": full.optimizer.warmup_steps,
            "weight_decay": full.optimizer.weight_decay,
        }
        full = RunConfig.model_validate(payload)
    train(full, run_id="full")
    split = config(tmp_path / "split")
    split = split.model_copy(
        update={
            "model": split.model.model_copy(update=memory_settings),
            "optimizer": full.optimizer,
        }
    )
    train(split, run_id="part", stop_after_step=5)
    train(
        split,
        run_id="resumed",
        resume=split.logging.root_dir / "part/checkpoints/latest.json",
    )
    left = CheckpointManager(full.logging.root_dir / "full").load(
        full.logging.root_dir / "full/checkpoints/latest.json"
    )
    right = CheckpointManager(split.logging.root_dir / "resumed").load(
        split.logging.root_dir / "resumed/checkpoints/latest.json"
    )
    assert left.tokens_seen == right.tokens_seen == 384
    assert left.cursor == right.cursor
    equal(left.optimizer, right.optimizer)
    equal(left.model, right.model)
    equal(left.rng, right.rng)
    parent = CheckpointManager(split.logging.root_dir / "part").load(
        split.logging.root_dir / "part/checkpoints/latest.json"
    )
    manifest = read_manifest(split.logging.root_dir / "resumed/manifest.json")
    assert manifest["checkpoint_sha256"] == parent.checkpoint_sha256
    assert right.parent_checkpoint_sha256 == parent.checkpoint_sha256


def test_training_pairs_validation_with_verified_checkpoints(tmp_path: Path) -> None:
    original = config(tmp_path)
    measured = original.model_copy(
        update={
            "evaluation": original.evaluation.model_copy(
                update={"every_steps": 2, "max_batches": 1}
            )
        }
    )
    run_id = train(measured, run_id="measured")

    evidence = experiment_evidence(measured.logging.root_dir / run_id)
    observations = evidence["quality_observations"]
    assert evidence["evidence_level"] == "checkpointed_held_out"
    assert evidence["verified_checkpoints"]
    assert isinstance(observations, list)
    assert [item["step"] for item in observations] == [0, 2, 4, 6, 8, 10, 12]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_interrupted_checkpoint_resumes_locally(tmp_path: Path) -> None:
    original = config(tmp_path)
    mps = original.model_copy(
        update={"runtime": original.runtime.model_copy(update={"backend": "auto"})}
    )
    train(mps, run_id="part", stop_after_step=2)
    train(
        mps,
        run_id="resumed",
        resume=mps.logging.root_dir / "part/checkpoints/latest.json",
    )

    resumed = CheckpointManager(mps.logging.root_dir / "resumed").load(
        mps.logging.root_dir / "resumed/checkpoints/latest.json"
    )
    assert resumed.step == mps.training.max_steps
    assert resumed.backend == "mps"
    manifest = read_manifest(mps.logging.root_dir / "resumed/manifest.json")
    assert manifest["requested_config"]["runtime"]["backend"] == "auto"
    assert manifest["effective_config"]["runtime"]["backend"] == "mps"


def test_adafactor_state_offload_is_rejected(tmp_path: Path) -> None:
    payload = config(tmp_path).model_dump(mode="json")
    payload["optimizer"] = {
        "name": "adafactor",
        "peak": 0.003,
        "floor": 0.0003,
        "warmup_steps": 2,
        "weight_decay": 0.1,
        "state_offload": True,
    }

    with pytest.raises(ValueError, match="state_offload is deferred"):
        RunConfig.model_validate(payload)


def test_resume_rejects_model_configuration_mismatch(tmp_path: Path) -> None:
    original = config(tmp_path / "original")
    train(original, run_id="part", stop_after_step=5)
    incompatible = original.model_copy(
        update={"model": original.model.model_copy(update={"ffn_dim": 48})}
    )
    with pytest.raises(ValueError, match="configuration differs"):
        train(
            incompatible,
            run_id="rejected",
            resume=original.logging.root_dir / "part/checkpoints/latest.json",
        )


def test_non_multiple_token_budget_commits_exactly_77_targets(tmp_path: Path) -> None:
    original = config(tmp_path)
    bounded = original.model_copy(
        update={"training": original.training.model_copy(update={"max_tokens": 77})}
    )
    run_id = train(bounded, run_id="bounded")
    snapshot = CheckpointManager(bounded.logging.root_dir / run_id).load(
        bounded.logging.root_dir / run_id / "checkpoints/latest.json"
    )
    assert snapshot.tokens_seen == 77


def test_allow_runtime_drift_does_not_allow_dataset_change(tmp_path: Path) -> None:
    original = config(tmp_path / "original")
    train(original, run_id="part", stop_after_step=5)
    changed = original.model_copy(
        update={"dataset": original.dataset.model_copy(update={"synthetic_seed": 99})}
    )
    with pytest.raises(ValueError, match="configuration differs"):
        train(
            changed,
            run_id="rejected",
            resume=original.logging.root_dir / "part/checkpoints/latest.json",
            allow_runtime_drift=True,
        )


def test_promotion_prepares_destination_dataset(tmp_path: Path) -> None:
    source = config(tmp_path / "source")
    train(source, run_id="source", stop_after_step=2)
    destination = config(tmp_path / "destination").model_copy(
        update={"dataset": source.dataset.model_copy(update={"synthetic_seed": 91})}
    )
    train(
        destination,
        run_id="promoted",
        promote=source.logging.root_dir / "source/checkpoints/latest.json",
        stop_after_step=1,
    )
    source_manifest = (
        source.logging.root_dir / "source/data/manifest.json"
    ).read_text()
    destination_manifest = (
        destination.logging.root_dir / "promoted/data/manifest.json"
    ).read_text()
    assert source_manifest != destination_manifest


def test_tampered_validation_report_is_not_held_out_evidence(tmp_path: Path) -> None:
    original = config(tmp_path)
    run_id = train(original, run_id="measured")
    report = next((original.logging.root_dir / run_id / "evaluations").glob("*.json"))
    payload = json.loads(report.read_text())
    payload["loss"] = 999
    report.write_text(json.dumps(payload))
    evidence = experiment_evidence(original.logging.root_dir / run_id)
    assert evidence["evidence_level"] != "checkpointed_held_out"
    assert report.name in {item["path"] for item in evidence["rejected_reports"]}
    assert payload["checkpoint"] in evidence["missing_reports"]
    assert all(
        item["checkpoint"] != payload["checkpoint"]
        for item in evidence["quality_observations"]
    )


def test_source_drift_requires_explicit_best_effort_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = config(tmp_path)
    train(original, run_id="part", stop_after_step=1)
    parent = original.logging.root_dir / "part"
    previous_source = read_manifest(parent / "manifest.json")["source_identity"]
    changed_source = {**previous_source, "sha256": "0" * 64}
    monkeypatch.setattr(
        "sparselab.training.trainer.source_identity", lambda: changed_source
    )
    with pytest.raises(ValueError, match="allow-runtime-drift"):
        train(original, run_id="rejected", resume=parent / "checkpoints/latest.json")
    assert not (original.logging.root_dir / "rejected").exists()
    train(
        original,
        run_id="allowed",
        resume=parent / "checkpoints/latest.json",
        allow_runtime_drift=True,
        stop_after_step=2,
    )
    manifest = read_manifest(original.logging.root_dir / "allowed/manifest.json")
    decision = next(
        item
        for item in manifest["resource_decisions"]
        if item["kind"] == "runtime_drift"
    )
    assert decision["resume_level"] == "best_effort"
    assert decision["changes"][0]["requested"] == previous_source["sha256"]
    assert decision["changes"][0]["effective"] == changed_source["sha256"]


def test_preexisting_cancel_marker_commits_no_update(tmp_path: Path) -> None:
    original = config(tmp_path)
    cancel = tmp_path / "cancel"
    cancel.touch()
    train(original, run_id="cancelled", cancel_path=cancel)
    snapshot = CheckpointManager(original.logging.root_dir / "cancelled").load(
        original.logging.root_dir / "cancelled/checkpoints/latest.json"
    )
    assert snapshot.step == snapshot.tokens_seen == 0
    assert snapshot.optimizer["state"] == {}
