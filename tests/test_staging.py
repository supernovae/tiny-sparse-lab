from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from test_training import config as training_config
from test_training import equal

import sparselab.staging as staging_module
from sparselab.config.models import RunConfig
from sparselab.model.portable_engram import export_portable_engram
from sparselab.staging import _read_sealed, stage
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import read_manifest
from sparselab.training.stages import ExperimentStage, StageHistory
from sparselab.training.trainer import train

FIXTURES = Path(__file__).with_name("fixtures")


def _config(root: Path) -> RunConfig:
    base = training_config(root)
    return base.model_copy(
        update={
            "training": base.training.model_copy(
                update={
                    "micro_batch_size": 1,
                    "max_steps": 3,
                    "max_tokens": 48,
                }
            ),
            "optimizer": base.optimizer.model_copy(update={"warmup_steps": 1}),
            "checkpoint": base.checkpoint.model_copy(update={"every_steps": 2}),
            "staging": base.staging.model_copy(
                update={"smoke_steps": 2, "warmup_steps": 3}
            ),
        }
    )


def test_warmup_pilots_are_isolated_and_bundle_is_self_contained(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    bundle_root = stage(config, tmp_path / "stage", through="warmup")
    bundle = _read_sealed(bundle_root / "bundle.json")

    direct_id = train(config, run_id="direct")
    direct_manifest = read_manifest(
        config.logging.root_dir / direct_id / "manifest.json"
    )
    direct_progress = json.loads(
        (config.logging.root_dir / direct_id / "progress.json").read_text(
            encoding="utf-8"
        )
    )
    assert direct_manifest["stage_bundle_sha256"] is None
    assert direct_manifest["pilot_reports"] == []
    assert "sha256" not in direct_manifest
    assert {record["stage"] for record in direct_progress["stages"]}.isdisjoint(
        {"SMOKE_TEST", "WARMUP"}
    )

    moved_tokenizer = tmp_path / "moved-tokenizer.json"
    shutil.move(config.tokenizer.path, moved_tokenizer)
    moved_cache = tmp_path / "moved-cache"
    shutil.move(config.dataset.cache_dir, moved_cache)
    assert not config.tokenizer.path.exists()
    assert not config.dataset.cache_dir.exists()

    bundled_id = train(config, run_id="bundled", stage_bundle=bundle_root)
    bundled_manifest = read_manifest(
        config.logging.root_dir / bundled_id / "manifest.json"
    )
    assert bundled_manifest["stage_bundle_sha256"] == bundle["sha256"]
    assert [report["purpose"] for report in bundled_manifest["pilot_reports"]] == [
        "smoke",
        "warmup",
    ]
    assert (config.logging.root_dir / bundled_id / "tokenizer.json").is_file()
    assert (config.logging.root_dir / bundled_id / "data" / "train.npy").is_file()

    direct = CheckpointManager(config.logging.root_dir / direct_id).load(
        config.logging.root_dir / direct_id / "checkpoints/latest.json"
    )
    bundled = CheckpointManager(config.logging.root_dir / bundled_id).load(
        config.logging.root_dir / bundled_id / "checkpoints/latest.json"
    )
    assert (direct.step, direct.tokens_seen, direct.cursor) == (
        bundled.step,
        bundled.tokens_seen,
        bundled.cursor,
    )
    equal(direct.model, bundled.model)
    equal(direct.optimizer, bundled.optimizer)
    equal(direct.rng, bundled.rng)


def test_tampered_stage_bundle_rejects_before_run_creation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    bundle_root = stage(config, tmp_path / "stage", through="validate")
    asset = bundle_root / "assets" / "tokenizer.json"
    asset.write_bytes(asset.read_bytes() + b"tampered")

    with pytest.raises(ValueError):
        train(config, run_id="rejected", stage_bundle=bundle_root)

    assert not (config.logging.root_dir / "rejected").exists()


def test_inspect_needs_no_assets_but_failed_preflight_is_not_resumable(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    missing = config.model_copy(
        update={
            "tokenizer": config.tokenizer.model_copy(
                update={"path": tmp_path / "missing.json"}
            )
        }
    )
    inspected = stage(missing, tmp_path / "inspect-only", through="inspect")
    report = _read_sealed(inspected / "stage.json")
    assert report["status"] == "complete"
    assert not (inspected / "assets").exists()

    failed_root = tmp_path / "failed-validate"
    with pytest.raises(Exception, match="missing.json"):
        stage(missing, failed_root, through="validate")

    failed = _read_sealed(failed_root / "stage.json")
    assert failed["status"] == "failed"
    assert not (failed_root / "bundle.json").exists()
    assert {record["stage"] for record in failed["stages"]} == {
        "CONFIGURED",
        "INSPECTED",
        "VALIDATED",
        "FAILED",
    }
    assert all(record["status"] != "resumable" for record in failed["stages"])


def test_over_budget_staging_writes_proposal_without_mutating_config(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    constrained = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={
                    "memory": config.runtime.memory.model_copy(
                        update={"budget_bytes": 1}
                    )
                }
            )
        }
    )
    before = constrained.model_dump(mode="json")
    output = tmp_path / "over-budget"

    with pytest.raises(MemoryError, match="exceeds the explicit safe ceiling"):
        stage(constrained, output, through="validate")

    assert constrained.model_dump(mode="json") == before
    assert (output / "proposal.yaml").is_file()
    assert _read_sealed(output / "stage.json")["status"] == "failed"
    assert not (output / "bundle.json").exists()


@pytest.mark.parametrize("package_ngram", [2, 3])
def test_portable_file_bundle_validates_config_and_survives_source_removal(
    tmp_path, package_ngram
):
    base = _config(tmp_path)
    package = tmp_path / "memory.engram"
    table = torch.arange(136, dtype=torch.float32).reshape(17, 8) / 136
    export_portable_engram(table, package, ngram_size=package_ngram)
    payload = base.model_dump(mode="json")
    payload["model"].update(
        memory="portable",
        memory_table_size=17,
        memory_dim=8,
        memory_ngram_size=2,
        memory_package_path=str(package),
    )
    config = RunConfig.model_validate(payload)
    output = tmp_path / "portable-stage"
    if package_ngram != 2:
        with pytest.raises(ValueError):
            stage(config, output, through="validate")
        assert _read_sealed(output / "stage.json")["status"] == "failed"
        assert not (output / "bundle.json").exists()
        return
    stage(config, output, through="validate")
    assert (output / "assets" / "portable_package").is_file()
    package.unlink()
    config.tokenizer.path.unlink()
    shutil.rmtree(config.dataset.cache_dir)
    run_id = train(config, run_id="portable-bundled", stage_bundle=output)
    run = config.logging.root_dir / run_id
    snapshot = CheckpointManager(run).load(run / "checkpoints" / "latest.json")
    assert (snapshot.step, snapshot.tokens_seen) == (3, 48)
    assert torch.equal(snapshot.model["memory.embedding.weight"], table)


@pytest.mark.parametrize("kind", ["out_of_memory", "pilot_failed"])
def test_classified_pilot_failure_only_proposes_for_memory_errors(
    tmp_path, monkeypatch, kind
):
    config = _config(tmp_path)
    original = config.model_dump(mode="json")
    real_run = subprocess.run

    def failed_pilot(argv, **kwargs):
        if len(argv) > 2 and argv[2] == "sparselab.training.pilot":
            root, purpose = Path(argv[3]), argv[4]
            staging_module._seal(
                root / "pilots" / purpose / "failure.json",
                {
                    "format_version": 1,
                    "kind": kind,
                    "message": "injected pilot failure",
                },
            )
            return subprocess.CompletedProcess(argv, returncode=1)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", failed_pilot)
    output = tmp_path / "pilot-failure"
    with pytest.raises(MemoryError if kind == "out_of_memory" else RuntimeError):
        stage(config, output, through="smoke")
    assert config.model_dump(mode="json") == original
    assert (output / "proposal.yaml").exists() == (kind == "out_of_memory")
    assert _read_sealed(output / "stage.json")["status"] == "failed"
    assert not (output / "bundle.json").exists()


def test_stage_history_allows_repeated_evaluation_checkpoint_boundaries() -> None:
    history = StageHistory()
    for stage_name, step, tokens in (
        (ExperimentStage.CONFIGURED, 0, 0),
        (ExperimentStage.INSPECTED, 0, 0),
        (ExperimentStage.VALIDATED, 0, 0),
        (ExperimentStage.TRAINING, 0, 0),
        (ExperimentStage.EVALUATING, 1, 16),
        (ExperimentStage.CHECKPOINTED, 1, 16),
        (ExperimentStage.TRAINING, 1, 16),
        (ExperimentStage.CHECKPOINTED, 2, 32),
        (ExperimentStage.TRAINING, 2, 32),
        (ExperimentStage.EVALUATING, 3, 48),
        (ExperimentStage.CHECKPOINTED, 3, 48),
        (ExperimentStage.COMPLETE, 3, 48),
    ):
        history.start(stage_name, step=step, tokens_seen=tokens)
        history.finish(step=step, tokens_seen=tokens)

    assert [record.stage for record in history.records].count(
        ExperimentStage.CHECKPOINTED
    ) == 3
    assert history.records[-1].stage is ExperimentStage.COMPLETE


def test_stage_history_rejects_backward_counters_and_terminal_reentry() -> None:
    history = StageHistory()
    for stage_name, step, tokens in (
        (ExperimentStage.CONFIGURED, 0, 0),
        (ExperimentStage.INSPECTED, 0, 0),
        (ExperimentStage.VALIDATED, 0, 0),
        (ExperimentStage.TRAINING, 2, 32),
    ):
        history.start(stage_name, step=step, tokens_seen=tokens)
        history.finish(step=step, tokens_seen=tokens)

    with pytest.raises(ValueError, match="counters cannot move backward"):
        history.start(ExperimentStage.CHECKPOINTED, step=1, tokens_seen=32)

    history.start(ExperimentStage.FAILED, step=2, tokens_seen=32)
    history.finish("failed", step=2, tokens_seen=32)
    with pytest.raises(ValueError, match="illegal stage transition"):
        history.start(ExperimentStage.TRAINING, step=2, tokens_seen=32)


@pytest.mark.parametrize(
    ("fixture_name", "supported"),
    [
        ("runtime_stage_wire_v1.json", True),
        ("runtime_stage_wire_v2.json", False),
    ],
)
def test_stage_wire_version_compatibility(
    fixture_name: str, supported: bool, tmp_path: Path
) -> None:
    wire = tmp_path / "bundle.json"
    shutil.copy2(FIXTURES / fixture_name, wire)

    if supported:
        sealed = _read_sealed(wire)
        assert sealed["format_version"] == 1
        assert sealed["status"] == "complete"
    else:
        with pytest.raises(ValueError, match="unsupported stage bundle version"):
            _read_sealed(wire)
