from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_training import config as training_config

from sparselab import workspace_preflight
from sparselab.data import packing
from sparselab.data.tokenizer import load_tokenizer
from sparselab.staging import stage
from sparselab.training import trainer


def configured(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(work))
    base = training_config(tmp_path)
    return base.model_copy(
        update={
            "training": base.training.model_copy(
                update={"max_steps": 3, "max_tokens": 96}
            ),
            "optimizer": base.optimizer.model_copy(update={"warmup_steps": 1}),
            "checkpoint": base.checkpoint.model_copy(update={"every_steps": 2}),
        }
    )


def available_space(monkeypatch, amount):
    real = workspace_preflight.os.statvfs

    def observed(path):
        stats = real(path)
        return SimpleNamespace(
            f_bavail=amount // stats.f_frsize,
            f_frsize=stats.f_frsize,
            f_favail=stats.f_favail,
            f_files=stats.f_files,
        )

    monkeypatch.setattr(workspace_preflight.os, "statvfs", observed)


def test_training_uses_authenticated_cache_footprint(tmp_path, monkeypatch):
    config = configured(tmp_path, monkeypatch)
    prepared = packing.prepare_data(config, load_tokenizer(config.tokenizer.path))
    checks = workspace_preflight.training_storage_checks(
        config, verified_prepared=prepared.receipt
    )
    assert len(checks) == 1
    needed = checks[0].projected_bytes + checks[0].reserve_bytes
    available_space(monkeypatch, needed + 4096)
    assert (
        workspace_preflight.training_storage_checks(config)[0].status == "insufficient"
    )

    def no_repacking(*args, **kwargs):
        pytest.fail("training retokenized already verified prepared inputs")

    monkeypatch.setattr(packing, "iter_documents", no_repacking)
    run_id = trainer.train(config, run_id="actual-storage")
    root = config.logging.root_dir / run_id
    progress = json.loads((root / "progress.json").read_text())
    assert progress["status"] == "completed"
    assert progress["tokens_seen"] == 96
    assert (root / "data/train.npy").read_bytes() == (
        prepared.root / "train.npy"
    ).read_bytes()


def test_training_refuses_insufficient_future_growth_before_model(
    tmp_path, monkeypatch
):
    config = configured(tmp_path, monkeypatch)
    packing.prepare_data(config, load_tokenizer(config.tokenizer.path))
    available_space(monkeypatch, 1)

    def no_model(*args, **kwargs):
        pytest.fail(
            "insufficient run-copy/checkpoint space reached model initialization"
        )

    monkeypatch.setattr(trainer.PyTorchEngine, "initialize", no_model)
    with pytest.raises(OSError, match="storage preflight failed"):
        trainer.train(config, run_id="insufficient")
    assert not (config.logging.root_dir / "insufficient").exists()


def test_staged_training_measures_bound_assets_without_original_cache(
    tmp_path, monkeypatch
):
    config = configured(tmp_path, monkeypatch)
    bundle = stage(config, tmp_path / "stage", through="validate")
    assets = bundle / "assets"
    staged_config = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(update={"cache_dir": assets}),
            "tokenizer": config.tokenizer.model_copy(
                update={"path": assets / "tokenizer.json"}
            ),
        }
    )
    proofs = {}
    from sparselab.staging import _load_inventory_prepared, verify_stage_bundle

    verify_stage_bundle(bundle, config, _proofs=proofs)
    data = _load_inventory_prepared(assets, proofs, byte_enabled=False)
    checks = workspace_preflight.training_storage_checks(
        staged_config, verified_prepared=data.receipt
    )
    needed = checks[0].projected_bytes + checks[0].reserve_bytes
    shutil.move(config.tokenizer.path, tmp_path / "retained-tokenizer.json")
    shutil.move(config.dataset.cache_dir, tmp_path / "retained-cache")
    available_space(monkeypatch, needed + 4096)

    def no_preparation(*args, **kwargs):
        pytest.fail("staged training prepared inputs again")

    monkeypatch.setattr(trainer, "prepare_data", no_preparation)
    run_id = trainer.train(config, run_id="staged-storage", stage_bundle=bundle)
    root = config.logging.root_dir / run_id
    assert json.loads((root / "progress.json").read_text())["status"] == "completed"
    saved = yaml.safe_load((root / "resolved_config.yaml").read_text())
    assert Path(saved["dataset"]["cache_dir"]) == config.dataset.cache_dir
    assert (root / "data/train.npy").read_bytes() == (
        assets / "data/train.npy"
    ).read_bytes()


def test_fresh_preparation_retains_preallocation_space_check(tmp_path, monkeypatch):
    config = configured(tmp_path, monkeypatch)
    available_space(monkeypatch, 1)

    def no_allocation(*args, **kwargs):
        pytest.fail("insufficient fresh-cache space reached token array allocation")

    monkeypatch.setattr(packing, "_collect", no_allocation)
    monkeypatch.setattr(trainer.PyTorchEngine, "initialize", no_allocation)
    with pytest.raises(OSError, match="storage preflight failed"):
        trainer.train(config, run_id="fresh-insufficient")
    assert not list(config.dataset.cache_dir.rglob("*.npy"))
    assert not (config.logging.root_dir / "fresh-insufficient").exists()


@pytest.mark.parametrize("mode", ["plain", "staged", "staged-with-bundle"])
def test_interrupted_resume_uses_saved_tokenizer(tmp_path, monkeypatch, mode):
    config = configured(tmp_path, monkeypatch)
    bundle = (
        stage(config, tmp_path / "resume-stage", through="validate")
        if mode != "plain"
        else None
    )
    if bundle is not None:
        shutil.move(config.tokenizer.path, tmp_path / "retained-tokenizer.json")
        shutil.move(config.dataset.cache_dir, tmp_path / "retained-cache")
    part_id = trainer.train(
        config, run_id="part", stop_after_step=2, stage_bundle=bundle
    )
    part = config.logging.root_dir / part_id
    if bundle is None:
        shutil.move(config.tokenizer.path, tmp_path / "retained-tokenizer.json")
        shutil.move(config.dataset.cache_dir, tmp_path / "retained-cache")
    assert not config.tokenizer.path.exists()
    assert not config.dataset.cache_dir.exists()
    assert (part / "tokenizer.json").is_file()

    def no_preparation(*args, **kwargs):
        pytest.fail("resume rebuilt already saved training inputs")

    monkeypatch.setattr(trainer, "prepare_data", no_preparation)
    run_id = trainer.train(
        config,
        run_id="resumed",
        resume=part / "checkpoints/latest.json",
        stage_bundle=bundle if mode == "staged-with-bundle" else None,
    )
    root = config.logging.root_dir / run_id
    progress = json.loads((root / "progress.json").read_text())
    assert progress["status"] == "completed"
    assert (progress["step"], progress["tokens_seen"]) == (3, 96)
    assert (root / "data/train.npy").read_bytes() == (
        part / "data/train.npy"
    ).read_bytes()
    assert (root / "tokenizer.json").read_bytes() == (
        part / "tokenizer.json"
    ).read_bytes()


def test_cancel_during_data_open_stops_before_model(tmp_path, monkeypatch):
    config = configured(tmp_path, monkeypatch)
    cancel = tmp_path / "cancel"
    prepare = trainer.prepare_data

    def prepared_then_cancel(*args, **kwargs):
        data = prepare(*args, **kwargs)
        cancel.write_text("cancel after native input loading")
        return data

    def no_model(*args, **kwargs):
        pytest.fail("data-open cancellation reached model initialization")

    monkeypatch.setattr(trainer, "prepare_data", prepared_then_cancel)
    monkeypatch.setattr(trainer.PyTorchEngine, "initialize", no_model)
    with pytest.raises(InterruptedError, match="before run creation"):
        trainer.train(config, run_id="cancelled-data-open", cancel_path=cancel)
    assert not (config.logging.root_dir / "cancelled-data-open").exists()
    assert cancel.read_text() == "cancel after native input loading"
