from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_staging import _config

import sparselab.staging as staging_module
from sparselab.config.models import MemoryConfig, RunConfig
from sparselab.staging import _read_sealed, stage
from sparselab.training import trainer as trainer_module
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.metrics import ExperimentStore
from sparselab.training.trainer import train


def _bounded_config(root: Path, *, policy: str = "enforce", budget: int | None = 1):
    raw = _config(root).model_dump(mode="json")
    raw["runtime"]["memory"].update(estimate_policy=policy, budget_bytes=budget)
    raw["training"].update(max_steps=1, max_tokens=16)
    raw["optimizer"]["warmup_steps"] = 0
    return RunConfig.model_validate(raw)


def _records(config: RunConfig):
    return ExperimentStore(config.logging.root_dir).export_records()["records"]


@pytest.mark.parametrize("value", ["warn", "disabled", False, None])
def test_estimate_policy_is_explicit_and_rejects_unknown_values(value):
    assert MemoryConfig().estimate_policy == "enforce"
    with pytest.raises(ValueError, match="estimate_policy"):
        MemoryConfig.model_validate({"estimate_policy": value})


def test_policy_serialization_preserves_default_and_records_observation():
    default = MemoryConfig().model_dump(mode="json")
    assert "estimate_policy" not in default
    assert MemoryConfig(estimate_policy="enforce").model_dump(mode="json") == default
    observed = MemoryConfig(estimate_policy="observe").model_dump(mode="json")
    assert observed["estimate_policy"] == "observe"
    assert MemoryConfig.model_validate(observed).estimate_policy == "observe"
    assert MemoryConfig.model_validate(default).estimate_policy == "enforce"


def test_default_training_still_refuses_estimated_exceedance(tmp_path):
    config = _bounded_config(tmp_path)
    with pytest.raises(MemoryError, match="estimated training memory exceeds"):
        train(config, run_id="enforced")
    assert not (config.logging.root_dir / "enforced/checkpoints/latest.json").exists()


def test_observe_trains_and_retains_estimate_metrics_and_configuration(
    tmp_path, caplog
):
    config = _bounded_config(tmp_path, policy="observe")
    before = config.model_dump(mode="json")
    with caplog.at_level(logging.WARNING):
        run_id = train(config, run_id="observed")
    records = _records(config)
    created = next(row["payload"] for row in records if row["kind"] == "run_created")
    assert created["metadata"]["memory_estimate"]["result"] == "LIKELY_TO_EXCEED"
    assert created["config"]["runtime"]["memory"]["estimate_policy"] == "observe"
    for key in ("model", "attention", "optimizer", "training", "seed"):
        assert created["config"][key] == before[key]
    assert config.model_dump(mode="json") == before
    assert "estimate_policy=observe" in caplog.text
    assert "memory/process_rss_bytes" in json.dumps(records)
    finished = next(row["payload"] for row in records if row["kind"] == "run_finished")
    assert finished["status"] == "completed"
    run = config.logging.root_dir / run_id
    saved = CheckpointManager(run).load(run / "checkpoints/latest.json")
    assert (saved.step, saved.tokens_seen) == (1, 16)


def test_observe_preserves_allocator_failure_and_failed_record(tmp_path, monkeypatch):
    config = _bounded_config(tmp_path, policy="observe")

    def fail_update(*args, **kwargs):
        raise torch.OutOfMemoryError("fixture allocator exhausted")

    monkeypatch.setattr(trainer_module.PyTorchEngine, "train_update", fail_update)
    with pytest.raises(torch.OutOfMemoryError, match="fixture allocator exhausted"):
        train(config, run_id="allocator-failed")
    records = _records(config)
    finished = next(row["payload"] for row in records if row["kind"] == "run_finished")
    assert finished["status"] == "failed"
    assert "fixture allocator exhausted" in json.dumps(records)
    run = config.logging.root_dir / "allocator-failed"
    saved = CheckpointManager(run).load(run / "checkpoints/latest.json")
    assert (saved.step, saved.tokens_seen) == (0, 0)


def test_observe_staging_keeps_exceedance_and_publishes_unchanged_bundle(
    tmp_path, caplog
):
    config = _bounded_config(tmp_path, policy="observe")
    before = config.model_dump(mode="json")
    output = tmp_path / "observational-stage"
    with caplog.at_level(logging.WARNING):
        stage(config, output, through="validate")
    report = _read_sealed(output / "stage.json")
    assert report["status"] == "complete"
    assert report["estimate"]["result"] == "LIKELY_TO_EXCEED"
    assert report["requested_config"] == before
    assert config.model_dump(mode="json") == before
    assert _read_sealed(output / "bundle.json")["pilot_reports"] == []
    assert not (output / "proposal.yaml").exists()
    assert "estimate_policy=observe" in caplog.text


@pytest.mark.parametrize("policy", ["enforce", "observe"])
def test_pilot_peak_above_planning_ceiling_respects_estimate_policy(
    tmp_path, monkeypatch, caplog, policy
):
    config = _bounded_config(tmp_path, policy=policy, budget=None)
    original = staging_module.calibrated_estimate

    def optimistic_estimate(*args, **kwargs):
        # Only the parent's forecast is substituted. The real bounded CPU pilot
        # retains native training, measurements, sealed evidence and validation.
        return replace(
            original(*args, **kwargs),
            result="LIKELY_TO_FIT",
            capacity_ceiling_bytes=1,
        )

    monkeypatch.setattr(staging_module, "calibrated_estimate", optimistic_estimate)
    output = tmp_path / "peak-stage"
    with caplog.at_level(logging.WARNING):
        if policy == "enforce":
            with pytest.raises(MemoryError, match="smoke measured or calibrated"):
                stage(config, output, through="smoke")
            assert _read_sealed(output / "stage.json")["status"] == "failed"
            assert not (output / "bundle.json").exists()
        else:
            stage(config, output, through="smoke")
            report = _read_sealed(output / "stage.json")
            assert report["status"] == "complete"
            bundle = _read_sealed(output / "bundle.json")
            assert bundle["pilot_reports"][0]["report"]["observed_peak_bytes"] > 1
            assert "estimate_policy=observe" in caplog.text
