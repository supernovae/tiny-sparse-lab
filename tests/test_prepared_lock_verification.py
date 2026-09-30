from __future__ import annotations

from pathlib import Path

import pytest
from test_training import config as training_config

from sparselab.data import packing
from sparselab.data.tokenizer import load_tokenizer
from sparselab.experiments.lock import resolve_plan
from sparselab.experiments.plan import ExperimentPlan
from sparselab.training import manifest as manifest_module


def test_repeated_prepared_lock_inputs_scan_once_and_reject_tampering(
    tmp_path, monkeypatch
):
    config = training_config(tmp_path)
    prepared = packing.prepare_data(config, load_tokenizer(config.tokenizer.path))
    packed = {
        "kind": "prepared_data",
        "version": 1,
        "producer": "sparselab",
        "identifier": prepared.manifest["settings_sha256"],
        "sha256": prepared.manifest["manifest_sha256"],
        "path": str(prepared.root),
    }
    tokenizer = {
        "kind": "tokenizer",
        "version": 1,
        "producer": "sparselab",
        "identifier": config.tokenizer.path.parent.name,
        "sha256": manifest_module.sha256_file(config.tokenizer.path),
        "path": str(config.tokenizer.path),
    }
    plan = ExperimentPlan.model_validate(
        {
            "plan_version": 1,
            "id": "deep-once",
            "base_run": config,
            "artifacts": {
                "tokenizer": tokenizer,
                "packed": packed,
                "packed_again": packed,
            },
            "inputs": {
                "tokenizer": "tokenizer",
                "training": "packed",
                "audit": "packed_again",
            },
        }
    )
    original = manifest_module.sha256_file
    reads: list[Path] = []

    def counted(path, *args, **kwargs):
        if path.suffix == ".npy":
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(manifest_module, "sha256_file", counted)
    locked = resolve_plan(plan, tmp_path / "plan.yaml")
    assert locked.cells[0].config == config
    assert sorted(path.name for path in reads) == ["train.npy", "validation.npy"]
    reads.clear()
    assert (
        resolve_plan(plan, tmp_path / "plan.yaml").scientific_sha256
        == locked.scientific_sha256
    )
    assert sorted(path.name for path in reads) == ["train.npy", "validation.npy"]
    path = prepared.root / "train.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError):
        resolve_plan(plan, tmp_path / "plan.yaml")
