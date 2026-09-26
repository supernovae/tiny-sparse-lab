from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from sparselab.config.loading import load_config
from sparselab.config.models import RunConfig
from sparselab.model.portable_engram import (
    export_portable_engram,
    load_portable_engram,
)
from sparselab.research.learned_portability_campaign import (
    _digest,
    _learned_config,
    _materialize_control_package,
    _resume_config_for_checkpoint,
)
from sparselab.training.manifest import canonical_json


def test_control_package_provenance_is_shared_across_recipient_widths(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "exports" / "source-s17.engram"
    source_path.parent.mkdir(parents=True)
    source_table = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    export_portable_engram(source_table, source_path, ngram_size=3)

    package64 = _materialize_control_package(
        tmp_path,
        source_path,
        {
            "role": "recipient",
            "condition": "constant",
            "recipient": "width64",
            "seed": 17,
        },
    )
    package_sha = hashlib.sha256(package64.read_bytes()).hexdigest()

    package128 = _materialize_control_package(
        tmp_path,
        source_path,
        {
            "role": "recipient",
            "condition": "constant",
            "recipient": "width128",
            "seed": 17,
        },
    )

    provenance = json.loads(package128.with_suffix(".provenance.json").read_text())
    transformed = load_portable_engram(package128).table
    assert provenance["coordinate_id"] == "control-source-s17-constant"
    assert provenance["package"]["sha256"] == package_sha
    assert hashlib.sha256(package128.read_bytes()).hexdigest() == package_sha
    assert torch.equal(
        transformed,
        source_table.mean(dim=0, keepdim=True).expand_as(source_table),
    )


def test_resume_uses_parent_owned_asset_paths_without_changing_settings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    parent_run = root / "runs" / "source-run"
    parent_run.mkdir(parents=True)
    checkpoint = parent_run / "checkpoints" / "latest.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text("{}\n", encoding="utf-8")

    source_config = load_config(Path("configs/runtime_smoke_cpu.yaml"))
    parent_payload = source_config.model_dump(mode="json")
    parent_payload["tokenizer"]["path"] = str(parent_run / "tokenizer.json")
    (parent_run / "resolved_config.yaml").write_text(
        yaml.safe_dump(parent_payload), encoding="utf-8"
    )

    requested_payload = source_config.model_dump(mode="json")
    requested_payload["tokenizer"]["path"] = str(root / "tokenizer.json")
    requested_payload["name"] = "resume-child"
    requested_payload["logging"]["root_dir"] = str(root / "child-runs")
    requested = RunConfig.model_validate(requested_payload)
    resumed = _resume_config_for_checkpoint(root, requested, checkpoint)

    assert resumed.tokenizer.path == parent_run / "tokenizer.json"
    assert resumed.name == "resume-child"
    assert resumed.logging.root_dir == root / "child-runs"
    changed_payload = dict(requested_payload)
    changed_payload["seed"] += 1
    changed = RunConfig.model_validate(changed_payload)
    with pytest.raises(ValueError, match="saved coordinate"):
        _resume_config_for_checkpoint(root, changed, checkpoint)

def test_learned_config_keeps_document_sentinel_and_bounds_checkpoints(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    protocol = {
        "observation_steps": {
            "source_native": [0, 32, 128, 512, 2048, 8192],
            "preparation": [0, 128, 512, 2048],
            "adapter": [0, 8, 32, 128, 512, 2048, 8192],
        }
    }
    (root / "portability_protocol.json").parent.mkdir(parents=True)
    protocol_path = root / "portability_protocol.json"
    protocol_path.write_bytes(
        canonical_json({**protocol, "sha256": _digest(protocol)}) + b"\n"
    )

    run_root = root / "runs" / "source-run"
    data_root = run_root / "portability" / "data"
    data_root.mkdir(parents=True)
    train_path = data_root / "source_train.jsonl"
    validation_path = data_root / "preparation_validation.jsonl"
    train_path.write_text("{}\n" * 4097, encoding="utf-8")
    validation_path.write_text("{}\n" * 129, encoding="utf-8")
    manifest_path = run_root / "portability_manifest.json"

    config = _learned_config(
        root,
        {"role": "source", "condition": "source-real", "recipient": "source", "seed": 17},
        manifest_path,
        max_steps=32768,
        backend_override="cpu",
    )

    assert config.dataset.train_max_documents == len(train_path.read_text().splitlines())
    assert config.dataset.validation_max_documents == len(
        validation_path.read_text().splitlines()
    )

    assert config.evaluation.every_steps == 32768
    assert config.checkpoint.every_steps is None
    assert config.checkpoint.steps == (0, 32, 128, 512, 2048, 8192, 32768)
    assert config.checkpoint.keep_periodic is True
