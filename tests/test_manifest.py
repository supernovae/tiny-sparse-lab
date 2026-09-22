from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sparselab.runtime import RuntimeInfo
from sparselab.training.manifest import (
    ArtifactIdentity,
    RunManifest,
    architecture_sha256,
    canonical_json,
    config_sha256,
    read_manifest,
    write_manifest,
)


def _runtime() -> RuntimeInfo:
    return RuntimeInfo(
        "pytorch",
        "cpu",
        "cpu",
        0,
        "CPU",
        None,
        "test",
        None,
        None,
        "test",
        None,
        None,
        None,
        None,
        None,
        "test",
        (),
        (),
    )


def _config(root: Path) -> dict[str, object]:
    return {
        "schema_version": 2,
        "model": {
            "vocab_size": 512,
            "hidden_dim": 16,
            "memory_package_path": str(root / "package"),
        },
        "attention": {"kind": "dense", "rope_base": 10000.0},
        "tokenizer": {"path": str(root / "tokenizer.json")},
        "dataset": {"cache_dir": str(root / "cache")},
        "logging": {"root_dir": str(root / "runs")},
    }


def test_identity_ignores_machine_paths_but_binds_architecture() -> None:
    left, right = _config(Path("/machine-a")), _config(Path("/machine-b"))
    assert config_sha256(left) == config_sha256(right)
    assert architecture_sha256(left) == architecture_sha256(right)

    changed = _config(Path("/machine-a"))
    changed["model"] = {**changed["model"], "hidden_dim": 32}  # type: ignore[index]
    assert architecture_sha256(left) != architecture_sha256(changed)
    assert config_sha256(left) != config_sha256(changed)


def test_canonical_json_rejects_arbitrary_objects() -> None:
    with pytest.raises(TypeError):
        canonical_json(Path("not-json"))


def test_manifest_rejects_duplicate_nonfinite_and_unknown_versions(
    tmp_path: Path,
) -> None:
    artifact = ArtifactIdentity("assets/tokenizer.json", "a" * 64, 1)
    manifest = RunManifest(
        "run",
        "name",
        _runtime(),
        _config(tmp_path),
        _config(tmp_path),
        architecture_sha256(_config(tmp_path)),
        {"sha256": "source"},
        "worker",
        artifacts=(artifact,),
    )
    path = tmp_path / "manifest.json"
    digest = write_manifest(path, manifest)
    loaded = read_manifest(path)
    assert digest == hashlib.sha256(canonical_json(loaded)).hexdigest()
    assert loaded["artifacts"] == [
        {
            "relative_path": artifact.relative_path,
            "sha256": artifact.sha256,
            "size_bytes": 1,
        }
    ]

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"manifest_version":1,"manifest_version":1}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"manifest_version":1,"x":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        read_manifest(nonfinite)

    unknown = dict(loaded)
    unknown["manifest_version"] = 99
    unknown["sha256"] = hashlib.sha256(canonical_json(unknown)).hexdigest()
    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported manifest version"):
        read_manifest(unknown_path)


def test_resigned_architecture_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest = RunManifest(
        "run",
        "name",
        _runtime(),
        _config(tmp_path),
        _config(tmp_path),
        architecture_sha256(_config(tmp_path)),
        {"sha256": "source"},
        "worker",
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    resigned = json.loads(path.read_text(encoding="utf-8"))
    resigned["architecture_sha256"] = "0" * 64
    resigned["sha256"] = hashlib.sha256(
        canonical_json(
            {key: value for key, value in resigned.items() if key != "sha256"}
        )
    ).hexdigest()
    path.write_bytes(canonical_json(resigned) + b"\n")

    with pytest.raises(ValueError, match="architecture hash mismatch"):
        read_manifest(path)


@pytest.mark.parametrize(
    "artifacts",
    [
        [
            {"relative_path": "same.json", "sha256": "a" * 64, "size_bytes": 1},
            {"relative_path": "same.json", "sha256": "b" * 64, "size_bytes": 2},
        ],
        [{"relative_path": "", "sha256": "a" * 64, "size_bytes": True}],
    ],
)
def test_resigned_invalid_artifact_inventory_is_rejected(
    tmp_path: Path, artifacts: list[dict[str, object]]
) -> None:
    manifest = RunManifest(
        "run",
        "name",
        _runtime(),
        _config(tmp_path),
        _config(tmp_path),
        architecture_sha256(_config(tmp_path)),
        {"sha256": "source"},
        "worker",
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    resigned = json.loads(path.read_text(encoding="utf-8"))
    resigned["artifacts"] = artifacts
    resigned["sha256"] = hashlib.sha256(
        canonical_json(
            {key: value for key, value in resigned.items() if key != "sha256"}
        )
    ).hexdigest()
    path.write_bytes(canonical_json(resigned) + b"\n")

    with pytest.raises(ValueError, match="artifact inventory"):
        read_manifest(path)


def test_legacy_v1_manifest_digest_is_verified_without_reinterpretation(
    tmp_path: Path,
) -> None:
    legacy = {
        "manifest_version": 1,
        "run_id": "historic",
        "requested_config_sha256": "historical-digest",
    }
    legacy["sha256"] = hashlib.sha256(canonical_json(legacy)).hexdigest()
    path = tmp_path / "legacy.json"
    path.write_bytes(canonical_json(legacy) + b"\n")

    assert read_manifest(path) == {
        "manifest_version": 1,
        "run_id": "historic",
        "requested_config_sha256": "historical-digest",
    }
