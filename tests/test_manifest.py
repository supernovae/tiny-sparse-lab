from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sparselab.config.loading import load_config
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


def test_legacy_final_memory_injection_preserves_hashes_and_embedding_binds() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "runtime_smoke_cpu.yaml": (
            "2843e32cd3b64ec6049a20881ada0512e561b9fa4c1b05dcc8910dde01390c4f",
            "abee11a5910dcd65942047b98a65b1c600125c7142fae5099c80771d9eebc949",
        ),
        "context_study_dense_s17_b24k.yaml": (
            "93db49867e51f63089c903ae02d913bc28ab3d8b9e9ae6d30305c0abb9d8ce9f",
            "32569ae52919b99b04ddb4882dd4c74536998a058a418db48e9c6e4bd8dcfaba",
        ),
        "context_study_engram_s17_b24k.yaml": (
            "0573c402becc5129e8c7185d5c8be0afb8612b10333e311f505a947c3f8d8181",
            "9da4c0a2159a703bd0ddcdcc9d6f53b414833793d3ab2a039f47e70f719e9af2",
        ),
    }
    for filename, digests in expected.items():
        config = load_config(root / "configs" / filename).model_dump(mode="json")
        assert (config_sha256(config), architecture_sha256(config)) == digests

    legacy = load_config(
        root / "configs" / "context_study_engram_s17_b24k.yaml"
    ).model_dump(mode="json")
    explicit_final = {
        **legacy,
        "model": {**legacy["model"], "memory_injection": "final"},
    }
    assert config_sha256(explicit_final) == config_sha256(legacy)
    assert architecture_sha256(explicit_final) == architecture_sha256(legacy)
    assert architecture_sha256(explicit_final["model"]) == architecture_sha256(
        legacy["model"]
    )
    assert explicit_final["model"]["memory_injection"] == "final"

    embedded = {
        **legacy,
        "model": {**legacy["model"], "memory_injection": "embedding"},
    }
    assert config_sha256(embedded) != config_sha256(legacy)
    assert architecture_sha256(embedded) != architecture_sha256(legacy)


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
