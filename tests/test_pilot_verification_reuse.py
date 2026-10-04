"""Operational stage/pilot proof reuse never changes the cold verification contract."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from test_staging import _config

from sparselab.staging import (
    materialize_prepared_inputs,
    stage,
    verify_prepared_inputs,
    verify_stage_bundle,
)
from sparselab.training import manifest as manifest_module
from sparselab.verification_proofs import verification_options


def test_prepared_inputs_signed_reuse_and_explicit_cold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    prepared = materialize_prepared_inputs(config, tmp_path / "prepared")
    private_config = tmp_path / "private-config"
    private_config.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(private_config))
    options = verification_options(tmp_path)
    assert options["verification_mode"] == "verified_reuse"

    original = manifest_module.sha256_file
    hashed: list[Path] = []

    def count_hash(path: Path, **kwargs: object) -> str:
        hashed.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(manifest_module, "sha256_file", count_hash)
    verify_prepared_inputs(prepared, config, **options)
    first = [p for p in hashed if p.suffix == ".npy"]
    assert first
    hashed.clear()
    verify_prepared_inputs(prepared, config, **options)
    assert not [p for p in hashed if p.suffix == ".npy"]

    hashed.clear()
    verify_prepared_inputs(prepared, config)
    assert sorted(p.name for p in hashed if p.suffix == ".npy") == sorted(
        p.name for p in first
    )

    array = prepared / "assets" / "data" / "train.npy"
    metadata = array.stat()
    content = bytearray(array.read_bytes())
    content[-1] ^= 1
    array.write_bytes(content)
    os.utime(array, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_prepared_inputs(prepared, config, **options)


def test_cold_mode_does_not_use_proof_even_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    prepared = materialize_prepared_inputs(config, tmp_path / "prepared")
    private_config = tmp_path / "private-config"
    private_config.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(private_config))
    options = verification_options(tmp_path)
    verify_prepared_inputs(prepared, config, **options)
    store = options["proof_store"]
    before = (store.hits, store.misses, store.recorded)
    cold = verification_options(tmp_path, cold=True)
    assert cold == {"proof_store": None, "verification_mode": "cold"}
    verify_prepared_inputs(prepared, config, **cold)
    assert (store.hits, store.misses, store.recorded) == before


def test_stage_bundle_warm_skips_payload_but_cold_rehashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    bundle = stage(config, tmp_path / "stage", through="validate")
    private_config = tmp_path / "private-config"
    private_config.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(private_config))
    options = verification_options(tmp_path)
    original = manifest_module.sha256_file
    hashed: list[Path] = []

    def count_hash(path: Path, **kwargs: object) -> str:
        hashed.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(manifest_module, "sha256_file", count_hash)
    first = verify_stage_bundle(bundle, config, **options)
    assert [path for path in hashed if path.suffix == ".npy"]
    hashed.clear()
    second = verify_stage_bundle(bundle, config, **options)
    assert first["sha256"] == second["sha256"]
    assert not [path for path in hashed if path.suffix == ".npy"]
    hashed.clear()
    cold = verify_stage_bundle(bundle, config)
    assert cold["sha256"] == first["sha256"]
    assert [path for path in hashed if path.suffix == ".npy"]
