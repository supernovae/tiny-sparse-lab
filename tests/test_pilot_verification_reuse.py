"""Operational stage/pilot proof reuse never changes the cold verification contract."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from test_staging import _config

from sparselab.staging import (
    materialize_prepared_inputs,
    verify_prepared_inputs,
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
