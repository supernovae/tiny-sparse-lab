"""Prepared-array trust is bounded to an in-process write or byte scan."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from test_training import config

from sparselab.data import packing
from sparselab.data.packing import load_prepared_data, prepare_data
from sparselab.data.tokenizer import load_tokenizer
from sparselab.data.verification import VerifiedFile, VerifiedPreparedData, verify_file
from sparselab.training import manifest as manifest_module


def test_write_has_zero_array_rereads_and_cold_cache_hashes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = config(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    actual_hash = manifest_module.sha256_file
    reads: list[str] = []
    wrapper_reads: list[str] = []

    def counted(path: Path) -> str:
        if path.suffix == ".npy":
            reads.append(path.name)
        return actual_hash(path)

    original_wrapper = packing._sha256

    def counted_wrapper(path: Path) -> str:
        wrapper_reads.append(path.name)
        return original_wrapper(path)

    monkeypatch.setattr(manifest_module, "sha256_file", counted)
    monkeypatch.setattr(packing, "_sha256", counted_wrapper)
    created = prepare_data(run, tokenizer)
    assert reads == wrapper_reads == []
    assert created.manifest["supervision"] == {"kind": "all_tokens"}
    assert created.train_supervision is created.validation_supervision is None
    assert not list(created.root.glob("*supervision.npy"))
    reference = io.BytesIO()
    np.save(reference, np.asarray(created.train), allow_pickle=False)
    assert (created.root / "train.npy").read_bytes() == reference.getvalue()
    assert (
        load_prepared_data(
            created.root,
            byte_enabled=False,
            verification="structural",
            receipt=created.receipt,
        ).receipt.manifest_sha256
        == created.receipt.manifest_sha256
    )
    assert reads == []
    cold = load_prepared_data(created.root, byte_enabled=False)
    assert sorted(reads) == sorted(wrapper_reads) == ["train.npy", "validation.npy"]
    assert cold.receipt.root == created.root.resolve()
    reads.clear()
    wrapper_reads.clear()
    cached = prepare_data(run, tokenizer)
    assert sorted(reads) == sorted(wrapper_reads) == ["train.npy", "validation.npy"]
    assert cached.receipt.manifest_sha256 == created.receipt.manifest_sha256


def test_tamper_same_size_and_forged_persisted_receipt_rejected(tmp_path: Path) -> None:
    run = config(tmp_path)
    prepared = prepare_data(run, load_tokenizer(run.tokenizer.path))
    with pytest.raises(TypeError, match="cannot be constructed"):
        VerifiedPreparedData(
            prepared.root, prepared.receipt.manifest_sha256, prepared.receipt.proofs
        )
    with pytest.raises(TypeError, match="cannot be constructed"):
        VerifiedFile(
            prepared.root / "train.npy",
            prepared.manifest["train"]["sha256"],
            (0, 0, 0, 0),
        )
    with pytest.raises(TypeError, match="hash_file"):
        verify_file(prepared.root / "train.npy", hash_file=lambda _: "0" * 64)
    with pytest.raises(AttributeError):
        prepared.receipt.proofs["train.npy"].sha256 = "0" * 64
    path = prepared.root / "train.npy"
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        final = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([final[0] ^ 1]))
    forged = {
        "manifest_sha256": prepared.receipt.manifest_sha256,
        "files": {
            key: {"sha256": proof.sha256, "size_bytes": proof.size_bytes}
            for key, proof in prepared.receipt.proofs.items()
        },
    }
    (tmp_path / "forged-receipt.json").write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="stale or forged"):
        load_prepared_data(
            prepared.root,
            byte_enabled=False,
            verification="structural",
            receipt=prepared.receipt,
        )
    with pytest.raises(ValueError, match="digest mismatch"):
        load_prepared_data(prepared.root, byte_enabled=False)
    with pytest.raises(ValueError, match="sealed receipt"):
        load_prepared_data(
            prepared.root, byte_enabled=False, verification="structural", receipt=forged
        )


def test_unexpected_symlink_inventory_rejected(tmp_path: Path) -> None:
    run = config(tmp_path)
    prepared = prepare_data(run, load_tokenizer(run.tokenizer.path))
    (prepared.root / "unexpected.npy").symlink_to(prepared.root / "train.npy")
    with pytest.raises(ValueError, match="unexpected prepared file inventory"):
        load_prepared_data(
            prepared.root,
            byte_enabled=False,
            verification="structural",
            receipt=prepared.receipt,
        )
