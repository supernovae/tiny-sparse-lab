from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_training import config

from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer
from sparselab.training.manifest import canonical_json, sha256_file


def _resign(manifest: dict[str, object]) -> None:
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()


def test_resigned_metadata_tamper_cannot_reuse_cache(tmp_path: Path) -> None:
    configured = config(tmp_path)
    prepared = prepare_data(configured, load_tokenizer(configured.tokenizer.path))
    manifest_path = prepared.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["train"]["shape"] = [999]  # type: ignore[index]
    _resign(manifest)
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")

    with pytest.raises(ValueError, match="cache integrity"):
        prepare_data(configured, load_tokenizer(configured.tokenizer.path))


def test_resigned_dtype_change_cannot_reuse_cache(tmp_path: Path) -> None:
    configured = config(tmp_path)
    prepared = prepare_data(configured, load_tokenizer(configured.tokenizer.path))
    array_path = prepared.root / "train.npy"
    np.save(array_path, np.load(array_path).astype(np.int64), allow_pickle=False)
    manifest_path = prepared.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = manifest["train"]
    assert isinstance(train, dict)
    train.update({"dtype": "int64", "sha256": sha256_file(array_path)})
    _resign(manifest)
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")

    with pytest.raises(ValueError, match="cache integrity"):
        prepare_data(configured, load_tokenizer(configured.tokenizer.path))
