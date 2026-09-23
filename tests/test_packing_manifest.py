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


def test_resigned_supervision_mask_tamper_cannot_reuse_cache(tmp_path: Path) -> None:
    configured = config(tmp_path)
    prepared = prepare_data(configured, load_tokenizer(configured.tokenizer.path))
    mask_path = prepared.root / "train_supervision.npy"
    mask = np.load(mask_path)
    np.save(mask_path, mask.astype(np.uint8), allow_pickle=False)
    manifest_path = prepared.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supervision = manifest["supervision"]
    assert isinstance(supervision, dict)
    train = supervision["train"]
    assert isinstance(train, dict)
    train.update({"dtype": "uint8", "sha256": sha256_file(mask_path)})
    _resign(manifest)
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")

    with pytest.raises(ValueError, match="cache integrity"):
        prepare_data(configured, load_tokenizer(configured.tokenizer.path))


@pytest.mark.parametrize("truncated", [False, True])
def test_supervision_never_crosses_roles_or_teaches_truncated_prompt_eos(
    tmp_path, truncated
):
    from tokenizers import Tokenizer, models, pre_tokenizers

    from sparselab.data.conversations import iter_rendered_conversations
    from sparselab.data.packing import _collect

    configured = config(tmp_path)
    path = tmp_path / "masked.jsonl"
    path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "loss_mode": "assistant_only",
                "messages": [
                    {"role": "user", "content": "question " * 20},
                    {"role": "assistant", "content": "answer"},
                ],
            }
        )
        + "\n"
    )
    rendered = next(iter_rendered_conversations(path))
    tokenizer = Tokenizer(
        models.WordLevel({"<unk>": 0, "<eos>": 1, rendered.text: 2}, unk_token="<unk>")
    )
    if truncated:
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    dataset = configured.dataset.model_copy(
        update={
            "source": "local_chat",
            "train_path": path,
            "validation_path": path,
            "train_max_documents": 1,
            "train_max_tokens": 4 if truncated else 100,
        }
    )
    _, mask, _, _ = _collect(dataset, tokenizer, "train")
    if truncated:
        assert not mask.any()
    else:
        # A tokenizer token spanning user, role marker and answer cannot be an
        # assistant-only target. The completed document's EOS remains supervised.
        assert mask.tolist() == [False, True]
