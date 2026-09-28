from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from test_training import config

from sparselab.data.conversations import RenderedConversation
from sparselab.data.local_stories import LICENSE, REVISION
from sparselab.data.packing import _collect, _collect_streaming, prepare_data
from sparselab.data.tokenizer import load_tokenizer


def _stories(monkeypatch: pytest.MonkeyPatch, texts: list[str]) -> None:
    monkeypatch.setattr(
        "sparselab.data.packing._source_documents",
        lambda _config, _split: (
            RenderedConversation(text, ((0, len(text)),), "all_tokens")
            for text in texts
        ),
    )


def test_streaming_arrays_match_legacy_bytes_and_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = config(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    texts = [f"The tiny fox found a blue stone number {index}." for index in range(23)]
    _stories(monkeypatch, texts)
    dataset = run.dataset.model_copy(
        update={"train_max_documents": len(texts), "train_max_tokens": 10_000}
    )
    settings = {"byte_table_size": 1024, "byte_ngram_size": 3}
    expected_ids, expected_mask, expected_byte, expected_stats = _collect(
        dataset, tokenizer, "train", **settings
    )
    stats = _collect_streaming(
        dataset,
        tokenizer,
        "train",
        tmp_path,
        selected_documents=len(texts),
        **settings,
    )
    for name, values in (
        ("train.npy", expected_ids),
        ("train_supervision.npy", expected_mask),
        ("train_byte_addresses.npy", expected_byte),
    ):
        expected = io.BytesIO()
        np.save(expected, values, allow_pickle=False)
        assert (tmp_path / name).read_bytes() == expected.getvalue()
    assert stats["retained_documents"] == expected_stats["retained_documents"]
    assert stats["truncated_documents"] == 0
    assert stats["peak_host_rss_bytes"] > 0
    assert not list(tmp_path.glob("*.raw"))


def test_streaming_rejects_partial_story_and_short_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = config(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    texts = ["The fox ran to the river.", "The fox returned home safely."]
    _stories(monkeypatch, texts)
    dataset = run.dataset.model_copy(update={"train_max_tokens": 10_000})
    first_ids = tokenizer.encode(texts[0], add_special_tokens=False).ids
    capped = dataset.model_copy(update={"train_max_tokens": len(first_ids) + 1})
    (tmp_path / "capped").mkdir()
    (tmp_path / "short").mkdir()
    with pytest.raises(ValueError, match="token cap would truncate selected story"):
        _collect_streaming(
            capped, tokenizer, "train", tmp_path / "capped", selected_documents=2
        )
    with pytest.raises(ValueError, match="selected distinct stories; required 3"):
        _collect_streaming(
            dataset, tokenizer, "train", tmp_path / "short", selected_documents=3
        )


def test_source_verified_before_cached_packing_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = config(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    train = [f"The fox explored grove number {n}." for n in range(15)]
    validation = [f"The fox rested in meadow number {n}." for n in range(15)]
    snapshot = {
        "splits": {
            "train": {"count": len(train), "sha256": "original"},
            "validation": {"count": len(validation), "sha256": "original"},
        }
    }
    monkeypatch.setattr("sparselab.data.packing.verify_snapshot", lambda _cfg: snapshot)
    monkeypatch.setattr(
        "sparselab.data.packing._source_documents",
        lambda _cfg, split: (
            RenderedConversation(text, ((0, len(text)),), "all_tokens")
            for text in (train if split == "train" else validation)
        ),
    )
    dataset = run.dataset.model_copy(
        update={
            "source": "local_stories",
            "train_max_documents": 15,
            "validation_max_documents": 15,
            "train_max_tokens": 10_000,
            "validation_max_tokens": 10_000,
        }
    )
    local = run.model_copy(update={"dataset": dataset})
    first = prepare_data(local, tokenizer)
    assert prepare_data(local, tokenizer).root == first.root
    snapshot["splits"]["train"]["sha256"] = "changed"
    assert prepare_data(local, tokenizer).root != first.root

    def corrupted(_cfg):
        raise ValueError("snapshot content digest mismatch")

    monkeypatch.setattr("sparselab.data.packing.verify_snapshot", corrupted)
    with pytest.raises(ValueError, match="snapshot content digest mismatch"):
        prepare_data(local, tokenizer)


def test_cached_local_stories_rejects_corrupted_saved_source(
    tmp_path: Path,
) -> None:
    run = config(tmp_path)
    source = tmp_path / "snapshot"
    source.mkdir()
    splits: dict[str, dict[str, int | str]] = {}
    for split in ("train", "validation"):
        stories = [
            f"{split} story {number} has a happy ending." for number in range(15)
        ]
        path = source / f"{split}.jsonl"
        path.write_bytes(
            b"".join(
                (json.dumps({"ordinal": number, "text": story}) + "\n").encode("utf-8")
                for number, story in enumerate(stories)
            )
        )
        splits[split] = {
            "path": path.name,
            "count": len(stories),
            "text_bytes": sum(len(story.encode("utf-8")) for story in stories),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "content_sha256": hashlib.sha256(
                b"".join(
                    len(story.encode("utf-8")).to_bytes(8, "big")
                    + story.encode("utf-8")
                    for story in stories
                )
            ).hexdigest(),
            "source_records_consumed": len(stories),
            "duplicates": 0,
            "overlap": 0,
            "empty": 0,
        }
    (source / "excluded.jsonl").write_bytes(b"")
    manifest_path = source / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "roneneldan/TinyStories",
                "revision": REVISION,
                "license": LICENSE,
                "splits": splits,
                "excluded": {
                    "path": "excluded.jsonl",
                    "count": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    dataset = run.dataset.model_copy(
        update={
            "source": "local_stories",
            "revision": REVISION,
            "license": LICENSE,
            "train_path": source / "train.jsonl",
            "validation_path": source / "validation.jsonl",
            "source_manifest_path": manifest_path,
            "train_max_documents": 15,
            "validation_max_documents": 15,
            "train_max_tokens": 10_000,
            "validation_max_tokens": 10_000,
        }
    )
    local = run.model_copy(update={"dataset": dataset})
    tokenizer = load_tokenizer(run.tokenizer.path)
    cached = prepare_data(local, tokenizer)
    assert cached.manifest["train"]["retained_documents"] == 15
    assert prepare_data(local, tokenizer).root == cached.root
    train_path = source / "train.jsonl"
    train_path.write_bytes(train_path.read_bytes().replace(b"happy", b"quiet", 1))
    with pytest.raises(ValueError, match="snapshot content or digest mismatch"):
        prepare_data(local, tokenizer)
