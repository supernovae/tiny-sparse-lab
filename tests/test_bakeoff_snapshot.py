from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparselab.config.models import DatasetConfig
from sparselab.data import bakeoff as bakeoff_module
from sparselab.data.local_stories import LICENSE, REVISION, snapshot


def test_bakeoff_reads_only_development_validation_stories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    remote_calls: list[str] = []
    source = {
        "train": [
            "a little red fox ran along the path",
            "a little blue fox ran along the path",
            "a little green fox ran along the path",
        ],
        "validation": [
            "a little red fox returned",
            "a little blue fox returned",
            "SEALED TEST STORY",
        ],
    }

    def remote(_name: str, *, split: str, **_kwargs: object):
        remote_calls.append(split)
        return ({"text": story} for story in source[split])

    monkeypatch.setattr("datasets.load_dataset", remote)
    root = tmp_path / "snapshot"
    snapshot(root, train_count=3, validation_count=3)
    config = DatasetConfig(
        source="local_stories",
        revision=REVISION,
        license=LICENSE,
        cache_dir=tmp_path,
        train_path=root / "train.jsonl",
        validation_path=root / "validation.jsonl",
        source_manifest_path=root / "manifest.json",
        train_max_documents=3,
        validation_max_documents=3,
        train_max_tokens=1000,
        validation_max_tokens=1000,
    )
    monkeypatch.setattr(bakeoff_module, "VOCABS", (260, 261))
    monkeypatch.setattr(bakeoff_module, "TRAIN_DOCS", 3)
    monkeypatch.setattr(bakeoff_module, "TRAIN_BYTES", 1000)
    monkeypatch.setattr(bakeoff_module, "DEV_DOCS", 2)
    original = bakeoff_module.iter_documents
    observed: list[str] = []

    def documents(dataset: DatasetConfig, split: str):
        for story in original(dataset, split):
            if split == "validation":
                observed.append(story)
            yield story

    monkeypatch.setattr(bakeoff_module, "iter_documents", documents)
    receipt_path = bakeoff_module.bakeoff(config, tmp_path / "bakeoff")
    receipt = json.loads(receipt_path.read_text())
    assert remote_calls == ["train", "validation"]
    assert observed == source["validation"][:2] * 2
    assert receipt["selected_vocab_size"] in (260, 261)
    assert (
        len({entry["training_content_sha256"] for entry in receipt["candidates"]}) == 1
    )
