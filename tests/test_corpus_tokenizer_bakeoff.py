"""Corpus Forge tokenizer fitting and held-out family selection invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sparselab.config.models import TokenizerTrainConfig
from sparselab.corpus.tokenizer_bakeoff import (
    GROUPS,
    Declaration,
    _bind_manifest,
    _dataset,
    _selection,
    choose_candidate,
)
from sparselab.data.tokenizer import train_tokenizer, verify_tokenizer_artifact


def _spec() -> Declaration:
    return Declaration(
        schema_version=1,
        release_path="release",
        vocab_sizes=(16384, 24576, 32768),
        max_fit_bytes=268435456,
        eval_split="validation",
        eval_max_docs_per_group=200,
        groups=GROUPS,
        near_best_ratio=0.98,
    )


def _fixture(tmp_path: Path, *, overlap: bool = False) -> Path:
    release = tmp_path / "release"
    (release / "lm").mkdir(parents=True)
    docs = []
    for split in ("train", "validation"):
        ids = []
        for group in GROUPS:
            text = f"{split}/{group}: one source document."
            ids.append(f"{split}-{group}")
            docs.append(
                {
                    "document_id": ids[-1],
                    "split": split,
                    "text": text,
                    "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "document_kind": group,
                    "source_location": f"{group}.txt",
                    "source_family": ("train" if overlap else split) + "_family",
                }
            )
        (release / "lm" / f"{split}.lineage.jsonl").write_text(
            "".join(
                json.dumps({"record_id": key, "split": split}) + "\n" for key in ids
            )
        )
    (release / "documents.jsonl").write_text(
        "".join(json.dumps(doc) + "\n" for doc in docs)
    )
    return release


def test_selection_stable_per_kind_and_family_disjoint(tmp_path: Path) -> None:
    release = _fixture(tmp_path)
    first = _selection(release, _spec())
    second = _selection(release, _spec())
    assert first == second
    fit, heldout, train = first
    for group in GROUPS:
        assert len(fit[group]) == len(heldout[group]) == len(train[group]) == 1
        assert fit[group][0]["split"] == "train"
        assert heldout[group][0]["split"] == "validation"
        assert fit[group][0]["source_family"] != heldout[group][0]["source_family"]


def test_selection_rejects_shared_family(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="families overlap"):
        _selection(_fixture(tmp_path, overlap=True), _spec())


def test_selection_rejects_shared_content_across_families(tmp_path: Path) -> None:
    release = _fixture(tmp_path)
    path = release / "documents.jsonl"
    docs = [json.loads(line) for line in path.read_text().splitlines()]
    docs[9]["text"] = docs[0]["text"]
    docs[9]["content_sha256"] = docs[0]["content_sha256"]
    path.write_text("".join(json.dumps(doc) + "\n" for doc in docs))
    with pytest.raises(ValueError, match="share normalized content"):
        _selection(release, _spec())


def test_selection_rule_near_best_boundary() -> None:
    candidates = [
        {"vocab_size": 16384, "weighted_bytes_per_token": 1.959},
        {"vocab_size": 24576, "weighted_bytes_per_token": 1.96},
        {"vocab_size": 32768, "weighted_bytes_per_token": 2.0},
    ]
    assert choose_candidate(candidates) == 24576
    candidates[0]["weighted_bytes_per_token"] = 1.96
    assert choose_candidate(candidates) == 16384


def test_real_tokenizer_manifest_tamper_rejected(tmp_path: Path) -> None:
    sample = tmp_path / "fit.jsonl"
    validation = tmp_path / "validation.jsonl"
    sample.write_text(
        "".join(
            json.dumps({"text": f"example {i}: some text and digits {i * 37}"}) + "\n"
            for i in range(80)
        )
    )
    validation.write_text(json.dumps({"text": "independent held-out text"}) + "\n")
    dataset = _dataset(
        sample,
        validation,
        hashlib.sha256(sample.read_bytes()).hexdigest(),
        sum(
            len(json.loads(line)["text"].encode())
            for line in sample.read_text().splitlines()
        ),
        80,
    )
    config = TokenizerTrainConfig(
        schema_version=1,
        vocab_size=260,
        max_documents=80,
        output_dir=tmp_path / "candidate",
        dataset=dataset,
    )
    artifact = train_tokenizer(config)
    binding = {"release_id": "a" * 64, "sample_receipt_sha256": "b" * 64}
    manifest = _bind_manifest(artifact, binding)
    assert (
        verify_tokenizer_artifact(
            artifact,
            source="local_text",
            revision=dataset.revision,
            vocab_size=260,
            dataset=dataset,
        )
        == manifest
    )
    path = artifact.with_name("tokenizer_manifest.json")
    path.write_text(
        path.read_text().replace(
            '"release_id":"' + "a" * 64, '"release_id":"' + "c" * 64
        )
    )
    with pytest.raises(ValueError, match="incompatible Corpus Forge identity"):
        _bind_manifest(artifact, binding)
