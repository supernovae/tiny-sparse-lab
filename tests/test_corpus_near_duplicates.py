"""Bounded read-only candidate audit for normalized corpus documents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from sparselab.corpus.cli import add_commands
from sparselab.corpus.near_duplicates import audit_release, candidates


def _document(identifier: str, split: str, text: str) -> dict[str, str]:
    return {
        "document_id": identifier,
        "split": split,
        "text": text,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def test_near_duplicate_reports_lexical_evidence_but_not_unrelated_text() -> None:
    common = (
        "The controller schedules workload replicas when available resources "
        "satisfy requested limits. "
    ) * 8
    rows = [
        _document("original", "train", common + "Inspect the deployment status."),
        _document("similar", "test", common + "Inspect the rollout status."),
        _document(
            "unrelated",
            "validation",
            "Bright planets orbit distant stars and astronomers chart their motions. "
            * 8,
        ),
    ]
    result = candidates(rows)
    assert result == candidates(reversed(rows))
    assert result["algorithm"] == "word_shingle_minhash_lsh_v1"
    assert result["cross_split_comparisons"] >= 1
    assert len(result["candidates"]) == 1
    match = result["candidates"][0]
    assert (match["left_document_id"], match["right_document_id"]) == (
        "original",
        "similar",
    )
    assert {match["left_split"], match["right_split"]} == {"train", "test"}
    assert 0.65 <= match["jaccard"] < 1
    assert match["shared_shingles"] < match["union_shingles"]
    assert not match["identical_content_sha256"]


def test_near_duplicate_caps_fail_instead_of_truncating() -> None:
    text = "Important deployment rollback verification evidence remains available. " * 4
    rows = [
        _document("a", "train", text),
        _document("b", "validation", text),
        _document("c", "test", text),
    ]
    with pytest.raises(ValueError, match="document cap"):
        candidates(rows, max_documents=2)
    with pytest.raises(ValueError, match="shingle cap"):
        candidates(rows, max_shingles_per_document=2)
    with pytest.raises(ValueError, match="bucket cap"):
        candidates(rows, max_bucket_documents=1)
    with pytest.raises(ValueError, match="comparison cap"):
        candidates(rows, max_comparisons=1)
    with pytest.raises(ValueError, match="candidate cap"):
        candidates(rows, max_candidates=1)


def test_near_duplicate_stream_fails_before_materializing_unbounded_text() -> None:
    consumed = 0

    def documents():
        nonlocal consumed
        for index in range(1_000):
            consumed += 1
            yield _document(str(index), "train", "x" * 1_000)

    with pytest.raises(ValueError, match="text byte cap"):
        candidates(documents(), max_input_text_bytes=2_500)
    assert consumed == 3


def test_native_cli_exposes_fail_closed_scale_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser = argparse.ArgumentParser()
    add_commands(parser.add_subparsers(dest="command", required=True))
    arguments = parser.parse_args(
        [
            "corpus",
            "near-duplicates",
            str(tmp_path),
            "--max-documents",
            "2",
            "--max-input-text-bytes",
            "4096",
            "--max-total-shingles",
            "100",
            "--max-comparisons",
            "5",
            "--max-candidates",
            "5",
        ]
    )
    assert arguments.max_documents == 2
    assert arguments.max_input_text_bytes == 4096
    assert arguments.max_total_shingles == 100
    rows = [
        _document("a", "train", "one two three four"),
        _document("b", "validation", "one two three four"),
        _document("c", "test", "one two three four"),
    ]
    (tmp_path / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    monkeypatch.setattr(
        "sparselab.corpus.release.verify_release", lambda _: {"release_id": "fixture"}
    )
    with pytest.raises(ValueError, match="document cap"):
        audit_release(tmp_path, max_documents=arguments.max_documents)
