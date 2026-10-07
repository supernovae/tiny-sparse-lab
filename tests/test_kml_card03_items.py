"""Offline authoring checks for Card 03's held-out evaluation manifest."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from sparselab.evaluation import kml_card03_items as authoring
from sparselab.training.manifest import sha256_file


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path, monkeypatch):
    release = tmp_path / "release"
    release.mkdir()
    release_id = "a" * 64
    monkeypatch.setattr(
        authoring, "verify_release", lambda path: {"release_id": release_id}
    )
    docs = [
        {
            "document_id": "test-1",
            "source_id": "source",
            "source_revision": "rev",
            "content_sha256": hashlib.sha256(b"The alarm was red.").hexdigest(),
            "text": "The alarm was red.",
            "split": "test",
            "drop_reason": None,
        },
        {
            "document_id": "test-2",
            "source_id": "source",
            "source_revision": "rev",
            "content_sha256": hashlib.sha256(b"The alarm was blue.").hexdigest(),
            "text": "The alarm was blue.",
            "split": "test",
            "drop_reason": None,
        },
        {
            "document_id": "train-1",
            "source_id": "source",
            "source_revision": "rev",
            "content_sha256": hashlib.sha256(b"Only training prose.").hexdigest(),
            "text": "Only training prose.",
            "split": "train",
            "drop_reason": None,
        },
    ]
    with (release / "documents.jsonl").open("w", encoding="utf-8") as stream:
        for doc in docs:
            stream.write(json.dumps(doc) + "\n")
    with (release / "spans.jsonl").open("w", encoding="utf-8") as stream:
        for doc in docs:
            stream.write(
                json.dumps(
                    {"record_id": doc["document_id"], "snapshot_sha256": "b" * 64}
                )
                + "\n"
            )
    families = tmp_path / "families.jsonl"
    with families.open("w", encoding="utf-8") as stream:
        for doc in docs:
            stream.write(
                json.dumps(
                    {
                        "document_id": doc["document_id"],
                        "family_id": doc["document_id"],
                        "split": doc["split"],
                        "stratum": "incident",
                        "content_sha256": doc["content_sha256"],
                    }
                )
                + "\n"
            )
    chunks = []
    for doc in docs[:2]:
        chunk = {
            "document_id": doc["document_id"],
            "start": 0,
            "end": len(doc["text"]),
            "text_sha256": hashlib.sha256(doc["text"].encode()).hexdigest(),
        }
        chunk["id"] = authoring._sha(chunk)
        chunks.append(chunk)
    item = {
        "id": "item-1",
        "suite": "open_book",
        "category": "direct_extraction",
        "split": "test",
        "parent_family": "test-1",
        "parent_document_ids": ["test-1"],
        "source_versions": {
            "test-1": {
                "source_id": "source",
                "source_revision": "rev",
                "snapshot_sha256": "b" * 64,
            }
        },
        "question": "What color was the alarm?",
        "answerability": "answerable",
        "required_claims": ["The alarm was red."],
        "prohibited_claims": ["It was blue."],
        "acceptable_paraphrases": ["red"],
        "numeric_tolerance": None,
        "support_chunk_ids": [chunks[0]["id"]],
        "precedence_rule": None,
        "clarification_target": None,
        "answer_length_max_words": 12,
        "reviewer": "fixture-reviewer",
        "review_status": "reviewed",
        "controls": {
            "no_evidence": [],
            "gold": [chunks[0]["id"]],
            "plausible_wrong": [chunks[1]["id"]],
            "shuffled_absent": [chunks[1]["id"]],
        },
    }
    item["content_sha256"] = authoring._sha(item)
    draft = {
        "schema_version": 1,
        "protocol": "kml-card03-evaluation-v1",
        "release_id": release_id,
        "family_inventory_sha256": sha256_file(families),
        "decoder": {"temperature": 0.0, "max_new_tokens": 64},
        "evidence_token_budget": 1024,
        "chunks": chunks,
        "items": [item],
    }
    return release, families, draft


def test_reviewed_heldout_draft_binds_content_and_reports_incomplete(
    tmp_path, monkeypatch
):
    release, families, data = _fixture(tmp_path, monkeypatch)
    draft = tmp_path / "draft.json"
    _write(draft, data)
    output = tmp_path / "frozen.json"
    result = authoring.freeze_card03_items(
        release, families, draft, output, require_complete=False
    )
    assert result["category_denominators"] == {"open_book/direct_extraction": 1}
    assert result["complete_denominators"] is False
    assert result["content_sha256"] == authoring._sha(
        {key: value for key, value in result.items() if key != "content_sha256"}
    )
    assert json.loads(output.read_text())["draft_sha256"] == sha256_file(draft)
    with pytest.raises(FileExistsError):
        authoring.freeze_card03_items(
            release, families, draft, output, require_complete=False
        )
    with pytest.raises(ValueError, match="200 closed-book and 400 open-book"):
        authoring.freeze_card03_items(release, families, draft, tmp_path / "full.json")


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda d: d["items"][0].update(parent_document_ids=["train-1"]), "held-out"),
        (lambda d: d["chunks"][0].update(text_sha256="0" * 64), "exact held-out"),
        (
            lambda d: d["items"][0]["source_versions"]["test-1"].update(
                snapshot_sha256="0" * 64
            ),
            "source versions",
        ),
        (lambda d: d["items"][0]["controls"].update(gold=[]), "gold and no-evidence"),
        (lambda d: d.update(release_id="0" * 64), "verified release"),
    ],
)
def test_rejects_unbound_or_leaking_items(tmp_path, monkeypatch, mutation, message):
    release, families, data = _fixture(tmp_path, monkeypatch)
    data = deepcopy(data)
    mutation(data)
    if message != "verified release":
        data["items"][0]["content_sha256"] = authoring._sha(
            {
                key: value
                for key, value in data["items"][0].items()
                if key != "content_sha256"
            }
        )
    draft = tmp_path / "draft.json"
    _write(draft, data)
    with pytest.raises(ValueError, match=message):
        authoring.freeze_card03_items(
            release, families, draft, tmp_path / "frozen.json", require_complete=False
        )
    assert not (tmp_path / "frozen.json").exists()
