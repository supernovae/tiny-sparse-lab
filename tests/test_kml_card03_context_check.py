"""Exact-node CPU checks for the Card 03 evaluation context screen."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _checker():
    path = (
        Path(__file__).parents[1]
        / "experiments/research/kernel-memory-lab/corpus-scale/check-evaluation-context.py"
    )
    spec = importlib.util.spec_from_file_location("card03_context_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(tmp_path, monkeypatch):
    checker = _checker()
    family_inventory = tmp_path / "families.jsonl"
    family_inventory.write_text("fixture\n")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("fixture\n")
    docs = {
        "heldout": {
            "document_id": "heldout",
            "drop_reason": None,
            "text": "Mode is manual. Mode is automatic. No mode discussed.",
        }
    }
    chunks = {
        "gold": {"document_id": "heldout", "start": 0, "end": 15},
        "wrong": {"document_id": "heldout", "start": 16, "end": 34},
        "absent": {"document_id": "heldout", "start": 35, "end": 53},
    }
    monkeypatch.setattr(checker, "verify_release", lambda path: {"release_id": "r"})
    monkeypatch.setattr(
        checker,
        "_rows",
        lambda path: (
            [{"text": "unrelated training text"}]
            if path.name == "train.jsonl"
            else list(docs.values())
        ),
    )
    monkeypatch.setattr(
        checker,
        "_families",
        lambda path, rows: {"heldout": {"split": "test"}},
    )
    monkeypatch.setattr(checker, "_chunks", lambda raw, rows, families: chunks)
    monkeypatch.setattr(
        checker,
        "load_tokenizer",
        lambda path: SimpleNamespace(
            encode=lambda value, add_special_tokens: SimpleNamespace(ids=list(value))
        ),
    )
    item = {
        "id": "one",
        "content_sha256": "fixture",
        "suite": "open_book",
        "question": "What mode?",
        "parent_document_ids": ["heldout"],
        "answerability": "answerable",
        "required_claims": ["Mode is manual"],
        "acceptable_paraphrases": ["Manual mode"],
        "support_chunk_ids": ["gold"],
        "controls": {
            "no_evidence": [],
            "gold": ["gold"],
            "plausible_wrong": ["wrong"],
            "shuffled_absent": ["absent"],
        },
    }
    draft = {
        "release_id": "r",
        "family_inventory_sha256": checker.sha256_file(family_inventory),
        "evidence_token_budget": 512,
        "decoder": {"max_output_tokens": 128},
        "chunks": list(chunks.values()),
        "items": [item],
    }
    return checker, draft, family_inventory, tokenizer_path


def test_context_screen_counts_each_condition_and_preserves_generation_space(
    tmp_path, monkeypatch
) -> None:
    checker, draft, families, tokenizer = _inputs(tmp_path, monkeypatch)
    result = checker.measure(draft, tmp_path, families, tokenizer)
    assert result["hard_error_count"] == 0
    assert result["review_flag_count"] == 0
    conditions = result["items"][0]["conditions"]
    assert set(conditions) == {
        "no_evidence",
        "gold",
        "plausible_wrong",
        "shuffled_absent",
    }
    assert all(value["prompt_tokens"] + 128 <= 1024 for value in conditions.values())


def test_context_screen_flags_long_question_and_answer_bearing_wrong_context(
    tmp_path, monkeypatch
) -> None:
    checker, draft, families, tokenizer = _inputs(tmp_path, monkeypatch)
    item = draft["items"][0]
    item["question"] = "Q" * 65
    item["controls"]["plausible_wrong"] = ["gold"]
    item["controls"]["shuffled_absent"] = ["gold"]
    result = checker.measure(draft, tmp_path, families, tokenizer)
    row = result["items"][0]
    assert "question_exceeds_64_tokens" in row["hard_errors"]
    assert "control_literal_answer_overlap:plausible_wrong" in row["review_flags"]
    assert "wrong_and_absent_controls_identical" in row["review_flags"]


def test_context_screen_flags_partial_train_phrase_overlap(
    tmp_path, monkeypatch
) -> None:
    checker, draft, families, tokenizer = _inputs(tmp_path, monkeypatch)
    draft["items"][0]["question"] = (
        "Which step follows alpha beta gamma delta epsilon zeta eta theta?"
    )
    original_rows = checker._rows
    monkeypatch.setattr(
        checker,
        "_rows",
        lambda path: (
            [{"text": "prefix alpha beta gamma delta epsilon zeta eta theta suffix"}]
            if path.name == "train.jsonl"
            else original_rows(path)
        ),
    )
    result = checker.measure(draft, tmp_path, families, tokenizer)
    assert "train_8gram_overlap:question" in result["items"][0]["review_flags"]
