"""Offline exact-version checks for Card 03 independent review receipts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sparselab.evaluation.kml_card03_items import _sha


def _validator():
    path = (
        Path(__file__).parents[1]
        / "experiments/research/kernel-memory-lab/corpus-scale/validate-evaluation-reviews.py"
    )
    spec = importlib.util.spec_from_file_location("card03_review_receipts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample():
    draft = {
        "id": "one",
        "suite": "open_book",
        "category": "direct_extraction",
        "question": "Which mode?",
        "required_claims": ["manual"],
        "support_chunk_ids": ["chunk-one"],
        "reviewer": "PENDING-INDEPENDENT-REVIEW",
        "review_status": "draft",
    }
    draft["content_sha256"] = _sha(draft)
    solved = {
        "id": "one",
        "draft_item_sha256": draft["content_sha256"],
        "independent_answer": "Manual mode.",
        "support_used": ["chunk-one"],
        "reasoning_note": "The cited passage names manual mode.",
    }
    reviewed = dict(draft)
    reviewed["reviewer"] = "reviewer-b"
    reviewed["review_status"] = "reviewed"
    reviewed["content_sha256"] = _sha(
        {key: value for key, value in reviewed.items() if key != "content_sha256"}
    )
    checks = {
        key: True
        for key in (
            "source_support",
            "category_fit",
            "answer_rubric",
            "alternatives",
            "controls",
            "lineage",
            "leakage",
            "context_fit",
            "question_clarity",
        )
    }
    decision = {
        "id": "one",
        "author": "author-a",
        "reviewer": "reviewer-b",
        "draft_item_sha256": draft["content_sha256"],
        "blind_row_sha256": _sha(solved),
        "verdict": "pass",
        "checks": checks,
        "notes": "Gold passage names manual mode; wrong context does not.",
        "reviewed_item_sha256": reviewed["content_sha256"],
    }
    return draft, solved, decision, reviewed


def test_review_receipt_requires_other_reviewer_and_exact_final_semantics() -> None:
    validator = _validator()
    draft, solved, decision, reviewed = _sample()
    result = validator.validate_partition(
        [draft], [solved], [decision], [reviewed],
        author="author-a", reviewer="reviewer-b", expected=1,
    )
    assert result["item_count"] == 1
    with pytest.raises(ValueError, match="must differ"):
        validator.validate_partition(
            [draft], [solved], [decision], [reviewed],
            author="author-a", reviewer="author-a", expected=1,
        )
    reviewed["question"] = "Changed after review?"
    with pytest.raises(ValueError, match="changed semantic content"):
        validator.validate_partition(
            [draft], [solved], [decision], [reviewed],
            author="author-a", reviewer="reviewer-b", expected=1,
        )


def test_review_receipt_rejects_stale_blind_solution() -> None:
    validator = _validator()
    draft, solved, decision, reviewed = _sample()
    solved["draft_item_sha256"] = "older-version"
    with pytest.raises(ValueError, match="invalid gold-blind solution"):
        validator.validate_partition(
            [draft], [solved], [decision], [reviewed],
            author="author-a", reviewer="reviewer-b", expected=1,
        )
