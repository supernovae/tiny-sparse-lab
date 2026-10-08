"""Bind independent Card 03 item reviews to exact final item versions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sparselab.evaluation.kml_card03_items import _rows, _sha, _unique
from sparselab.training.manifest import canonical_json, sha256_file

_BASE_CHECKS = {
    "source_support",
    "category_fit",
    "answer_rubric",
    "alternatives",
    "controls",
    "lineage",
    "leakage",
    "context_fit",
    "question_clarity",
}
_BLIND_FIELDS = {
    "id",
    "draft_item_sha256",
    "independent_answer",
    "support_used",
    "reasoning_note",
}
_REVIEW_FIELDS = {
    "id",
    "author",
    "reviewer",
    "draft_item_sha256",
    "blind_row_sha256",
    "verdict",
    "checks",
    "notes",
    "reviewed_item_sha256",
}


def _by_id(rows: list[dict], expected: int, label: str) -> dict[str, dict]:
    if len(rows) != expected or any(
        not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
        for row in rows
    ):
        raise ValueError(f"{label} needs {expected} identified rows")
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != expected:
        raise ValueError(f"{label} contains duplicate item IDs")
    return indexed


def validate_partition(
    authored: list[dict],
    blind: list[dict],
    reviews: list[dict],
    final: list[dict],
    *,
    author: str,
    reviewer: str,
    expected: int = 200,
) -> dict:
    if author == reviewer or not author or not reviewer:
        raise ValueError("item author and independent reviewer must differ")
    authors = _by_id(authored, expected, "author draft")
    blind_rows = _by_id(blind, expected, "blind solutions")
    review_rows = _by_id(reviews, expected, "review decisions")
    final_rows = _by_id(final, expected, "reviewed items")
    if not (authors.keys() == blind_rows.keys() == review_rows.keys() == final_rows.keys()):
        raise ValueError("item IDs differ between author, blind and review files")
    categories: Counter[str] = Counter()
    for item_id, draft in authors.items():
        solved = blind_rows[item_id]
        review = review_rows[item_id]
        reviewed = final_rows[item_id]
        if (
            draft.get("review_status") != "draft"
            or draft.get("reviewer") != "PENDING-INDEPENDENT-REVIEW"
            or draft.get("content_sha256") != _sha(
                {key: value for key, value in draft.items() if key != "content_sha256"}
            )
        ):
            raise ValueError(f"invalid authored draft: {item_id}")
        if (
            set(solved) != _BLIND_FIELDS
            or solved["draft_item_sha256"] != draft["content_sha256"]
            or not isinstance(solved["independent_answer"], str)
            or not solved["independent_answer"].strip()
            or not isinstance(solved["reasoning_note"], str)
            or not solved["reasoning_note"].strip()
            or not isinstance(solved["support_used"], list)
            or any(not isinstance(value, str) for value in solved["support_used"])
            or not set(solved["support_used"]).issubset(set(draft["support_chunk_ids"]))
        ):
            raise ValueError(f"invalid gold-blind solution: {item_id}")
        if (
            set(review) != _REVIEW_FIELDS
            or review["author"] != author
            or review["reviewer"] != reviewer
            or review["draft_item_sha256"] != draft["content_sha256"]
            or review["blind_row_sha256"] != _sha(solved)
            or review["verdict"] != "pass"
            or not isinstance(review["notes"], str)
            or len(review["notes"].strip()) < 20
            or not isinstance(review["checks"], dict)
            or not _BASE_CHECKS.issubset(review["checks"])
            or any(review["checks"][key] is not True for key in _BASE_CHECKS)
        ):
            raise ValueError(f"independent review incomplete: {item_id}")
        if draft["category"] == "two_source_inference" and (
            review["checks"].get("both_sources_required") is not True
            or len(set(solved["support_used"])) < 2
        ):
            raise ValueError(f"two-source necessity not reviewed: {item_id}")
        if draft["category"] == "missing_ambiguous_evidence" and review["checks"].get(
            "whole_parent_absence_checked"
        ) is not True:
            raise ValueError(f"whole-parent absence not reviewed: {item_id}")
        expected_final = dict(draft)
        expected_final["reviewer"] = reviewer
        expected_final["review_status"] = "reviewed"
        expected_final["content_sha256"] = _sha(
            {key: value for key, value in expected_final.items() if key != "content_sha256"}
        )
        if (
            reviewed != expected_final
            or review["reviewed_item_sha256"] != expected_final["content_sha256"]
        ):
            raise ValueError(f"reviewed item changed semantic content: {item_id}")
        categories[f"{draft['suite']}/{draft['category']}"] += 1
    return {
        "schema_version": 1,
        "status": "STRUCTURAL_REVIEW_RECEIPTS_COMPLETE_NOT_SEMANTIC_CERTIFICATION",
        "author": author,
        "reviewer": reviewer,
        "item_count": expected,
        "category_counts": dict(sorted(categories.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-items", type=Path, required=True)
    parser.add_argument("--blind-solutions", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path, required=True)
    parser.add_argument("--reviewed-items", type=Path, required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    authored = json.loads(args.author_items.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    if isinstance(authored, dict):
        authored = authored["items"]
    final = json.loads(args.reviewed_items.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    if isinstance(final, dict):
        final = final["items"]
    result = validate_partition(
        authored,
        _rows(args.blind_solutions),
        _rows(args.review_decisions),
        final,
        author=args.author,
        reviewer=args.reviewer,
    )
    result["input_sha256"] = {
        name: sha256_file(path)
        for name, path in (
            ("author_items", args.author_items),
            ("blind_solutions", args.blind_solutions),
            ("review_decisions", args.review_decisions),
            ("reviewed_items", args.reviewed_items),
        )
    }
    with args.output.open("xb") as stream:
        stream.write(canonical_json(result) + b"\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
