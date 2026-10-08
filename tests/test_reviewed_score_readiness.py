"""Offline reviewed-score binding tests; no model fixture or generation runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sparselab.campaign.plan import ModelReadiness
from sparselab.evaluation import readiness, reviewed_scores
from sparselab.training.manifest import canonical_json, sha256_file


def _write_canonical(path: Path, value: dict) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def _scores_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, correct: int = 120
) -> tuple[Path, dict, dict]:
    index_sha = "a" * 64
    checkpoint_sha = "b" * 64
    panel_path = tmp_path / "panel.json"
    panel_path.write_text("mocked")
    prompts = [f"question {i}" for i in range(200)]
    panel = {
        "record_sha256": "d" * 64,
        "evaluation_index_sha256": index_sha,
        "checkpoint_sha256": checkpoint_sha,
        "rows": [{"status": "COMPLETED", "prompt": prompt} for prompt in prompts],
    }
    monkeypatch.setattr(reviewed_scores, "verify_panel_result", lambda _p: panel)
    order_path = tmp_path / "order.json"
    frozen_path = tmp_path / "frozen.json"
    frozen = {
        "format": "kml-card03-evaluation-manifest-v1",
        "items": [
            {
                "id": f"item-{i}",
                "content_sha256": hashlib.sha256(f"item {i}".encode()).hexdigest(),
                "suite": "closed_book",
                "split": "test",
                "category": f"axis_{i // 20}",
            }
            for i in range(200)
        ],
    }
    frozen_sha = hashlib.sha256(canonical_json(frozen)).hexdigest()
    frozen["content_sha256"] = frozen_sha
    _write_canonical(frozen_path, frozen)
    order = {
        "frozen_content_sha256": frozen_sha,
        "item_ids": [f"item-{i}" for i in range(200)],
        "item_content_sha256": [
            hashlib.sha256(f"item {i}".encode()).hexdigest() for i in range(200)
        ],
    }
    order_path.write_text(json.dumps(order))
    review_paths = [tmp_path / "review_a.jsonl", tmp_path / "review_b.jsonl"]
    for reviewer, source in zip(("author", "checker"), review_paths, strict=True):
        rows = [
            {
                "index": i,
                "id": order["item_ids"][i],
                "content_sha256": order["item_content_sha256"][i],
                "prompt_sha256": hashlib.sha256(prompts[i].encode()).hexdigest(),
                "category": f"axis_{i // 20}",
                "reviewer": reviewer,
                "score": int(i % 20 < correct // 10),
                "format_ok": i % 20 < 19,
            }
            for i in range(200)
        ]
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "format": "reviewed-panel-scores-v1",
        "evaluation_index_sha256": index_sha,
        "checkpoint_sha256": checkpoint_sha,
        "frozen_content_sha256": frozen_sha,
        "panel_path": str(panel_path),
        "panel_record_sha256": panel["record_sha256"],
        "order_path": str(order_path),
        "frozen_items_path": str(frozen_path),
        "order_file_sha256": sha256_file(order_path),
        "review_paths": [str(path) for path in review_paths],
        "review_file_sha256": [sha256_file(path) for path in review_paths],
        "reviewers": ["author", "checker"],
        "adjudications": [],
    }
    _write_canonical(receipt_path, receipt)
    kwargs = {
        "index_sha256": index_sha,
        "checkpoint_sha256": checkpoint_sha,
        "frozen_content_sha256": frozen_sha,
        "expected_items": 200,
        "expected_axes": 10,
        "items_per_axis": 20,
        "min_correct": 120,
        "min_axis_correct": 10,
        "min_format": 190,
        "min_axis_format": 18,
    }
    return receipt_path, kwargs, receipt


def test_complete_independent_scores_pass_and_changed_ledger_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, kwargs, receipt = _scores_fixture(tmp_path, monkeypatch)
    summary = reviewed_scores.verify_reviewed_scores(path, **kwargs)
    assert summary["passed"] is True
    assert summary["correct"] == 120
    assert summary["format_ok"] == 190
    assert len(summary["axes"]) == 10
    source = Path(receipt["review_paths"][1])
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="identity changed"):
        reviewed_scores.verify_reviewed_scores(path, **kwargs)


def test_missing_panel_row_or_changed_order_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, kwargs, receipt = _scores_fixture(tmp_path, monkeypatch)
    order = Path(receipt["order_path"])
    order.write_text(order.read_text() + " ")
    with pytest.raises(ValueError, match="item order changed"):
        reviewed_scores.verify_reviewed_scores(path, **kwargs)
    order.write_text(order.read_text().rstrip())
    panel = {
        "record_sha256": receipt["panel_record_sha256"],
        "evaluation_index_sha256": kwargs["index_sha256"],
        "checkpoint_sha256": kwargs["checkpoint_sha256"],
        "rows": [
            {"status": "COMPLETED", "prompt": f"question {i}"} for i in range(199)
        ],
    }
    monkeypatch.setattr(reviewed_scores, "verify_panel_result", lambda _p: panel)
    with pytest.raises(ValueError, match="coverage differs"):
        reviewed_scores.verify_reviewed_scores(path, **kwargs)


def test_disagreement_needs_adjudication_and_changed_prompt_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, kwargs, receipt = _scores_fixture(tmp_path, monkeypatch)
    source = Path(receipt["review_paths"][1])
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0]["score"] = 0
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt["review_file_sha256"][1] = sha256_file(source)
    _write_canonical(path, receipt)
    with pytest.raises(ValueError, match="unresolved"):
        reviewed_scores.verify_reviewed_scores(path, **kwargs)
    receipt["adjudications"] = [
        {
            "index": 0,
            "id": rows[0]["id"],
            "content_sha256": rows[0]["content_sha256"],
            "score": 1,
            "format_ok": True,
            "adjudicator": "arbiter",
            "note": "independent resolution",
        }
    ]
    _write_canonical(path, receipt)
    assert reviewed_scores.verify_reviewed_scores(path, **kwargs)["passed"] is True
    rows[0]["prompt_sha256"] = "e" * 64
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt["review_file_sha256"][1] = sha256_file(source)
    _write_canonical(path, receipt)
    with pytest.raises(ValueError, match="frozen item or panel prompt"):
        reviewed_scores.verify_reviewed_scores(path, **kwargs)


def test_threshold_failure_and_missing_scores_cannot_promote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, kwargs, _ = _scores_fixture(tmp_path, monkeypatch, correct=100)
    scores = reviewed_scores.verify_reviewed_scores(path, **kwargs)
    assert scores["passed"] is False
    policy = readiness.ModelReadinessPolicy.model_validate(
        {
            "readiness_version": 1,
            "id": "language",
            "require_verified_checkpoint": True,
            "required_gate_ids": ["loss"],
            "min_completed_evaluations": 1,
            "max_heldout_loss": None,
            "require_human_review": True,
            "reviewed_scores": {
                key: kwargs[key]
                for key in (
                    "frozen_content_sha256",
                    "expected_items",
                    "expected_axes",
                    "items_per_axis",
                    "min_correct",
                    "min_axis_correct",
                    "min_format",
                    "min_axis_format",
                )
            },
        }
    )
    index = {"evaluations": [{"id": "loss", "status": "COMPLETED"}]}
    assert (
        readiness._decision(policy, index, {"decision": "approve"}, None)[0]
        == "INCONCLUSIVE"
    )
    assert (
        readiness._decision(policy, index, {"decision": "approve"}, scores)[0]
        == "DO_NOT_ADVANCE"
    )
    passing = {**scores, "passed": True}
    assert readiness._decision(policy, index, None, passing)[0] == "NEEDS_REVIEW"
    assert (
        readiness._decision(policy, index, {"decision": "approve"}, passing)[0]
        == "READY_FOR_NEXT_STAGE"
    )


def test_absent_optional_policy_and_campaign_stage_keep_legacy_json() -> None:
    source = {
        "readiness_version": 1,
        "id": "legacy",
        "require_verified_checkpoint": True,
        "required_gate_ids": ["loss"],
        "min_completed_evaluations": 1,
        "max_heldout_loss": None,
        "require_human_review": True,
    }
    assert (
        readiness.ModelReadinessPolicy.model_validate(source).model_dump(mode="json")
        == source
    )
    stage = ModelReadiness.model_validate(
        {
            "id": "readiness",
            "kind": "model_readiness",
            "scope": "model",
            "evaluation": "evaluation",
            "policy": "policy.yaml",
            "review": None,
        }
    )
    assert "reviewed_scores" not in stage.model_dump(mode="json")


def test_readiness_result_reverifies_score_source_and_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score_path, kwargs, receipt = _scores_fixture(tmp_path, monkeypatch)
    policy = readiness.ModelReadinessPolicy.model_validate(
        {
            "readiness_version": 1,
            "id": "score_binding",
            "require_verified_checkpoint": True,
            "required_gate_ids": ["loss"],
            "min_completed_evaluations": 1,
            "max_heldout_loss": None,
            "require_human_review": True,
            "reviewed_scores": {
                key: kwargs[key]
                for key in (
                    "frozen_content_sha256",
                    "expected_items",
                    "expected_axes",
                    "items_per_axis",
                    "min_correct",
                    "min_axis_correct",
                    "min_format",
                    "min_axis_format",
                )
            },
        }
    )
    index_path = tmp_path / "index.json"
    index = {
        "index_sha256": kwargs["index_sha256"],
        "checkpoint_sha256": kwargs["checkpoint_sha256"],
        "evaluations": [{"id": "loss", "status": "COMPLETED"}],
    }
    monkeypatch.setattr(readiness, "verify_evaluation_index", lambda _p: index)
    monkeypatch.setattr(readiness, "_validate_policy", lambda _p, _i: None)
    result_path = readiness.assess_readiness(
        policy, index_path, reviewed_score_receipt=score_path
    )
    assert readiness.verify_readiness_result(result_path)["state"] == "NEEDS_REVIEW"
    source = Path(receipt["review_paths"][0])
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="identity changed"):
        readiness.verify_readiness_result(result_path)
