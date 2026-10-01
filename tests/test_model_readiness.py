"""Readiness is checkpoint-bound, coverage-aware and explicitly reviewed."""

from __future__ import annotations

import hashlib
import json

import pytest
from test_evaluation_suite import evaluated_run as _evaluated_run

from sparselab.evaluation.capabilities import describe_capability_card
from sparselab.evaluation.readiness import (
    ModelReadinessPolicy,
    assess_readiness,
    issue_review,
    verify_readiness_result,
    verify_review_receipt,
)
from sparselab.evaluation.suite import run_suite, verify_evaluation_index
from sparselab.training.manifest import canonical_json

evaluated_run = _evaluated_run


def _policy(root, **overrides):
    payload = {
        "readiness_version": 1,
        "id": "tiny",
        "require_verified_checkpoint": True,
        "required_gate_ids": ["loss"],
        "min_completed_evaluations": 1,
        "max_heldout_loss": {"evaluation_id": "loss", "value": 1000},
        "require_human_review": True,
        **overrides,
    }
    path = root / "policy.json"
    path.write_text(json.dumps(payload))
    return path


def test_readiness_review_and_replay(evaluated_run, tmp_path):
    suite, runs = evaluated_run
    generation = min((runs / "suite-run/checkpoints").glob("step_*")).name
    index = run_suite(suite, "suite-run", generation, runs, backend="cpu")
    policy = _policy(tmp_path)
    missing_review = assess_readiness(policy, index)
    assert verify_readiness_result(missing_review)["state"] == "NEEDS_REVIEW"
    receipt_path = tmp_path / "human-review.json"
    review = issue_review(
        index, "reviewerA", "approve", "Reviewed results", receipt_path
    )
    assert (
        verify_review_receipt(receipt_path)["receipt_sha256"]
        == review["receipt_sha256"]
    )
    assert (
        issue_review(index, "reviewerA", "approve", "Reviewed results", receipt_path)
        == review
    )
    approved = assess_readiness(policy, index, receipt_path)
    assert verify_readiness_result(approved)["state"] == "READY_FOR_NEXT_STAGE"
    relocated_policy = tmp_path / "moved-policy.json"
    relocated_policy.write_bytes(policy.read_bytes())
    relocated_review = tmp_path / "moved-human-review.json"
    relocated_review.write_bytes(receipt_path.read_bytes())
    assert assess_readiness(relocated_policy, index, relocated_review) == approved
    assert (
        verify_review_receipt(relocated_review)["receipt_sha256"]
        == review["receipt_sha256"]
    )
    with pytest.raises(ValueError, match="conflicting"):
        issue_review(index, "reviewerB", "reject", "Different decision", receipt_path)
    rejected_path = tmp_path / "rejected.json"
    issue_review(index, "reviewerB", "reject", "Reject model", rejected_path)
    assert (
        verify_readiness_result(assess_readiness(policy, index, rejected_path))["state"]
        == "DO_NOT_ADVANCE"
    )


def test_invalid_gate_coverage_and_threshold(evaluated_run, tmp_path):
    suite, runs = evaluated_run
    generation = min((runs / "suite-run/checkpoints").glob("step_*")).name
    index = run_suite(suite, "suite-run", generation, runs, backend="cpu")
    for values in (
        {"required_gate_ids": ["surface"]},
        {"required_gate_ids": ["external"]},
        {"required_gate_ids": ["loss", "loss"]},
        {"min_completed_evaluations": 4},
        {
            "required_gate_ids": ["loss"],
            "max_heldout_loss": {"evaluation_id": "external", "value": 2},
        },
    ):
        with pytest.raises(ValueError):
            assess_readiness(_policy(tmp_path, **values), index)
    loss = verify_evaluation_index(index)["evaluations"][0]["result"]["loss"]
    if loss > 0:
        assert (
            verify_readiness_result(
                assess_readiness(
                    _policy(
                        tmp_path, max_heldout_loss={"evaluation_id": "loss", "value": 0}
                    ),
                    index,
                )
            )["state"]
            == "DO_NOT_ADVANCE"
        )
    assert (
        verify_readiness_result(
            assess_readiness(_policy(tmp_path, min_completed_evaluations=2), index)
        )["state"]
        == "INCONCLUSIVE"
    )


def test_declared_but_unavailable_gate_is_inconclusive(evaluated_run, tmp_path):
    _, runs = evaluated_run
    card = describe_capability_card("chat-alias-recall-v1")
    card["cases"] = [{**card["cases"][0], "prompt": "long " * 100}]
    card["digest"] = hashlib.sha256(
        canonical_json(
            {
                key: value
                for key, value in card.items()
                if key not in {"format", "digest"}
            }
        )
    ).hexdigest()
    (tmp_path / "card.json").write_text(json.dumps(card))
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "evaluation_suite_version": 1,
                "id": "missing-gate",
                "evaluations": [
                    {
                        "id": "card",
                        "role": "gate",
                        "kind": "capability_card",
                        "source": "card.json",
                    }
                ],
            }
        )
    )
    generation = min((runs / "suite-run/checkpoints").glob("step_*")).name
    index = run_suite(suite, "suite-run", generation, runs, backend="cpu")
    assert verify_evaluation_index(index)["evaluations"][0]["status"] == "UNAVAILABLE"
    policy = _policy(tmp_path, required_gate_ids=["card"], max_heldout_loss=None)
    assert (
        verify_readiness_result(assess_readiness(policy, index))["state"]
        == "INCONCLUSIVE"
    )


def test_declaration_schema_and_nonfinite_threshold():
    from pathlib import Path

    from sparselab.evaluation.suite import EvaluationSuite

    assert (
        json.loads(Path("schemas/model-readiness-v1.schema.json").read_text())
        == ModelReadinessPolicy.model_json_schema()
    )
    assert (
        json.loads(Path("schemas/evaluation-suite-v1.schema.json").read_text())
        == EvaluationSuite.model_json_schema()
    )
    with pytest.raises(ValueError):
        ModelReadinessPolicy.model_validate(
            {
                "readiness_version": 1,
                "id": "x",
                "require_verified_checkpoint": True,
                "required_gate_ids": ["loss"],
                "min_completed_evaluations": 1,
                "max_heldout_loss": {"evaluation_id": "loss", "value": float("nan")},
                "require_human_review": False,
            }
        )
