"""Lineage identity, graph validity and absence are distinct from model quality."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_evaluation_suite import evaluated_run as _evaluated_run

from sparselab.family.cli import compare, graph, show
from sparselab.family.manifest import FamilyManifest, load_family

SHA = "a" * 64
OTHER = "b" * 64

evaluated_run = _evaluated_run


def node(
    name: str, parent: str | None = None, *, path: str = "missing/generation"
) -> dict:
    return {
        "id": name,
        "parent": parent,
        "parent_checkpoint_sha256": SHA if parent else None,
        "corpus": {"id": "corpus", "sha256": SHA},
        "tokenizer": {"id": "tokenizer", "sha256": SHA},
        "plan": {"id": "plan", "sha256": SHA},
        "architecture_sha256": SHA,
        "objective": "next_token",
        "budget": {"max_steps": 1, "max_tokens": 12},
        "checkpoint": {"sha256": SHA, "path": path},
        "evaluation_index": None,
        "readiness_result": None,
    }


def fixture(path: Path, nodes: list[dict]) -> Path:
    path.write_text(json.dumps({"family_version": 1, "id": "lineage", "nodes": nodes}))
    return path


def test_absent_pinned_child_preserves_graph_and_node_identity(tmp_path: Path) -> None:
    first = fixture(tmp_path / "family.json", [node("parent"), node("child", "parent")])
    family = load_family(first)
    prior = family.identities()["child"]
    assert graph(first)["edges"] == [{"parent": "parent", "child": "child"}]
    assert (
        show(first)["nodes"][1]["availability"]["checkpoint"]
        == "MISSING_NONRECONSTRUCTABLE"
    )
    changed = [
        node("parent", path="another/location"),
        node("child", "parent", path="another/child"),
    ]
    fixture(first, changed)
    assert load_family(first).identities()["child"] == prior
    facts = compare(first, "parent", "child")
    assert facts["second"]["ancestors"] == ["parent"]
    assert "causality" not in facts


def test_invalid_parents_and_declarations(tmp_path: Path) -> None:
    first = node("first")
    for nodes in (
        [node("child", "parent"), node("parent")],
        [first, node("first")],
        [first, {**node("child", "first"), "parent_checkpoint_sha256": OTHER}],
        [
            first,
            {
                **node("child", "first"),
                "checkpoint": {"sha256": SHA, "path": "../escape"},
            },
        ],
    ):
        with pytest.raises(ValueError):
            load_family(fixture(tmp_path / "family.json", nodes))
    with pytest.raises(ValueError):
        load_family(fixture(tmp_path / "family.json", [{**first, "invented": "field"}]))
    assert (
        json.loads(Path("schemas/model-family-v1.schema.json").read_text())
        == FamilyManifest.model_json_schema()
    )


def test_missing_checkpoint_is_not_a_new_training_authorization(tmp_path: Path) -> None:
    source = fixture(
        tmp_path / "family.json", [node("parent"), node("child", "parent")]
    )
    result = show(source)
    assert result["graph_valid"] is True
    assert all(
        row["availability"]["checkpoint"] == "MISSING_NONRECONSTRUCTABLE"
        for row in result["nodes"]
    )


def test_reviewed_promotion_is_immutable_and_rejects_conflict(
    evaluated_run, tmp_path: Path
) -> None:
    from sparselab.evaluation.readiness import (
        assess_readiness,
        issue_review,
        verify_readiness_result,
    )
    from sparselab.evaluation.suite import run_suite, verify_evaluation_index
    from sparselab.family.receipts import (
        decide,
        verify_lifecycle_metadata,
        verify_lifecycle_receipt,
    )
    from sparselab.training.manifest import canonical_json, read_manifest

    suite, runs = evaluated_run
    checkpoint = min((runs / "suite-run/checkpoints").glob("step_*"))
    index_path = run_suite(suite, "suite-run", checkpoint.name, runs, backend="cpu")
    index = verify_evaluation_index(index_path)
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "readiness_version": 1,
                "id": "policy",
                "require_verified_checkpoint": True,
                "required_gate_ids": ["loss"],
                "min_completed_evaluations": 1,
                "max_heldout_loss": {"evaluation_id": "loss", "value": 1000},
                "require_human_review": True,
            }
        )
    )
    review_path = tmp_path / "review.json"
    issue_review(
        index_path, "human", "approve", "Inspected checkpoint-bound result", review_path
    )
    result_path = assess_readiness(policy, index_path, review_path)
    readiness = verify_readiness_result(result_path)
    declaration = node("base", path=str(checkpoint))
    declaration["checkpoint"]["sha256"] = index["checkpoint_sha256"]
    declaration["evaluation_index"] = {
        "sha256": index["index_sha256"],
        "path": str(index_path),
    }
    declaration["readiness_result"] = {
        "sha256": readiness["result_sha256"],
        "path": str(result_path),
    }
    run_manifest = read_manifest(checkpoint.parent.parent / "manifest.json")
    declaration["architecture_sha256"] = run_manifest["architecture_sha256"]
    declaration["budget"] = {
        "max_steps": run_manifest["effective_config"]["training"]["max_steps"],
        "max_tokens": run_manifest["effective_config"]["training"]["max_tokens"],
    }
    declaration["tokenizer"]["sha256"] = next(
        item["sha256"]
        for item in run_manifest["artifacts"]
        if item["relative_path"] == "tokenizer.json"
    )
    source = fixture(tmp_path / "family.json", [declaration])
    receipt = decide(
        source,
        "promote",
        "base",
        result_path,
        index_path,
        review_path,
        "Reviewed next-stage candidate",
        work_root=tmp_path / "state",
    )
    assert (
        decide(
            source,
            "promote",
            "base",
            result_path,
            index_path,
            review_path,
            "Reviewed next-stage candidate",
            work_root=tmp_path / "state",
        )
        == receipt
    )
    with pytest.raises(ValueError, match="conflicting"):
        decide(
            source,
            "promote",
            "base",
            result_path,
            index_path,
            review_path,
            "Different action note",
            work_root=tmp_path / "state",
        )
    for altered, error in (
        ({**declaration, "architecture_sha256": SHA}, "architecture/objective/budget"),
        (
            {
                **declaration,
                "budget": {
                    "max_steps": declaration["budget"]["max_steps"] + 1,
                    "max_tokens": declaration["budget"]["max_tokens"],
                },
            },
            "architecture/objective/budget",
        ),
        (
            {**declaration, "objective": "unverified-objective"},
            "architecture/objective/budget",
        ),
        ({**declaration, "tokenizer": {"id": "tokenizer", "sha256": SHA}}, "tokenizer"),
    ):
        fixture(source, [altered])
        with pytest.raises(ValueError, match=error):
            show(source, work_root=tmp_path / "state")
    fixture(source, [declaration])
    assert (
        show(source, work_root=tmp_path / "state")["nodes"][0]["availability"][
            "checkpoint"
        ]
        == "PRESENT"
    )
    from sparselab.family.manifest import family_digest

    original = family_digest(source)
    published = list(
        (
            tmp_path
            / "state"
            / "family"
            / "lineage"
            / load_family(source).identities()["base"]
        ).glob("decision-*.json")
    )
    assert len(published) == 1
    (tmp_path / "promoted.json").write_bytes(published[0].read_bytes())
    assert (
        verify_lifecycle_receipt(tmp_path / "promoted.json")["receipt_sha256"]
        == receipt["receipt_sha256"]
    )
    relocated = dict(receipt)
    relocated["approval_path"] = str(tmp_path / "different-review.json")
    relocated["record_sha256"] = hashlib.sha256(
        canonical_json(
            {key: value for key, value in relocated.items() if key != "record_sha256"}
        )
    ).hexdigest()
    relocated_path = tmp_path / "relocated.json"
    relocated_path.write_bytes(canonical_json(relocated) + b"\n")
    assert (
        verify_lifecycle_metadata(relocated_path)["receipt_sha256"]
        == receipt["receipt_sha256"]
    )
    with pytest.raises(OSError):
        verify_lifecycle_receipt(relocated_path)
    relocated["approval_path"] = str(review_path)
    relocated_path.write_bytes(canonical_json(relocated) + b"\n")
    with pytest.raises(ValueError, match="invalid lifecycle receipt"):
        verify_lifecycle_metadata(relocated_path)
    declaration["lifecycle_receipts"] = ["promoted.json"]
    fixture(source, [declaration])
    assert family_digest(source) == original
    sibling = node("sibling")
    sibling["checkpoint"] = None
    fixture(source, [declaration, sibling])
    assert family_digest(source) != original
    rejected_review = tmp_path / "reject-review.json"
    issue_review(
        index_path, "other-human", "reject", "Reviewed and rejected", rejected_review
    )
    with pytest.raises(ValueError, match="conflicting"):
        decide(
            source,
            "reject",
            "base",
            result_path,
            index_path,
            rejected_review,
            "Different outcome after sibling declaration",
            work_root=tmp_path / "state",
        )
    from sparselab.campaign.state import digest

    for forged, reason in (
        (
            {
                **receipt,
                "approval_path": str(rejected_review),
                "approval_sha256": json.loads(rejected_review.read_text())[
                    "receipt_sha256"
                ],
            },
            "promotion/supersession",
        ),
        (
            {**receipt, "action": "supersede", "successor": "sibling"},
            "successor is not a descendant",
        ),
    ):
        forged["receipt_sha256"] = digest(
            "sparselab-model-lifecycle-receipt-v1",
            {
                key: value
                for key, value in forged.items()
                if key
                not in {
                    "approval_path",
                    "readiness_path",
                    "evaluation_path",
                    "family_path",
                    "issued_at_utc",
                    "receipt_sha256",
                    "record_sha256",
                }
            },
        )
        forged["record_sha256"] = hashlib.sha256(
            canonical_json(
                {key: value for key, value in forged.items() if key != "record_sha256"}
            )
        ).hexdigest()
        counterexample = (
            tmp_path / f"forged-{forged['action']}-{forged['approval_sha256']}.json"
        )
        counterexample.write_bytes(canonical_json(forged) + b"\n")
        verify_lifecycle_metadata(counterexample)
        with pytest.raises(ValueError, match=reason):
            verify_lifecycle_receipt(counterexample)
    assert published[0].read_bytes() == (tmp_path / "promoted.json").read_bytes()
