"""Offline inspection and exact review coverage; no model or source transport."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_kml_50m_preparation_sequence import _policy, _sources

from sparselab.corpus.admission_draft import draft_admission_manifest
from sparselab.corpus.admission_inspection import (
    inspect_admission,
    verify_admission_inspection,
)
from sparselab.corpus.project import load_project
from sparselab.corpus.release_review import verify_admission_review
from sparselab.training.manifest import canonical_json, sha256_file


def _fixture(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    recipe, strata = _sources(root, "inspection-fixture")
    template, policy = _policy(recipe, strata)
    draft = recipe / "draft.json"
    draft_admission_manifest(
        load_project(recipe / "acquire.yaml"), root, template, policy, draft
    )
    selection = recipe / "selection.json"
    selection.write_bytes(
        canonical_json(
            {
                "format": "sparselab-admission-inspection-selection-v1",
                "seed": "fixture-inspection-v1",
                "normalizer": "normalizer-structure-v3",
                "sources": [
                    {
                        "source_id": source_id,
                        "exception_count": 1,
                        "strata": [
                            {
                                "id": "all",
                                "count": 1,
                                "path_prefix": None,
                                "length_band": None,
                                "issues_nonempty": None,
                            }
                        ],
                    }
                    for source_id in strata
                ],
            }
        )
        + b"\n"
    )
    inspection = recipe / "inspection.json"
    inspect_admission(
        load_project(recipe / "acquire.yaml"),
        recipe / "acquire.yaml",
        root,
        draft,
        policy,
        selection,
        inspection,
        max_input_bytes=1048576,
        max_excerpt_bytes=80,
        max_output_bytes=262144,
    )
    return recipe, policy, draft, selection, inspection


def _review(
    admission: Path, draft: Path, inspection: Path, decisions: list[dict]
) -> None:
    admission.write_bytes(draft.read_bytes())
    receipt = {
        "format": "sparselab-admission-review-v2",
        "decision": "ACCEPTED",
        "reviewer": "Fixture reviewer",
        "reviewed_on": "2026-10-10",
        "draft_path": str(draft),
        "draft_sha256": sha256_file(draft),
        "admission_sha256": sha256_file(admission),
        "inspection_path": str(inspection),
        "inspection_sha256": sha256_file(inspection),
        "item_decisions": decisions,
    }
    admission.with_name(admission.name + ".review.json").write_bytes(
        canonical_json(receipt) + b"\n"
    )


def test_verified_pre_admission_inspection_and_exact_review(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    recipe, _, draft, _, inspection = _fixture(root)
    observed = verify_admission_inspection(inspection, root)
    lock = json.loads(
        (root / "corpora/inspection-fixture/acquisition.json").read_text()
    )
    actual_snapshot_bytes = sum(
        file.stat().st_size
        for source in lock["sources"].values()
        for file in (Path(source["snapshot_path"]) / "files").rglob("*")
        if file.is_file()
    )
    assert observed["input_bytes"] == actual_snapshot_bytes
    assert {item["source_id"] for item in observed["items"]} == {
        "books",
        "books_rows",
        "incident",
        "wiki",
    }
    assert len(observed["items"]) == 4
    assert len(observed["quarantine_exceptions"]) == 4
    assert all(item["decision"] == "qualify" for item in observed["items"])
    assert all(
        item["decision"] == "quarantine" for item in observed["quarantine_exceptions"]
    )
    assert all(item["raw_truncated"] for item in observed["items"])
    row = next(item for item in observed["items"] if item["source_id"] == "books_rows")
    assert row["row_index"] == 0 and row["physical_line"] == 1
    assert row["source_span"]["row_index"] == 0
    assert row["snapshot_sha256"] and row["rights"]["license_references"]
    decisions = [
        {
            "item_id": item["item_id"],
            "outcome": "pass",
            "note": "Fixture content and source notice checked.",
        }
        for item in observed["items"]
    ]
    admission = recipe / "admission.json"
    _review(admission, draft, inspection, decisions)
    assert (
        verify_admission_review(admission, root)["format"]
        == "sparselab-admission-review-v2"
    )
    for invalid in (
        decisions[:-1],
        decisions + [decisions[0]],
        [{**decisions[0], "item_id": "0" * 64}, *decisions[1:]],
        [{**decisions[0], "outcome": "block"}, *decisions[1:]],
    ):
        _review(admission, draft, inspection, invalid)
        with pytest.raises(ValueError, match="incomplete, substituted or blocking"):
            verify_admission_review(admission, root)
    _review(admission, draft, inspection, decisions)
    inspection.rename(recipe / "inspection-removed.json")
    with pytest.raises((FileNotFoundError, ValueError), match="inspection|missing"):
        verify_admission_review(admission, root)
    admission.with_name(admission.name + ".review.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_admission_review(admission, root)


def test_inspection_rejects_changed_inputs_and_exhausted_caps(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    recipe, policy, draft, selection, inspection = _fixture(root)
    project = load_project(recipe / "acquire.yaml")
    with pytest.raises(ValueError, match="input-byte cap"):
        inspect_admission(
            project,
            recipe / "acquire.yaml",
            root,
            draft,
            policy,
            selection,
            recipe / "too-small-input.json",
            max_input_bytes=100,
            max_excerpt_bytes=80,
            max_output_bytes=262144,
        )
    with pytest.raises(ValueError, match="output-byte cap"):
        inspect_admission(
            project,
            recipe / "acquire.yaml",
            root,
            draft,
            policy,
            selection,
            recipe / "too-small-output.json",
            max_input_bytes=1048576,
            max_excerpt_bytes=80,
            max_output_bytes=100,
        )
    changed = json.loads(selection.read_text())
    changed["normalizer"] = "normalizer-structure-v2"
    selection.write_bytes(canonical_json(changed) + b"\n")
    with pytest.raises(ValueError, match="differs from verified inputs"):
        verify_admission_inspection(inspection, root)
    changed["normalizer"] = "normalizer-structure-v3"
    selection.write_bytes(canonical_json(changed) + b"\n")
    policy.write_text("Changed source rights claim.\n")
    with pytest.raises(ValueError, match="policy identity mismatch"):
        verify_admission_inspection(inspection, root)
    policy.write_text("Reviewed fixture source policy, Apache-2.0 local use.\n")
    altered = json.loads(draft.read_text())
    altered["sources"][0]["files"][0]["reason"] = "Changed review reason"
    draft.write_bytes(canonical_json(altered) + b"\n")
    with pytest.raises(ValueError, match="differs from verified inputs"):
        verify_admission_inspection(inspection, root)
    draft_admission_manifest(
        load_project(recipe / "acquire.yaml"),
        root,
        recipe / "application.json",
        policy,
        recipe / "restored-draft.json",
    )
    draft.write_bytes((recipe / "restored-draft.json").read_bytes())
    lock = root / "corpora" / "inspection-fixture" / "acquisition.json"
    source = json.loads(lock.read_text())["sources"]["books_rows"]
    snapshot = Path(source["snapshot_path"])
    sample = next((snapshot / "files").glob("*.jsonl"))
    sample.write_bytes(
        sample.read_bytes().replace(b"Fixture book 0", b"Changed book 0", 1)
    )
    with pytest.raises(ValueError, match="snapshot byte mismatch"):
        verify_admission_inspection(inspection, root)


def test_admission_review_rejects_obsolete_unbound_decisions(tmp_path: Path) -> None:
    admission = tmp_path / "admission.json"
    admission.write_text("{}")
    receipt = admission.with_name(admission.name + ".review.json")
    receipt.write_text(
        json.dumps(
            {
                "format": "sparselab-admission-review-v1",
                "decision": "ACCEPTED",
                "reviewer": "Synthetic fixture only",
                "reviewed_on": "2026-10-10",
                "draft_path": str(admission),
                "draft_sha256": sha256_file(admission),
                "admission_sha256": sha256_file(admission),
                "spot_audits": [
                    {
                        "source_id": "fixture",
                        "location": "line 1",
                        "outcome": "pass",
                        "note": "Unbound synthetic review",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="inspection-bound decision"):
        verify_admission_review(admission, tmp_path)
