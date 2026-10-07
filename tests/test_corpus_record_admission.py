"""Offline checks for reviewed source-policy decisions on acquired HF rows."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sparselab.corpus.pipeline import _records_for_file
from sparselab.corpus.project import (
    ReleaseDeclaration,
    SourceDeclaration,
    release_declaration_payload,
)
from sparselab.corpus.rights import (
    reported_spdx_expression,
    verify_record_admission,
)
from sparselab.training.manifest import canonical_json


def _fixture(tmp_path: Path) -> tuple[dict, dict, dict, Path, SourceDeclaration]:
    revision = "a" * 40
    shard_path = "project_gutenberg-dolma-0014.json.gz"
    sample_path = shard_path + ".sample.jsonl"
    shard_sha = "b" * 64
    snapshot_sha = "c" * 64
    source = SourceDeclaration.model_validate(
        {
            "schema_version": 2,
            "id": "gutenberg_fixture",
            "kind": "huggingface_dataset",
            "canonical_uri": "https://huggingface.co/datasets/common-pile/project_gutenberg_filtered",
            "revision": revision,
            "license": "Public Domain claim, review required",
            "license_url": "https://www.gutenberg.org/policy/license.html",
            "rights": {
                "training_eligibility": "review_required",
                "redistribution_mode": "review_required",
            },
            "domains": ["general_prose"],
            "document_kinds": ["prose"],
            "source_family": "gutenberg_fixture",
            "acquisition": {
                "config": "default",
                "split": "train",
                "text_field": "text",
                "max_rows": 4,
                "max_bytes": 100000,
                "bounded_shards": [
                    {
                        "path": shard_path,
                        "expected_sha256": shard_sha,
                        "max_shard_bytes": 100000,
                        "max_scanned_rows": 4,
                        "hash_modulus": 1,
                        "hash_remainders": [0],
                        "declared_config": "default",
                        "declared_split": "train",
                    }
                ],
            },
        }
    )
    source_rows = [
        {
            "id": "100",
            "metadata": {
                "license": "Public Domain",
                "url": "https://www.gutenberg.org/ebooks/100.txt.utf-8",
                "title": "Covered book",
                "provenance": shard_path + ":1",
                "language": "en",
            },
            "text": "A covered paragraph with enough ordinary prose to pass the record screen. "
            * 2,
        },
        {
            "id": "101",
            "metadata": {
                "license": "Copyrighted",
                "url": "https://www.gutenberg.org/ebooks/101.txt.utf-8",
                "title": "Contrary book",
                "provenance": shard_path + ":2",
            },
            "text": "This row is outside the reviewed source license scope. " * 2,
        },
    ]
    selected = []
    rendered = []
    for index, row in enumerate(source_rows):
        row_sha = hashlib.sha256(canonical_json(row)).hexdigest()
        selected.append({"source_row_index": index, "source_row_sha256": row_sha})
        rendered.append(
            {
                **row,
                "_sparselab_source": {
                    "id": row["id"],
                    "dataset_revision": revision,
                    "source_shard_path": shard_path,
                    "source_shard_sha256": shard_sha,
                    "source_row_index": index,
                    "source_row_sha256": row_sha,
                },
            }
        )
    raw = b"".join(canonical_json(row) + b"\n" for row in rendered)
    sample = tmp_path / "gutenberg_fixture" / snapshot_sha / "files" / sample_path
    sample.parent.mkdir(parents=True)
    sample.write_bytes(raw)
    sample_sha = hashlib.sha256(raw).hexdigest()
    snapshot = {
        "snapshot_sha256": snapshot_sha,
        "declaration": source.model_dump(mode="json"),
        "files": [{"path": sample_path, "sha256": sample_sha, "size": len(raw)}],
        "retrieval": {
            "shards": [
                {
                    "output_path": sample_path,
                    "source_shard_path": shard_path,
                    "source_shard_sha256": shard_sha,
                    "selected_rows": selected,
                }
            ]
        },
    }
    manifest = {
        "schema_version": 1,
        "policy_id": "test-source-policy-v1",
        "policy_sha256": "d" * 64,
        "sources": [
            {
                "source_id": source.id,
                "snapshot_sha256": snapshot_sha,
                "source_revision": revision,
                "sample_path": sample_path,
                "sample_sha256": sample_sha,
                "source_shard_path": shard_path,
                "source_shard_sha256": shard_sha,
                "license_label": "US public-domain claim; local research",
                "rights": {
                    "training_eligibility": "eligible_with_obligations",
                    "redistribution_mode": "metadata_reconstruction_only",
                    "license_references": [
                        "https://www.gutenberg.org/policy/license.html"
                    ],
                    "notices": ["Retain source attribution"],
                },
                "records": [
                    {
                        "source_row_index": 0,
                        "source_row_sha256": selected[0]["source_row_sha256"],
                        "decision": "qualify",
                        "reason": "source policy covers this row",
                        "issues": [
                            {
                                "field": "edition",
                                "owner": "source audit",
                                "remedy": "retain book URL for later edition lookup",
                                "decision_impact": "nonblocking under reviewed source policy",
                            }
                        ],
                    },
                    {
                        "source_row_index": 1,
                        "source_row_sha256": selected[1]["source_row_sha256"],
                        "decision": "quarantine",
                        "reason": "contrary license",
                        "issues": [],
                    },
                ],
            }
        ],
    }
    return manifest, source.model_dump(mode="json"), snapshot, raw, source


def _verify(manifest: dict, source: dict, snapshot: dict, tmp_path: Path) -> dict:
    return verify_record_admission(
        manifest,
        {source["id"]: source},
        {source["id"]: snapshot},
        tmp_path,
    )


def test_covered_row_inherits_policy_while_conflict_stays_quarantined(
    tmp_path: Path,
) -> None:
    manifest, source_dict, snapshot, raw, source = _fixture(tmp_path)
    result = _verify(manifest, source_dict, snapshot, tmp_path)
    rights = result[source.id, snapshot["files"][0]["path"]]
    assert rights["rights"].training_eligibility == "eligible_with_obligations"
    assert rights["decisions"][0]["issues"][0]["field"] == "edition"
    assert (
        reported_spdx_expression(
            {
                "rights": rights["rights"].model_dump(mode="json"),
                "admission_policy_spdx_expression": rights["policy_spdx_expression"],
            },
            None,
        )
        == "unknown"
    )
    rejected: list[dict] = []
    pairs = _records_for_file(
        raw,
        snapshot["files"][0]["path"],
        source,
        snapshot["snapshot_sha256"],
        file_rights=rights["rights"],
        record_admission=rights,
        rejected_records=rejected,
    )
    assert len(pairs) == 1
    assert pairs[0][0]["license"] == manifest["sources"][0]["license_label"]
    assert rejected[0]["row"] == 1
    assert "quarantine" in rejected[0]["reason"]


def test_admission_spdx_summary_uses_inherited_license_not_old_source_claim() -> None:
    file = {
        "rights": {"detected_spdx_expression": None},
        "admission_policy_spdx_expression": "CC-BY-SA-4.0",
    }
    assert reported_spdx_expression(file, None) == "CC-BY-SA-4.0"
    assert reported_spdx_expression(file, "MIT") == "CC-BY-SA-4.0"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "cover selected rows"),
        ("duplicate", "duplicate or invalid"),
        ("row_hash", "row provenance mismatch"),
        ("snapshot", "source provenance mismatch"),
        ("qualify_conflict", "material exception"),
        ("incomplete_issue", "invalid record admission decision value"),
        ("missing_policy", "invalid versioned"),
    ],
)
def test_admission_fails_closed_on_incomplete_or_conflicting_evidence(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest, source, snapshot, _, _ = _fixture(tmp_path)
    manifest = copy.deepcopy(manifest)
    entry = manifest["sources"][0]
    if mutation == "missing":
        entry["records"].pop()
    elif mutation == "duplicate":
        entry["records"].append(copy.deepcopy(entry["records"][0]))
    elif mutation == "row_hash":
        entry["records"][0]["source_row_sha256"] = "0" * 64
    elif mutation == "snapshot":
        entry["snapshot_sha256"] = "0" * 64
    elif mutation == "qualify_conflict":
        entry["records"][1]["decision"] = "qualify"
    elif mutation == "incomplete_issue":
        entry["records"][0]["issues"][0].pop("owner")
    else:
        manifest.pop("policy_sha256")
    with pytest.raises(ValueError, match=message):
        _verify(manifest, source, snapshot, tmp_path)


def test_flagged_qualify_requires_documented_manual_review(tmp_path: Path) -> None:
    manifest, source, snapshot, raw, _ = _fixture(tmp_path)
    sample = (
        tmp_path
        / source["id"]
        / snapshot["snapshot_sha256"]
        / "files"
        / snapshot["files"][0]["path"]
    )
    rows = [json.loads(line) for line in raw.splitlines()]
    rows[0]["text"] += " Caption printed by permission of the rights holder."
    original = {
        key: value for key, value in rows[0].items() if key != "_sparselab_source"
    }
    digest = hashlib.sha256(canonical_json(original)).hexdigest()
    rows[0]["_sparselab_source"]["source_row_sha256"] = digest
    updated = b"".join(canonical_json(row) + b"\n" for row in rows)
    sample.write_bytes(updated)
    new_file_sha = hashlib.sha256(updated).hexdigest()
    snapshot["files"][0]["sha256"] = new_file_sha
    snapshot["retrieval"]["shards"][0]["selected_rows"][0]["source_row_sha256"] = digest
    entry = manifest["sources"][0]
    entry["sample_sha256"] = new_file_sha
    entry["records"][0]["source_row_sha256"] = digest
    with pytest.raises(ValueError, match="lacks manual review"):
        _verify(manifest, source, snapshot, tmp_path)
    entry["records"][0]["manual_review"] = {
        "reviewer": "Fixture reviewer",
        "reviewed_on": "2026-10-07",
        "evidence_url": "https://www.gutenberg.org/ebooks/100",
        "finding": "Caption exception affects an image omitted from this text row.",
    }
    _verify(manifest, source, snapshot, tmp_path)


def test_wikimedia_rights_conflict_cannot_be_hidden_as_exclusion(
    tmp_path: Path,
) -> None:
    manifest, source, snapshot, raw, _ = _fixture(tmp_path)
    source["canonical_uri"] = (
        "https://huggingface.co/datasets/common-pile/wikimedia_filtered"
    )
    snapshot["declaration"]["canonical_uri"] = source["canonical_uri"]
    entry = manifest["sources"][0]
    entry["license_label"] = "CC-BY-SA-4.0; local research"
    entry["rights"]["spdx_expression"] = "CC-BY-SA-4.0"
    entry["records"][1]["decision"] = "exclude"
    rows = [json.loads(line) for line in raw.splitlines()]
    for index, row in enumerate(rows):
        metadata = row["metadata"]
        metadata["url"] = "https://wikipedia.com/wiki/" + metadata["title"].replace(
            " ", "_"
        )
        metadata["namespace"] = "0"
        metadata["license"] = (
            "Creative Commons - Attribution Share-Alike - "
            "https://creativecommons.org/licenses/by-sa/4.0/"
            if index == 0
            else "Proprietary"
        )
        original = {
            key: value for key, value in row.items() if key != "_sparselab_source"
        }
        digest = hashlib.sha256(canonical_json(original)).hexdigest()
        row["_sparselab_source"]["source_row_sha256"] = digest
        snapshot["retrieval"]["shards"][0]["selected_rows"][index][
            "source_row_sha256"
        ] = digest
        entry["records"][index]["source_row_sha256"] = digest
    updated = b"".join(canonical_json(row) + b"\n" for row in rows)
    sample = (
        tmp_path
        / source["id"]
        / snapshot["snapshot_sha256"]
        / "files"
        / snapshot["files"][0]["path"]
    )
    sample.write_bytes(updated)
    sample_sha = hashlib.sha256(updated).hexdigest()
    snapshot["files"][0]["sha256"] = sample_sha
    entry["sample_sha256"] = sample_sha
    with pytest.raises(
        ValueError, match="material rights exception must be quarantined"
    ):
        _verify(manifest, source, snapshot, tmp_path)
    entry["records"][1]["decision"] = "quarantine"
    _verify(manifest, source, snapshot, tmp_path)


def test_optional_admission_reference_preserves_historical_release_payload() -> None:
    value = {
        "schema_version": 2,
        "mixture": {"general_prose": 1.0},
        "publication_mode": "metadata_reconstruction_only",
        "lm": {"selected": False, "training_splits": []},
        "chat": {"selected": False, "training_splits": []},
    }
    release = ReleaseDeclaration.model_validate(value)
    payload = release_declaration_payload(release)
    assert "record_admission" not in payload
    assert hashlib.sha256(canonical_json(payload)).hexdigest() == (
        "01fab54388883d4218a7c1b1ce5bbbae7755d18a06d706fd5a2788df5bfb40fe"
    )
