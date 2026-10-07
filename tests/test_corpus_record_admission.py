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


def _mixed_fixture(tmp_path: Path) -> tuple[dict, dict, dict]:
    manifest, hf_source, hf_snapshot, _, _ = _fixture(tmp_path)
    git_source = SourceDeclaration.model_validate(
        {
            "schema_version": 2,
            "id": "incident_fixture",
            "kind": "git",
            "canonical_uri": "https://github.com/example/incident-docs",
            "revision": "e" * 40,
            "license": "Apache-2.0; reviewed local research",
            "license_url": "https://github.com/example/incident-docs/blob/"
            + "e" * 40
            + "/LICENSE",
            "rights": {
                "training_eligibility": "review_required",
                "redistribution_mode": "review_required",
            },
            "domains": ["incident_response"],
            "document_kinds": ["documentation"],
            "source_family": "incident_fixture",
            "acquisition": {
                "max_bytes": 100000,
                "bounded_blobs": [
                    {
                        "path": "docs/covered.md",
                        "git_blob_oid": "a" * 40,
                        "max_bytes": 25,
                    },
                    {
                        "path": "docs/exception.md",
                        "git_blob_oid": "b" * 40,
                        "max_bytes": 40,
                    },
                ],
                "tree_oid": "c" * 40,
            },
        }
    ).model_dump(mode="json")
    snapshot_sha = "f" * 64
    contents = {
        "docs/covered.md": b"# Covered\nIncident response.\n",
        "docs/exception.md": b"<!-- SPDX-License-Identifier: MIT -->\nNo.\n",
    }
    files = []
    decisions = []
    for path, raw in contents.items():
        target = tmp_path / git_source["id"] / snapshot_sha / "files" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        files.append({"path": path, "sha256": digest, "size": len(raw)})
        decisions.append(
            {
                "path": path,
                "sha256": digest,
                "decision": "qualify" if path.endswith("covered.md") else "quarantine",
                "reason": "reviewed source policy"
                if path.endswith("covered.md")
                else "file license exception",
                "issues": [],
            }
        )
    git_snapshot = {
        "snapshot_sha256": snapshot_sha,
        "declaration": git_source,
        "files": files,
        "retrieval": {"commit": git_source["revision"]},
    }
    manifest["schema_version"] = 2
    manifest["sources"].append(
        {
            "source_id": git_source["id"],
            "snapshot_sha256": snapshot_sha,
            "source_revision": git_source["revision"],
            "license_label": "Apache-2.0 reviewed local research",
            "rights": {
                "training_eligibility": "eligible_with_obligations",
                "redistribution_mode": "metadata_reconstruction_only",
                "spdx_expression": "Apache-2.0",
                "license_references": [git_source["license_url"]],
                "notices": ["Retain source attribution"],
            },
            "files": decisions,
        }
    )
    return (
        manifest,
        {hf_source["id"]: hf_source, git_source["id"]: git_source},
        {
            hf_source["id"]: hf_snapshot,
            git_source["id"]: git_snapshot,
        },
    )


def test_mixed_admission_qualifies_covered_git_file_and_quarantines_exception(
    tmp_path: Path,
) -> None:
    manifest, sources, snapshots = _mixed_fixture(tmp_path)
    admitted = verify_record_admission(manifest, sources, snapshots, tmp_path)
    covered = admitted["incident_fixture", "docs/covered.md"]
    exception = admitted["incident_fixture", "docs/exception.md"]
    assert covered["decision"] == "qualify"
    assert covered["rights"].training_eligibility == "eligible_with_obligations"
    assert exception["decision"] == "quarantine"
    assert exception["rights"].training_eligibility == "review_required"
    assert (
        admitted[
            "gutenberg_fixture", "project_gutenberg-dolma-0014.json.gz.sample.jsonl"
        ]["decisions"][1]["decision"]
        == "quarantine"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "do not cover pinned files"),
        ("unknown", "do not cover pinned files"),
        ("sha", "SHA-256 mismatch"),
        ("qualify_exception", "material rights exception"),
        ("tamper", "bytes differ from snapshot"),
    ],
)
def test_mixed_admission_fails_closed_on_git_file_exception(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest, sources, snapshots = _mixed_fixture(tmp_path)
    git = manifest["sources"][1]
    if mutation == "missing":
        git["files"].pop()
    elif mutation == "unknown":
        git["files"][0]["path"] = "docs/unknown.md"
    elif mutation == "sha":
        git["files"][0]["sha256"] = "0" * 64
    elif mutation == "qualify_exception":
        git["files"][1]["decision"] = "qualify"
    else:
        (
            tmp_path / "incident_fixture" / ("f" * 64) / "files/docs/covered.md"
        ).write_bytes(b"tampered")
    with pytest.raises(ValueError, match=message):
        verify_record_admission(manifest, sources, snapshots, tmp_path)


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        (b"\nCopyright 2024 Third Party\n", "lacks manual review"),
        (b"\n<!-- SPDX-License-Identifier: MIT -->\n", "material rights exception"),
        (b"\nAKIAABCDEFGHIJKLMNOP\n", "material rights exception"),
    ],
)
def test_git_exception_screen_reads_beyond_header(
    tmp_path: Path, suffix: bytes, expected: str
) -> None:
    manifest, sources, snapshots = _mixed_fixture(tmp_path)
    target = tmp_path / "incident_fixture" / ("f" * 64) / "files/docs/covered.md"
    target.write_bytes(b"ordinary text\n" * 30 + suffix)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    snapshots["incident_fixture"]["files"][0]["sha256"] = digest
    manifest["sources"][1]["files"][0]["sha256"] = digest
    with pytest.raises(ValueError, match=expected):
        verify_record_admission(manifest, sources, snapshots, tmp_path)


def test_flagged_git_file_can_qualify_only_after_documented_review(
    tmp_path: Path,
) -> None:
    manifest, sources, snapshots = _mixed_fixture(tmp_path)
    target = tmp_path / "incident_fixture" / ("f" * 64) / "files/docs/covered.md"
    target.write_bytes(b"# Covered\nCopyright 2024 example, repository owner.\n")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    snapshots["incident_fixture"]["files"][0]["sha256"] = digest
    decision = manifest["sources"][1]["files"][0]
    decision["sha256"] = digest
    decision["manual_review"] = {
        "reviewer": "Fixture reviewer",
        "reviewed_on": "2026-10-07",
        "evidence_url": "https://github.com/example/incident-docs/blob/"
        + "e" * 40
        + "/LICENSE",
        "finding": "Copyright notice belongs to the pinned repository owner.",
    }
    result = verify_record_admission(manifest, sources, snapshots, tmp_path)
    assert result["incident_fixture", "docs/covered.md"]["decision"] == "qualify"
