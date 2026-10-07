"""Offline checks for admission-aware deterministic family split freezing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sparselab.corpus import split_freeze
from sparselab.corpus.mixture import _inventory as verify_mixture_inventory
from sparselab.corpus.project import SplitDeclaration
from sparselab.training.manifest import canonical_json, sha256_file


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sources = [SimpleNamespace(id=source) for source in ("books", "pages", "incidents")]
    project_root = tmp_path / "project"
    project_root.mkdir()
    admission = {"schema_version": 2, "sources": []}
    admission_bytes = canonical_json(admission)
    (project_root / "admission.json").write_bytes(admission_bytes)
    project = SimpleNamespace(
        config=SimpleNamespace(id="freeze-fixture"),
        root=project_root,
        release=SimpleNamespace(
            record_admission=SimpleNamespace(
                path="admission.json",
                sha256=hashlib.sha256(admission_bytes).hexdigest(),
            )
        ),
        sources=sources,
    )
    inventory = tmp_path / "inventory.jsonl"
    rows = []
    decisions = {}
    snapshots = {}
    for source in sources:
        snapshot = tmp_path / source.id
        snapshot.mkdir()
        (snapshot / "manifest.json").write_text("{}")
        snapshots[source.id] = {
            "snapshot_path": str(snapshot),
            "snapshot_sha256": "a" * 64,
        }
        for index in range(10):
            path = (
                f"docs/{index}.md"
                if source.id == "incidents"
                else f"{source.id}.sample.jsonl"
            )
            rows.append(
                {
                    "document_id": f"{source.id}-{index}",
                    "source_id": source.id,
                    "source_location": path
                    + (
                        "#lines=1-2"
                        if source.id == "incidents"
                        else f"#row={index + 1}"
                    ),
                    "source_row_index": None if source.id == "incidents" else index,
                    "family_hint": f"{source.id}-family:{index}",
                    "content_sha256": hashlib.sha256(
                        f"{source.id}-{index}".encode()
                    ).hexdigest(),
                }
            )
            decisions[source.id, path] = (
                {"decision": "qualify"}
                if source.id == "incidents"
                else {"decisions": {i: {"decision": "qualify"} for i in range(10)}}
            )
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    clusters_path = tmp_path / "clusters.json"
    clusters = {
        "schema_version": 1,
        "inventory_sha256": sha256_file(inventory),
        "seed": "reviewed-seed-v1",
        "reviewer": "Fixture reviewer",
        "reviewed_on": "2026-10-07",
        "source_strata": {
            "books": "general_prose",
            "pages": "explanatory_prose",
            "incidents": "incident_response_docs",
        },
        "merges": [],
    }
    clusters_path.write_bytes(canonical_json(clusters))
    monkeypatch.setattr(
        split_freeze,
        "write_split_inventory",
        lambda *_: {"sha256": sha256_file(inventory)},
    )
    monkeypatch.setattr(
        split_freeze,
        "verify_acquisition",
        lambda *_: {"sources": snapshots},
    )
    monkeypatch.setattr(split_freeze, "source_declaration_payload", lambda _: {})
    monkeypatch.setattr(split_freeze, "verify_record_admission", lambda *_: decisions)
    return project, inventory, clusters_path, rows, decisions


def test_freeze_splits_is_deterministic_and_validates_integer_family_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters, rows, _ = _fixture(tmp_path, monkeypatch)
    first = tmp_path / "splits-first.yaml"
    second = tmp_path / "splits-second.yaml"
    report = split_freeze.freeze_splits(project, tmp_path, inventory, clusters, first)
    split_freeze.freeze_splits(project, tmp_path, inventory, clusters, second)
    assert first.read_bytes() == second.read_bytes()
    declaration = yaml.safe_load(first.read_text())
    SplitDeclaration.model_validate(declaration)
    assert set(declaration["assignments"]) == {row["document_id"] for row in rows}
    assert report["eligible_documents"] == 30
    assert all(
        counts
        == {
            "families": 10,
            "train": 8,
            "validation": 1,
            "test": 1,
            "fractions": {
                "train": "8/10",
                "validation": "1/10",
                "test": "1/10",
            },
        }
        for counts in report["family_counts"].values()
    )
    candidates = [
        json.loads(line)
        for line in Path(report["family_candidates"]).read_text().splitlines()
    ]
    assert len(candidates) == 30
    assert all(
        set(row) == {"document_id", "family_id", "split", "stratum", "content_sha256"}
        for row in candidates
    )


def test_finalized_family_inventory_matches_kept_release_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters, _, _ = _fixture(tmp_path, monkeypatch)
    split_path = tmp_path / "splits.yaml"
    report = split_freeze.freeze_splits(
        project, tmp_path, inventory, clusters, split_path
    )
    release = tmp_path / "release"
    release.mkdir()
    candidates = [
        json.loads(line)
        for line in Path(report["family_candidates"]).read_text().splitlines()
    ]
    source_by_stratum = {
        "general_prose": "books",
        "explanatory_prose": "pages",
        "incident_response_docs": "incidents",
    }
    documents = {
        row["document_id"]: {
            "document_id": row["document_id"],
            "source_id": source_by_stratum[row["stratum"]],
            "domains": [row["stratum"]],
            "split": row["split"],
            "content_sha256": row["content_sha256"],
            "drop_reason": "duplicate" if index == 0 else None,
        }
        for index, row in enumerate(candidates)
    }
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in documents.values())
    )
    monkeypatch.setattr(
        split_freeze,
        "verify_release",
        lambda _: {
            "release_id": "fixture-release",
            "build_identity": {"split": yaml.safe_load(split_path.read_text())},
        },
    )
    final = tmp_path / "final-family-inventory.jsonl"
    result = split_freeze.finalize_family_inventory(release, split_path, final)
    assert result["kept_documents"] == 29
    kept = {key: doc for key, doc in documents.items() if doc["drop_reason"] is None}
    assert set(
        verify_mixture_inventory(
            final, kept, {v: k for k, v in source_by_stratum.items()}
        )
    ) == set(kept)


def test_freeze_splits_stops_on_missing_decision_unresolved_family_and_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters_path, rows, decisions = _fixture(tmp_path, monkeypatch)
    decisions.pop(("incidents", "docs/0.md"))
    with pytest.raises(ValueError, match="lacks file admission"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters_path, tmp_path / "missing.yaml"
        )
    decisions["incidents", "docs/0.md"] = {"decision": "qualify"}
    rows[0]["family_hint"] = "unresolved-family:books-0"
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    clusters = json.loads(clusters_path.read_text())
    clusters["inventory_sha256"] = sha256_file(inventory)
    clusters_path.write_bytes(canonical_json(clusters))
    with pytest.raises(ValueError, match="unresolved family"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters_path, tmp_path / "unresolved.yaml"
        )


def test_cross_split_duplicate_qualifying_content_fails_but_quarantine_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters_path, rows, decisions = _fixture(tmp_path, monkeypatch)
    first = tmp_path / "baseline.yaml"
    split_freeze.freeze_splits(project, tmp_path, inventory, clusters_path, first)
    assignments = yaml.safe_load(first.read_text())["assignments"]
    train = next(row for row in rows if assignments[row["document_id"]] == "train")
    heldout = next(
        row
        for row in rows
        if row["source_id"] == "books"
        and assignments[row["document_id"]] == "validation"
    )
    heldout["content_sha256"] = train["content_sha256"]
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    clusters = json.loads(clusters_path.read_text())
    clusters["inventory_sha256"] = sha256_file(inventory)
    clusters_path.write_bytes(canonical_json(clusters))
    with pytest.raises(ValueError, match="cross-split exact content duplicate"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters_path, tmp_path / "duplicate.yaml"
        )
    source_id = heldout["source_id"]
    path = heldout["source_location"].split("#", 1)[0]
    sibling = {
        **heldout,
        "document_id": "books-sibling",
        "source_location": path + "#row=11",
        "source_row_index": 10,
        "content_sha256": hashlib.sha256(b"sibling").hexdigest(),
    }
    rows.append(sibling)
    decisions[source_id, path]["decisions"][10] = {"decision": "qualify"}
    decisions[source_id, path]["decisions"][heldout["source_row_index"]] = {
        "decision": "quarantine"
    }
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    clusters["inventory_sha256"] = sha256_file(inventory)
    clusters_path.write_bytes(canonical_json(clusters))
    split_freeze.freeze_splits(
        project, tmp_path, inventory, clusters_path, tmp_path / "quarantined.yaml"
    )
