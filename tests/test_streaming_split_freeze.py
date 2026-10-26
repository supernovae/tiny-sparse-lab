"""Tiny equivalence and interruption fixtures; no corpus-scale qualification."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sparselab.corpus import split_freeze as old
from sparselab.corpus import streaming_split_freeze as streaming
from sparselab.corpus.split_runs import SplitStages, external_sort, rows, write_rows
from sparselab.training.manifest import canonical_json, sha256_file


def fixture(tmp_path, monkeypatch, *, states=None, merges=None):
    """Substitute closed execution/acquisition evidence only; compare algorithms."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    source = SimpleNamespace(id="fixture", revision="a" * 40)
    records = [
        {
            "document_id": hashlib.sha256(f"document-{i}".encode()).hexdigest(),
            "content_sha256": hashlib.sha256(f"content-{i}".encode()).hexdigest(),
            "family_hint": f"family-{i}",
            "source_id": source.id,
            "source_location": f"sample.jsonl#row={i + 1}",
            "source_row_index": i,
            "snapshot_sha256": "b" * 64,
        }
        for i in range(120)
    ]
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in records))
    states = states or {}
    decisions = [
        {
            "source_row_index": i,
            "source_row_sha256": "c" * 64,
            "decision": states.get(i, "qualify"),
            "reason": "fixture review",
            "issues": [],
        }
        for i in range(len(records))
    ]
    rights = {
        "training_eligibility": "eligible",
        "redistribution_mode": "metadata_reconstruction_only",
        "spdx_expression": "MIT",
        "license_references": ["https://example.invalid/rights"],
    }
    manifest = {
        "schema_version": 2,
        "policy_id": "fixture-reviewed",
        "policy_sha256": "d" * 64,
        "sources": [
            {
                "source_id": source.id,
                "snapshot_sha256": "b" * 64,
                "source_revision": source.revision,
                "sample_path": "sample.jsonl",
                "license_label": "MIT",
                "rights": rights,
                "records": decisions,
            }
        ],
    }
    admission = tmp_path / "admission.json"
    admission.write_bytes(canonical_json(manifest))
    reference = SimpleNamespace(path=admission.name, sha256=sha256_file(admission))
    project = SimpleNamespace(
        root=tmp_path,
        config=SimpleNamespace(id="stream-fixture"),
        sources=[source],
        release=SimpleNamespace(record_admission=reference),
        splits=SimpleNamespace(family_fields_by_source={}),
    )
    clusters = {
        "schema_version": 1,
        "inventory_sha256": sha256_file(inventory),
        "seed": "retained-seed",
        "source_strata": {source.id: "web"},
        "split_weights": {"train": 98, "validation": 1, "test": 1},
        "merges": merges or [],
        "reviewer": "fixture",
        "reviewed_on": "2026-10-26",
        "inventory_completion": {"path": "fixture-evidence.json", "sha256": "e" * 64},
    }
    clusters_path = tmp_path / "clusters.json"
    clusters_path.write_bytes(canonical_json(clusters))
    declaration = {
        "id": source.id,
        "kind": "huggingface_dataset",
        "schema_version": 2,
        "revision": source.revision,
    }
    monkeypatch.setattr(streaming, "source_declaration_payload", lambda _: declaration)
    monkeypatch.setattr(old, "source_declaration_payload", lambda _: declaration)
    adopted = SimpleNamespace(
        report={"documents": len(records), "sha256": sha256_file(inventory)},
        provenance={
            "mode": "fixture_closed_execution",
            "source_validation": "native acquisition and admission checks remain required before split publication",
        },
        check=lambda *_: None,
    )
    monkeypatch.setattr(
        "sparselab.corpus.split_inventory_adoption.adopt_completed_inventory",
        lambda *_: adopted,
    )

    def verified(*args, work, **kwargs):
        work.update(adopted=adopted, completion_mode="adopted_execution")
        return adopted.report, True

    monkeypatch.setattr(old, "verified_inventory", verified)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_bytes(b"{}")
    monkeypatch.setattr(
        old,
        "verify_acquisition",
        lambda *_: {"sources": {source.id: {"snapshot_path": str(snapshot)}}},
    )
    monkeypatch.setattr(
        old,
        "verify_record_admission",
        lambda *_: {
            (source.id, "sample.jsonl"): {
                "decisions": {i: row for i, row in enumerate(decisions)}
            }
        },
    )
    return project, inventory, clusters_path, records


def assignments(output):
    reference = yaml.safe_load(output.read_bytes())["assignment_file"]
    return list(rows(output.parent / ("assignments-" + reference["sha256"] + ".jsonl")))


def test_native_equivalence_with_quarantined_rows_and_approved_merge(
    tmp_path, monkeypatch
):
    merges = [
        {"family_id": "reviewed-component", "member_hints": ["family-0", "family-1"]}
    ]
    project, inventory, clusters, _records = fixture(
        tmp_path, monkeypatch, states={118: "exclude", 119: "quarantine"}, merges=merges
    )
    before = tmp_path / "old.yaml"
    after = tmp_path / "new.yaml"
    old.freeze_splits(project, tmp_path, inventory, clusters, before)
    result = old.freeze_splits(
        project, tmp_path, inventory, clusters, after, reuse_completed_preparation=True
    )
    assert before.read_bytes() == after.read_bytes()
    assert assignments(before) == assignments(after)
    assert (
        before.with_suffix(".family-candidates.jsonl").read_bytes()
        == after.with_suffix(".family-candidates.jsonl").read_bytes()
    )
    assert result["eligible_documents"] == 118
    assert result["placeholder_documents"] == 2
    assert result["deferred_verification"]
    receipt = json.loads(after.with_suffix(".receipt.json").read_bytes())
    assert (
        "acquisition-lock and snapshot-manifest verification"
        in receipt["deferred_verification"]
    )
    assert "were not rerun" in receipt["preparation_reuse"]["source_validation"]
    assert "inventory-to-admission" in receipt["verification_scope"]
    assert not list((tmp_path / "corpora").rglob("*.sqlite"))


def test_resume_export_reuses_completed_phases_and_chunks(tmp_path, monkeypatch):
    project, inventory, clusters, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / "splits.yaml"
    monkeypatch.setattr(streaming, "BUFFER_BYTES", 600)
    original = SplitStages.phase

    def interrupt(self, name, produce):
        if name == "assignments-export-000002":
            raise RuntimeError("fixture process interrupted")
        return original(self, name, produce)

    monkeypatch.setattr(SplitStages, "phase", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        old.freeze_splits(
            project,
            tmp_path,
            inventory,
            clusters,
            output,
            reuse_completed_preparation=True,
        )
    assert not output.exists()
    monkeypatch.setattr(SplitStages, "phase", original)
    monkeypatch.setattr(
        streaming, "_admission_phase", lambda *_: pytest.fail("repeated admission")
    )
    monkeypatch.setattr(
        streaming, "_inventory_phase", lambda *_: pytest.fail("repeated inventory")
    )
    result = old.freeze_splits(
        project, tmp_path, inventory, clusters, output, reuse_completed_preparation=True
    )
    assert "assignments-export-000001" in result["reused_phases"]
    assert len(assignments(output)) == 120


def test_changed_snapshot_binding_is_refused_without_source_body_reads(
    tmp_path, monkeypatch
):
    project, inventory, clusters, records = fixture(tmp_path, monkeypatch)
    records[0]["snapshot_sha256"] = "f" * 64
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in records))
    declaration = json.loads(clusters.read_bytes())
    declaration["inventory_sha256"] = sha256_file(inventory)
    clusters.write_bytes(canonical_json(declaration))
    with pytest.raises(ValueError, match="snapshots disagree"):
        old.freeze_splits(
            project,
            tmp_path,
            inventory,
            clusters,
            tmp_path / "splits.yaml",
            reuse_completed_preparation=True,
        )


def test_noneligible_rights_remain_refused(tmp_path, monkeypatch):
    project, inventory, clusters, _ = fixture(tmp_path, monkeypatch)
    path = tmp_path / "admission.json"
    manifest = json.loads(path.read_bytes())
    manifest["sources"][0]["rights"]["training_eligibility"] = "ineligible"
    path.write_bytes(canonical_json(manifest))
    project.release.record_admission.sha256 = sha256_file(path)
    with pytest.raises(ValueError, match="rights do not permit"):
        old.freeze_splits(
            project,
            tmp_path,
            inventory,
            clusters,
            tmp_path / "splits.yaml",
            reuse_completed_preparation=True,
        )


def test_external_sort_has_bounded_fan_in_and_stable_order(tmp_path):
    result = external_sort(
        ([i % 5, i] for i in reversed(range(110))), tmp_path / "runs", buffer_bytes=1
    )
    assert list(rows(result["path"])) == sorted([i % 5, i] for i in range(110))
    assert result["rows"] == 110


def test_changed_stage_output_recomputes_only_invalidated_suffix(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    request = {"producer": "fixture", "input": "fixed"}
    stages = SplitStages(tmp_path, request)
    assert "corpus" in stages.cache.authority
    assert "inspection" in stages.cache.authority
    calls = []

    def produce(name):
        def run(directory):
            calls.append(name)
            return {"files": [write_rows(directory / "rows.jsonl", [[name]])]}

        return run

    first = stages.phase("first", produce("first"))
    stages.phase("second", produce("second"))
    stages = SplitStages(tmp_path, request)
    stages.phase("first", produce("first"))
    Path(first["files"][0]["path"]).write_bytes(b"corrupt")
    stages = SplitStages(tmp_path, request)
    stages.phase("first", produce("first"))
    stages.phase("second", produce("second"))
    assert calls == ["first", "second", "first", "second"]


def test_torn_readable_journal_does_not_destroy_committed_phase(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    stages = SplitStages(tmp_path, {"producer": "fixture"})
    stages.phase(
        "first", lambda directory: {"files": [write_rows(directory / "rows", [[1]])]}
    )
    with stages.journal.open("ab") as stream:
        stream.write(b'{"incomplete":')
    resumed = SplitStages(tmp_path, {"producer": "fixture"})
    resumed.phase("first", lambda _: pytest.fail("repeated completed phase"))


def test_rehashed_completion_without_native_proof_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    request = {"producer": "fixture"}
    stages = SplitStages(tmp_path, request)
    calls = []

    def produce(directory):
        calls.append(1)
        return {"files": [write_rows(directory / "rows", [[1]])], "verified": True}

    stages.phase("first", produce)
    path = next(stages.root.glob("completion-*.json"))
    event = json.loads(path.read_bytes())
    event["record"]["result"]["verified"] = False
    event["sha256"] = hashlib.sha256(canonical_json(event["record"])).hexdigest()
    path.write_bytes(canonical_json(event) + b"\n")
    resumed = SplitStages(tmp_path, request)
    assert resumed.phase("first", produce)["verified"] is True
    assert len(calls) == 2
