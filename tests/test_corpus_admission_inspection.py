"""Offline inspection and exact review coverage; no model or source transport."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from test_preparation_sequence import _policy, _sources

from sparselab.corpus.admission_draft import draft_admission_manifest
from sparselab.corpus.admission_inspection import (
    inspect_admission,
    verify_admission_inspection,
)
from sparselab.corpus.project import load_project
from sparselab.corpus.release_review import verify_admission_review
from sparselab.training.manifest import canonical_json, sha256_file


def _selected_bytes(report, lock):
    total = 0
    for item in report["items"] + report["quarantine_exceptions"]:
        path = (
            Path(lock["sources"][item["source_id"]]["snapshot_path"])
            / "files"
            / item["path"]
        )
        if item["row_index"] is None:
            total += path.stat().st_size
        else:
            total += len(
                io.BytesIO(path.read_bytes()).readlines()[item["physical_line"] - 1]
            )
    return total


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
    assert observed["read_bytes"] == actual_snapshot_bytes + _selected_bytes(
        observed, lock
    )
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


def test_inspection_does_not_retain_source_candidates_between_records(
    tmp_path, monkeypatch
):
    import weakref

    from sparselab.corpus import admission_inspection

    root = tmp_path / "work"
    root.mkdir()
    recipe, policy, draft, selection, _ = _fixture(root)
    monkeypatch.setattr(
        admission_inspection.InspectionCache, "load", lambda *a, **k: None
    )
    previous = None
    calls = 0
    original = admission_inspection._candidate

    class Candidate(dict):
        pass

    def tracked(*args, **kwargs):
        nonlocal previous, calls
        assert previous is None or previous() is None, "whole source candidate retained"
        row = Candidate(original(*args, **kwargs))
        previous = weakref.ref(row)
        calls += 1
        return row

    monkeypatch.setattr(admission_inspection, "_candidate", tracked)
    inspect_admission(
        load_project(recipe / "acquire.yaml"),
        recipe / "acquire.yaml",
        root,
        draft,
        policy,
        selection,
        recipe / "streamed.json",
        max_input_bytes=1048576,
        max_excerpt_bytes=80,
        max_output_bytes=262144,
    )
    assert calls > 8 and previous() is None


def test_bounded_jsonl_reads_stop_before_oversized_record_is_materialized():
    import io

    from sparselab.corpus.jsonl_records import iter_jsonl_records

    class Limited(io.BytesIO):
        def readline(self, size=-1):
            assert size == 5
            return super().readline(size)

    assert (
        next(
            iter_jsonl_records(Limited(b"{}\n"), source="fixture", max_record_bytes=4)
        ).value
        == {}
    )
    with pytest.raises(ValueError, match="record-byte cap"):
        list(
            iter_jsonl_records(
                Limited(b"123456789\n"), source="fixture", max_record_bytes=4
            )
        )
    with pytest.raises(ValueError, match="positive integer"):
        list(
            iter_jsonl_records(
                io.BytesIO(b"{}\n"), source="fixture", max_record_bytes=True
            )
        )


def test_streaming_median_strata_and_fallback_match_full_sort_reference(
    tmp_path, monkeypatch
):
    import hashlib

    from sparselab.corpus import admission_inspection as inspection

    root = tmp_path / "work"
    root.mkdir()
    recipe, policy, draft, selection, _ = _fixture(root)
    project = load_project(recipe / "acquire.yaml")
    declaration = json.loads(selection.read_text())
    for source in declaration["sources"]:
        source["exception_count"] = 0
        source["strata"] = [
            {
                "id": "absent",
                "count": 2,
                "path_prefix": "absent/",
                "length_band": None,
                "issues_nonempty": None,
            },
            {
                "id": "short",
                "count": 1,
                "path_prefix": None,
                "length_band": "short",
                "issues_nonempty": None,
            },
            {
                "id": "long",
                "count": 1,
                "path_prefix": None,
                "length_band": "long",
                "issues_nonempty": None,
            },
        ]
    selection.write_bytes(canonical_json(declaration))
    entries = {
        entry["source_id"]: entry for entry in json.loads(draft.read_text())["sources"]
    }
    rows = {}
    # This sampler-only substitution follows real acquisition/draft verification;
    # it supplies five tiny lengths to exercise median and undersupplied strata.
    for source in project.sources:
        rows[source.id] = [
            inspection._candidate(
                source,
                entries[source.id],
                {"decision": "qualify", "reason": "sampler fixture", "issues": []},
                path=f"item-{number}.txt",
                raw="Fixture prose words. " * number,
                digest=hashlib.sha256(str(number).encode()).hexdigest(),
                row_index=number,
            )
            for number in range(1, 6)
        ]

    def scan(
        _project, _root, _draft, _lock, budget, visit, *, only_source=None, **kwargs
    ):
        assert budget >= 100
        for source_id, values in rows.items():
            if only_source is not None and source_id != only_source:
                continue
            for ordinal, row in enumerate(values):
                visit(source_id, {**row, "_ordinal": ordinal, "_offset": 0})
        return 100

    def fetch(source, entry, snapshot, row, stats):
        return {
            **rows[source.id][row["_ordinal"]],
            "retained_length_bytes": row["retained_length_bytes"],
        }

    monkeypatch.setattr(inspection, "_scan_candidates", scan)
    monkeypatch.setattr(inspection, "_fetch_candidate", fetch)
    # This is deliberately sampler-only synthetic data, never signed cold proof.
    monkeypatch.setattr(inspection.InspectionCache, "load", lambda *a, **k: None)
    monkeypatch.setattr(inspection.InspectionCache, "publish", lambda *a, **k: None)
    output = recipe / "ranked.json"
    inspect_admission(
        project,
        recipe / "acquire.yaml",
        root,
        draft,
        policy,
        selection,
        output,
        max_input_bytes=1048576,
        max_excerpt_bytes=80,
        max_output_bytes=262144,
    )
    report = json.loads(output.read_text())
    assert report["input_bytes"] == 100 * len(project.sources)
    assert report["read_bytes"] == report["input_bytes"]
    for source_id, values in rows.items():
        short = sorted(
            values[:3],
            key=lambda row: inspection._rank(
                declaration["seed"], source_id, "short", row
            ),
        )
        long = sorted(
            values[3:],
            key=lambda row: inspection._rank(
                declaration["seed"], source_id, "long", row
            ),
        )
        remaining = sorted(
            [("short", row) for row in short[1:]] + [("long", row) for row in long[1:]],
            key=lambda pair: inspection._rank(declaration["seed"], source_id, *pair),
        )
        expected = [("short", short[0]), ("long", long[0]), *remaining[:2]]
        assert [
            (item["stratum"], item["item_id"])
            for item in report["items"]
            if item["source_id"] == source_id
        ] == [(name, row["item_id"]) for name, row in expected]


def _inspect_again(root, fixture, name="again.json", **caps):
    recipe, policy, draft, selection, _ = fixture
    return inspect_admission(
        load_project(recipe / "acquire.yaml"),
        recipe / "acquire.yaml",
        root,
        draft,
        policy,
        selection,
        recipe / name,
        max_input_bytes=caps.get("max_input_bytes", 1048576),
        max_excerpt_bytes=80,
        max_output_bytes=262144,
    )


def test_snapshot_cap_boundary_and_single_pass_precheck(tmp_path, monkeypatch):
    from sparselab.corpus import admission_inspection as inspection

    fixture = _fixture(tmp_path)
    recipe, _, draft, _, receipt = fixture
    project = load_project(recipe / "acquire.yaml")
    lock = inspection.verify_acquisition(project, tmp_path)
    manifest = json.loads(draft.read_text())
    size = json.loads(receipt.read_text())["input_bytes"]
    assert (
        inspection._scan_candidates(
            project, tmp_path, manifest, lock, size, lambda *_: None
        )
        == size
    )
    with pytest.raises(ValueError, match="input-byte cap exceeded"):
        inspection._scan_candidates(
            project, tmp_path, manifest, lock, size - 1, lambda *_: None
        )

    cap = sum(source.acquisition.max_bytes for source in project.sources)
    original = inspection._scan_candidates
    budgets = []

    def at_declared_boundary(project, root, draft, lock, budget, visit, **kwargs):
        budgets.append(budget)
        original(project, root, draft, lock, budget, visit, **kwargs)
        # Fixture substitution: charge this source's full declared capacity.
        return next(
            source.acquisition.max_bytes
            for source in project.sources
            if source.id == kwargs["only_source"]
        )

    monkeypatch.setattr(inspection, "_scan_candidates", at_declared_boundary)
    monkeypatch.setattr(inspection.InspectionCache, "load", lambda *a, **k: None)
    monkeypatch.setattr(inspection.InspectionCache, "publish", lambda *a, **k: None)
    _inspect_again(tmp_path, fixture, max_input_bytes=cap)
    report = json.loads((recipe / "again.json").read_text())
    assert budgets == [cap] * len(project.sources)
    assert report["input_bytes"] == cap
    assert report["read_bytes"] == cap + _selected_bytes(report, lock)
    budgets.clear()
    with pytest.raises(ValueError, match="below declared source maxima"):
        _inspect_again(tmp_path, fixture, "below.json", max_input_bytes=cap - 1)
    assert budgets == []


@pytest.mark.parametrize(
    "guard,message",
    [
        ("_MAX_RECORD_BYTES", "record-byte cap"),
        ("_MAX_FILE_BYTES", "file-byte cap"),
        ("_MAX_POOL_BYTES", "selection-pool byte cap"),
    ],
)
def test_inspection_limit_callsites_refuse_without_output(
    tmp_path, monkeypatch, guard, message
):
    from sparselab.corpus import admission_inspection as inspection

    fixture = _fixture(tmp_path)
    monkeypatch.setattr(inspection, guard, 1)
    with pytest.raises(ValueError, match=message):
        _inspect_again(tmp_path, fixture)
    assert not (fixture[0] / "again.json").exists()


def test_draft_record_cap_is_enforced_at_reader(tmp_path, monkeypatch):
    from sparselab.corpus import admission_draft

    recipe, policy, _, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(admission_draft, "_MAX_RECORD_BYTES", 1)
    output = recipe / "oversize-draft.json"
    with pytest.raises(ValueError, match="record-byte cap"):
        draft_admission_manifest(
            load_project(recipe / "acquire.yaml"),
            tmp_path,
            recipe / "application.json",
            policy,
            output,
        )
    assert not output.exists()


def test_quarantine_coverage_counts_unshown_rows(tmp_path):
    fixture = _fixture(tmp_path)
    selection = fixture[3]
    declaration = json.loads(selection.read_text())
    for source in declaration["sources"]:
        source["exception_count"] = 0
    selection.write_bytes(canonical_json(declaration))
    _inspect_again(tmp_path, fixture)
    report = json.loads((fixture[0] / "again.json").read_text())
    summaries = [row for row in report["coverage"] if row["stratum"] == "__summary__"]
    assert len(summaries) == 4
    assert all(row["quarantine_population"] > 0 for row in summaries)
    assert all(row["exceptions_incomplete"] is True for row in summaries)
    assert all(row["exceptions_shown"] == 0 for row in summaries)
    assert report["quarantine_exceptions"] == []


def test_inspection_closes_reader_when_candidate_refuses(tmp_path, monkeypatch):
    from sparselab.corpus import admission_inspection as inspection
    from sparselab.corpus import jsonl_records

    fixture = _fixture(tmp_path)
    original_reader = jsonl_records.records_from_path
    original_candidate = inspection._candidate
    readers = []
    closed = []

    def tracked(*args, **kwargs):
        def iterate():
            try:
                yield from original_reader(*args, **kwargs)
            finally:
                closed.append(True)

        reader = iterate()
        readers.append(reader)  # keep alive: GC must not supply the close
        return reader

    def refuse(source, *args, **kwargs):
        if source.kind == "huggingface_dataset":
            raise ValueError("fixture candidate refusal")
        return original_candidate(source, *args, **kwargs)

    monkeypatch.setattr(jsonl_records, "records_from_path", tracked)
    monkeypatch.setattr(inspection, "_candidate", refuse)
    monkeypatch.setattr(inspection.InspectionCache, "load", lambda *a, **k: None)
    with pytest.raises(ValueError, match="fixture candidate refusal"):
        _inspect_again(tmp_path, fixture)
    assert readers and closed == [True]


def test_old_inspection_and_bound_review_require_regeneration(tmp_path):
    recipe, _, draft, _, inspection = _fixture(tmp_path)
    payload = json.loads(inspection.read_text())
    del payload["read_bytes"]
    inspection.write_bytes(canonical_json(payload) + b"\n")
    with pytest.raises(ValueError, match="differs from verified inputs"):
        verify_admission_inspection(inspection, tmp_path)
    admission = recipe / "admission.json"
    _review(admission, draft, inspection, [])
    with pytest.raises(ValueError, match="differs from verified inputs"):
        verify_admission_review(admission, tmp_path)


@pytest.mark.parametrize("kind", ["git", "huggingface_dataset"])
def test_draft_refuses_row_policy_without_declared_qualification(tmp_path, kind):
    recipe, policy, _, _, _ = _fixture(tmp_path)
    project = load_project(recipe / "acquire.yaml")
    source = next(source for source in project.sources if source.kind == kind)
    template = recipe / "application.json"
    application = json.loads(template.read_text())
    entry = next(
        entry for entry in application["sources"] if entry["source_id"] == source.id
    )
    entry["row_use_policy"] = {"mode": "record_attribution"}
    template.write_bytes(canonical_json(application))
    with pytest.raises(
        ValueError, match="row use policy requires declared HF row qualification"
    ):
        draft_admission_manifest(
            project, tmp_path, template, policy, recipe / "unsupported.json"
        )


def test_inspection_changed_input_refuses_without_output(tmp_path, monkeypatch):
    from sparselab.corpus import admission_inspection as inspection

    fixture = _fixture(tmp_path)
    original = inspection._scan_candidates

    def changed(project, root, draft, lock, *args, **kwargs):
        count = original(project, root, draft, lock, *args, **kwargs)
        folder = Path(lock["sources"][kwargs["only_source"]]["snapshot_path"]) / "files"
        target = next(path for path in folder.rglob("*") if path.is_file())
        target.write_bytes(target.read_bytes() + b"\n")
        return count

    monkeypatch.setattr(inspection, "_scan_candidates", changed)
    monkeypatch.setattr(inspection.InspectionCache, "load", lambda *a, **k: None)
    with pytest.raises(ValueError, match="changed"):
        _inspect_again(tmp_path, fixture)
    assert not (fixture[0] / "again.json").exists()


@pytest.mark.parametrize("kind", ["git", "huggingface_dataset"])
def test_inspection_refuses_before_reading_over_budget_payload(
    tmp_path, monkeypatch, kind
):
    from types import SimpleNamespace

    from sparselab.corpus import admission_inspection as inspection
    from sparselab.corpus import jsonl_records

    recipe, _, draft, _, _ = _fixture(tmp_path)
    project = load_project(recipe / "acquire.yaml")
    source = next(source for source in project.sources if source.kind == kind)
    lock = inspection.verify_acquisition(project, tmp_path)
    folder = Path(lock["sources"][source.id]["snapshot_path"])
    manifest = json.loads((folder / "manifest.json").read_text())
    file = manifest["files"][0]

    def forbid(*args, **kwargs):
        raise AssertionError("over-budget source payload read")

    if kind == "huggingface_dataset":
        monkeypatch.setattr(jsonl_records, "records_from_path", forbid)
    else:
        monkeypatch.setattr(Path, "read_bytes", forbid)
    with pytest.raises(ValueError, match="input-byte cap exceeded"):
        inspection._scan_candidates(
            SimpleNamespace(sources=(source,), config=project.config),
            tmp_path,
            json.loads(draft.read_text()),
            lock,
            file["size"] - 1,
            lambda *_: None,
        )


def test_inspection_cli_summary_includes_sampling_reads(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "private-settings"))
    previous = os.umask(0o077)
    try:
        fixture = _fixture(tmp_path)
        summary = _inspect_again(tmp_path, fixture)
    finally:
        os.umask(previous)
    report = json.loads((fixture[0] / "again.json").read_text())
    lock = json.loads(
        (tmp_path / "corpora/inspection-fixture/acquisition.json").read_text()
    )
    assert summary["read_bytes"] == report["read_bytes"]
    assert summary["read_bytes"] == summary["input_bytes"] + _selected_bytes(
        report, lock
    )
    assert summary["work"]["full_source_passes"] == 0


def test_explicit_large_streaming_input_budget_does_not_allocate_or_refuse_by_native_ceiling(
    tmp_path,
):
    fixture = _fixture(tmp_path)
    _inspect_again(tmp_path, fixture, "large-budget.json", max_input_bytes=3 * 1024**3)
    observed = json.loads((fixture[0] / "large-budget.json").read_text())
    assert observed["max_input_bytes"] == 3 * 1024**3
    assert observed["input_bytes"] < 1048576
