"""Tiny real native validation, deterministic selection, and authenticated reuse."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from test_corpus_admission_inspection import _fixture, _inspect_again

from sparselab.corpus import admission_inspection as inspection
from sparselab.corpus.admission_inspection_cache import InspectionCache
from sparselab.corpus.project import load_project
from sparselab.corpus.structure_cleaning import clean_structure
from sparselab.training.manifest import canonical_json


@pytest.fixture
def private_inspection(tmp_path, monkeypatch):
    # Match the native owned-store contract; change only this fixture process.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    previous = os.umask(0o077)
    try:
        root = tmp_path / "work"
        root.mkdir()
        fixture = _fixture(root)
    finally:
        os.umask(previous)
    return root, fixture


def _discard_operation_cache(root):
    for path in (root / "cache/admission-inspection").glob("operation-*.json"):
        path.unlink()  # Disposable test cache only; published receipts stay intact.


def test_warm_completed_output_and_fresh_process_skip_full_passes(
    private_inspection, monkeypatch
):
    root, fixture = private_inspection
    recipe, policy, draft, selection, receipt = fixture
    before = receipt.stat().st_mtime_ns
    summary = _inspect_again(root, fixture)
    assert summary["work"]["operation_cache_hits"] == 1
    assert summary["work"]["full_source_passes"] == 0
    assert summary["work"]["selected_records_read"] == 0
    same = inspection.inspect_admission(
        load_project(recipe / "acquire.yaml"),
        recipe / "acquire.yaml",
        root,
        draft,
        policy,
        selection,
        receipt,
        max_input_bytes=1048576,
        max_excerpt_bytes=80,
        max_output_bytes=262144,
    )
    assert same["sha256"] == summary["sha256"]
    assert same["work"]["full_source_passes"] == 0
    assert receipt.stat().st_mtime_ns == before

    def forbid(*args, **kwargs):
        raise AssertionError("warm verification repeated a full pass or metadata load")

    monkeypatch.setattr(inspection, "_scan_candidates", forbid)
    monkeypatch.setattr(inspection, "read_metadata", forbid)
    assert inspection.verify_admission_inspection(receipt, root)["items"]
    script = """import sys
from pathlib import Path
from sparselab.corpus import admission_inspection as module
def forbid(*args, **kwargs):raise AssertionError('cold work in fresh warm process')
module._scan_candidates=forbid
module.read_metadata=forbid
assert module.verify_admission_inspection(Path(sys.argv[1]),Path(sys.argv[2]))['items']
print('warm verified')
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(receipt), str(root)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "warm verified" in child.stdout


def test_cold_validation_parses_each_hf_shard_once(private_inspection, monkeypatch):
    from sparselab.corpus import jsonl_records

    root, fixture = private_inspection
    monkeypatch.setattr(InspectionCache, "load", lambda *args, **kwargs: None)
    original = jsonl_records.records_from_path
    opened = []

    def tracked(path, **limits):
        opened.append(Path(path))
        yield from original(path, **limits)

    monkeypatch.setattr(jsonl_records, "records_from_path", tracked)
    summary = _inspect_again(root, fixture, "cold-once.json")
    assert len(opened) == len(set(opened)) == 1
    assert summary["work"]["full_source_passes"] == 4
    assert summary["work"]["records_validated"] == 38
    assert summary["work"]["selected_records_read"] == 8


def test_inspection_materializes_each_metadata_input_once_per_operation(
    private_inspection, monkeypatch
):
    from sparselab.corpus import bounded_metadata

    root, fixture = private_inspection
    monkeypatch.setattr(InspectionCache, "load", lambda *args, **kwargs: None)
    original = bounded_metadata._Reader.read
    parsed = Counter()

    def tracked(reader):
        stream = getattr(reader.stream, "stream", reader.stream)
        parsed[(reader.store.temporary.name, str(stream.name))] += 1
        return original(reader)

    monkeypatch.setattr(bounded_metadata._Reader, "read", tracked)
    summary = _inspect_again(root, fixture, "metadata-once.json")
    assert parsed and all(count == 1 for count in parsed.values())
    assert len({name for _, name in parsed if name.endswith("manifest.json")}) == 4
    assert summary["work"]["full_source_passes"] == 4
    assert summary["work"]["records_validated"] == 38
    assert (fixture[0] / "metadata-once.json").read_bytes() == fixture[4].read_bytes()


def test_interruption_reuses_only_completed_sources(private_inspection, monkeypatch):
    root, fixture = private_inspection
    directory = root / "cache/admission-inspection"
    for path in directory.glob("*.json"):
        path.unlink()
    original = inspection._scan_candidates
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(kwargs["only_source"])
        if len(calls) != 2:
            return original(*args, **kwargs)
        arguments = list(args)
        visit = arguments[5]

        def stop(source_id, row):
            visit(source_id, row)
            raise KeyboardInterrupt

        arguments[5] = stop
        return original(*arguments, **kwargs)

    monkeypatch.setattr(inspection, "_scan_candidates", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _inspect_again(root, fixture, "interrupted.json")
    assert not (fixture[0] / "interrupted.json").exists()
    assert len(list(directory.glob("source-*.json"))) == 1
    monkeypatch.setattr(inspection, "_scan_candidates", original)
    resumed = _inspect_again(root, fixture, "resumed.json")
    assert resumed["work"]["source_cache_hits"] == 1
    assert resumed["work"]["full_source_passes"] == 3
    assert (fixture[0] / "resumed.json").read_bytes() == fixture[4].read_bytes()


@pytest.mark.parametrize(
    "damage", ["truncated", "unsigned_payload", "bad_hmac", "no_proof"]
)
def test_corrupt_or_unauthenticated_source_is_revalidated(
    private_inspection, monkeypatch, damage
):
    root, fixture = private_inspection
    _discard_operation_cache(root)
    target = next((root / "cache/admission-inspection").glob("source-*.json"))
    value = json.loads(target.read_text())
    source_id = value["payload"]["items"][0]["source_id"]
    if damage == "truncated":
        target.write_bytes(b'{"dependencies":')
    elif damage == "unsigned_payload":
        value["payload"]["items"][0]["cleaned_excerpt"] = "forged derived result"
        target.write_bytes(canonical_json(value))
    else:
        store = InspectionCache(root).proofs
        receipts = [
            path
            for path in store.directory.glob("*.json")
            if json.loads(path.read_text())["proof"]["binding"]["path"] == str(target)
        ]
        assert receipts
        for path in receipts:
            if damage == "no_proof":
                path.unlink()
                continue
            proof = json.loads(path.read_text())
            proof["hmac"] = "0" * 64
            path.write_bytes(canonical_json(proof))
    calls = []
    original = inspection._scan_candidates

    def tracked(*args, **kwargs):
        calls.append(kwargs["only_source"])
        return original(*args, **kwargs)

    monkeypatch.setattr(inspection, "_scan_candidates", tracked)
    summary = _inspect_again(root, fixture, "revalidated.json")
    assert calls == [source_id]
    assert summary["work"]["full_source_passes"] == 1
    assert summary["work"]["source_cache_hits"] == 3
    assert (fixture[0] / "revalidated.json").read_bytes() == fixture[4].read_bytes()


def test_changed_source_decision_recomputes_only_that_source(private_inspection):
    root, fixture = private_inspection
    draft = json.loads(fixture[2].read_text())
    entry = next(row for row in draft["sources"] if row["source_id"] == "books_rows")
    entry["records"][0]["reason"] = "New fixture review rationale"
    fixture[2].write_bytes(canonical_json(draft))
    summary = _inspect_again(root, fixture, "changed.json")
    assert summary["work"]["full_source_passes"] == 1
    assert summary["work"]["source_cache_hits"] == 3
    report = json.loads((fixture[0] / "changed.json").read_text())
    assert (
        next(row for row in report["items"] if row["source_id"] == "books_rows")[
            "reason"
        ]
        == "New fixture review rationale"
    )
    with pytest.raises(ValueError, match="differs from verified inputs"):
        inspection.verify_admission_inspection(fixture[4], root)


def test_changed_selection_recomputes_only_affected_source(private_inspection):
    root, fixture = private_inspection
    selection = json.loads(fixture[3].read_text())
    source = next(row for row in selection["sources"] if row["source_id"] == "books")
    source["strata"][0]["count"] = 2
    fixture[3].write_bytes(canonical_json(selection))
    summary = _inspect_again(root, fixture, "selection.json.out")
    assert summary["work"]["full_source_passes"] == 1
    assert summary["work"]["source_cache_hits"] == 3


def test_changed_verifier_code_invalidates_completed_work(
    private_inspection, monkeypatch
):
    root, fixture = private_inspection
    original = InspectionCache.__init__

    def changed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.authority["inspection"]["rights.py"] = "0" * 64

    monkeypatch.setattr(InspectionCache, "__init__", changed)
    summary = _inspect_again(root, fixture, "new-code.json")
    assert summary["work"]["full_source_passes"] == 4
    assert summary["work"]["source_cache_hits"] == 0


def test_foreign_or_lost_verification_key_requires_native_checks(
    private_inspection, monkeypatch, tmp_path
):
    root, fixture = private_inspection
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "other-settings"))
    summary = _inspect_again(root, fixture, "foreign-key.json")
    assert summary["work"]["full_source_passes"] == 4
    assert summary["work"]["source_cache_hits"] == 0


def test_interrupted_output_publication_reuses_completed_operation(
    private_inspection, monkeypatch
):
    root, fixture = private_inspection
    original = inspection.os.link

    def stop(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(inspection.os, "link", stop)
    with pytest.raises(KeyboardInterrupt):
        _inspect_again(root, fixture, "publication.json")
    assert not (fixture[0] / "publication.json").exists()
    assert not list(fixture[0].glob(".inspection-*"))
    monkeypatch.setattr(inspection.os, "link", original)
    summary = _inspect_again(root, fixture, "publication.json")
    assert summary["work"]["operation_cache_hits"] == 1
    assert summary["work"]["full_source_passes"] == 0


def test_same_size_input_change_with_restored_mtime_refuses(private_inspection):
    root, fixture = private_inspection
    lock = json.loads(
        (root / "corpora/inspection-fixture/acquisition.json").read_text()
    )
    snapshot = Path(lock["sources"]["books_rows"]["snapshot_path"])
    path = next((snapshot / "files").iterdir())
    before = path.stat()
    raw = path.read_bytes()
    altered = raw.replace(b"Fixture book 0", b"Changed book 0", 1)
    assert altered != raw and len(altered) == len(raw)
    path.write_bytes(altered)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError, match="snapshot byte mismatch"):
        _inspect_again(root, fixture, "tampered.json")
    assert not (fixture[0] / "tampered.json").exists()


def test_changed_snapshot_inventory_cannot_reuse_proof(private_inspection):
    root, fixture = private_inspection
    lock = json.loads(
        (root / "corpora/inspection-fixture/acquisition.json").read_text()
    )
    snapshot = Path(lock["sources"]["books_rows"]["snapshot_path"])
    (snapshot / "files/extra.txt").write_text("unexpected fixture input")
    with pytest.raises(ValueError, match="snapshot file inventory mismatch"):
        _inspect_again(root, fixture, "extra.json")


def test_published_receipt_tamper_is_not_hidden_by_reuse(private_inspection):
    root, fixture = private_inspection
    payload = json.loads(fixture[4].read_text())
    payload["items"][0]["cleaned_excerpt"] = "tampered public receipt"
    fixture[4].write_bytes(canonical_json(payload))
    with pytest.raises(ValueError, match="differs from verified inputs"):
        inspection.verify_admission_inspection(fixture[4], root)
    with pytest.raises(ValueError, match="existing inspection output differs"):
        _inspect_again(root, fixture, fixture[4].name)


def test_native_selection_matches_independent_full_sort(private_inspection):
    root, fixture = private_inspection
    recipe, _, draft_path, selection_path, _ = fixture
    draft = json.loads(draft_path.read_text())
    books = next(row for row in draft["sources"] if row["source_id"] == "books")
    books["files"][0]["decision"] = "exclude"
    books["files"][0]["reason"] = "Fixture exclusion exercises an even population"
    draft_path.write_bytes(canonical_json(draft))
    selection = json.loads(selection_path.read_text())
    for spec in selection["sources"]:
        spec["strata"] = [
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
                "issues_nonempty": False,
            },
            {
                "id": "long",
                "count": 1,
                "path_prefix": None,
                "length_band": "long",
                "issues_nonempty": None,
            },
        ]
    selection_path.write_bytes(canonical_json(selection))
    project = load_project(recipe / "acquire.yaml")
    lock = inspection.verify_acquisition(project, root)
    # Independent reference materializes only this tiny fixture. It uses native
    # provenance validation, but no production median/pool/ranking implementation.
    rows = {source.id: [] for source in project.sources}
    inspection._scan_candidates(
        project,
        root,
        draft,
        lock,
        1048576,
        lambda source, row: rows[source].append(row),
    )
    summary = _inspect_again(root, fixture, "reference.json")
    assert summary["work"]["full_source_passes"] == len(project.sources)
    report = json.loads((recipe / "reference.json").read_text())
    populations = {}
    for spec in selection["sources"]:
        source_id = spec["source_id"]
        qualified = [row for row in rows[source_id] if row["decision"] == "qualify"]
        lengths = {
            row["item_id"]: len(
                clean_structure(
                    row["raw"], source_id=source_id, normalizer=selection["normalizer"]
                )["text"].encode()
            )
            for row in qualified
        }
        ordered = sorted(lengths.values())
        n = len(ordered)
        median = ordered[(n - 1) // 2] + ordered[n // 2] if n else 0
        populations[source_id] = n
        assigned = {group["id"]: [] for group in spec["strata"]}
        for row in qualified:
            for group in spec["strata"]:
                band, prefix, issues = (
                    group["length_band"],
                    group["path_prefix"],
                    group["issues_nonempty"],
                )
                short = 2 * lengths[row["item_id"]] <= median
                if (
                    (prefix is None or row["path"].startswith(prefix))
                    and (band is None or short == (band == "short"))
                    and (issues is None or bool(row["issues"]) == issues)
                ):
                    assigned[group["id"]].append(row)
                    break

        def rank(name, row, source_id=source_id):
            digest = hashlib.sha256(
                f"{selection['seed']} | {source_id} | {name} | {row['digest']}".encode()
            ).hexdigest()
            return digest, row["path"], row["row_index"] or 0

        selected, remaining = [], []
        for group in spec["strata"]:
            name, count = group["id"], group["count"]
            ordered_rows = sorted(
                assigned[name], key=lambda row, name=name, rank=rank: rank(name, row)
            )
            selected.extend((name, row["item_id"]) for row in ordered_rows[:count])
            if count:
                remaining.extend(
                    (rank(name, row), name, row["item_id"])
                    for row in ordered_rows[count:]
                )
            coverage = next(
                row
                for row in report["coverage"]
                if row["source_id"] == source_id and row["stratum"] == name
            )
            assert coverage["population"] == len(assigned[name])
        requested = sum(group["count"] for group in spec["strata"])
        selected.extend(
            (name, item)
            for _, name, item in sorted(remaining)[: requested - len(selected)]
        )
        assert [
            (row["stratum"], row["item_id"])
            for row in report["items"]
            if row["source_id"] == source_id
        ] == selected
        held = sorted(
            (row for row in rows[source_id] if row["decision"] == "quarantine"),
            key=lambda row, rank=rank: rank("quarantine", row),
        )[: spec["exception_count"]]
        assert [
            row["item_id"]
            for row in report["quarantine_exceptions"]
            if row["source_id"] == source_id
        ] == [row["item_id"] for row in held]
    assert populations["books"] == 10 and populations["incident"] == 11


def test_jsonl_offsets_and_hash_cover_blank_lines_and_unicode(tmp_path):
    from sparselab.corpus.jsonl_records import records_from_path

    path = tmp_path / "physical.jsonl"
    first = canonical_json({"text": "snowman ☃ and LF \\n"}) + b"\n"
    raw = b"\n \n" + first + b"\n" + b'{"text":"last"}'
    path.write_bytes(raw)
    records = list(
        records_from_path(path, expected_sha256=hashlib.sha256(raw).hexdigest())
    )
    assert [(row.physical_line, row.byte_offset) for row in records] == [
        (3, 3),
        (5, 4 + len(first)),
    ]
    assert records[1].raw == b'{"text":"last"}'
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        list(records_from_path(path, expected_sha256="0" * 64))


def test_public_cli_reuses_completed_inspection(private_inspection):
    root, fixture = private_inspection
    recipe, policy, draft, selection, _ = fixture
    declaration = json.loads(selection.read_text())
    declaration["seed"] = "fixture-public-cli"
    selection.write_bytes(canonical_json(declaration))
    command = [
        sys.executable,
        "-m",
        "sparselab",
        "--work-dir",
        str(root),
        "corpus",
        "inspect-admission",
        str(recipe / "acquire.yaml"),
        "--draft",
        str(draft),
        "--policy-document",
        str(policy),
        "--selection",
        str(selection),
        "--output",
        str(recipe / "cli.json"),
        "--max-input-bytes",
        "1048576",
        "--max-excerpt-bytes",
        "80",
        "--max-output-bytes",
        "262144",
    ]
    results = []
    for _ in range(2):
        child = subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=False
        )
        assert child.returncode == 0, child.stdout + child.stderr
        results.append(json.loads(child.stdout))
    assert results[0]["work"]["full_source_passes"] == 4
    assert results[1]["work"]["full_source_passes"] == 0
    assert results[1]["work"]["operation_cache_hits"] == 1
    assert results[0]["sha256"] == results[1]["sha256"]


def test_unavailable_cache_authority_falls_back_to_native_validation(
    private_inspection, monkeypatch
):
    from sparselab.corpus import admission_inspection_cache as cache_module

    root, fixture = private_inspection

    def unavailable(*args, **kwargs):
        raise ValueError("fixture unavailable verifier authority")

    monkeypatch.setattr(cache_module, "verifier_authority", unavailable)
    summary = _inspect_again(root, fixture, "no-authority.json")
    assert summary["work"]["full_source_passes"] == 4
    assert summary["work"]["cache_read_available"] is False
    assert (
        summary["work"]["cache_unavailable_reason"] == "verifier_authority_unavailable"
    )
    assert (fixture[0] / "no-authority.json").read_bytes() == fixture[4].read_bytes()


def test_cache_write_failure_preserves_other_completed_source_reuse(
    private_inspection, monkeypatch
):
    from sparselab.corpus import admission_inspection_cache as cache_module

    root, fixture = private_inspection
    draft = json.loads(fixture[2].read_text())
    entry = draft["sources"][0]
    decision = entry.get("records", entry.get("files"))[0]
    decision["reason"] = "Fixture changed review rationale"
    fixture[2].write_bytes(canonical_json(draft))

    def unavailable(*args, **kwargs):
        raise OSError("fixture read-only cache")

    monkeypatch.setattr(cache_module.os, "replace", unavailable)
    summary = _inspect_again(root, fixture, "cache-readonly.json")
    assert summary["work"]["full_source_passes"] == 1
    assert summary["work"]["source_cache_hits"] == 3
    assert summary["work"]["cache_read_available"] is True
    assert summary["work"]["cache_write_available"] is False
    assert summary["work"]["cache_unavailable_reason"] == "cache_write_unavailable"
    assert (fixture[0] / "cache-readonly.json").is_file()


def test_symlinked_optional_cache_is_not_followed_or_required(
    private_inspection, tmp_path
):
    root, fixture = private_inspection
    directory = root / "cache/admission-inspection"
    directory.rename(directory.with_name("retained-cache"))
    outside = tmp_path / "outside"
    outside.mkdir()
    directory.symlink_to(outside, target_is_directory=True)
    summary = _inspect_again(root, fixture, "unsafe-cache.json")
    assert summary["work"]["full_source_passes"] == 4
    assert summary["work"]["source_cache_hits"] == 0
    assert summary["work"]["cache_write_available"] is False
    assert summary["work"]["cache_unavailable_reason"] == "untrusted_cache_directory"
    assert not list(outside.iterdir())


@pytest.mark.parametrize("unavailable", ["key", "store", "store_file"])
def test_unavailable_verification_state_reports_cold_fallback(
    private_inspection, unavailable
):
    root, fixture = private_inspection
    directory = (
        Path(os.environ["XDG_CONFIG_HOME"]) / "sparselab"
        if unavailable == "key"
        else root / "cache/verification-v1"
    )
    if unavailable == "store_file":
        directory.rename(directory.with_name("retained-verification"))
        directory.write_bytes(b"not a proof directory")
        directory.chmod(0o600)
    else:
        directory.chmod(0o755)
    summary = _inspect_again(root, fixture, "unavailable-state.json")
    work = summary["work"]
    assert work["full_source_passes"] == 4
    assert work["operation_cache_hits"] == work["source_cache_hits"] == 0
    assert work["cache_read_available"] is False
    assert work["cache_write_available"] is False
    reason = "key" if unavailable == "key" else "store"
    assert work["cache_unavailable_reason"] == f"verification_{reason}_unavailable"
    assert (fixture[0] / "unavailable-state.json").read_bytes() == fixture[
        4
    ].read_bytes()


def test_git_file_cap_refuses_before_native_file_read(private_inspection, monkeypatch):
    root, fixture = private_inspection
    project = load_project(fixture[0] / "acquire.yaml")
    source = next(source for source in project.sources if source.kind == "git")
    lock = json.loads(
        (root / "corpora" / project.config.id / "acquisition.json").read_text()
    )
    files = Path(lock["sources"][source.id]["snapshot_path"]) / "files"
    original = Path.read_bytes

    def no_native_file_read(path):
        if path.is_relative_to(files):
            raise AssertionError("oversized Git file reached native read_bytes")
        return original(path)

    monkeypatch.setattr(inspection, "_MAX_FILE_BYTES", 1)
    monkeypatch.setattr(Path, "read_bytes", no_native_file_read)
    with pytest.raises(ValueError, match="inspection file-byte cap exceeded"):
        _inspect_again(root, fixture, "over-file-cap.json")
    assert not (fixture[0] / "over-file-cap.json").exists()
