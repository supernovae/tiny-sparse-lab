"""Tiny v2 policy fixtures: disk joins must not inherit v3 promotion rules."""

import json
import sqlite3

import pytest

from sparselab.corpus.large_build import _deduplicate
from sparselab.corpus.progress import BuildProgress
from sparselab.training.manifest import canonical_json


def index(tmp_path, *, split="train", incompatible=False):
    db = sqlite3.connect(tmp_path / "index.sqlite")
    db.executescript("""
        CREATE TABLE documents(document_id TEXT UNIQUE, payload BLOB, split TEXT,
            representative_id TEXT, drop_reason TEXT);
        CREATE TABLE origins(method TEXT, origin TEXT, document_number INTEGER);
        CREATE INDEX origins_key ON origins(method,origin,document_number);
    """)
    rights = {
        "training_eligibility": "eligible_with_obligations",
        "license_references": ["https://example.invalid/license"],
        "notices": ["Fixture attribution"],
        "training_restriction": None,
    }
    for number, identifier in enumerate(("a", "b", "c", "d"), 1):
        payload = {
            "license": "MIT",
            "redistribution": "metadata_reconstruction_only",
            "rights": {
                **rights,
                "notices": ["Different"]
                if incompatible and identifier == "c"
                else rights["notices"],
            },
        }
        db.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?)",
            (
                identifier,
                canonical_json(payload),
                split if identifier == "c" else "train",
                None,
                None,
            ),
        )
        # Raw a/b plus normalized b/c exercises native transitive compression.
        db.execute(
            "INSERT INTO origins VALUES(?,?,?)",
            ("raw", "raw-ab" if number < 3 else identifier, number),
        )
        db.execute(
            "INSERT INTO origins VALUES(?,?,?)",
            ("normalized", "norm-bc" if number in (2, 3) else identifier, number),
        )
        # v2 must NOT apply the v3-only canonical-origin relation.
        db.execute(
            "INSERT INTO origins VALUES(?,?,?)", ("canonical_origin", "all", number)
        )
    db.commit()
    return db


def test_v2_disk_dedup_keeps_native_representatives_and_ignores_origin_union(tmp_path):
    db = index(tmp_path)
    try:
        result = _deduplicate(
            db,
            tmp_path / "groups.jsonl",
            diagnostics=tmp_path / "diagnostics.json",
            progress=BuildProgress(tmp_path / "progress.jsonl", "fixture"),
            schema_version=2,
        )
        assert result == (2, 2)
        assert list(
            db.execute(
                "SELECT document_id,representative_id,drop_reason FROM documents ORDER BY document_id"
            )
        ) == [
            ("a", "a", None),
            ("b", "a", "duplicate"),
            ("c", "a", "duplicate"),
            ("d", "d", None),
        ]
        groups = [
            json.loads(row)
            for row in (tmp_path / "groups.jsonl").read_text().splitlines()
        ]
        assert groups == [
            {
                "method": "raw",
                "sha256": "raw-ab",
                "origins": ["a", "b"],
                "representative": "a",
                "splits": ["train"],
            },
            {
                "method": "normalized",
                "sha256": "norm-bc",
                "origins": ["b", "c"],
                "representative": "b",
                "splits": ["train"],
            },
        ]
        assert (tmp_path / "groups.jsonl").read_bytes() == b"".join(
            canonical_json(row) + b"\n" for row in groups
        )
    finally:
        db.close()


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("heldout", "cross-split exact text overlap"),
        ("rights", "incompatible rights evidence"),
    ],
)
def test_v2_disk_dedup_refuses_v3_style_cross_split_or_incompatible_rights(
    tmp_path, fault, expected
):
    db = index(
        tmp_path,
        split="validation" if fault == "heldout" else "train",
        incompatible=fault == "rights",
    )
    try:
        with pytest.raises(ValueError, match=expected):
            _deduplicate(
                db,
                tmp_path / "groups.jsonl",
                diagnostics=tmp_path / "diagnostics.json",
                progress=BuildProgress(tmp_path / "progress.jsonl", "fixture"),
                schema_version=2,
            )
        diagnostic = json.loads((tmp_path / "diagnostics.json").read_text())
        assert expected in diagnostic["error"]
        assert diagnostic["duplicates"][-1]["origins"] == ["b", "c"]
    finally:
        db.close()


@pytest.mark.parametrize("observed", [None, 8, 7])
def test_admitted_inventory_sql_joins_require_exact_qualified_rows(tmp_path, observed):
    from sparselab.corpus.large_build import _verify_admitted_inventory

    db = sqlite3.connect(tmp_path / "coverage.sqlite")
    db.execute("CREATE TABLE documents(source_id TEXT,payload BLOB)")
    if observed is not None:
        db.execute(
            "INSERT INTO documents VALUES(?,?)",
            (
                "fixture",
                canonical_json(
                    {
                        "source_location": "sample.jsonl#row=1",
                        "metadata": {"source_row_index": observed},
                    }
                ),
            ),
        )
    admissions = {
        ("fixture", "sample.jsonl"): {"decisions": {7: {"decision": "qualify"}}}
    }
    try:
        if observed == 7:
            _verify_admitted_inventory(db, admissions)
            assert not list(
                db.execute(
                    "SELECT name FROM sqlite_master WHERE name LIKE 'admission_coverage_%'"
                )
            )
        else:
            with pytest.raises(
                ValueError, match="admitted source row inventory differs"
            ):
                _verify_admitted_inventory(db, admissions)
    finally:
        db.close()


def test_cold_admission_coverage_rejects_duplicates_and_cleans_on_failure(tmp_path):
    from sparselab.corpus.admission_coverage import AdmissionCoverage
    from sparselab.corpus.bounded_metadata import metadata_operation

    with metadata_operation(tmp_path) as store:
        with (
            pytest.raises(ValueError, match="duplicate admitted source row"),
            AdmissionCoverage(store.connection) as coverage,
        ):
            coverage.expect("s", "p", 1)
            coverage.observe("s", "p", 1, unique=True)
            coverage.observe("s", "p", 1, unique=True)
        assert not list(
            store.connection.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'admission_coverage_%'"
            )
        )
    assert not list(tmp_path.glob(".metadata-*"))


def test_admission_coverage_uses_the_metadata_page_cap(tmp_path):
    from sparselab.corpus.admission_coverage import AdmissionCoverage
    from sparselab.corpus.bounded_metadata import metadata_operation

    with (
        metadata_operation(tmp_path, max_database_bytes=32768) as store,
        pytest.raises(ValueError, match="database byte cap exceeded"),
        AdmissionCoverage(store.connection) as coverage,
    ):
        for index in range(100):
            coverage.expect("s", "x" * 2048, index)
    assert not list(tmp_path.glob(".metadata-*"))
