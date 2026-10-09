"""Offline, zero-model-work checks for physical LF JSONL record identity."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_corpus_record_admission import _fixture

from sparselab.corpus import split_inventory
from sparselab.corpus.jsonl_records import iter_jsonl_records, records_from_bytes
from sparselab.corpus.large_build import _prepare_file
from sparselab.corpus.pipeline import _records_for_file
from sparselab.corpus.progress import BuildProgress
from sparselab.corpus.release import _iter_rows, _rows
from sparselab.corpus.rights import resolve_file_rights, verify_record_admission
from sparselab.training.manifest import canonical_json, sha256_file

SPECIAL = "paragraph\u0085separator\u2028line\u2029end"


def test_reader_uses_only_lf_and_preserves_blank_physical_lines():
    raw = (
        b" \r\n"
        + canonical_json({"text": SPECIAL})
        + b"\r\n\t\n"
        + canonical_json({"text": "last"})
    )
    records = list(records_from_bytes(raw, source="fixture.jsonl"))
    assert [record.physical_line for record in records] == [2, 4]
    assert [record.value["text"] for record in records] == [SPECIAL, "last"]
    assert records[0].raw.endswith(b"\r\n")
    assert records[1].raw == canonical_json({"text": "last"})
    assert list(iter_jsonl_records(io.BytesIO(raw), source="fixture.jsonl")) == records


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"text":', "malformed JSON.*physical line 1"),
        (b'{"text":"\xff"}', "invalid UTF-8.*physical line 1"),
        (b"\n{broken}\n", "malformed JSON.*physical line 2"),
    ],
)
def test_reader_fails_clearly_on_bad_physical_record(raw: bytes, message: str):
    with pytest.raises(ValueError, match=message):
        list(records_from_bytes(raw, source="fixture.jsonl"))


def test_ingestion_inventory_stream_build_and_release_readback_keep_row_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admission, source_payload, snapshot, original, source = _fixture(tmp_path)
    rows = [
        json.loads(record.raw) for record in records_from_bytes(original, source="old")
    ]
    rows[0]["text"] = SPECIAL + " ordinary prose to preserve the document. " * 4
    for index, row in enumerate(rows):
        original_row = {
            key: value for key, value in row.items() if key != "_sparselab_source"
        }
        digest = hashlib.sha256(canonical_json(original_row)).hexdigest()
        row["_sparselab_source"]["source_row_sha256"] = digest
        snapshot["retrieval"]["shards"][0]["selected_rows"][index][
            "source_row_sha256"
        ] = digest
        admission["sources"][0]["records"][index]["source_row_sha256"] = digest
    # A blank physical line is ignored, CRLF is accepted, and the last record
    # has no LF; neither source bytes nor selected row ordinals are rewritten.
    raw = b" \r\n" + canonical_json(rows[0]) + b"\r\n" + canonical_json(rows[1])
    snapshot["files"][0].update(sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))
    admission["sources"][0]["sample_sha256"] = snapshot["files"][0]["sha256"]
    sample = (
        tmp_path
        / source.id
        / snapshot["snapshot_sha256"]
        / "files"
        / snapshot["files"][0]["path"]
    )
    sample.write_bytes(raw)
    (sample.parent.parent / "manifest.json").write_text(json.dumps(snapshot))
    before = sha256_file(sample)
    verify_record_admission(
        admission,
        {source.id: source_payload},
        {source.id: snapshot},
        tmp_path,
    )
    project = SimpleNamespace(
        config=SimpleNamespace(id="jsonl-fixture"), sources=[source]
    )
    monkeypatch.setattr(
        split_inventory,
        "verify_acquisition",
        lambda *_: {
            "sources": {
                source.id: {
                    "snapshot_path": str(sample.parent.parent),
                    "snapshot_sha256": snapshot["snapshot_sha256"],
                }
            }
        },
    )
    inventory_path = tmp_path / "inventory.jsonl"
    split_inventory.write_split_inventory(project, tmp_path, inventory_path)
    inventory = [record.value for record in _iter_records(inventory_path)]
    assert [row["source_row_index"] for row in inventory] == [0, 1]
    rights = resolve_file_rights(source.rights, sample.name, raw)
    legacy_pairs = _records_for_file(
        raw, sample.name, source, snapshot["snapshot_sha256"], file_rights=rights
    )
    assert [row["document_id"] for row, _ in legacy_pairs] == [
        row["document_id"] for row in inventory
    ]
    assert [row["metadata"]["source_row_sha256"] for row, _ in legacy_pairs] == [
        item["source_row_sha256"]
        for item in snapshot["retrieval"]["shards"][0]["selected_rows"]
    ]
    assert SPECIAL in legacy_pairs[0][0]["text"]
    prepared = _prepare_file(
        build_id="lf-jsonl-test",
        prepared_root=tmp_path / "prepared",
        source=source,
        file=snapshot["files"][0],
        snapshot_sha=snapshot["snapshot_sha256"],
        source_path=sample,
        decision=rights,
        progress=BuildProgress(tmp_path / "progress.jsonl", "lf-jsonl-test"),
    )
    prepared_docs = _rows(prepared / "docs.jsonl")
    assert prepared_docs == [row for row, _ in legacy_pairs]
    assert list(_iter_rows(prepared / "docs.jsonl")) == prepared_docs
    assert sha256_file(sample) == before == snapshot["files"][0]["sha256"]


def _iter_records(path: Path):
    from sparselab.corpus.jsonl_records import records_from_path

    return records_from_path(path)
