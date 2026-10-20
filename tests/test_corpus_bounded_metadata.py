"""Tiny fixtures for bounded metadata parsing; no corpus or runtime work."""

import json
from pathlib import Path

import pytest

from sparselab.corpus.bounded_metadata import (
    DiskRows,
    MetadataStore,
    metadata_operation,
    read_metadata,
)
from sparselab.training.manifest import canonical_json


def test_disk_rows_preserve_json_semantics_and_close(tmp_path):
    value = {
        "sources": [{"records": [{"id": i, "text": 'é\\"\n'} for i in range(12)]}],
        "selected_rows": [{"index": 2}, {"index": 5}],
        "empty": [],
        "metadata": {"bool": True, "null": None},
    }
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_json(value))
    original = path.read_bytes()
    with metadata_operation(tmp_path) as store:
        actual = read_metadata(path)
        rows = actual["sources"][0]["records"]
        assert isinstance(rows, DiskRows) and isinstance(rows, list)
        assert len(rows) == 12 and rows[-1] == value["sources"][0]["records"][-1]
        assert actual == value and canonical_json(actual) == original
        assert rows[0] in rows
        with pytest.raises(TypeError, match="read-only"):
            rows.append({})
        with metadata_operation(tmp_path) as nested:
            assert nested is store
        directory = Path(store.temporary.name)
        assert directory.exists()
    assert not directory.exists() and path.read_bytes() == original


@pytest.mark.parametrize(
    "raw",
    [
        b'{"records":[{},]}',
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":1e999}',
        b'{"records":[{"x":-1e999}]}',
        b'{"records":[',
        b'{"records":[]}junk',
        b'{"x":"\xff"}',
    ],
)
def test_malformed_metadata_refuses_and_cleans(tmp_path, raw):
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    with pytest.raises((ValueError, UnicodeError)), metadata_operation(tmp_path):
        read_metadata(path)
    assert not list(tmp_path.glob(".metadata-*"))


@pytest.mark.parametrize(
    "limit,raw,error",
    [
        ({"max_input_bytes": 3}, b'{"a":1}', "input-byte"),
        ({"max_header_bytes": 8}, b'{"a":"too long"}', "header-byte"),
        ({"max_record_bytes": 8}, b'{"records":[{"a":"too long"}]}', "record-byte"),
    ],
)
def test_metadata_limits_refuse(tmp_path, limit, raw, error):
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    with (
        MetadataStore(tmp_path, max_database_bytes=65536) as store,
        pytest.raises(ValueError, match=error),
    ):
        store.load(path, **{"max_input_bytes": 1024, **limit})


def test_database_limit_refuses(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"records": [{"text": "x" * 1024} for _ in range(32)]}))
    with (
        MetadataStore(tmp_path, max_database_bytes=16384) as store,
        pytest.raises(ValueError, match="database byte cap"),
    ):
        store.load(path, max_input_bytes=65536)


def test_chunk_boundaries_and_later_large_arrays_do_not_fill_header(tmp_path):
    value = {
        "records": [{"text": "é" * 2000, "id": i} for i in range(8)],
        "tail": "verified",
    }
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_json(value))
    with MetadataStore(tmp_path, max_database_bytes=131072) as store:
        actual = store.load(
            path, max_input_bytes=65536, max_header_bytes=128, max_record_bytes=8192
        )
        assert actual == value
        assert actual["tail"] == "verified"


def test_disk_decision_lookup_coverage_and_lifetime(tmp_path):
    from sparselab.corpus.bounded_metadata import DiskDecisions, decision_index

    with metadata_operation(tmp_path):
        index = decision_index()
        assert isinstance(index, DiskDecisions)
        index[7] = {"decision": "qualify"}
        index[2] = {"decision": "quarantine"}
        assert len(index) == 2 and set(index) == {2, 7}
        assert index.get(7) == {"decision": "qualify"} and index.get(9) is None
        assert list(index.items()) == [
            (2, {"decision": "quarantine"}),
            (7, {"decision": "qualify"}),
        ]
        assert list(index.values()) == [
            {"decision": "quarantine"},
            {"decision": "qualify"},
        ]
        with pytest.raises(ValueError, match="duplicate"):
            index[7] = {}
    assert decision_index() == {}


@pytest.mark.parametrize("key", ["7", 7.0, True, None, -1, 2**63, []])
def test_decision_index_does_not_coerce_keys(tmp_path, key):
    from sparselab.corpus.bounded_metadata import decision_index

    with metadata_operation(tmp_path):
        index = decision_index()
        index[7] = {"decision": "qualify"}
        assert key not in index
        assert index.get(key) is None
        with pytest.raises(KeyError):
            index[key]
        with pytest.raises(ValueError, match="64-bit integer"):
            index[key] = {}
        assert list(index.items()) == [(7, {"decision": "qualify"})]


def test_nesting_and_symlink_refusal(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("[" * 66 + "0" + "]" * 66)
    with metadata_operation(tmp_path), pytest.raises(ValueError, match="nesting cap"):
        read_metadata(path)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with metadata_operation(tmp_path), pytest.raises(ValueError, match="unsafe"):
        read_metadata(link)
    assert not list(tmp_path.glob(".metadata-*"))
