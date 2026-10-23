"""Operation-local metadata reuse; tiny files, exact bytes and parse counts."""

import json
import os

import pytest

from sparselab.corpus import bounded_metadata as metadata
from sparselab.training.manifest import canonical_json


def _input(tmp_path):
    path = tmp_path / "manifest.json"
    value = {
        "records": [{"id": i, "text": 'é\\"\n'} for i in range(12)],
        "header": {"items": [1, {"ok": True}]},
    }
    path.write_bytes(canonical_json(value))
    return path, value


def _parses(monkeypatch):
    original = metadata._Reader.read
    calls = []

    def tracked(reader):
        calls.append(reader.store.next_identifier)
        return original(reader)

    monkeypatch.setattr(metadata._Reader, "read", tracked)
    return calls


def test_unchanged_input_hashes_but_does_not_parse_or_insert_again(
    tmp_path, monkeypatch
):
    path, value = _input(tmp_path)
    original = path.read_bytes()
    calls = _parses(monkeypatch)
    digests = []
    content_digest = metadata._content_digest

    def tracked_hash(stream, limit):
        digests.append(stream.name)
        return content_digest(stream, limit)

    monkeypatch.setattr(metadata, "_content_digest", tracked_hash)
    with metadata.metadata_operation(tmp_path) as store:
        first = metadata.read_metadata(path)
        collections = store.next_identifier
        second = metadata.read_metadata(path)
        assert len(calls) == 1
        assert len(digests) == 1
        assert store.next_identifier == collections
        assert second == first == value
        assert canonical_json(second) == original
        assert first is not second and first["records"] is not second["records"]
    assert path.read_bytes() == original


def test_headers_rows_and_owning_store_appends_do_not_poison_other_readers(tmp_path):
    path, value = _input(tmp_path)
    with metadata.metadata_operation(tmp_path) as store:
        first = metadata.read_metadata(path)
        other = metadata.read_metadata(path)
        first["header"]["items"][1]["ok"] = False
        first["records"][0]["id"] = 1000
        store.append(first["records"], {"id": 20})
        assert len(first["records"]) == 13
        assert first["records"][-1] == {"id": 20}
        assert other == metadata.read_metadata(path) == value
        assert len(list(other["records"])) == len(other["records"]) == 12
        assert first["records"].identifier != other["records"].identifier


def test_warm_hash_stops_reading_at_bound_when_input_grows(tmp_path, monkeypatch):
    path, _ = _input(tmp_path)
    limit = path.stat().st_size
    original = type(path).open
    reads = []

    class GrowingInput:
        def __init__(self, stream):
            self.stream = stream
            self.grown = False

        def read(self, size):
            if not self.grown:
                with original(path, "ab") as out:
                    out.write(b" " * 4096)
                self.grown = True
            raw = self.stream.read(size)
            reads.append(len(raw))
            return raw

        def fileno(self):
            return self.stream.fileno()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.stream.close()

    def grow(selected, mode="r", *args, **kwargs):
        stream = original(selected, mode, *args, **kwargs)
        return GrowingInput(stream) if selected == path and mode == "rb" else stream

    with metadata.MetadataStore(tmp_path, max_database_bytes=None) as store:
        store.load(path, max_input_bytes=limit)
        monkeypatch.setattr(type(path), "open", grow)
        with pytest.raises(ValueError, match="input-byte cap"):
            store.load(path, max_input_bytes=limit)
    assert sum(reads) <= limit + 1


@pytest.mark.parametrize("change", ["overwrite", "replace", "coarse_timestamps"])
def test_same_size_input_changes_never_reuse_stale_rows(tmp_path, monkeypatch, change):
    path, value = _input(tmp_path)
    calls = _parses(monkeypatch)
    if change == "coarse_timestamps":
        monkeypatch.setattr(metadata, "_fingerprint", lambda info: "unchanged")
    with metadata.metadata_operation(tmp_path):
        assert metadata.read_metadata(path) == value
        before = path.stat()
        raw = path.read_bytes().replace(b'"id":0', b'"id":9')
        assert raw != path.read_bytes() and len(raw) == before.st_size
        if change == "replace":
            other = path.with_name("replacement.json")
            other.write_bytes(raw)
            other.replace(path)
        else:
            path.write_bytes(raw)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert metadata.read_metadata(path)["records"][0]["id"] == 9
        assert len(calls) == 2


@pytest.mark.parametrize(
    "limit,match",
    [
        ("max_input_bytes", "input-byte"),
        ("max_header_bytes", "header-byte"),
        ("max_record_bytes", "record-byte"),
    ],
)
def test_cache_does_not_bypass_stricter_parser_limits(tmp_path, limit, match):
    path, _ = _input(tmp_path)
    with metadata.MetadataStore(tmp_path, max_database_bytes=131072) as store:
        store.load(path, max_input_bytes=65536)
        with pytest.raises(ValueError, match=match):
            store.load(path, **{"max_input_bytes": 65536, limit: 1})


def test_changed_during_read_and_interrupted_import_are_not_cached(
    tmp_path, monkeypatch
):
    path, _ = _input(tmp_path)
    original = metadata._Reader.read
    with metadata.metadata_operation(tmp_path) as store:

        def changed(reader):
            value = original(reader)
            path.write_bytes(path.read_bytes() + b" ")
            return value

        monkeypatch.setattr(metadata._Reader, "read", changed)
        with pytest.raises(ValueError, match="changed while reading"):
            metadata.read_metadata(path)
        assert not store._loaded

        def interrupted(reader):
            original(reader)
            raise KeyboardInterrupt

        monkeypatch.setattr(metadata._Reader, "read", interrupted)
        with pytest.raises(KeyboardInterrupt):
            metadata.read_metadata(path)
        assert not store._loaded
        monkeypatch.setattr(metadata._Reader, "read", original)
        assert metadata.read_metadata(path) == json.loads(path.read_bytes())
    assert not list(tmp_path.glob(".metadata-*"))


def test_reuse_scope_nested_context_eviction_and_new_operation(tmp_path, monkeypatch):
    path, value = _input(tmp_path)
    calls = _parses(monkeypatch)
    monkeypatch.setattr(metadata, "_REUSED_INPUTS", 1)
    with metadata.metadata_operation(tmp_path) as store:
        assert metadata.read_metadata(path) == value
        with metadata.metadata_operation(tmp_path):
            assert metadata.read_metadata(path) == value
        assert len(calls) == 1
        another = path.with_name("another.json")
        another.write_bytes(path.read_bytes())
        metadata.read_metadata(another)
        assert len(store._loaded) == 1
        assert metadata.read_metadata(path) == value
        assert len(calls) == 3
    with metadata.metadata_operation(tmp_path):
        assert metadata.read_metadata(path) == value
    assert len(calls) == 4


def test_header_budget_evicts_or_skips_instead_of_refusing_work(tmp_path, monkeypatch):
    path, value = _input(tmp_path)
    calls = _parses(monkeypatch)
    monkeypatch.setattr(metadata, "_REUSED_HEADER_BYTES", 1)
    with metadata.metadata_operation(tmp_path) as store:
        assert metadata.read_metadata(path) == metadata.read_metadata(path) == value
        assert len(calls) == 2 and not store._loaded


def test_released_rows_invalidate_only_affected_materializations(tmp_path, monkeypatch):
    path, value = _input(tmp_path)
    calls = _parses(monkeypatch)
    with metadata.metadata_operation(tmp_path) as store:
        assert metadata.read_metadata(path) == value
        boundary = store.next_identifier
        other = path.with_name("another.json")
        other.write_bytes(path.read_bytes())
        assert metadata.read_metadata(other) == value
        assert len(calls) == 2
        store.release_after(boundary)
        assert metadata.read_metadata(path) == value
        assert len(calls) == 2  # Earlier cached rows survived scoped cleanup.
        assert metadata.read_metadata(other) == value
        assert len(calls) == 3  # Released rows must be materialized again.


def test_cached_input_still_refuses_symlinks_and_malformed_replacement(tmp_path):
    path, _ = _input(tmp_path)
    with metadata.metadata_operation(tmp_path):
        metadata.read_metadata(path)
        target = path.with_name("original.json")
        path.rename(target)
        path.symlink_to(target)
        with pytest.raises(ValueError, match="unsafe"):
            metadata.read_metadata(path)
        path.unlink()
        path.write_bytes(b'{"records":[{},]}')
        with pytest.raises(ValueError):
            metadata.read_metadata(path)


@pytest.mark.parametrize("failed_write", ["load", "append"])
def test_sqlite_rollback_cannot_leave_a_stale_reuse_hit(tmp_path, failed_write):
    path = tmp_path / "small.json"
    value = {"records": [{"id": 1}]}
    path.write_bytes(canonical_json(value))
    with metadata.MetadataStore(tmp_path, max_database_bytes=16384) as store:
        first = store.load(path, max_input_bytes=100000)
        with pytest.raises(ValueError, match="database byte cap"):
            if failed_write == "load":
                large = tmp_path / "large.json"
                large.write_bytes(
                    canonical_json({"records": [{"text": "x" * 1024}] * 64})
                )
                store.load(large, max_input_bytes=100000)
            else:
                store.append(first["records"], {"text": "x" * 40000})
        assert store.load(path, max_input_bytes=100000) == value


def test_owner_append_copy_keeps_other_collections_independent(tmp_path):
    path, value = _input(tmp_path)
    with metadata.metadata_operation(tmp_path) as store:
        first = metadata.read_metadata(path)["records"]
        second = metadata.read_metadata(path)["records"]
        store.append(first, {"id": 13})
        store.append(second, {"id": 14})
        assert first[-1] == {"id": 13} and second[-1] == {"id": 14}
        assert metadata.read_metadata(path)["records"] == value["records"]
        assert len({first.identifier, second.identifier}) == 2
