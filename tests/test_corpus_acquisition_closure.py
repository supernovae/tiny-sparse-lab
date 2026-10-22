"""Native full closure retains disk-backed rows through durable JSON and stdout."""

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from test_corpus_acquisition_steps import fixture, run

from sparselab.cli.main import main
from sparselab.corpus.acquisition import verify_acquisition
from sparselab.corpus.bounded_metadata import DiskRows, MetadataStore


def prepared(tmp_path, monkeypatch):
    recipe, project, root, ledger, calls, _ = fixture(tmp_path, monkeypatch)
    run(project, root, "one")
    run(project, root, "two")
    assert not (root / "corpora/example/acquisition.json").exists()
    before = len(calls)
    return recipe, project, root, ledger, calls, before


def invoke(monkeypatch, root, recipe):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sparselab", "--work-dir", str(root), "corpus", "acquire", str(recipe)],
    )
    main()


def test_full_cli_streams_receipt_rows_and_preserves_json(tmp_path, monkeypatch):
    recipe, project, root, _ledger, calls, before = prepared(tmp_path, monkeypatch)
    native_load = MetadataStore.load
    loaded = set()

    def tracked(store, path, **limits):
        value = native_load(store, path, **limits)
        assert store.max_database_bytes is None
        if path.name == "manifest.json":
            loaded.add(Path(path))
            assert isinstance(value["files"], DiskRows)
            assert isinstance(
                value["retrieval"]["shards"][0]["selected_rows"], DiskRows
            )
        elif path.name == "acquisition.json":
            for source in value["sources"].values():
                assert isinstance(
                    source["receipt"]["retrieval"]["shards"][0]["selected_rows"],
                    DiskRows,
                )
        return value

    original_dumps = json.dumps

    def no_whole_lock(value, *args, **kwargs):
        assert not (
            isinstance(value, dict) and "project_sha256" in value and "sources" in value
        ), "whole acquisition lock must be streamed"
        return original_dumps(value, *args, **kwargs)

    class ChunkedOutput(io.StringIO):
        def write(self, value):
            assert len(value) <= 1024, "stdout must be emitted in bounded chunks"
            return super().write(value)

    monkeypatch.setattr(MetadataStore, "load", tracked)
    monkeypatch.setattr(json, "dumps", no_whole_lock)
    output = ChunkedOutput()
    with redirect_stdout(output):
        invoke(monkeypatch, root, recipe)
    snapshots = set((root / "corpora/example/snapshots").glob("*/*/manifest.json"))
    assert snapshots and snapshots <= loaded
    lock_path = root / "corpora/example/acquisition.json"
    lock = json.loads(lock_path.read_bytes())
    assert json.loads(output.getvalue()) == lock
    assert output.getvalue() == original_dumps(lock, sort_keys=True, default=str) + "\n"
    assert (
        lock_path.read_bytes()
        == original_dumps(
            lock,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    assert all(
        entry["receipt"]["retrieval"]["shards"][0]["selected_rows"]
        for entry in lock["sources"].values()
    )
    monkeypatch.setattr(MetadataStore, "load", native_load)
    assert verify_acquisition(project, root) == lock
    assert len(calls) == before
    assert not list(root.rglob(".metadata-*"))


def test_partial_lock_write_refuses_publication_and_cleans_scratch(
    tmp_path, monkeypatch
):
    recipe, _project, root, _ledger, calls, before = prepared(tmp_path, monkeypatch)
    native_iterencode = json.JSONEncoder.iterencode

    def fail_lock(self, value, *args, **kwargs):
        if isinstance(value, dict) and "project_sha256" in value and "sources" in value:
            yield "{"
            raise OSError("fixture lock write failure")
        yield from native_iterencode(self, value, *args, **kwargs)

    monkeypatch.setattr(json.JSONEncoder, "iterencode", fail_lock)
    with pytest.raises(OSError, match="fixture lock write failure"):
        invoke(monkeypatch, root, recipe)
    base = root / "corpora/example"
    assert not (base / "acquisition.json").exists()
    assert not list(base.glob(".acquisition-*"))
    assert len(list((base / "snapshots").glob("*/*/manifest.json"))) == 2
    assert not list(root.rglob(".metadata-*"))
    assert len(calls) == before
