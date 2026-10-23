"""Strict JSON strings and byte limits across streaming buffer boundaries."""

import json
from pathlib import Path

import pytest

from sparselab.corpus.bounded_metadata import DiskRows, MetadataStore


@pytest.mark.parametrize("padding", range(4088, 4105))
@pytest.mark.parametrize("ascii_only", [True, False])
def test_strings_preserve_escapes_utf8_and_spilled_rows(tmp_path, padding, ascii_only):
    value = {
        "padding": "x" * padding,
        "selected_rows": [
            {'key\\"雪': '雪\\"\n\t\u2028' * 300},
            {"empty": "", "tail": "\\" * 8 + '"'},
        ],
        "tail": "done",
    }
    raw = json.dumps(value, ensure_ascii=ascii_only, separators=(",", ":")).encode()
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    with MetadataStore(tmp_path / "scratch", max_database_bytes=1024**2) as store:
        loaded = store.load(path, max_input_bytes=len(raw))
        assert isinstance(loaded["selected_rows"], DiskRows)
        assert loaded == value


@pytest.mark.parametrize(
    "raw",
    [
        b'{"records":[{"value":"bad\\q"}]}',
        b'{"records":[{"value":"bad\nline"}]}',
        b'{"records":[{"value":"bad\x00byte"}]}',
        b'{"records":[{"value":"bad\xffutf8"}]}',
        b'{"records":[{"value":"truncated\\',
        b'{"records":[{"value":"unterminated',
        rb'{"records":[{"same":1,"sa\u006de":2}]}',
    ],
)
def test_malformed_strings_and_decoded_duplicate_keys_refuse(tmp_path, raw):
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    with (
        MetadataStore(tmp_path / "scratch", max_database_bytes=1024**2) as store,
        pytest.raises(ValueError),
    ):
        store.load(path, max_input_bytes=len(raw))


@pytest.mark.parametrize("field", ["header", "record"])
def test_exact_string_byte_limits_and_overflow_inside_run(tmp_path, monkeypatch, field):
    item = {"label": "雪" + "x" * 8192 + '\\"'}
    value = item if field == "header" else {"records": [item]}
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    row = json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode()
    # A spilled record also accounts for its array delimiter.
    exact = len(raw) if field == "header" else len(row) + 1
    key = "max_header_bytes" if field == "header" else "max_record_bytes"
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    reads = []
    original = Path.open

    class ObservedInput:
        def __init__(self, stream):
            self.stream = stream

        def read(self, size):
            data = self.stream.read(size)
            reads.append(len(data))
            return data

        def fileno(self):
            return self.stream.fileno()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.stream.close()

    def observe(selected, mode="r", *args, **kwargs):
        stream = original(selected, mode, *args, **kwargs)
        return ObservedInput(stream) if selected == path and mode == "rb" else stream

    monkeypatch.setattr(Path, "open", observe)
    with MetadataStore(tmp_path / "scratch", max_database_bytes=1024**2) as store:
        assert store.load(path, max_input_bytes=len(raw), **{key: exact}) == value
        for limit in (exact - 1, 64):
            reads.clear()
            with pytest.raises(ValueError, match=f"metadata {field}-byte cap exceeded"):
                store.load(path, max_input_bytes=len(raw), **{key: limit})
            # Refuse during the oversized span, not after reading it all.
            assert sum(reads) <= limit + 4096


def test_growing_input_still_refuses_at_input_byte_limit(tmp_path, monkeypatch):
    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    raw = json.dumps({"label": "x" * 8192}).encode()
    original = Path.open

    def grow_on_open(selected, mode="r", *args, **kwargs):
        if selected == path and mode == "rb":
            path.write_bytes(raw)
        return original(selected, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", grow_on_open)
    with (
        MetadataStore(tmp_path / "scratch", max_database_bytes=1024**2) as store,
        pytest.raises(ValueError, match="metadata input-byte cap exceeded"),
    ):
        store.load(path, max_input_bytes=64)
