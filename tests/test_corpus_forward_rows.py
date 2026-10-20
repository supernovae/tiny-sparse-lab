"""Bounded cold-replay cursor: source ordinals, full validation and closure."""

import hashlib

import pytest

from sparselab.corpus.jsonl_records import forward_rows


def cursor(path, payload, cap=1024):
    path.write_bytes(payload)
    return forward_rows(
        path, max_record_bytes=cap, expected_sha256=hashlib.sha256(payload).hexdigest()
    )


def test_forward_replay_keeps_nonblank_ordinals_and_current_row(tmp_path):
    with cursor(
        tmp_path / "rows.jsonl", b'\n{"text":"a"}\n\n{"text":"b"}\n{"text":"c"}'
    ) as rows:
        assert rows.get(2).value == {"text": "b"}
        assert rows.get(2).physical_line == 4
        assert rows.get(3).value == {"text": "c"}
        with pytest.raises(ValueError, match="increasing"):
            rows.get(1)


@pytest.mark.parametrize("suffix", [b"broken\n", b'{"text":"' + b"x" * 100 + b'"}\n'])
def test_unselected_tail_still_validated(tmp_path, suffix):
    with (
        pytest.raises(ValueError),
        cursor(tmp_path / "rows.jsonl", b'{"text":"a"}\n' + suffix, cap=64) as rows,
    ):
        assert rows.get(1).value == {"text": "a"}


def test_skipped_bad_row_and_out_of_range_refuse(tmp_path):
    with (
        pytest.raises(ValueError, match="malformed"),
        cursor(tmp_path / "rows.jsonl", b"broken\n{}\n") as rows,
    ):
        rows.get(2)
    with (
        pytest.raises(ValueError, match="exceeds"),
        cursor(tmp_path / "rows.jsonl", b"{}\n") as rows,
    ):
        rows.get(2)


def test_changed_source_refuses_at_completion(tmp_path):
    path = tmp_path / "rows.jsonl"
    with pytest.raises(ValueError, match="changed"), cursor(path, b"{}\n") as rows:
        rows.get(1)
        path.write_bytes(b"[]\n")


def test_exception_closes_reader_without_masking_original_failure(
    tmp_path, monkeypatch
):
    from sparselab.corpus import jsonl_records

    closed = []
    original = jsonl_records.records_from_path

    def tracked(*args, **kwargs):
        try:
            yield from original(*args, **kwargs)
        finally:
            closed.append(True)

    monkeypatch.setattr(jsonl_records, "records_from_path", tracked)
    with (
        pytest.raises(RuntimeError, match="caller failed"),
        cursor(tmp_path / "rows.jsonl", b"{}\nbroken\n") as rows,
    ):
        rows.get(1)
        raise RuntimeError("caller failed")
    assert closed == [True]
