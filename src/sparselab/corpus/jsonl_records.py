"""Read physical LF-delimited JSON records without Unicode line splitting.

Blank physical lines are ignored, as in the existing corpus JSONL readers.
The physical line number still includes them; source-row ordinals count only
nonblank records. Source bytes are never rewritten by this reader.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


@dataclass(frozen=True)
class JSONLRecord:
    physical_line: int
    value: Any
    raw: bytes


def iter_lf_lines(
    stream: BinaryIO, *, max_record_bytes: int | None = None
) -> Iterator[tuple[int, bytes]]:
    """Yield raw physical lines, including blank lines and a final unterminated line."""
    if max_record_bytes is None:
        yield from enumerate(stream, 1)
        return
    if type(max_record_bytes) is not int or max_record_bytes <= 0:
        raise ValueError("record-byte cap must be a positive integer")
    physical_line = 0
    while raw := stream.readline(max_record_bytes + 1):
        if len(raw) > max_record_bytes:
            raise ValueError("JSONL record-byte cap exceeded")
        physical_line += 1
        yield physical_line, raw


def iter_jsonl_records(
    stream: BinaryIO, *, source: str, max_record_bytes: int | None = None
) -> Iterator[JSONLRecord]:
    """Decode UTF-8 strictly and parse one JSON value per nonblank LF record."""
    for physical_line, raw in iter_lf_lines(stream, max_record_bytes=max_record_bytes):
        if not raw.strip():
            continue
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"invalid UTF-8 in {source} physical line {physical_line}"
            ) from error
        try:
            value = json.loads(decoded)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"malformed JSON in {source} physical line {physical_line}: {error.msg}"
            ) from error
        yield JSONLRecord(physical_line, value, raw)


def records_from_bytes(raw: bytes, *, source: str) -> Iterator[JSONLRecord]:
    return iter_jsonl_records(io.BytesIO(raw), source=source)


def records_from_path(
    path: Path, *, max_record_bytes: int | None = None
) -> Iterator[JSONLRecord]:
    with path.open("rb") as stream:
        yield from iter_jsonl_records(
            stream, source=str(path), max_record_bytes=max_record_bytes
        )
