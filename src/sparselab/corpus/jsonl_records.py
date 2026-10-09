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


def iter_lf_lines(stream: BinaryIO) -> Iterator[tuple[int, bytes]]:
    """Yield raw physical lines, including blank lines and a final unterminated line."""
    yield from enumerate(stream, 1)


def iter_jsonl_records(stream: BinaryIO, *, source: str) -> Iterator[JSONLRecord]:
    """Decode UTF-8 strictly and parse one JSON value per nonblank LF record."""
    for physical_line, raw in iter_lf_lines(stream):
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


def records_from_path(path: Path) -> Iterator[JSONLRecord]:
    with path.open("rb") as stream:
        yield from iter_jsonl_records(stream, source=str(path))
