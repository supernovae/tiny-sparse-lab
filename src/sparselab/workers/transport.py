"""Protocol-v1 bounded binary frames for worker stdio transport."""

from __future__ import annotations

import hashlib
import json
import struct
from typing import BinaryIO

MAX_HEADER = 1024 * 1024


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _read(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("truncated frame")
    return value


def write_frame(stream: BinaryIO, header: dict[str, object], attachments: list[tuple[str, bytes]] | None = None) -> None:
    attachments = attachments or []
    inventory = [{"name": name, "length": len(body), "sha256": hashlib.sha256(body).hexdigest()} for name, body in attachments]
    encoded = canonical({**header, "attachments": inventory})
    if len(encoded) > MAX_HEADER:
        raise ValueError("oversized header")
    stream.write(struct.pack(">I", len(encoded)) + encoded)
    stream.writelines(struct.pack(">Q", len(body)) + body for _, body in attachments)


def read_frame(stream: BinaryIO) -> tuple[dict[str, object], list[tuple[str, bytes]]]:
    size = struct.unpack(">I", _read(stream, 4))[0]
    if size > MAX_HEADER:
        raise ValueError("oversized header")
    header = json.loads(_read(stream, size))
    if not isinstance(header, dict) or header.get("protocol_version") != 1:
        raise ValueError("unsupported protocol version")
    result: list[tuple[str, bytes]] = []
    for item in header.get("attachments", []):
        if not isinstance(item, dict):
            raise TypeError("invalid attachment inventory")
        length = struct.unpack(">Q", _read(stream, 8))[0]
        body = _read(stream, length)
        if length != item.get("length") or hashlib.sha256(body).hexdigest() != item.get(
            "sha256"
        ):
            raise ValueError("attachment integrity failure")
        name = item.get("name")
        if not isinstance(name, str):
            raise TypeError("invalid attachment name")
        result.append((name, body))
    return header, result
