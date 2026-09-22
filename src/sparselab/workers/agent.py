"""Finite protocol-v1 stdio worker endpoint."""
from __future__ import annotations

import sys

from sparselab.runtime import discover_runtimes
from sparselab.workers.transport import read_frame, write_frame


def serve_once() -> None:
    header, _ = read_frame(sys.stdin.buffer)
    request_id = header.get("request_id")
    operation = header.get("op")
    if operation == "discover":
        result = {"runtimes": [info.as_dict() for info in discover_runtimes()]}
        write_frame(
            sys.stdout.buffer,
            {"protocol_version": 1, "request_id": request_id, "ok": True, "result": result, "error": None},
        )
        return
    write_frame(
        sys.stdout.buffer,
        {"protocol_version": 1, "request_id": request_id, "ok": False, "result": None, "error": {"code": "UNKNOWN_OP", "message": "unsupported operation", "retryable": False}},
    )

if __name__ == "__main__":
    serve_once()
