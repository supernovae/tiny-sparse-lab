"""Human-visible preprocessing progress without changing scientific inputs."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def heartbeat(stage: str, *, interval_seconds: float = 30.0) -> Iterator[None]:
    """Emit flushed start/heartbeat/finish records around blocking work."""
    started = time.monotonic()
    stop = threading.Event()

    def emit(kind: str, **values: object) -> None:
        print(
            json.dumps(
                {"preprocessing": stage, "event": kind, **values}, sort_keys=True
            ),
            file=sys.stderr,
        )

    def beat() -> None:
        while not stop.wait(interval_seconds):
            emit("heartbeat", elapsed_seconds=round(time.monotonic() - started, 1))

    emit("started")
    thread = threading.Thread(target=beat, name=f"progress-{stage}", daemon=True)
    thread.start()
    try:
        yield
    except BaseException:
        emit("failed", elapsed_seconds=round(time.monotonic() - started, 1))
        raise
    else:
        emit("finished", elapsed_seconds=round(time.monotonic() - started, 1))
    finally:
        stop.set()
        thread.join()
