"""Emit an absolute phase deadline from the one persistent Card 05 ledger."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

from sparselab.training.attempt_budget import AttemptBudget


def deadline_ns(phase: str, ledger: Path, root: Path) -> int:
    if phase not in {"stage", "train", "evaluate"}:
        raise ValueError("invalid full-tranche phase")
    status = AttemptBudget(ledger).status()
    if status["max_updates"] != 4883 or status["max_wall_seconds"] > 10800:
        raise ValueError("wrong full-tranche ledger")
    start = int(datetime.fromisoformat(status["started_at_utc"]).timestamp() * 1e9)
    aggregate = start + 10_800_000_000_000
    if phase == "evaluate":
        marker = root / "evaluation-start-ns.txt"
        value = time.time_ns()
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"{value}\n")
            stream.flush()
            os.fsync(stream.fileno())
        phase_deadline = value + 7_200_000_000_000
    else:
        phase_deadline = start + 3_000_000_000_000
    deadline = min(phase_deadline, aggregate)
    if time.time_ns() >= deadline:
        raise ValueError("full-tranche phase deadline reached")
    return deadline


if __name__ == "__main__":
    print(
        deadline_ns(
            sys.argv[1],
            Path(os.environ["SPARSELAB_ATTEMPT_BUDGET_LEDGER"]),
            Path(os.environ["KML_PROFILE_ROOT"]),
        )
    )
