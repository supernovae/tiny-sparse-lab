"""Append-only experiment stage records and validated transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class ExperimentStage(StrEnum):
    CONFIGURED = "CONFIGURED"
    INSPECTED = "INSPECTED"
    VALIDATED = "VALIDATED"
    SMOKE_TEST = "SMOKE_TEST"
    WARMUP = "WARMUP"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    CHECKPOINTED = "CHECKPOINTED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True)
class StageRecord:
    stage: ExperimentStage
    status: str
    started_at: str
    finished_at: str | None = None
    step: int = 0
    tokens_seen: int = 0
    reason: str | None = None


class StageHistory:
    def __init__(self) -> None:
        self.records: list[StageRecord] = []

    def start(
        self, stage: ExperimentStage, *, step: int = 0, tokens_seen: int = 0
    ) -> StageRecord:
        record = StageRecord(
            stage,
            "running",
            datetime.now(UTC).isoformat(),
            step=step,
            tokens_seen=tokens_seen,
        )
        self.records.append(record)
        return record

    def finish(
        self,
        status: str = "complete",
        *,
        reason: str | None = None,
        step: int = 0,
        tokens_seen: int = 0,
    ) -> StageRecord:
        if not self.records:
            raise ValueError("no active stage")
        last = self.records[-1]
        record = StageRecord(
            last.stage,
            status,
            last.started_at,
            datetime.now(UTC).isoformat(),
            step,
            tokens_seen,
            reason,
        )
        self.records[-1] = record
        return record
