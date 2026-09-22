"""Frozen capability and attempt records for independent workers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sparselab.runtime import RuntimeInfo


@dataclass(frozen=True)
class WorkerCapabilities:
    schema_version: int
    worker_id: str
    name: str
    transport: Literal['local','ssh']
    engine: str
    backend: str
    device_index: int
    runtime: RuntimeInfo
    supported_precisions: tuple[str,...]
    features: tuple[str,...]
    max_concurrent_runs: int = 1
    validation_status: Literal['unverified','passed','failed'] = 'unverified'
    status: Literal['idle','busy','offline','unknown'] = 'unknown'
    protocol_versions: tuple[int,...] = (1,)

@dataclass(frozen=True)
class AttemptReceipt:
    attempt_id: str
    run_id: str
    status: Literal['PREPARED','RUNNING','COMPLETE','FAILED','INTERRUPTED','UNKNOWN']
    worker_id: str
