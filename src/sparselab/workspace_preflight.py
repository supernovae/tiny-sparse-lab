"""Conservative filesystem headroom checks before durable run publication."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from sparselab.config.models import RunConfig, TokenizerTrainConfig
from sparselab.model.inspection import inspection_report
from sparselab.workdir import resolve_work_dir

_RESERVE_BYTES = 256 * 1024 * 1024
_RESERVE_INODES = 128


@dataclass(frozen=True)
class StorageCheck:
    path: str
    filesystem_path: str
    available_bytes: int
    available_inodes: int | None
    projected_bytes: int
    projected_inodes: int
    reserve_bytes: int
    reserve_inodes: int
    status: str


def _existing_ancestor(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        if candidate == candidate.parent:
            raise FileNotFoundError(f"no existing ancestor for {path}")
        candidate = candidate.parent
    if not candidate.is_dir():
        raise NotADirectoryError(candidate)
    return candidate


def check_storage(
    path: Path,
    *,
    projected_bytes: int,
    projected_inodes: int = 64,
    reserve_bytes: int = _RESERVE_BYTES,
    reserve_inodes: int = _RESERVE_INODES,
) -> StorageCheck:
    """Check a path's actual filesystem without creating the destination."""
    if min(projected_bytes, projected_inodes, reserve_bytes, reserve_inodes) < 0:
        raise ValueError("storage projections and reserves must be non-negative")
    ancestor = _existing_ancestor(path)
    stats = os.statvfs(ancestor)
    available_bytes = stats.f_bavail * stats.f_frsize
    available_inodes = stats.f_favail if stats.f_files else None
    enough_bytes = available_bytes >= projected_bytes + reserve_bytes
    enough_inodes = (
        available_inodes is None
        or available_inodes >= projected_inodes + reserve_inodes
    )
    return StorageCheck(
        path=str(path.expanduser().resolve()),
        filesystem_path=str(ancestor),
        available_bytes=available_bytes,
        available_inodes=available_inodes,
        projected_bytes=projected_bytes,
        projected_inodes=projected_inodes,
        reserve_bytes=reserve_bytes,
        reserve_inodes=reserve_inodes,
        status="adequate" if enough_bytes and enough_inodes else "insufficient",
    )


def require_storage(checks: list[StorageCheck]) -> list[dict[str, object]]:
    failures = [item for item in checks if item.status == "insufficient"]
    if failures:
        details = "; ".join(
            f"{item.path}: free {item.available_bytes} bytes / "
            f"{item.available_inodes} inodes, need {item.projected_bytes + item.reserve_bytes} "
            f"bytes / {item.projected_inodes + item.reserve_inodes} inodes"
            for item in failures
        )
        raise OSError(
            f"workspace storage preflight failed before publication: {details}"
        )
    return [asdict(item) for item in checks]


def projected_data_bytes(config: RunConfig) -> int:
    """Upper planning allowance for prepared token and common sidecar arrays."""
    targets = config.dataset.train_max_tokens + config.dataset.validation_max_tokens
    per_token = 32 + 4 * (config.model.semantic_memory_dim or 0)
    return math.ceil(1.25 * targets * per_token) + 32 * 1024 * 1024


def training_storage_checks(
    config: RunConfig, *, work_dir: Path | None = None, run_dir: Path | None = None
) -> list[StorageCheck]:
    checkpoint = int(inspection_report(config)["estimated_checkpoint_bytes"])
    # Two retained generations and one in-flight generation. The estimate omits
    # serialization metadata and temporary buffers, hence the 25% allowance.
    checkpoint_growth = math.ceil(3.75 * checkpoint)
    cache_growth = projected_data_bytes(config)
    destinations = (
        (
            run_dir if run_dir is not None else config.logging.root_dir,
            checkpoint_growth,
            256,
        ),
        (config.dataset.cache_dir, cache_growth, 128),
        (
            work_dir if work_dir is not None else resolve_work_dir(),
            max(64 * 1024 * 1024, checkpoint // 4),
            128,
        ),
    )
    grouped: dict[int, tuple[Path, int, int]] = {}
    for path, bytes_needed, inodes_needed in destinations:
        ancestor = _existing_ancestor(path)
        device = ancestor.stat().st_dev
        prior = grouped.get(device)
        if prior is None:
            grouped[device] = (path, bytes_needed, inodes_needed)
        else:
            grouped[device] = (
                prior[0],
                prior[1] + bytes_needed,
                prior[2] + inodes_needed,
            )
    return [
        check_storage(
            path, projected_bytes=bytes_needed, projected_inodes=inodes_needed
        )
        for path, bytes_needed, inodes_needed in grouped.values()
    ]


def tokenizer_storage_checks(config: TokenizerTrainConfig) -> list[StorageCheck]:
    # Input text is bounded, but a remote source's download cache is not. Keep
    # that uncertainty visible in documentation; this checks known local growth.
    known_bytes = max(64 * 1024 * 1024, config.dataset.train_max_tokens * 2)
    if (
        _existing_ancestor(config.output_dir).stat().st_dev
        == _existing_ancestor(config.dataset.cache_dir).stat().st_dev
    ):
        return [check_storage(config.output_dir, projected_bytes=2 * known_bytes)]
    return [
        check_storage(config.output_dir, projected_bytes=known_bytes),
        check_storage(config.dataset.cache_dir, projected_bytes=known_bytes),
    ]
