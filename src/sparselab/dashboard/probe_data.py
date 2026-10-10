"""Read-only probe history for the dashboard (no Streamlit import here).

Sources under the lab root: ``tries/*/try.json`` (the ``probe`` block of a lab
try) and ``probes/*/probe.json`` (``sparselab probe``). Live progress comes from
``tries/*/probe-progress.json`` and ``probes/*/progress.json``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LIVE_STATES = {"loading", "running"}
STALE_SECONDS = 600


@dataclass(frozen=True)
class ProbeEntry:
    key: str
    source: str  # "try" or "probe"
    created_at: str
    path: Path
    result: dict[str, Any]
    question: str | None
    delta: dict[str, Any]
    comparison: dict[str, Any] | None

    @property
    def label(self) -> str:
        verdict = self.result.get("verdict") or {}
        title = (
            self.question
            or ", ".join(self.delta)
            or self.result["target"].get("run_id")
        )
        return (
            f"{self.key} · {title} · {str(verdict.get('status', '?')).upper()} → "
            f"{verdict.get('action', '?')}"
        )


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return value if isinstance(value, dict) else None


def load_entries(lab_dir: Path, limit: int = 500) -> list[ProbeEntry]:
    """Newest first; unreadable or probe-less records are skipped."""
    entries: list[ProbeEntry] = []
    for path in sorted((lab_dir / "tries").glob("*/try.json"))[-limit:]:
        record = _read(path)
        probe = (record or {}).get("probe")
        if not isinstance(probe, dict) or "format" not in probe:
            continue
        entries.append(
            ProbeEntry(
                key=str(record.get("try_id")),
                source="try",
                created_at=str(record.get("created_at") or probe.get("created_at")),
                path=path,
                result=probe,
                question=record.get("question"),
                delta=record.get("delta") or {},
                comparison=record.get("comparison"),
            )
        )
    for path in sorted((lab_dir / "probes").glob("*/probe.json"))[-limit:]:
        result = _read(path)
        if not result or "verdict" not in result:
            continue
        entries.append(
            ProbeEntry(
                key=str(result.get("probe_id") or path.parent.name),
                source="probe",
                created_at=str(result.get("created_at")),
                path=path,
                result=result,
                question=None,
                delta={},
                comparison=None,
            )
        )
    entries.sort(key=lambda entry: entry.created_at, reverse=True)
    return entries


def live_batteries(lab_dir: Path, now: datetime | None = None) -> list[dict[str, Any]]:
    """Batteries whose progress file says running and was updated recently."""
    now = now or datetime.now(UTC)
    live = []
    files = [
        *(lab_dir / "tries").glob("*/probe-progress.json"),
        *(lab_dir / "probes").glob("*/progress.json"),
    ]
    for path in files:
        state = _read(path)
        if not state or state.get("state") not in LIVE_STATES:
            continue
        try:
            updated = datetime.fromisoformat(str(state.get("updated_at")))
        except ValueError:
            updated = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        if (now - updated).total_seconds() > STALE_SECONDS:
            continue
        live.append({**state, "key": path.parent.name})
    return live


def probe_row(result: dict[str, Any], probe_id: str) -> dict[str, Any] | None:
    for row in result.get("probes") or []:
        if row.get("id") == probe_id:
            return row
    return None


def history_rows(entries: Iterable[ProbeEntry]) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        result = entry.result
        verdict = result.get("verdict") or {}
        loss = probe_row(result, "heldout_loss") or {}
        recall = probe_row(result, "fact_recall") or {}
        needle = probe_row(result, "needle") or {}
        target = result.get("target") or {}
        details = loss.get("details") or {}
        rows.append(
            {
                "when": entry.created_at,
                "id": entry.key,
                "source": entry.source,
                "idea": entry.question
                or ", ".join(entry.delta)
                or target.get("run_id"),
                "verdict": verdict.get("status"),
                "action": verdict.get("action"),
                "tier": result.get("tier"),
                "suite": (result.get("suite") or {}).get("sha256", "")[:8],
                "target": target.get("run_id"),
                "loss": loss.get("value"),
                "loss_delta": loss.get("delta"),
                "recall": recall.get("value"),
                "needle": needle.get("value"),
                "parameters": target.get("parameters"),
                "parameter_bytes": target.get("parameter_bytes"),
                "tokens_seen": target.get("tokens_seen"),
                "ms_per_token": details.get("ms_per_token"),
                "overfit_suspected": (result.get("guard") or {}).get(
                    "overfit_suspected"
                ),
            }
        )
    return rows


def pareto_frontier(
    points: Sequence[tuple[float, float]],
) -> list[int]:
    """Indices of points not dominated when minimizing both coordinates."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    frontier: list[int] = []
    best = float("inf")
    for index in order:
        if points[index][1] < best:
            frontier.append(index)
            best = points[index][1]
    return frontier
