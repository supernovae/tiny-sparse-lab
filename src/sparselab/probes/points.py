"""Comparable points: one checkpoint's quality and cost, from sealed records.

Shared by ``sparselab compare`` and the dashboard's Pareto view (no Streamlit
import here). Sources are verified lab records only (``lab_records``):

* try records: each arm's held-out score (with its ``eval_group``) and cost,
  so runs without a probe battery still appear;
* probe records: the battery's target and baseline, including lm-eval;
* packaged reference results (``probes/reference_results/*.json``): sealed
  probe records of pinned public checkpoints, so nothing is downloaded to plot
  or compare against them.

A metric value only compares within its group: held-out loss within one
``eval_group`` (validation data, tokenizer, loss mask, protocol), lm-eval
accuracy within one ``benchmark_group`` (tasks, task versions, shots, items).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from sparselab.lab_records import iter_lab_records, read_lab_record


@dataclass(frozen=True)
class Metric:
    id: str
    label: str
    group_label: str
    higher_is_better: bool
    not_comparable: str
    missing_hint: str


METRICS = {
    "heldout_loss": Metric(
        id="heldout_loss",
        label="held-out loss",
        group_label="eval group",
        higher_is_better=False,
        not_comparable="different eval group (validation data, tokenizer, loss "
        "mask or eval protocol differ)",
        missing_hint="score it on our validation split (`sparselab try` or "
        "`sparselab probe`)",
    ),
    "lm_eval": Metric(
        id="lm_eval",
        label="lm-eval accuracy",
        group_label="benchmark group",
        higher_is_better=True,
        not_comparable="different benchmark group (tasks, limit, few-shot or "
        "task versions differ)",
        missing_hint="run `sparselab probe RUN --tier full` (needs the lmeval extra)",
    ),
}

# Cost axes: resident counts every weight held in memory, active counts what
# one token touches (one embedding row, the routed experts and memory rows).
COSTS = {
    "parameters": "resident parameters",
    "active_parameters": "active parameters / token",
    "parameter_bytes": "resident weight bytes",
    "active_parameter_bytes": "active weight bytes / token",
    "tokens_seen": "training tokens",
    "ms_per_token": "scoring latency (ms / token)",
}
IDENTITY_FIELDS = (
    "run_id",
    "checkpoint_sha256",
    "step",
    "tokens_seen",
    *(c for c in COSTS if c not in {"tokens_seen", "ms_per_token"}),
)


def packaged_reference_records() -> list[tuple[Path, dict[str, Any]]]:
    """Sealed reference results shipped with SparseLab (verified on read)."""
    folder = resources.files("sparselab.probes") / "reference_results"
    out = []
    for item in sorted(folder.iterdir(), key=lambda p: p.name):
        if item.name.endswith(".json"):
            path = Path(str(item))
            out.append((path, read_lab_record(path, "probe")[1]))
    return out


def _row(result: Mapping[str, Any], probe: str) -> Mapping[str, Any]:
    for row in result.get("probes") or []:
        if row.get("id") == probe:
            return row
    return {}


def _lm_eval_items(tasks: Mapping[str, Any] | None) -> dict[str, list[float]]:
    return {task: list(row.get("items") or []) for task, row in (tasks or {}).items()}


def probe_points(
    result: Mapping[str, Any],
    *,
    source: str,
    path: Path | None,
    packaged: bool,
    try_record: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Target and baseline of one probe battery as points.

    A battery inside a lab try names its points like the try's arms
    (``TRY_ID:candidate``), so one checkpoint reads the same in every view.
    """
    loss = _row(result, "heldout_loss")
    lm = _row(result, "lm_eval")
    loss_details, lm_details = loss.get("details") or {}, lm.get("details") or {}
    points = []
    for role, who in (
        ("candidate", result.get("target")),
        ("baseline", result.get("baseline")),
    ):
        if not who or not who.get("checkpoint_sha256"):
            continue
        candidate = role == "candidate"
        metrics: dict[str, dict[str, Any]] = {}
        value = loss.get("value" if candidate else "baseline_value")
        if value is not None and loss.get("status") not in {"error", "not_comparable"}:
            metrics["heldout_loss"] = {
                "value": value,
                "group": who.get("eval_group"),
                "window_sums": loss_details.get("window_sums") if candidate else None,
                "window_counts": loss_details.get("window_counts")
                if candidate
                else None,
            }
        value = lm.get("value" if candidate else "baseline_value")
        if value is not None and lm_details.get("benchmark_group"):
            tasks = lm_details.get("tasks" if candidate else "baseline_tasks")
            metrics["lm_eval"] = {
                "value": value,
                # A baseline row only exists when both sides shared the group.
                "group": lm_details["benchmark_group"],
                "tasks": {t: r.get("acc") for t, r in (tasks or {}).items()},
                "items": _lm_eval_items(tasks),
                "chance": lm_details.get("chance"),
            }
        reference = who.get("reference")
        label = f"{who.get('run_id')}@{who.get('step')}"
        if reference:
            label = reference["name"]
        elif try_record is not None:
            label = f"{try_record.get('try_id')}:{role}"
        points.append(
            {
                **{k: who.get(k) for k in IDENTITY_FIELDS},
                "label": label,
                "kind": "reference"
                if reference
                else "try"
                if try_record is not None
                else "probe",
                "role": role,
                "reference": reference,
                "packaged": packaged,
                "ms_per_token": loss_details.get("ms_per_token") if candidate else None,
                "verdict": (result.get("verdict") or {}).get("status")
                if candidate
                else "baseline",
                "source": source,
                "path": str(path) if path else None,
                "created_at": str(
                    (try_record or result).get("created_at")  # type: ignore[union-attr]
                ),
                "metrics": metrics,
            }
        )
    return points


def try_points(record: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    """Each scored arm of a lab try (whether or not it was probed)."""
    points = []
    for role, arm in (record.get("arms") or {}).items():
        heldout = arm.get("heldout") or {}
        if not arm.get("checkpoint_sha256") or heldout.get("loss") is None:
            continue
        points.append(
            {
                **{k: arm.get(k) for k in IDENTITY_FIELDS},
                "label": f"{record.get('try_id')}:{role}",
                "kind": "try",
                "role": role,
                "reference": None,
                "packaged": False,
                "ms_per_token": heldout.get("ms_per_token"),
                "verdict": (record.get("comparison") or {}).get("verdict"),
                "source": str(record.get("try_id")),
                "path": str(path),
                "created_at": str(record.get("created_at")),
                "question": record.get("question"),
                "metrics": {
                    "heldout_loss": {
                        "value": heldout["loss"],
                        # Older records carry no group: shown, never compared.
                        "group": arm.get("eval_group"),
                        "window_sums": heldout.get("window_sums"),
                        "window_counts": heldout.get("window_counts"),
                    }
                },
            }
        )
    return points


def points_from_record(
    kind: str, record: Mapping[str, Any], path: Path, *, packaged: bool = False
) -> list[dict[str, Any]]:
    if kind == "try":
        out = try_points(record, path)
        probe = record.get("probe")
        if isinstance(probe, dict) and "format" in probe:
            out += probe_points(
                probe,
                source=str(record.get("try_id")),
                path=path,
                packaged=False,
                try_record=record,
            )
        return out
    return probe_points(
        record,
        source=str(record.get("probe_id") or path.parent.name),
        path=path,
        packaged=packaged,
    )


def collect_points(
    lab_dir: Path | None, *, references: bool = True, limit: int = 500
) -> list[dict[str, Any]]:
    """All points, newest first (packaged references last, i.e. oldest)."""
    points: list[dict[str, Any]] = []
    if lab_dir is not None and lab_dir.is_dir():
        accepted, _ = iter_lab_records(lab_dir, limit)
        for kind, path, record in accepted:
            points += points_from_record(kind, record, path)
    points.sort(key=lambda p: p["created_at"], reverse=True)
    if references:
        for path, record in packaged_reference_records():
            points += points_from_record("probe", record, path, packaged=True)
    return points


def metric_points(points: Iterable[Mapping[str, Any]], metric: str) -> list[dict]:
    """Flat points for one metric, one per (group, checkpoint); first wins.

    Pass points newest first: the same checkpoint scored again in the same
    group is one point (the newest); scored under another group it is a
    separate point that never shares a chart or a comparison with this one.
    """
    seen: set[tuple[Any, Any]] = set()
    out = []
    for point in points:
        value = (point.get("metrics") or {}).get(metric)
        if not value or value.get("value") is None:
            continue
        key = (value.get("group"), point.get("checkpoint_sha256"))
        if key in seen:
            continue
        seen.add(key)
        flat = {k: v for k, v in point.items() if k != "metrics"}
        out.append({**flat, **value, "metric": metric})
    return out


def pareto_frontier(
    points: Sequence[tuple[float, float]], *, maximize: bool = False
) -> list[int]:
    """Indices of points not dominated (lower cost, better quality).

    Cost (x) is always minimized; quality (y) is minimized unless MAXIMIZE.
    """
    sign = -1.0 if maximize else 1.0
    order = sorted(
        range(len(points)), key=lambda i: (points[i][0], sign * points[i][1])
    )
    frontier: list[int] = []
    best = float("inf")
    for index in order:
        if sign * points[index][1] < best:
            frontier.append(index)
            best = sign * points[index][1]
    return frontier
