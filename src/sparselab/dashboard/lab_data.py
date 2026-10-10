"""What the lab pages show, computed without Streamlit (testable).

Everything comes from verified lab records (``lab_records``), the shared points
pool (``probes.points``) and the explorer cache; nothing here re-scores or
re-judges a result. Each view answers one researcher question:

* Home: what happened, and what should I do next (:func:`next_steps`);
* Experiments: did my change help (:func:`probe_trends`, activity);
* Models: where does each checkpoint sit (:func:`checkpoint_catalog`,
  :func:`unscored_runs`);
* Behaviors: what does the model actually do (:func:`generations`,
  :func:`recall_items`, :func:`needle_curves`, :func:`calibration_curves`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sparselab.dashboard.probe_data import (
    ProbeEntry,
    load_history,
    probe_row,
    result_group,
)
from sparselab.explorer import cached_explorations
from sparselab.lab_records import iter_lab_records
from sparselab.probes.points import METRICS, collect_points

# Lab-try comparison verdicts in plain words (``comparison.verdict``).
TRY_VERDICT_TEXT = {
    "CANDIDATE_LOWER_LOSS": "candidate has lower held-out loss",
    "CANDIDATE_HIGHER_LOSS": "candidate has higher held-out loss",
    "NO_LOSS_DIFFERENCE": "no held-out loss difference",
    "NOT_COMPARABLE": "arms are not comparable",
    "INCOMPLETE": "try did not finish",
}


@dataclass
class LabSnapshot:
    lab_dir: Path
    entries: list[ProbeEntry]
    tries: list[tuple[Path, dict[str, Any]]]
    probes: list[tuple[Path, dict[str, Any]]]
    rejected: list[tuple[Path, str]]
    points: list[dict[str, Any]]
    explorations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.tries and not self.probes


def snapshot(lab_dir: Path) -> LabSnapshot:
    accepted, rejected = iter_lab_records(lab_dir) if lab_dir.is_dir() else ([], [])
    entries, _ = load_history(lab_dir) if lab_dir.is_dir() else ([], [])
    newest = lambda item: str(item[1].get("created_at"))
    return LabSnapshot(
        lab_dir=lab_dir,
        entries=entries,
        tries=sorted(
            ((p, r) for k, p, r in accepted if k == "try"), key=newest, reverse=True
        ),
        probes=sorted(
            ((p, r) for k, p, r in accepted if k == "probe"), key=newest, reverse=True
        ),
        rejected=rejected,
        points=collect_points(lab_dir),
        explorations=cached_explorations(lab_dir) if lab_dir.is_dir() else [],
    )


# --- Home ---------------------------------------------------------------------


def activity(snap: LabSnapshot, limit: int = 12) -> list[dict[str, Any]]:
    """Recent tries and probe batteries, newest first, verdict first."""
    rows = []
    for _, record in snap.tries:
        comparison = record.get("comparison") or {}
        probe = record.get("probe") if isinstance(record.get("probe"), dict) else {}
        verdict = (probe or {}).get("verdict") or {}
        rows.append(
            {
                "when": str(record.get("created_at")),
                "id": str(record.get("try_id")),
                "kind": "try",
                "what": record.get("question")
                or ", ".join(record.get("delta") or {})
                or "lab try",
                "outcome": TRY_VERDICT_TEXT.get(
                    str(comparison.get("verdict")),
                    comparison.get("verdict") or record.get("status"),
                ),
                "loss_delta": comparison.get("heldout_loss_delta"),
                "verdict": verdict.get("status"),
                "action": verdict.get("action"),
            }
        )
    for _, record in snap.probes:
        verdict = record.get("verdict") or {}
        target = record.get("target") or {}
        rows.append(
            {
                "when": str(record.get("created_at")),
                "id": str(record.get("probe_id")),
                "kind": "probe",
                "what": f"probe {target.get('run_id')} ({record.get('tier')} tier)",
                "outcome": verdict.get("suggestion"),
                "loss_delta": (probe_row(record, "heldout_loss") or {}).get("delta"),
                "verdict": verdict.get("status"),
                "action": verdict.get("action"),
            }
        )
    rows.sort(key=lambda r: r["when"], reverse=True)
    return rows[:limit]


def _run_id(point: Mapping[str, Any]) -> str | None:
    run = point.get("run_id")
    return str(run) if run and not str(run).startswith("ref:") else None


def next_steps(
    snap: LabSnapshot, runs_dirs: Iterable[Path] = (), limit: int = 4
) -> list[dict[str, str]]:
    """Concrete next commands, most useful first, each with why.

    Derived only from recorded verdicts and which evidence is missing; the
    latest verdict's own suggestion always comes first.
    """
    steps: list[dict[str, str]] = []
    if snap.empty:
        return [
            {
                "title": "Run your first experiment",
                "why": "Train a baseline and one change side by side; the fast "
                "probe tier runs automatically and gives a verdict.",
                "command": "sparselab try DELTA.yaml --vs BASELINE.yaml",
            },
            {
                "title": "See the reference curve",
                "why": "Pinned public models (Pythia, SmolLM2) ship with sealed "
                "scores, so you can read the Models page before training.",
                "command": "sparselab compare --list-references",
            },
        ]
    if snap.entries:
        # The latest verdict leads when it names a command; otherwise (abandon,
        # tweak, longer run) the verdict banner already says what to do.
        latest = snap.entries[0]
        verdict = latest.result.get("verdict") or {}
        target = (latest.result.get("target") or {}).get("run_id")
        action = verdict.get("action")
        baseline = (latest.result.get("baseline") or {}).get("run_id")
        command = {
            "escalate": f"sparselab probe {target} --tier {verdict.get('next_tier')}"
            + (f" --vs {baseline}" if baseline else ""),
            "compare": f"sparselab compare {latest.key} --references",
            "rerun": f"sparselab probe {target} --tier {latest.result.get('tier')}"
            + (f" --vs {baseline}" if baseline else ""),
        }.get(str(action))
        if command:
            steps.append(
                {
                    "title": f"Latest verdict: {str(verdict.get('status', '?')).upper()}"
                    f" → {action}",
                    "why": str(verdict.get("suggestion") or ""),
                    "command": command,
                }
            )
    probed = {
        p.get("checkpoint_sha256")
        for p in snap.points
        if "fact_recall" in (p.get("metrics") or {})
    }
    unprobed = [
        p
        for p in snap.points
        if p["kind"] == "try"
        and p["role"] == "candidate"
        and p.get("checkpoint_sha256") not in probed
        and _run_id(p)
    ]
    if unprobed:
        run = _run_id(unprobed[0])
        steps.append(
            {
                "title": "Check behavior, not only loss",
                "why": f"{len(unprobed)} candidate checkpoint(s) have held-out loss "
                "but no fact-recall/needle evidence yet.",
                "command": f"sparselab probe {run} --tier standard",
            }
        )
    lm = {
        p.get("checkpoint_sha256")
        for p in snap.points
        if "lm_eval" in (p.get("metrics") or {})
    }
    lab_runs = [p for p in snap.points if _run_id(p) and p["role"] == "candidate"]
    if lab_runs and not any(p.get("checkpoint_sha256") in lm for p in lab_runs):
        steps.append(
            {
                "title": "Place a run on the reference curve",
                "why": "No lab checkpoint has lm-eval accuracy yet, so none can "
                "sit next to Pythia/SmolLM2.",
                "command": f"sparselab probe {_run_id(lab_runs[0])} --tier full",
            }
        )
    plain = unscored_runs(runs_dirs, snap.points)
    if plain:
        steps.append(
            {
                "title": "Score your plain training runs",
                "why": f"{len(plain)} run(s) from `sparselab train` have no lab "
                "record yet, so they are missing from the catalog and the Pareto "
                "view.",
                "command": plain[0]["command"],
            }
        )
    explored = {
        (e.get("target") or {}).get("checkpoint_sha256") for e in snap.explorations
    }
    small = [p for p in lab_runs if p.get("checkpoint_sha256") not in explored]
    if small:
        steps.append(
            {
                "title": "Look inside a model",
                "why": "See its attention patterns, per-token loss and (for MoE "
                "or memory runs) routing and table lookups.",
                "command": f"sparselab explore {_run_id(small[0])}",
            }
        )
    return steps[:limit]


# --- Experiments --------------------------------------------------------------


def probe_trends(entries: Iterable[ProbeEntry]) -> list[dict[str, Any]]:
    """One row per (result, probe) with a value: plot a probe over time.

    ``trend_group`` is the comparison identity (suite digest + the probe's
    eval/benchmark/item group); only rows sharing it lie on one scale and may
    be joined into a trend. ``None`` when the record predates its group.
    """
    rows = []
    for entry in entries:
        suite_sha = (entry.result.get("suite") or {}).get("sha256", "")
        suite = suite_sha[:8]
        for row in entry.result.get("probes") or []:
            if row.get("value") is None:
                continue
            group = result_group(entry.result, row)
            rows.append(
                {
                    "when": entry.created_at,
                    "id": entry.key,
                    "probe": row["id"],
                    "title": row.get("title"),
                    "value": row["value"],
                    "baseline_value": row.get("baseline_value"),
                    "delta": row.get("delta"),
                    "delta_se": row.get("delta_se"),
                    "status": row.get("status"),
                    "higher_is_better": row.get("higher_is_better"),
                    "suite": suite,
                    "group": group,
                    "trend_group": f"{suite_sha}:{group}" if group else None,
                }
            )
    return rows


def tries_without_probes(snap: LabSnapshot) -> list[dict[str, Any]]:
    rows = []
    for _, record in snap.tries:
        if isinstance(record.get("probe"), dict) and "format" in record["probe"]:
            continue
        candidate = (record.get("arms") or {}).get("candidate") or {}
        rows.append(
            {
                "id": record.get("try_id"),
                "what": record.get("question") or ", ".join(record.get("delta") or {}),
                "outcome": TRY_VERDICT_TEXT.get(
                    str((record.get("comparison") or {}).get("verdict")), None
                ),
                "command": f"sparselab probe {candidate.get('run_id')} --tier standard",
            }
        )
    return rows


# --- Models -------------------------------------------------------------------


def checkpoint_catalog(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per checkpoint: cost, newest value per metric, where from."""
    rows: dict[str, dict[str, Any]] = {}
    for point in points:  # newest first
        sha = point.get("checkpoint_sha256")
        if not sha:
            continue
        row = rows.setdefault(
            sha,
            {
                "label": point["label"],
                "kind": point["kind"],
                "run_id": point.get("run_id"),
                "checkpoint": str(sha)[:12],
                "step": point.get("step"),
                "parameters": point.get("parameters"),
                "active_parameters": point.get("active_parameters"),
                "tokens_seen": point.get("tokens_seen"),
                "sources": [],
                **{metric: None for metric in METRICS},
            },
        )
        if point["source"] not in row["sources"]:
            row["sources"].append(point["source"])
        for metric, value in (point.get("metrics") or {}).items():
            if row.get(metric) is None and value.get("value") is not None:
                row[metric] = value["value"]
        if row["kind"] != "reference" and point["kind"] == "reference":
            row["kind"] = "reference"
    out = list(rows.values())
    for row in out:
        row["sources"] = ", ".join(row["sources"][:3]) + (
            f" (+{len(row['sources']) - 3})" if len(row["sources"]) > 3 else ""
        )
    return out


def unscored_runs(
    runs_dirs: Iterable[Path], points: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Plain `sparselab train` runs that no lab record has scored yet."""
    known = {str(p.get("run_id")) for p in points}
    out = []
    for root in runs_dirs:
        if not root or not root.is_dir():
            continue
        for run in sorted(root.iterdir()):
            if not (
                (run / "manifest.json").is_file()
                and (run / "resolved_config.yaml").is_file()
            ):
                continue
            if run.name in known:
                continue
            out.append(
                {
                    "run_id": run.name,
                    "path": str(run),
                    "command": f"sparselab probe {run.name} --runs-dir {root}",
                }
            )
    return out


# --- Behaviors ----------------------------------------------------------------


def generations(entry: ProbeEntry) -> list[dict[str, Any]]:
    """Greedy continuations of the same prompts, candidate next to baseline."""
    row = probe_row(entry.result, "repetition") or {}
    details = row.get("details") or {}
    base = {
        s["prompt"]: s["continuation"] for s in details.get("baseline_samples") or []
    }
    return [
        {
            "prompt": s["prompt"],
            "candidate": s["continuation"],
            "baseline": base.get(s["prompt"]),
        }
        for s in details.get("samples") or []
    ]


def recall_items(entry: ProbeEntry, probe_id: str) -> list[dict[str, Any]]:
    """Per held-out item of a ranking probe: asked, expected, each arm's pick."""
    row = probe_row(entry.result, probe_id) or {}
    return [dict(item) for item in (row.get("details") or {}).get("items") or []]


def needle_curves(entries: Iterable[ProbeEntry]) -> list[dict[str, Any]]:
    """Needle accuracy by context length, one curve per (checkpoint, group).

    Needle items are sized in each model's own tokens, so only results in one
    ``item_group`` (same rendered items, tokenizer and lengths) share a scale;
    ``group`` is that comparison identity (shared :func:`result_group`), None
    for older records without one. The newest result per (checkpoint, group)
    wins.
    """
    rows = []
    seen: set[tuple[str, str | None]] = set()
    for entry in sorted(entries, key=lambda e: e.created_at, reverse=True):
        row = probe_row(entry.result, "needle") or {}
        target = entry.result.get("target") or {}
        sha = str(target.get("checkpoint_sha256"))
        details = row.get("details") or {}
        by_length = details.get("by_length") or {}
        group = result_group(entry.result, row) if row else None
        if not by_length or (sha, group) in seen:
            continue
        seen.add((sha, group))
        for fraction, value in by_length.items():
            if value.get("accuracy") is None:
                continue
            rows.append(
                {
                    "checkpoint": f"{entry.key} ({target.get('run_id')})",
                    "checkpoint_sha256": sha,
                    "group": group,
                    "when": entry.created_at,
                    "fraction": float(fraction),
                    "tokens": value.get("tokens"),
                    "accuracy": value["accuracy"],
                    "chance": details.get("chance"),
                }
            )
    return rows


def calibration_curves(entry: ProbeEntry) -> list[dict[str, Any]]:
    """Reliability-diagram bins of the candidate (and baseline) of a result."""
    row = probe_row(entry.result, "calibration") or {}
    details = row.get("details") or {}
    out = []
    for arm, key in (
        ("candidate", "reliability"),
        ("baseline", "baseline_reliability"),
    ):
        for b in details.get(key) or []:
            out.append({"arm": arm, **b})
    return out
