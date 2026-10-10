"""``sparselab compare``: place sealed results against each other and against
pinned reference models, metric by metric, only within a comparison group.

Every pair is one of: ``higher``/``lower``/``within_noise`` (same group, with a
paired standard error when both sides kept per-item evidence),
``not_comparable`` (different eval or benchmark group, with the reason) or
``missing_evidence`` (one side never measured the metric, with how to get it).
Nothing is compared silently across groups.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sparselab.lab_records import read_lab_record, resolve_record
from sparselab.probes import metrics
from sparselab.probes.points import (
    METRICS,
    collect_points,
    metric_points,
    points_from_record,
    with_checkpoint_evidence,
)
from sparselab.probes.render import GLYPH, _compact, _num, _paint, params_text
from sparselab.reference_models import REFERENCES, is_reference, reference_for

REFERENCE_LOSS_NOTE = (
    "reference models have their own tokenizer and training data, so held-out "
    "loss on our split is never comparable; use lm-eval accuracy"
)
NOISE_SE = 2.0  # |Δ| within 2 paired SE reads as noise
STATUS_TEXT = {
    "higher": "HIGHER",
    "lower": "LOWER",
    "within_noise": "WITHIN NOISE",
    "not_comparable": "NOT COMPARABLE",
    "missing_evidence": "MISSING EVIDENCE",
}


def _merge(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One checkpoint seen by several records/roles -> one point (first wins)."""
    merged: dict[str, Any] = {**points[0], "metrics": {}}
    for point in points:
        for key, value in point.items():
            if merged.get(key) is None and value is not None:
                merged[key] = value
        for name, value in (point.get("metrics") or {}).items():
            merged["metrics"].setdefault(name, value)
    return merged


def _reference_point(spec: str, pool: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reference = reference_for(spec)
    found = [
        p
        for p in pool
        if p["kind"] == "reference"
        and (p.get("reference") or {}).get("name") == reference.name
    ]
    if not found:
        raise ValueError(
            f"no result for {spec}: run `sparselab probe {spec} --tier full`"
        )
    return _merge(found)


def resolve_point(
    spec: str, lab_dir: Path, pool: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """A try/probe id or record path (its candidate), or ``ref:NAME``.

    The record's own measurements come first; other verified records of the
    same checkpoint (e.g. a later ``probe --tier full``) add what it lacks.
    """
    if is_reference(spec):
        return with_checkpoint_evidence(_reference_point(spec, pool), pool)
    path = resolve_record(spec, lab_dir)
    kind, record = read_lab_record(path)
    points = points_from_record(kind, record, path)
    candidate = [p for p in points if p["role"] == "candidate"]
    if not candidate:
        raise ValueError(f"{spec}: no scored candidate in this record")
    sha = candidate[0]["checkpoint_sha256"]
    own = _merge([p for p in candidate if p["checkpoint_sha256"] == sha])
    return with_checkpoint_evidence(own, pool)


def _common_group(
    subject: Mapping[str, Any],
    other: Mapping[str, Any],
    metric_id: str,
    a: Mapping[str, Any],
    b: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Measurements of both checkpoints in one shared group, when they exist.

    A checkpoint scored in several groups (e.g. an older lm-eval record and a
    newer `probe --tier full`) is compared in the group the other side has.
    """
    if a.get("group") == b.get("group"):
        return a, b
    mine = (subject.get("by_group") or {}).get(metric_id) or {}
    theirs = (other.get("by_group") or {}).get(metric_id) or {}
    for group in (b.get("group"), a.get("group"), *mine):
        if group is not None and group in mine and group in theirs:
            return mine[group], theirs[group]
    if b.get("group") in mine:
        return mine[b["group"]], b
    if a.get("group") in theirs:
        return a, theirs[a["group"]]
    return a, b


def pair(
    subject: Mapping[str, Any], other: Mapping[str, Any], metric_id: str
) -> dict[str, Any]:
    """Compare one metric of SUBJECT against OTHER (never across groups)."""
    metric = METRICS[metric_id]
    a = (subject.get("metrics") or {}).get(metric_id)
    b = (other.get("metrics") or {}).get(metric_id)
    out: dict[str, Any] = {
        "metric": metric_id,
        "other": other["label"],
        "value": a and a["value"],
        "other_value": b and b["value"],
        "delta": None,
        "se": None,
        "note": None,
    }
    if metric_id == "heldout_loss" and "reference" in {subject["kind"], other["kind"]}:
        return {**out, "status": "not_comparable", "note": REFERENCE_LOSS_NOTE}
    if a is None or b is None:
        who = subject if a is None else other
        hint = metric.missing_hint
        if metric_id == "lm_eval" and who.get("run_id"):
            hint = f"run `sparselab probe {who['run_id']} --tier full` (lmeval extra)"
        return {
            **out,
            "status": "missing_evidence",
            "note": f"{who['label']} has no {metric.label}; {hint}",
        }
    a, b = _common_group(subject, other, metric_id, a, b)
    source = (a.get("evidence") or {}).get("value")
    out.update(
        value=a["value"],
        other_value=b["value"],
        evidence=source,
        group=a.get("group") if a.get("group") == b.get("group") else None,
    )
    if a.get("group") is None or b.get("group") is None:
        return {
            **out,
            "status": "not_comparable",
            "note": f"no {metric.group_label} recorded (older record); re-score it",
        }
    if a["group"] != b["group"]:
        return {**out, "status": "not_comparable", "note": metric.not_comparable}
    delta = a["value"] - b["value"]
    se = None
    if metric_id == "heldout_loss" and a.get("window_sums") and b.get("window_sums"):
        se = metrics.ratio_difference_se(
            a["window_sums"], a["window_counts"], b["window_sums"], b["window_counts"]
        )
    if metric_id == "lm_eval":
        paired = metrics.task_mean_difference(a.get("items") or {}, b["items"] or {})
        se = paired[1] if paired else None
    status = "higher" if delta > 0 else "lower"
    if se is not None and abs(delta) <= NOISE_SE * se or delta == 0:
        status = "within_noise"
    notes = []
    if source and source != subject.get("source"):
        notes.append(f"subject measured in {source}")
    if se is None:
        notes.append("no paired SE (per-item evidence missing)")
    note = "; ".join(notes) or None
    return {**out, "status": status, "delta": delta, "se": se, "note": note}


def compare(
    subject_spec: str,
    other_specs: Sequence[str],
    *,
    lab_dir: Path,
    references: bool = False,
) -> dict[str, Any]:
    pool = collect_points(lab_dir)
    subject = resolve_point(subject_spec, lab_dir, pool)
    others = [resolve_point(spec, lab_dir, pool) for spec in other_specs]
    if references:
        named = {o["checkpoint_sha256"] for o in others}
        for name in REFERENCES:
            point = _reference_point(f"ref:{name}", pool)
            if point["checkpoint_sha256"] not in named | {subject["checkpoint_sha256"]}:
                others.append(point)
    if not others:
        raise ValueError("nothing to compare against: name results or --references")
    comparisons = [
        {"other": other["label"], "pairs": [pair(subject, other, m) for m in METRICS]}
        for other in others
    ]
    return {
        "subject": subject,
        "points": [subject, *others],
        "comparisons": comparisons,
        "curve": reference_curve(subject, pool) if references else None,
    }


def reference_curve(
    subject: Mapping[str, Any], pool: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """References in the subject's benchmark group, by size, with the subject."""
    mine = (subject.get("metrics") or {}).get("lm_eval")
    refs = [p for p in metric_points(pool, "lm_eval") if p["kind"] == "reference"]
    if mine is None:
        return {"group": None, "points": [], "note": METRICS["lm_eval"].missing_hint}
    # Of the groups this checkpoint was scored in, use the one with the most
    # references (ties: the subject's own record).
    options = (subject.get("by_group") or {}).get("lm_eval") or {}
    ranked = sorted(
        options.values(),
        key=lambda m: -sum(r["group"] == m["group"] for r in refs),
    )
    if ranked and any(r["group"] == ranked[0]["group"] for r in refs):
        mine = ranked[0]
    same = [p for p in refs if p["group"] == mine["group"]]
    rows = sorted(
        [*same, {**subject, **mine, "label": subject["label"], "subject": True}],
        key=lambda p: (p.get("active_parameters") or 0, p["label"]),
    )
    chance = mine.get("chance") or {}
    return {
        "group": mine["group"],
        "chance": sum(chance.values()) / len(chance) if chance else None,
        "points": rows,
        "excluded": len(refs) - len(same),
    }


# --- Rendering ----------------------------------------------------------------

STATUS_COLOR = {
    "better": "32",
    "worse": "31",
    "within_noise": "36",
    "not_comparable": "35",
    "missing_evidence": "33",
}


def _verdict_word(status: str, metric_id: str) -> str:
    if status not in {"higher", "lower"}:
        return status
    good = (status == "higher") == METRICS[metric_id].higher_is_better
    return "better" if good else "worse"


def _metric_value(point: Mapping[str, Any], metric_id: str) -> str:
    value = (point.get("metrics") or {}).get(metric_id)
    return _num(value["value"]) if value else "–"


def _provenance(points: Sequence[Mapping[str, Any]], color: bool) -> list[str]:
    """Where a point's measurements came from when not its own record."""
    lines = []
    for point in points:
        for metric_id, value in (point.get("metrics") or {}).items():
            source = (value.get("evidence") or {}).get("value")
            if source and source != point.get("source"):
                lines.append(
                    _paint(
                        f"    ↳ {point['label']}: {METRICS[metric_id].label} from "
                        f"{source} (same checkpoint "
                        f"{str(point.get('checkpoint_sha256'))[:12]})",
                        "2",
                        color,
                    )
                )
    return lines


def render(report: Mapping[str, Any], *, color: bool = False) -> str:
    subject = report["subject"]
    points = report["points"]
    width = max(len(p["label"]) for p in points) + 2
    lines = [
        _paint("COMPARE", "1", color)
        + f"  {subject['label']} vs {len(points) - 1} point(s) · comparisons only "
        "within one eval/benchmark group",
        "",
        _paint(
            f"    {'point':<{width}} {'resident':>9} {'active':>9} {'tokens':>8}"
            f" {'held-out loss':>14} {'lm-eval acc':>12}",
            "2",
            color,
        ),
    ]
    for point in points:
        mark = "▶" if point is subject else "◆" if point["kind"] == "reference" else "•"
        lines.append(
            f"  {mark} {point['label']:<{width}} {_compact(point.get('parameters')):>9}"
            f" {_compact(point.get('active_parameters')):>9}"
            f" {_compact(point.get('tokens_seen')):>8}"
            f" {_metric_value(point, 'heldout_loss'):>14}"
            f" {_metric_value(point, 'lm_eval'):>12}"
        )
    lines += _provenance(points, color)
    for metric_id, metric in METRICS.items():
        direction = "higher" if metric.higher_is_better else "lower"
        lines += [
            "",
            _paint(f"  {metric.label}", "1", color) + f" ({direction} is better)",
        ]
        grouped: dict[tuple[str, str], list[str]] = {}
        for comparison in report["comparisons"]:
            row = next(p for p in comparison["pairs"] if p["metric"] == metric_id)
            word = _verdict_word(row["status"], metric_id)
            if row["status"] in {"not_comparable", "missing_evidence"}:
                grouped.setdefault((row["status"], row["note"]), []).append(
                    row["other"]
                )
                continue
            glyph = {"better": "✔", "worse": "✖"}.get(word, GLYPH["info"])
            text = f"Δ {row['delta']:+.3f}" + (
                f" ±{row['se']:.3f}" if row["se"] is not None else ""
            )
            lines.append(
                "    "
                + _paint(
                    f"{glyph} {word.upper().replace('_', ' '):<13}",
                    STATUS_COLOR[word],
                    color,
                )
                + f" vs {row['other']:<{width}} {text}"
                + (_paint(f"  ({row['note']})", "2", color) if row["note"] else "")
            )
        for (status, note), names in grouped.items():
            glyph = GLYPH["not_comparable"] if status == "not_comparable" else "○"
            lines.append(
                "    "
                + _paint(
                    f"{glyph} {STATUS_TEXT[status]:<13}", STATUS_COLOR[status], color
                )
                + f" vs {', '.join(names)}"
            )
            lines.append(_paint(f"      ↳ {note}", "2", color))
    curve = report.get("curve")
    if curve is not None:
        lines += ["", _paint("  reference curve", "1", color) + " (lm-eval accuracy)"]
        if curve["group"] is None:
            lines.append(
                _paint(f"    ○ {subject['label']}: {curve['note']}", "33", color)
            )
        else:
            for point in curve["points"]:
                bar = "█" * max(1, round(40 * point["value"]))
                label = point["label"]
                text = (
                    f"    {'▶' if point.get('subject') else '◆'} {label:<{width}}"
                    f" {params_text(point):<30} {point['value']:.3f} {bar}"
                )
                lines.append(_paint(text, "1", color) if point.get("subject") else text)
            if curve.get("chance") is not None:
                lines.append(
                    _paint(f"    chance (task mean) {curve['chance']:.3f}", "2", color)
                )
            if curve["excluded"]:
                lines.append(
                    _paint(
                        f"    {curve['excluded']} reference(s) in another benchmark "
                        "group not shown",
                        "2",
                        color,
                    )
                )
    lines.append(
        _paint(
            f"\n  legend  Δ = subject − other ± paired SE · within noise = |Δ| ≤ "
            f"{NOISE_SE:g} SE · resident = all weights · active = touched per token",
            "2",
            color,
        )
    )
    return "\n".join(lines)


def render_references(pool: Sequence[Mapping[str, Any]], *, color: bool = False) -> str:
    """The pinned reference registry and which results are available."""
    have = {
        (p.get("reference") or {}).get("name")
        for p in pool
        if p["kind"] == "reference" and p["metrics"].get("lm_eval")
    }
    lines = [_paint("REFERENCE MODELS", "1", color) + "  (pinned revisions)"]
    for ref in REFERENCES.values():
        mark = "✔" if ref.name in have else "○"
        lines.append(
            f"  {mark} ref:{ref.name:<22} {ref.repo_id}@{ref.revision[:12]}"
            f" · {_compact(ref.training_tokens)} tokens · {ref.license}"
        )
    lines.append(
        _paint(
            "  ✔ = sealed lm-eval result available (packaged or in this lab); "
            "probe one with `sparselab probe ref:NAME --tier full`",
            "2",
            color,
        )
    )
    return "\n".join(lines)


def as_json(report: Mapping[str, Any]) -> dict[str, Any]:
    """JSON contract: drop bulky per-item evidence."""

    def slim(point: Mapping[str, Any]) -> dict[str, Any]:
        metrics_ = {
            name: {
                k: v
                for k, v in value.items()
                if k not in {"window_sums", "window_counts", "items"}
            }
            for name, value in (point.get("metrics") or {}).items()
        }
        return {**point, "metrics": metrics_}

    curve = report.get("curve")
    return {
        "subject": slim(report["subject"]),
        "points": [slim(p) for p in report["points"]],
        "comparisons": report["comparisons"],
        "curve": None
        if curve is None
        else {
            **curve,
            "points": [
                {
                    k: v
                    for k, v in p.items()
                    if k not in {"window_sums", "window_counts", "items"}
                }
                for p in curve["points"]
            ],
        },
    }
