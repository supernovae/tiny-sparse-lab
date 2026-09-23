"""Deterministic, dependency-free SVG charts for static study reports."""

from __future__ import annotations

import html
import math
from collections.abc import Iterable


def _svg(title: str, points: Iterable[tuple[str, float]], *, y_label: str) -> str:
    values = list(points)
    width, height, left, bottom = 760, 300, 58, 45
    plot_w, plot_h = width - left - 20, height - 30 - bottom
    finite = [(label, value) for label, value in values if math.isfinite(value)]
    if not finite:
        body = '<text x="20" y="55">No observed values available.</text>'
    else:
        low, high = min(value for _, value in finite), max(value for _, value in finite)
        if low == high:
            low, high = low - 0.5, high + 0.5
        marks: list[str] = []
        for index, (label, value) in enumerate(finite):
            x = left + (plot_w * index / max(1, len(finite) - 1))
            y = 30 + plot_h * (1 - (value - low) / (high - low))
            marks.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4"/>')
            marks.append(
                f'<text x="{x:.2f}" y="{height - 18}" text-anchor="middle">{html.escape(label)}</text>'
            )
            marks.append(
                f'<text x="{x:.2f}" y="{y - 8:.2f}" text-anchor="middle">{value:.6g}</text>'
            )
        body = (
            f'<line x1="{left}" y1="30" x2="{left}" y2="{height - bottom}"/>'
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - 20}" y2="{height - bottom}"/>'
            f'<text x="5" y="20">{html.escape(y_label)} ({low:.6g} … {high:.6g})</text>'
            + "".join(marks)
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">'
        f"<title>{html.escape(title)}</title><style>text{{font:12px sans-serif}}line{{stroke:#555}}circle{{fill:#1769aa}}</style>"
        f"{body}</svg>\n"
    )


def render_charts(report: dict[str, object]) -> dict[str, str]:
    """Return stable SVG bytes keyed by safe fixed chart names.

    Charts intentionally render only endpoint observations supplied in the report;
    they never smooth, rank, or infer missing values.
    """
    charts: dict[str, str] = {}
    runs = report.get("runs")
    cards = report.get("cards")
    if isinstance(runs, list) and isinstance(cards, list):
        for card in cards:
            if not isinstance(card, dict) or not isinstance(card.get("name"), str):
                continue
            name = card["name"]
            reference = card.get("reference")
            values: list[tuple[str, float]] = []
            for run in runs:
                if not isinstance(run, dict):
                    continue
                caps = run.get("capabilities")
                cap = (
                    (
                        caps.get(reference)
                        if isinstance(reference, str)
                        else caps.get(name)
                    )
                    if isinstance(caps, dict)
                    else None
                )
                result = cap.get("result", cap) if isinstance(cap, dict) else None
                score = result.get("score") if isinstance(result, dict) else None
                if (
                    isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(float(score))
                ):
                    values.append((str(run.get("run_id", "run"))[:12], float(score)))
            if values:
                charts[f"card-{len(charts) + 1}.svg"] = _svg(
                    name, values, y_label="endpoint score"
                )
    comparisons = report.get("comparisons")
    if isinstance(comparisons, list):
        values = []
        for item in comparisons:
            if not isinstance(item, dict):
                continue
            caps = item.get("capabilities")
            if not isinstance(caps, dict):
                continue
            for cap in caps.values():
                if (
                    isinstance(cap, dict)
                    and isinstance(cap.get("score_delta"), (int, float))
                    and not isinstance(cap.get("score_delta"), bool)
                    and math.isfinite(float(cap["score_delta"]))
                ):
                    values.append(
                        (str(item.get("id", "pair")), float(cap["score_delta"]))
                    )
        charts["pair-deltas.svg"] = _svg(
            "Observed per-seed pair deltas", values, y_label="variant − baseline"
        )
    return charts
