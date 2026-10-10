"""Probe battery page: live progress, verdicts, per-probe results, history, Pareto."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from sparselab.dashboard.probe_data import (
    history_rows,
    live_batteries,
    load_entries,
    pareto_frontier,
)
from sparselab.probes.suite import PROBES, suite_identity

STATUS_COLOR = {
    "pass": "#1a7f37",
    "warn": "#b58100",
    "fail": "#cf222e",
    "info": "#0969da",
    "skipped": "#6e7781",
    "not_comparable": "#8250df",
    "error": "#cf222e",
    "incomplete": "#b58100",
}
STATUS_ICON = {
    "pass": "✔",
    "warn": "▲",
    "fail": "✖",
    "info": "•",
    "skipped": "⊘",
    "not_comparable": "≠",
    "error": "!",
    "incomplete": "…",
}
ACTION_TEXT = {
    "abandon": "Abandon this idea",
    "tweak": "Tweak and retry",
    "escalate": "Escalate to the next tier",
    "longer_run": "Schedule a longer run",
    "compare": "Compare against a baseline",
}
_CSS = """
<style>
.pb-banner {border-radius: 14px; padding: 18px 22px; margin: 6px 0 14px 0;
  color: white; box-shadow: 0 2px 10px rgba(0,0,0,.08);}
.pb-banner h3 {margin: 0 0 4px 0; color: white;}
.pb-banner p {margin: 2px 0; opacity: .95;}
.pb-chip {display: inline-block; padding: 1px 10px; border-radius: 999px;
  color: white; font-size: .78rem; font-weight: 600; letter-spacing: .02em;}
.pb-table {width: 100%; border-collapse: collapse; font-size: .92rem;}
.pb-table td, .pb-table th {padding: 7px 8px; border-bottom: 1px solid rgba(128,128,128,.18);
  vertical-align: top; text-align: left;}
.pb-table th {font-weight: 600; opacity: .7; font-size: .8rem; text-transform: uppercase;}
.pb-meter {position: relative; width: 120px; height: 10px; border-radius: 6px;
  background: rgba(128,128,128,.15); margin-top: 5px;}
.pb-meter .mid {position: absolute; left: 50%; top: -2px; width: 2px; height: 14px;
  background: rgba(128,128,128,.6);}
.pb-meter .bar {position: absolute; top: 0; height: 10px; border-radius: 6px;}
.pb-hint {opacity: .75; font-size: .82rem;}
.pb-live {border: 1px dashed rgba(128,128,128,.5); border-radius: 12px; padding: 10px 14px;
  margin-bottom: 10px;}
</style>
"""


def _chip(status: str) -> str:
    color = STATUS_COLOR.get(status, "#6e7781")
    label = html.escape(status.replace("_", " ").upper())
    return (
        f'<span class="pb-chip" style="background:{color}">'
        f"{STATUS_ICON.get(status, '?')} {label}</span>"
    )


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _meter(row: dict[str, Any]) -> str:
    regression = row.get("regression")
    scale = (row.get("thresholds") or {}).get("fail") or (
        row.get("thresholds") or {}
    ).get("warn")
    if regression is None or not scale:
        return ""
    share = min(1.0, abs(regression) / scale) * 50
    if row.get("within_noise"):
        share = min(share, 6)
    if regression < 0:
        style = f"right:50%;width:{share:.1f}%;background:{STATUS_COLOR['pass']}"
    else:
        tone = STATUS_COLOR.get(row.get("status", ""), "#b58100")
        if row.get("status") == "pass":
            tone = "#8c959f"
        style = f"left:50%;width:{share:.1f}%;background:{tone}"
    return (
        f'<div class="pb-meter" title="left = better, right = worse; '
        f'the full half is the fail threshold"><div class="bar" style="{style}"></div>'
        f'<div class="mid"></div></div>'
    )


def _verdict_banner(result: dict[str, Any]) -> None:
    verdict = result.get("verdict") or {}
    status = str(verdict.get("status", "info"))
    action = ACTION_TEXT.get(verdict.get("action", ""), verdict.get("action", ""))
    next_tier = verdict.get("next_tier")
    guard = result.get("guard") or {}
    guard_text = (
        "⚠ dev-split gain not shared by the held-out split (possible probe overfit)"
        if guard.get("overfit_suspected")
        else "held-out split decides; no dev/held-out divergence"
    )
    stop = result.get("fast_fail") or {}
    stop_text = (
        f"<p>⚡ {html.escape(stop['reason'])}</p>" if stop.get("stopped") else ""
    )
    st.markdown(
        f'<div class="pb-banner" style="background:{STATUS_COLOR.get(status, "#0969da")}">'
        f"<h3>{STATUS_ICON.get(status, '?')} {html.escape(status.upper())} · "
        f"next: {html.escape(str(action))}"
        + (f" (--tier {html.escape(next_tier)})" if next_tier else "")
        + "</h3>"
        f"<p>{html.escape(str(verdict.get('suggestion', '')))}</p>"
        f"<p style='font-size:.85rem'>Guard: {html.escape(guard_text)} · "
        f"tiers run: {html.escape(' → '.join(result.get('tiers_run') or []))} · "
        f"{result.get('seconds', 0):.1f}s</p>{stop_text}</div>",
        unsafe_allow_html=True,
    )


def _metrics(result: dict[str, Any]) -> None:
    rows = {row["id"]: row for row in result.get("probes") or []}
    picks = [
        ("heldout_loss", "Held-out loss", "inverse"),
        ("fact_recall", "Fact recall", "normal"),
        ("needle", "Needle", "normal"),
        ("repetition", "Repetition (seq-rep-4)", "inverse"),
    ]
    columns = st.columns(len(picks))
    for column, (probe_id, label, mode) in zip(columns, picks, strict=True):
        row = rows.get(probe_id) or {}
        value = row.get("value")
        delta = row.get("delta")
        column.metric(
            label,
            "–" if value is None else f"{value:.3f}",
            None if delta is None else f"{delta:+.3f} vs baseline",
            delta_color=mode,
            help=(next((p.explains for p in PROBES if p.id == probe_id), None)),
        )


def _detail(row: dict[str, Any]) -> str:
    details = row.get("details") or {}
    if row["id"] == "heldout_loss" and details.get("perplexity"):
        return (
            f"ppl {details['perplexity']:.1f} · {details.get('valid_targets')} targets"
        )
    if row["id"] == "token_agreement" and "js" in details:
        text = f"JS {details['js']:.3f} · KL {details['kl_base_to_candidate']:.3f}"
        if details.get("informative") is False:
            text += " · near-identical predictions"
        return text
    if row["id"] == "repetition" and details.get("distinct_2") is not None:
        return f"distinct-1 {details['distinct_1']:.2f} · distinct-2 {details['distinct_2']:.2f}"
    if row["id"] in {"fact_recall", "needle"} and details.get("chance") is not None:
        return f"chance {details['chance']:.2f} · n={details.get('n')}"
    if row["id"] == "lm_eval" and details.get("tasks"):
        return " · ".join(
            f"{task} {_num((v or {}).get('acc'), 2)}"
            for task, v in details["tasks"].items()
        )
    return ""


def _probe_table(result: dict[str, Any]) -> None:
    body = []
    for row in result.get("probes") or []:
        status = str(row.get("status"))
        value = _num(row.get("value"))
        if row.get("baseline_value") is not None:
            value += (
                f" <span class='pb-hint'>vs {_num(row.get('baseline_value'))}</span>"
            )
        delta = "" if row.get("delta") is None else f"{row['delta']:+.3f}"
        if row.get("delta_se") is not None:
            delta += f" <span class='pb-hint'>±{row['delta_se']:.3f}</span>"
        hint = ""
        if status in {"warn", "fail"}:
            hint = html.escape(row.get("suggests") or "")
        elif row.get("note"):
            hint = html.escape(str(row["note"]))
        body.append(
            "<tr>"
            f"<td>{_chip(status)}</td>"
            f"<td><b>{html.escape(row['title'])}</b><br>"
            f"<span class='pb-hint'>{html.escape(row['tier'])} tier · cost {row['cost']}"
            + (" · hard" if row.get("hard") else "")
            + "</span></td>"
            f"<td>{value}</td><td>{delta}{_meter(row)}</td>"
            f"<td>{html.escape(_detail(row))}<br><span class='pb-hint'>{hint}</span></td>"
            "</tr>"
        )
    st.markdown(
        "<table class='pb-table'><tr><th>Status</th><th>Probe</th>"
        "<th>Candidate</th><th>Δ vs baseline</th><th>Notes</th></tr>"
        + "".join(body)
        + "</table>",
        unsafe_allow_html=True,
    )


def _comparison_chart(result: dict[str, Any]) -> None:
    rows = [
        row
        for row in result.get("probes") or []
        if row.get("value") is not None and row.get("baseline_value") is not None
    ]
    if not rows:
        st.caption("Bars appear when a baseline is part of the result.")
        return
    figure = make_subplots(
        rows=1, cols=len(rows), subplot_titles=[row["title"] for row in rows]
    )
    for index, row in enumerate(rows, start=1):
        colors = ["#8c959f", STATUS_COLOR.get(row["status"], "#0969da")]
        figure.add_trace(
            go.Bar(
                x=["baseline", "candidate"],
                y=[row["baseline_value"], row["value"]],
                marker_color=colors,
                showlegend=False,
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=index,
        )
    figure.update_layout(height=260, margin={"l": 10, "r": 10, "t": 40, "b": 10})
    figure.update_annotations(font_size=11)
    st.plotly_chart(figure, use_container_width=True)
    needle = next((r for r in rows if r["id"] == "needle"), None)
    by_length = (needle or {}).get("details", {}).get("by_length") or {}
    if by_length:
        frame = pd.DataFrame(
            [
                {"context tokens": v["tokens"], "arm": arm, "accuracy": v[key]}
                for v in by_length.values()
                for arm, key in (
                    ("candidate", "accuracy"),
                    ("baseline", "baseline_accuracy"),
                )
                if v.get(key) is not None
            ]
        )
        chart = px.line(
            frame,
            x="context tokens",
            y="accuracy",
            color="arm",
            markers=True,
            title="Needle retrieval by context length",
            color_discrete_map={"candidate": "#0969da", "baseline": "#8c959f"},
        )
        chance = needle["details"].get("chance")
        if chance:
            chart.add_hline(y=chance, line_dash="dot", annotation_text="chance")
        chart.update_layout(
            height=280,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
            yaxis_range=[0, 1.05],
        )
        st.plotly_chart(chart, use_container_width=True)


def _explain() -> None:
    st.markdown(
        "Each probe is cheap and answers one question. They run cheapest first; a "
        "**hard** failure stops the battery (fast-fail) and the battery only escalates "
        "to the next tier when nothing failed. Verdicts use the **held-out** item "
        "split only; the **dev** split is free to look at, and a dev gain the "
        "held-out split does not share raises the overfit guard."
    )
    for spec in PROBES:
        fail = "—" if spec.fail is None else f"{spec.fail:g}"
        how = {
            "delta_rel": "relative change vs baseline",
            "delta_abs": "absolute change vs baseline",
            "value_min": "absolute value (minimum)",
        }[spec.mode]
        with st.expander(
            f"{spec.title} · {spec.tier} tier · cost {spec.cost}",
            expanded=spec is PROBES[0],
        ):
            st.markdown(spec.explains)
            st.markdown(
                f"**Judged on** {how}: warn beyond {spec.warn:g}, fail beyond {fail}"
                + (" (hard: stops the battery)" if spec.hard else "")
                + f". Changes within 2 standard errors count as noise.\n\n"
                f"**If it fails:** {spec.suggests}\n\n"
                f"*Reference:* {spec.reference}"
            )


def _history(entries: list[Any]) -> pd.DataFrame:
    frame = pd.DataFrame(history_rows(entries))
    if frame.empty:
        st.info("No probe results yet.")
        return frame
    frame["when"] = pd.to_datetime(frame["when"], errors="coerce", utc=True)
    shown = frame[
        [
            "when",
            "id",
            "idea",
            "verdict",
            "action",
            "tier",
            "loss",
            "loss_delta",
            "recall",
            "needle",
            "suite",
        ]
    ]
    st.dataframe(shown, use_container_width=True, hide_index=True)
    plotted = frame.dropna(subset=["loss_delta"])
    if not plotted.empty:
        chart = px.scatter(
            plotted,
            x="when",
            y="loss_delta",
            color="verdict",
            symbol="action",
            hover_data=["id", "idea"],
            color_discrete_map=STATUS_COLOR,
            title="Held-out loss change vs baseline over time (below 0 = better)",
        )
        chart.add_hline(y=0, line_color="#8c959f")
        chart.update_traces(marker_size=11)
        chart.update_layout(height=320, margin={"l": 10, "r": 10, "t": 40, "b": 10})
        st.plotly_chart(chart, use_container_width=True)
    if frame["suite"].nunique() > 1:
        st.warning(
            "History mixes probe suite versions; only compare rows with the same suite digest."
        )
    return frame


def _pareto(frame: pd.DataFrame) -> None:
    if frame.empty or frame["loss"].dropna().empty:
        st.info("The Pareto view needs at least one result with a held-out loss.")
        return
    axis = st.radio(
        "Cost axis",
        ("parameters", "parameter_bytes", "tokens_seen", "ms_per_token"),
        horizontal=True,
        key="probe_pareto_axis",
        format_func=lambda name: {
            "parameters": "parameters",
            "parameter_bytes": "resident weight bytes",
            "tokens_seen": "training tokens",
            "ms_per_token": "latency (ms/token)",
        }[name],
    )
    # One point per candidate checkpoint: the newest result wins.
    data = (
        frame.dropna(subset=["loss", axis])
        .drop_duplicates(subset=["target"], keep="first")
        .reset_index(drop=True)
    )
    if data.empty:
        st.info(f"No results record {axis}.")
        return
    front = pareto_frontier(list(zip(data[axis], data["loss"], strict=True)))
    data["frontier"] = data.index.isin(front)
    chart = px.scatter(
        data,
        x=axis,
        y="loss",
        color="verdict",
        hover_data=["id", "idea", "action"],
        color_discrete_map=STATUS_COLOR,
        title="Quality vs cost: lower-left is better; the line is the Pareto frontier",
    )
    frontier = data[data["frontier"]].sort_values(axis)
    chart.add_trace(
        go.Scatter(
            x=frontier[axis],
            y=frontier["loss"],
            mode="lines",
            line={"color": "#0969da", "dash": "dot"},
            name="frontier",
        )
    )
    chart.update_traces(marker_size=12, selector={"mode": "markers"})
    chart.update_layout(height=380, margin={"l": 10, "r": 10, "t": 40, "b": 10})
    st.plotly_chart(chart, use_container_width=True)


@st.fragment(run_every=2)
def _live(lab_dir: Path) -> None:
    for state in live_batteries(lab_dir):
        total = int(state.get("total") or 0)
        done = len(state.get("done") or [])
        chips = " ".join(_chip(s) for s in (state.get("statuses") or {}).values())
        st.markdown(
            f"<div class='pb-live'>⏳ <b>{html.escape(state['key'])}</b> · tier "
            f"{html.escape(str(state.get('tier')))} · running "
            f"<code>{html.escape(str(state.get('current') or state.get('state')))}</code>"
            f"<br>{chips}</div>",
            unsafe_allow_html=True,
        )
        if total:
            st.progress(done / total, text=f"{done}/{total} probes")


def probe_page(lab_dir: Path) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    identity = suite_identity()
    st.header("Probe battery")
    st.caption(
        f"{identity['name']} v{identity['version']} · suite {identity['sha256'][:12]} · "
        "cheap checks that filter ideas before a longer run · lab root "
        f"{lab_dir}"
    )
    _live(lab_dir)
    entries = load_entries(lab_dir)
    if not entries:
        st.info(
            "No probe results yet. Run `sparselab try DELTA.yaml --vs BASELINE.yaml` "
            "(the fast tier runs automatically) or `sparselab probe RUN --vs BASELINE`."
        )
        _explain()
        return
    choice = st.selectbox(
        "Result", range(len(entries)), format_func=lambda i: entries[i].label
    )
    entry = entries[choice]
    result = entry.result
    target = result.get("target") or {}
    baseline = result.get("baseline") or {}
    st.caption(
        f"candidate **{target.get('run_id')}** (step {target.get('step')}) · "
        + (
            f"baseline **{baseline.get('run_id')}** · "
            if baseline
            else "no baseline · "
        )
        + f"suite {result['suite']['sha256'][:12]}"
    )
    if entry.comparison:
        st.caption(
            f"lab try verdict: {entry.comparison.get('verdict')} · "
            + ", ".join(
                f"{k}: {v.get('base')!r} → {v.get('variant')!r}"
                for k, v in entry.delta.items()
            )
        )
    _verdict_banner(result)
    _metrics(result)
    results_tab, explain_tab, history_tab, pareto_tab = st.tabs(
        ["Results", "What the probes mean", "History", "Pareto"]
    )
    with results_tab:
        _probe_table(result)
        _comparison_chart(result)
        samples = (
            (next((r for r in result["probes"] if r["id"] == "repetition"), {}) or {})
            .get("details", {})
            .get("samples")
        )
        if samples:
            with st.expander("Greedy samples (degeneration probe)"):
                for sample in samples:
                    st.markdown(
                        f"**{html.escape(sample['prompt'])}** → "
                        f"{html.escape(sample['continuation'])}"
                    )
    with explain_tab:
        _explain()
    with history_tab:
        frame = _history(entries)
    with pareto_tab:
        _pareto(frame)
