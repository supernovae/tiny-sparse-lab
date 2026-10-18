"""Read-only live plots with data-independent Streamlit component identity.

Streamlit's built-in Plotly element hashes the figure into its identity. The v2
component instead keeps one DOM node per metric and lets Plotly.react preserve
interaction state. Plotly's installed bundle is served locally, without a CDN.
"""

from functools import lru_cache
from html import escape
from pathlib import Path
from textwrap import wrap

import plotly.graph_objects as go
import streamlit as st
from plotly.offline import get_plotlyjs

from sparselab.dashboard.ui import SERIES
from sparselab.training.metric_registry import metric_spec

AXES = {
    "step": "Optimizer step",
    "tokens_seen": "Tokens seen (recorded count)",
    "wall_time": "Wall time (seconds)",
}


def figure(points: list[dict[str, object]], name: str, axis: str) -> go.Figure:
    spec = metric_spec(name)
    run_ids = sorted({str(point["run_id"]) for point in points})
    chart = go.Figure()
    for index, run_id in enumerate(run_ids):
        rows = [point for point in points if point["run_id"] == run_id]
        chart.add_trace(
            go.Scatter(
                x=[row[axis] for row in rows],
                y=[row["value"] for row in rows],
                name="<br>".join(escape(part) for part in wrap(run_id, width=28)),
                uid=run_id,
                mode="lines+markers",
                line={"color": list(SERIES.values())[index % len(SERIES)]},
                connectgaps=False,
            )
        )
    revision = f"{name}:{axis}:{run_ids}"
    chart.update_layout(
        template="plotly_white",
        height=390,
        margin={"l": 65, "r": 15, "t": 15, "b": 65},
        font={"size": 14},
        showlegend=True,
        uirevision=revision,
        selectionrevision=revision,
        legend={"orientation": "h", "y": -0.25, "x": 0, "font": {"size": 14}},
        xaxis={"title": AXES[axis], "automargin": True, "nticks": 5},
        yaxis={
            "title": spec.unit if spec else "Recorded value (unit unavailable)",
            "automargin": True,
            "nticks": 5,
        },
    )
    return chart


@lru_cache(maxsize=1)
def _component():
    script = Path(__file__).with_name("training_charts.js").read_text()
    return st.components.v2.component(
        "training_live_chart",
        js=get_plotlyjs() + "\n" + script,
        isolate_styles=False,
    )


def render(points: list[dict[str, object]], name: str, axis: str, context: str) -> None:
    chart = figure(points, name, axis)
    _component()(
        key=f"training-chart:{context}:{name}",
        data=chart.to_json(remove_uids=False),
        height="content",
    )
    spec = metric_spec(name)
    st.markdown(
        "**What this shows:** "
        + (
            spec.summary + f" Unit: {spec.unit}."
            if spec
            else "Recorded observations for this metric; no canonical description or unit is available."
        )
    )
