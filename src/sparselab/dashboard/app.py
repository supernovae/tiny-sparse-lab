"""Local read-only Streamlit dashboard for SparseLab run telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import plotly.express as px
import streamlit as st

from sparselab.dashboard.queries import RunRecord, events, metrics, runs


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runs-dir", default="runs")
    args, _ = parser.parse_known_args()
    return args


def selected(root: Path) -> tuple[list[object], list[str]]:
    records = runs(root)
    ids = [record.run_id for record in records]
    selected_ids = st.sidebar.multiselect("Runs", ids, default=ids[:1])
    return records, selected_ids


def architecture(record: RunRecord) -> dict[str, str]:
    config = record.config
    model = config["model"]
    attention = config["attention"]
    dataset = config["dataset"]
    assert isinstance(model, dict)
    assert isinstance(attention, dict)
    assert isinstance(dataset, dict)
    return {
        "attention": str(attention.get("kind", "dense")),
        "dataset": str(dataset.get("source", "unknown")),
        "ffn": str(model.get("ffn", "dense")),
        "memory": str(model.get("memory", "none")),
    }


def overview(root: Path) -> None:
    st.header("Overview")
    records, selected_ids = selected(root)
    if not records:
        st.info("No runs yet. Run `sparselab train CONFIG --run-id NAME`.")
        return
    st.dataframe(
        [
            {
                "run_id": item.run_id,
                "name": item.name,
                "status": item.status,
                "parent": item.parent_run_id,
                "updated_at": item.updated_at,
                **architecture(item),
            }
            for item in records
        ],
        use_container_width=True,
    )
    if selected_ids:
        st.subheader("Events")
        st.dataframe(events(root, selected_ids), use_container_width=True)


def training(root: Path) -> None:
    st.header("Training")
    _, selected_ids = selected(root)
    points = metrics(root, selected_ids)
    if not points:
        st.info("Select a run with recorded metrics.")
        return
    axis = st.radio("X axis", ("step", "tokens_seen", "wall_time"), horizontal=True)
    names = sorted({str(point["name"]) for point in points})
    selected_names = st.multiselect("Metrics", names, default=names)
    filtered = [point for point in points if point["name"] in selected_names]
    figure = px.line(
        filtered, x=axis, y="value", color="run_id", facet_row="name", markers=True
    )
    figure.update_layout(height=max(350, 220 * len(selected_names)))
    st.plotly_chart(figure, use_container_width=True)
    with st.expander("What is this?"):
        st.markdown(
            "Charts show only stored observations. Loss is mean negative log-probability in nats; perplexity is `exp(loss)`. Throughput counts valid target tokens per timed update second."
        )


def evaluation(root: Path) -> None:
    st.header("Evaluation")
    records, selected_ids = selected(root)
    reports: list[dict[str, object]] = []
    for record in records:
        if record.run_id not in selected_ids:
            continue
        for path in (root / record.run_id / "evaluations").glob("*.json"):
            report = json.loads(path.read_text())
            if isinstance(report, dict):
                reports.append({"run_id": record.run_id, "path": str(path), **report})
    if reports:
        st.dataframe(reports, use_container_width=True)
    else:
        st.info(
            "No retained evaluations. Use `sparselab facts evaluate RUN_ID MANIFEST` "
            "or `sparselab facts transfer-evaluate SOURCE TARGET MANIFEST`."
        )


def learn() -> None:
    st.header("Learn")
    st.markdown(
        """### Metric guide

- **Loss:** mean negative log-probability in nats per next-token target. Lower is better only under comparable tokenizer, data, and budget conditions.
- **Perplexity:** `exp(loss)`; compare only with compatible tokenizers.
- **Learning rate:** actual AdamW update learning rate.
- **Gradient norm:** global L2 norm before clipping; spikes can indicate instability.
- **Throughput:** valid next-token targets per timed update second.
- **Parameter counts:** total/trainable/active-per-token use the dense inspection convention, not FLOPs.

### Reference guides

- `docs/architecture.md` — dense, sliding-window, MLA, MoE, and memory boundaries.
- `docs/training.md` — data packing, checkpoints, and resume behavior.
- `docs/metrics.md` — metric definitions and comparison limits.
- `docs/experiments.md` — controlled scale-comparison protocol.
- `docs/withheld-facts.md` — fact manifest, audit, and transfer-evaluation workflow.
"""
    )


def main() -> None:
    st.set_page_config(page_title="SparseLab", layout="wide")
    st.title("Tiny Sparse Lab")
    root = Path(arguments().runs_dir)
    if not (root / "experiments.sqlite3").is_file():
        st.info("No runs yet. Run `sparselab train CONFIG --run-id NAME`.")
        return
    page = st.navigation(
        [
            st.Page(lambda: overview(root), title="Overview", url_path="overview"),
            st.Page(lambda: training(root), title="Training", url_path="training"),
            st.Page(
                lambda: evaluation(root), title="Evaluation", url_path="evaluation"
            ),
            st.Page(learn, title="Learn", url_path="learn"),
        ]
    )
    page.run()


main()
