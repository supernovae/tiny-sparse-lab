"""Local read-only Streamlit dashboard."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import streamlit as st

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--runs-dir", default="runs")
args, _ = parser.parse_known_args()
st.set_page_config(page_title="SparseLab", layout="wide")
st.title("Tiny Sparse Lab")
db = Path(args.runs_dir) / "experiments.sqlite3"
if not db.exists():
    st.info("No runs yet. Run `sparselab train CONFIG --run-id NAME`.")
    st.stop()
with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
    runs = con.execute(
        "SELECT run_id,name,status,updated_at FROM runs ORDER BY updated_at DESC"
    ).fetchall()
    st.subheader("Overview")
    st.dataframe(runs)
    selected = st.multiselect(
        "Runs", [row[0] for row in runs], default=[row[0] for row in runs[:1]]
    )
    if selected:
        marks = ",".join("?" * len(selected))
        rows = con.execute(
            f"SELECT run_id,step,tokens_seen,name,value FROM metrics WHERE run_id IN ({marks}) ORDER BY step",
            selected,
        ).fetchall()
        st.subheader("Training")
        st.dataframe(rows)
st.subheader("Learn")
st.markdown(
    "**Loss** is mean negative log-probability in nats. **Perplexity** is `exp(loss)`. **Learning rate** is AdamW update LR. **Gradient norm** is global L2 before clipping. **Throughput** counts valid next-token targets per second. Parameter counts use the documented direct-use convention."
)
