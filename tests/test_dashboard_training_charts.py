"""Synthetic telemetry only: no tokenizer, model or training work."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from sparselab.dashboard.training_charts import AXES, figure


@pytest.fixture(autouse=True)
def isolated_component_registry():
    # AppTest creates a fresh Streamlit runtime/registry per test.
    from sparselab.dashboard.training_charts import _component, _loader

    _component.cache_clear()
    _loader.cache_clear()
    yield
    _component.cache_clear()
    _loader.cache_clear()


def points():
    return [
        {
            "run_id": run,
            "step": step,
            "tokens_seen": step * 32,
            "wall_time": step * 2.5,
            "name": "train/loss",
            "value": value,
        }
        for run in ("run-b", "run-a")
        for step, value in ((1, 2.0), (2, None), (3, 1.5))
    ]


@pytest.mark.parametrize("axis", AXES)
def test_observations_and_units_are_unchanged(axis):
    chart = figure(points(), "train/loss", axis)
    assert chart.layout.yaxis.title.text == "nats/target"
    for trace in chart.data:
        rows = [row for row in points() if row["run_id"] == trace.uid]
        assert list(trace.x) == [row[axis] for row in rows]
        assert list(trace.y) == [row["value"] for row in rows]
        assert trace.connectgaps is False
    assert chart.layout.legend.orientation == "h"
    assert chart.layout.font.size >= 14


def test_live_revision_changes_only_with_interpretation():
    initial = figure(points(), "train/loss", "step")
    appended = points() + [dict(points()[0], step=4, value=1.2)]
    changed = figure(appended, "train/loss", "step")
    assert changed.layout.uirevision == initial.layout.uirevision
    assert changed.layout.selectionrevision == initial.layout.selectionrevision
    assert [trace.uid for trace in changed.data] == [
        trace.uid for trace in initial.data
    ]
    assert (
        figure(points(), "train/loss", "wall_time").layout.uirevision
        != initial.layout.uirevision
    )
    assert (
        figure(points()[:3], "train/loss", "step").layout.uirevision
        != initial.layout.uirevision
    )


def test_component_identity_survives_appended_and_unchanged_data():
    def page():
        import streamlit as st

        from sparselab.dashboard.training_charts import render

        rows = [
            {"run_id": "fixture", "step": i, "value": 3 - i / 10}
            for i in range(st.session_state.get("count", 3))
        ]
        render(rows, "train/loss", "step", "fixture-root")
        render(rows, "custom/unknown", "step", "fixture-root")

    app = AppTest.from_function(page).run()
    assert not app.exception
    initial = app.get("bidi_component")[0].proto
    assert len(initial.js_content.encode()) < 64_000
    app.run()
    assert app.get("bidi_component")[0].proto.id == initial.id
    app.session_state["count"] = 4
    app.run()
    assert not app.exception
    updated = app.get("bidi_component")[0].proto
    assert updated.id == initial.id
    assert updated.json != initial.json
    assert len(updated.js_content.encode()) < 64_000
    assert json.loads(json.loads(updated.json))["data"][0]["uid"] == "fixture"
    assert len(json.loads(json.loads(updated.json))["data"][0]["x"]) == 4
    assert "nats/target" in app.markdown[0].value
    assert "no canonical description or unit" in app.markdown[1].value


def test_series_colors_match_when_a_metric_is_missing_for_one_run():
    both = figure(points(), "train/loss", "step", ["run-a", "run-b"])
    only_b = figure(points()[:3], "train/loss", "step", ["run-a", "run-b"])
    assert only_b.data[0].uid == both.data[1].uid == "run-b"
    assert only_b.data[0].line.color == both.data[1].line.color


def test_unknown_unit_and_incomplete_color_selector():
    chart = figure(points(), "custom/unknown", "step", ["run-a"])
    assert chart.layout.yaxis.title.text == "Recorded value (unit unavailable)"
    assert len(chart.data) == 2


def test_training_page_wires_consistent_colors_and_one_static_loader():
    def page():
        import streamlit as st

        from sparselab.dashboard.app import training
        from sparselab.dashboard.queries import DashboardSnapshot, RunRecord
        from sparselab.dashboard.training_charts import load_plotly

        load_plotly()
        st.session_state["selected_runs"] = ["run-a", "run-b"]
        # No real run directory or checkpoint is needed for this presentation test.
        records = tuple(
            RunRecord(
                run_id=r,
                name=r,
                status="running",
                parent_run_id=None,
                updated_at="t",
                config={},
                metadata={},
                purpose=None,
            )
            for r in ("run-a", "run-b")
        )
        rows = tuple(
            {
                "run_id": run,
                "name": metric,
                "step": 1,
                "tokens_seen": 32,
                "wall_time": 2.0,
                "value": 2.0,
            }
            for run, metric in (
                ("run-a", "train/loss"),
                ("run-b", "train/loss"),
                ("run-b", "validation/loss"),
            )
        )
        view = DashboardSnapshot(
            runs=records,
            metrics=rows,
            events=(),
            stages=(),
            checkpoints=(),
            runtime_progress=(),
            manifests={},
        )
        training(view)

    app = AppTest.from_function(page).run()
    assert not app.exception
    components = app.get("bidi_component")
    assert len(components) == 3
    loader = components[0].proto
    assert len(loader.js_content) > 1_000_000
    figures = [json.loads(json.loads(c.proto.json)) for c in components[1:]]
    assert all(len(c.proto.js_content.encode()) < 64_000 for c in components[1:])
    assert (
        figures[0]["data"][1]["line"]["color"] == figures[1]["data"][0]["line"]["color"]
    )
    app.run()
    assert app.get("bidi_component")[0].proto == loader
