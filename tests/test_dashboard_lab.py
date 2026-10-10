"""Lab dashboard: data views, explorer figures and page renders (CPU, no training)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from test_explorer import MOE_NGRAM, TEXT, _loaded

from sparselab import explorer
from sparselab.dashboard import explorer_view as xv
from sparselab.dashboard import lab_data
from sparselab.lab_records import jsonable, seal, write_json_atomic, write_sealed
from sparselab.probes.suite import suite_identity

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = {
    "run_id": "lab-try-1",
    "checkpoint_sha256": "a" * 64,
    "step": 60,
    "tokens_seen": 7680,
    "parameters": 98_720,
    "active_parameters": 43_424,
    "eval_group": "g" * 64,
}
BASELINE = {**CANDIDATE, "run_id": "lab-base-1", "checkpoint_sha256": "b" * 64}
ITEM_GROUP = "i" * 64


def _row(probe: str, value: float, baseline: float | None, **details: Any) -> dict:
    return {
        "id": probe,
        "title": probe.replace("_", " "),
        "tier": "standard",
        "cost": 1,
        "hard": False,
        "status": "pass",
        "value": value,
        "baseline_value": baseline,
        "delta": None if baseline is None else value - baseline,
        "higher_is_better": probe != "heldout_loss",
        "thresholds": {"warn": 0.05, "fail": 0.15},
        "details": details,
    }


def _probe_result() -> dict[str, Any]:
    items = [
        {"prompt": "Q: Where does Mira live? A:", "answer": " Oslo", "picked": " Oslo", "credit": 1.0, "baseline_picked": " Lima", "baseline_credit": 0.0},
        {"prompt": "Q: What does Tam keep? A:", "answer": " a fox", "picked": " a cat", "credit": 0.0, "baseline_picked": " a cat", "baseline_credit": 0.0},
    ]  # fmt: skip
    bins = [
        {"low": 0.0, "high": 0.1, "confidence": 0.05, "accuracy": 0.04, "share": 0.6},
        {"low": 0.1, "high": 0.2, "confidence": 0.15, "accuracy": 0.2, "share": 0.4},
    ]
    return {
        "format": "sparselab-probe-v1",
        "created_at": "2026-10-10T19:00:00+00:00",
        "suite": {"sha256": suite_identity()["sha256"]},
        "tier": "standard",
        "tiers_run": ["fast", "standard"],
        "seconds": 3.2,
        "target": CANDIDATE,
        "baseline": BASELINE,
        "guard": {"overfit_suspected": False},
        "verdict": {
            "status": "warn",
            "action": "tweak",
            "next_tier": None,
            "missing": [],
            "suggestion": "Calibration slipped: tweak and retry.",
        },
        "probes": [
            _row(
                "heldout_loss",
                3.247,
                3.278,
                window_sums=[32.0, 33.0],
                window_counts=[10, 10],
            ),
            _row(
                "calibration", 0.296, 0.263, reliability=bins, baseline_reliability=bins
            ),
            _row(
                "repetition",
                0.69,
                0.966,
                distinct_1=0.2,
                distinct_2=0.03,
                samples=[{"prompt": "One day", "continuation": " the the bird"}],
                baseline_samples=[{"prompt": "One day", "continuation": " 1 1 1 1"}],
            ),
            _row(
                "fact_recall",
                0.5,
                0.0,
                chance=0.25,
                n=2,
                items=items,
                item_group=ITEM_GROUP,
            ),
            _row(
                "parametric_recall",
                0.5,
                0.25,
                chance=0.25,
                n=2,
                items=items,
                control_accuracy=0.25,
            ),
            _row(
                "needle",
                0.5,
                0.25,
                chance=1 / 6,
                n=6,
                by_length={
                    "0.5": {"tokens": 27, "accuracy": 0.5, "baseline_accuracy": 0.25}
                },
            ),
        ],
    }


@pytest.fixture(scope="module")
def lab(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dash-lab")
    probe = {**_probe_result(), "probe_id": "probe-1"}
    write_sealed(_mk(root / "probes/probe-1/probe.json"), probe)
    write_sealed(
        _mk(root / "tries/try-0/try.json"),
        {
            "format": "sparselab-lab-try-v1",
            "try_id": "try-0",
            "created_at": "2026-10-10T18:00:00+00:00",
            "question": "Do 4 experts beat the dense FFN?",
            "delta": {"model.ffn": {"base": "dense", "variant": "moe"}},
            "arms": {
                "baseline": {**BASELINE, "heldout": {"loss": 3.278}},
                "candidate": {**CANDIDATE, "heldout": {"loss": 3.247}},
            },
            "comparison": {
                "verdict": "CANDIDATE_LOWER_LOSS",
                "heldout_loss_delta": -0.031,
            },
        },
    )
    exploration = explorer.explore_loaded(_loaded(**MOE_NGRAM), TEXT)
    path = explorer.cache_path(root, "c" * 64, TEXT)
    path.parent.mkdir(parents=True)
    write_json_atomic(path, seal(jsonable(exploration)))
    run = root / "runs" / "plain-run"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")
    (run / "resolved_config.yaml").write_text("{}")
    return root


def _mk(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# --- Data views ----------------------------------------------------------------


def test_activity_and_next_steps_are_verdict_first(lab: Path) -> None:
    snap = lab_data.snapshot(lab)
    rows = lab_data.activity(snap)
    assert [r["id"] for r in rows] == ["probe-1", "try-0"]  # newest first
    assert rows[0]["verdict"] == "warn" and rows[0]["action"] == "tweak"
    assert rows[1]["outcome"] == "candidate has lower held-out loss"
    steps = lab_data.next_steps(snap)
    # "tweak" names no command (the verdict banner says it); the rest do.
    assert not any(s["title"].startswith("Latest verdict") for s in steps)
    assert all(s["why"] and s["command"].startswith("sparselab ") for s in steps)
    assert steps[-1]["command"] == "sparselab explore lab-try-1"
    with_runs = lab_data.next_steps(snap, [lab / "runs"])
    assert any(
        s["command"] == f"sparselab probe plain-run --runs-dir {lab / 'runs'}"
        for s in with_runs
    )
    empty = lab_data.next_steps(lab_data.snapshot(lab / "nowhere"))
    assert empty[0]["command"].startswith("sparselab try")


def test_catalog_unscored_runs_and_behaviors(lab: Path) -> None:
    snap = lab_data.snapshot(lab)
    catalog = {r["checkpoint"]: r for r in lab_data.checkpoint_catalog(snap.points)}
    mine = catalog["a" * 12]
    assert mine["heldout_loss"] == pytest.approx(3.247)
    assert mine["fact_recall"] == pytest.approx(0.5)
    assert any(r["kind"] == "reference" for r in catalog.values())
    unscored = lab_data.unscored_runs([lab / "runs"], snap.points)
    assert [u["run_id"] for u in unscored] == ["plain-run"]
    assert (
        unscored[0]["command"] == f"sparselab probe plain-run --runs-dir {lab / 'runs'}"
    )
    entry = snap.entries[0]
    assert lab_data.generations(entry) == [
        {"prompt": "One day", "candidate": " the the bird", "baseline": " 1 1 1 1"}
    ]
    misses = [i for i in lab_data.recall_items(entry, "fact_recall") if i["credit"] < 1]
    assert [m["answer"] for m in misses] == [" a fox"]
    curve = lab_data.needle_curves(snap.entries)
    assert curve[0]["tokens"] == 27 and curve[0]["accuracy"] == 0.5
    arms = {r["arm"] for r in lab_data.calibration_curves(entry)}
    assert arms == {"candidate", "baseline"}


def test_explorer_figures_follow_the_exploration(lab: Path) -> None:
    (exploration,) = lab_data.snapshot(lab).explorations
    arch = exploration["architecture"]
    page = xv.architecture_html(arch)
    assert "Mixture of experts" in page and "Memory · ngram" in page
    assert "shared KV head" in page  # grouped-query attention is called out
    layer = exploration["routing"]["0"]
    experts = arch["layers"][0]["ffn"]["experts"]
    figure = xv.routing_figure(layer, exploration["tokens"], experts, "r")
    gates = figure.data[0].z
    assert len(gates) == experts and len(gates[0]) == len(exploration["tokens"])
    assert sum(map(sum, gates)) == pytest.approx(
        sum(map(sum, layer["weights"])), rel=1e-6
    )
    assert sum(xv.expert_load(layer)) == pytest.approx(1.0)
    # head_summary on a pure previous-token head
    previous = [[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]]
    summary = xv.head_summary(previous)
    assert summary["mean_distance"] == pytest.approx(1.0)
    assert summary["mean_entropy"] == pytest.approx(0.0, abs=1e-9)
    rows = xv.memory_rows(exploration["memory"], exploration["tokens"])
    assert len(rows) == len(exploration["tokens"]) and "gate" in rows[0]
    assert set(xv.memory_reuse(exploration["memory"])) == {
        s["table"] for s in exploration["memory"]["streams"]
    }
    strip = xv.token_strip_html(exploration["tokens"])
    assert strip.count("lab-tok") == len(exploration["tokens"])


# --- Page renders ----------------------------------------------------------------


def _page(name: str, lab: str) -> None:
    from pathlib import Path

    from sparselab.dashboard import lab_pages

    root = Path(lab)
    runs = [root / "runs"]
    {
        "home": lambda: lab_pages.home_page(root, runs, {}),
        "experiments": lambda: lab_pages.experiments_page(root),
        "models": lambda: lab_pages.models_page(root, runs),
        "behaviors": lambda: lab_pages.behaviors_page(root),
        "explorer": lambda: lab_pages.explorer_page(root, runs),
    }[name]()


def _render(name: str, lab: Path) -> AppTest:
    test = AppTest.from_function(_page, args=(name, str(lab)), default_timeout=60)
    test.run()
    assert not test.exception, test.exception
    return test


def _text(test: AppTest) -> str:
    return "\n".join(
        [m.value for m in test.markdown]
        + [c.value for c in test.code]
        + [c.value for c in test.caption]
        + [i.value for i in test.info]
    )


@pytest.mark.parametrize(
    ("page", "command"),
    [
        ("home", "sparselab try DELTA.yaml --vs BASELINE.yaml"),
        ("experiments", "sparselab try DELTA.yaml --vs BASELINE.yaml"),
        ("models", "sparselab try DELTA.yaml --vs BASELINE.yaml"),
        ("behaviors", "sparselab probe RUN --tier standard"),
        ("explorer", "sparselab explore RUN"),
    ],
)
def test_empty_pages_name_the_command_to_run(
    page: str, command: str, tmp_path: Path
) -> None:
    assert command in _text(_render(page, tmp_path))


def test_home_leads_with_the_verdict_and_next_steps(lab: Path) -> None:
    text = _text(_render("home", lab))
    assert "WARN" in text and "Calibration slipped" in text
    assert "Next steps" in "\n".join(s.value for s in _render("home", lab).subheader)
    assert "Do 4 experts beat the dense FFN?" in text


def test_experiments_page_shows_results_trends_and_compare(lab: Path) -> None:
    test = _render("experiments", lab)
    assert [t.label for t in test.tabs] == [
        "Results",
        "History & trends",
        "Compare",
        "What the probes mean",
    ]
    assert len(test.get("plotly_chart")) >= 2  # comparison bars + trend
    text = _text(test)
    assert "fact recall" in text.lower()
    # Compare against the references: held-out loss is refused, never mixed.
    assert "NOT COMPARABLE" in text


def test_models_page_lists_catalog_references_and_unscored_runs(lab: Path) -> None:
    test = _render("models", lab)
    assert len(test.dataframe) >= 2  # catalog + references
    text = _text(test)
    assert "sparselab probe plain-run --runs-dir" in text


def test_behaviors_page_shows_generations_misses_and_reliability(lab: Path) -> None:
    test = _render("behaviors", lab)
    text = _text(test)
    assert "the the bird" in text and "1 1 1 1" in text
    assert "a fox" in text and "MISS" in text
    assert "never-trained" in text.lower() or any(
        "Never-trained" in m.label for m in test.metric
    )
    assert len(test.get("plotly_chart")) >= 2  # needle + reliability


def test_explorer_page_renders_a_cached_moe_memory_exploration(lab: Path) -> None:
    test = _render("explorer", lab)
    labels = [t.label for t in test.tabs]
    assert labels == [
        "Architecture",
        "Tokens",
        "Attention",
        "Weights",
        "Expert routing",
        "Memory lookups",
    ]
    assert len(test.get("plotly_chart")) >= 6


def test_explorer_page_names_the_size_limit(lab: Path) -> None:
    exploration = lab_data.snapshot(lab).explorations[0]

    def app(payload: dict) -> None:
        from sparselab.dashboard.lab_pages import render_exploration

        render_exploration(payload)

    too_big = {
        "format": exploration["format"],
        "target": {"run_id": "big"},
        "architecture": exploration["architecture"],
        "unavailable": "120,000,000 parameters is above the explorer's limit",
    }
    test = AppTest.from_function(app, args=(too_big,), default_timeout=60)
    test.run()
    assert not test.exception
    assert "Too big to explore" in test.warning[0].value


def test_full_app_opens_on_lab_home(lab: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real entry point: navigation builds and the default page is Home."""
    monkeypatch.setattr(
        sys, "argv", ["app", "--runs-dir", str(lab / "runs"), "--lab-dir", str(lab)]
    )
    test = AppTest.from_file(
        str(ROOT / "src/sparselab/dashboard/app.py"), default_timeout=60
    )
    test.run()
    assert not test.exception
    assert [h.value for h in test.header] == ["Lab home"]
    assert any(str(lab) in c.value for c in test.sidebar.caption)


def _lm_result(probe_id: str, when: str, value: float, group: str) -> dict:
    tasks = {"piqa": {"acc": value, "items": [1.0, 0.0]}}
    lm = _row("lm_eval", value, None, tasks=tasks, benchmark_group=group)
    return {
        **_probe_result(),
        "probe_id": probe_id,
        "created_at": when,
        "probes": [_row("heldout_loss", 3.2, None), lm],
    }


def _values(data: Any) -> list[float]:
    """Plotly JSON arrays may be base64 typed arrays."""
    import base64

    import numpy as np

    if isinstance(data, dict):
        raw = base64.b64decode(data["bdata"])
        return [round(float(v), 6) for v in np.frombuffer(raw, dtype=data["dtype"])]
    return [round(float(v), 6) for v in data]


def test_trends_never_join_results_from_different_benchmark_groups(
    tmp_path: Path,
) -> None:
    """lm-eval with other tasks/limit is another benchmark group: its own trend."""
    import json

    default, custom = "d" * 64, "w" * 64
    for probe_id, when, value, group in (
        ("probe-11", "2026-10-10T10:00:00+00:00", 0.40, default),
        ("probe-12", "2026-10-10T11:00:00+00:00", 0.42, default),
        ("probe-13", "2026-10-10T12:00:00+00:00", 0.90, custom),
    ):
        path = _mk(tmp_path / f"probes/{probe_id}/probe.json")
        write_sealed(path, _lm_result(probe_id, when, value, group))
    snap = lab_data.snapshot(tmp_path)
    rows = [r for r in lab_data.probe_trends(snap.entries) if r["probe"] == "lm_eval"]
    by_id = {r["id"]: r["trend_group"] for r in rows}
    assert len(set(by_id.values())) == 2
    assert by_id["probe-11"] == by_id["probe-12"] != by_id["probe-13"]
    # Held-out loss of the same results shares the eval group.
    loss = [
        r for r in lab_data.probe_trends(snap.entries) if r["probe"] == "heldout_loss"
    ]
    assert len({r["trend_group"] for r in loss}) == 1

    def trend_values(test: AppTest) -> list[list[float]]:
        for chart in test.get("plotly_chart"):
            figure = json.loads(chart.proto.spec)
            title = figure["layout"].get("title", {}).get("text", "")
            if "per result" in title:
                return [
                    _values(trace["y"])
                    for trace in figure["data"]
                    if trace.get("name") == "candidate"
                ]
        raise AssertionError("no trend chart")

    test = _render("experiments", tmp_path)
    test.selectbox(key="experiments_trend_probe").set_value("lm_eval").run()
    assert not test.exception
    # Newest result's group by default: only the custom-task point is drawn.
    assert trend_values(test) == [[0.9]]
    assert "1 other comparison group(s) hidden" in _text(test)
    group = test.selectbox(key="experiments_trend_group_lm_eval")
    assert len(group.options) == 2
    assert group.index == 0
    group.select_index(1).run()
    assert trend_values(test) == [[0.4, 0.42]]
