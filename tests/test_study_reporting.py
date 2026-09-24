from __future__ import annotations

import hashlib
from typing import cast

import pytest
from pydantic import ValidationError

from sparselab.experiments.analysis import build_research_analysis
from sparselab.experiments.charts import render_charts
from sparselab.research.catalog import FactorialDesign

_FACTORIAL_DESIGN = {
    "format": "sparselab-factorial-design",
    "version": 1,
    "id": "width-memory",
    "factors": [
        {"axis": "width", "control": "4x", "treatment": "1x"},
        {"axis": "memory", "control": "none", "treatment": "lexical"},
    ],
}
_METRICS = [
    {"name": "held-out validation loss", "kind": "measured", "direction": "lower"},
    {"name": "capability-card score", "kind": "capability", "direction": "higher"},
    {
        "name": "parameter inventory",
        "kind": "architectural",
        "direction": "descriptive",
    },
]


def _run(
    run_id: str,
    width: str,
    memory: str,
    loss: float,
    score: float | None,
    *,
    tokens_seen: int = 128,
) -> dict[str, object]:
    table = 0 if memory == "none" else 32
    return {
        "run_id": run_id,
        "config_sha256": hashlib.sha256(run_id.encode("utf-8")).hexdigest(),
        "coordinate": {"seed": "s17", "width": width, "memory": memory},
        "endpoint_status": {"status": "complete", "step": 8, "tokens_seen": 128},
        "identity": {
            "run_id": run_id,
            "step": 8,
            "tokens_seen": tokens_seen,
            "tokenizer_sha256": "a" * 64,
            "data_sha256": {"train": "b" * 64, "validation": "c" * 64},
            "source_identity_sha256": "d" * 64,
            "runtime": {"engine": "pytorch", "backend": "cpu"},
            "training_runtime": {"device": "cpu"},
            "config": {"seed": 17},
        },
        "validation": {"loss": loss},
        "capabilities": {
            "card-a": {
                "result": {
                    "valid": score is not None,
                    "score": score,
                }
            }
        },
        "_inventory": {
            "total": 1000 + table,
            "active_per_token": 800 + table,
            "memory_table": table,
            "memory_adapter": 8 if memory != "none" else 0,
        },
    }


def _factorial_runs() -> list[dict[str, object]]:
    return [
        _run("y00", "4x", "none", 10.0, 0.25),
        _run("y10", "1x", "none", 8.0, 0.50),
        _run("y01", "4x", "lexical", 7.0, 0.50),
        _run("y11", "1x", "lexical", 4.0, 0.75),
    ]


def _analysis(runs: list[dict[str, object]]) -> dict[str, object]:
    quantities = []
    for run in runs:
        quantities.append(
            {
                "config_sha256": run["config_sha256"],
                "inventory": run["_inventory"],
            }
        )
    return build_research_analysis(
        runs,
        entry={"dependent_metrics": _METRICS},
        selection={
            "scale": "smoke",
            "data": "offline",
            "backend": "cpu",
            "design": "default",
        },
        factorial_designs=[_FACTORIAL_DESIGN],
        design_axes={
            "width": [
                {"label": "4x", "set": {"model.ffn_dim": 1024}},
                {"label": "1x", "set": {"model.ffn_dim": 256}},
            ],
            "memory": [
                {
                    "label": "none",
                    "set": {"model.memory": "none", "model.memory_table_size": 0},
                },
                {
                    "label": "lexical",
                    "set": {"model.memory": "ngram", "model.memory_table_size": 32},
                },
            ],
        },
        quantities=quantities,
        cards=[{"reference": "card-a"}],
    )


def _section(analysis: dict[str, object], name: str) -> dict[str, object]:
    section = analysis[name]
    assert isinstance(section, dict)
    return cast(dict[str, object], section)


def _rows(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return cast(list[dict[str, object]], value)


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _strings(value: object) -> list[str]:
    assert isinstance(value, list)
    assert all(isinstance(item, str) for item in value)
    return cast(list[str], value)


def test_factorial_reports_raw_cells_and_directional_effect_formulas() -> None:
    analysis = _analysis(_factorial_runs())
    factorial = _section(analysis, "factorial")
    designs = _rows(factorial["designs"])
    design = designs[0]
    contexts = _rows(design["contexts"])
    context = contexts[0]
    outcomes = _rows(context["outcomes"])
    loss = next(item for item in outcomes if item["id"] == "validation_loss")
    assert loss["direction"] == "lower"
    assert [cell["value"] for cell in _rows(loss["cells"])] == [
        10.0,
        8.0,
        7.0,
        4.0,
    ]
    assert loss["factor_a_main_effect"] == -2.5
    assert loss["factor_b_main_effect"] == -3.5
    assert loss["interaction_delta"] == -1.0
    assert loss["main_effects"] == {"width": -2.5, "memory": -3.5}
    score = next(item for item in outcomes if item["id"] == "capability:card-a")
    assert score["direction"] == "higher"
    assert score["interaction_delta"] == 0.0

    allocations = _section(analysis, "allocation_heatmaps")
    heatmaps = _rows(allocations["heatmaps"])
    heatmap = heatmaps[0]
    assert heatmap["kind"] == "architectural_estimate"
    cells = _rows(heatmap["cells"])
    assert _mapping(cells[0]["parameters"])["memory_share"] == 0.0
    assert _mapping(cells[2]["parameters"])["memory_table"] == 32

    sweeps = _section(analysis, "boundary_sweeps")
    assert "no threshold is inferred" in str(sweeps["interpretation"])
    sweep_axes = _rows(sweeps["axes"])
    assert {axis["axis"] for axis in sweep_axes} == {"width", "memory"}
    width_sweep = next(axis for axis in sweep_axes if axis["axis"] == "width")
    levels = _rows(_rows(width_sweep["groups"])[0]["levels"])
    assert levels[0]["configuration_patch"] == {"model.ffn_dim": 1024}


def test_static_charts_render_factorial_allocation_and_boundary_sweeps() -> None:
    charts = render_charts({"research_analysis": _analysis(_factorial_runs())})

    assert any(name.startswith("factorial-") for name in charts)
    assert any(name.startswith("allocation-") for name in charts)
    assert any(name.startswith("sweep-") for name in charts)
    allocation = next(
        content for name, content in charts.items() if name.startswith("allocation-")
    )
    assert "Architectural parameter allocation" in allocation
    assert "Not measured runtime cost" in allocation
    assert all(content.startswith("<?xml") for content in charts.values())


def test_missing_factorial_cell_is_inconclusive_without_effects() -> None:
    runs = _factorial_runs()
    runs.pop()

    analysis = _analysis(runs)
    factorial = _section(analysis, "factorial")
    design = _rows(factorial["designs"])[0]
    context = _rows(design["contexts"])[0]
    outcome = next(
        item for item in _rows(context["outcomes"]) if item["id"] == "validation_loss"
    )
    assert outcome["status"] == "inconclusive"
    assert "interaction_delta" not in outcome
    cells = _rows(outcome["cells"])
    assert cells[3]["cell"] == "y11"
    assert cells[3]["value"] is None
    assert "planned factorial cell is missing" in _strings(outcome["reasons"])[0]


def test_factorial_endpoint_identity_mismatch_is_inconclusive() -> None:
    runs = _factorial_runs()
    identity = cast(dict[str, object], runs[3]["identity"])
    runs[3]["identity"] = {**identity, "tokens_seen": 127}

    analysis = _analysis(runs)
    factorial = _section(analysis, "factorial")
    outcome = _rows(_rows(factorial["designs"])[0]["contexts"])[0]
    outcome = _rows(outcome["outcomes"])[0]
    assert outcome["status"] == "inconclusive"
    assert "interaction_delta" not in outcome
    assert any(
        "differs at tokens_seen" in reason for reason in _strings(outcome["reasons"])
    )


def test_nondominance_respects_directions_and_preserves_exact_ties() -> None:
    runs = [
        _run("front-a", "4x", "none", 1.0, 0.5),
        _run("front-b", "1x", "none", 1.0, 0.5),
        _run("dominated", "4x", "lexical", 2.0, 0.4),
        _run("tradeoff", "1x", "lexical", 0.5, 0.4),
        _run("missing-score", "4x", "none", 0.8, None),
    ]
    analysis = build_research_analysis(
        runs,
        entry={"dependent_metrics": _METRICS},
        selection={
            "scale": "smoke",
            "data": "offline",
            "backend": "cpu",
            "design": "default",
        },
        factorial_designs=[],
        design_axes={},
        quantities=[],
        cards=[{"reference": "card-a"}],
    )
    nondominance = _section(analysis, "nondominance")
    groups = _rows(nondominance["groups"])
    assert len(groups) == 1
    group = groups[0]
    assert group["nondominated_run_ids"] == ["front-a", "front-b", "tradeoff"]
    assert _rows(group["ties"]) == [
        {
            "values": {"validation_loss": 1.0, "capability:card-a": 0.5},
            "run_ids": ["front-a", "front-b"],
        }
    ]
    incomplete = _rows(group["inconclusive_candidates"])
    assert len(incomplete) == 1
    assert incomplete[0]["run_id"] == "missing-score"
    assert "capability:card-a" in _strings(incomplete[0]["reasons"])[0]


def test_nondominance_is_inconclusive_when_every_candidate_is_partial() -> None:
    run = _run("partial", "4x", "none", 1.0, 0.5)
    run["endpoint_status"] = {"status": "partial"}
    analysis = _analysis([run])

    nondominance = _section(analysis, "nondominance")
    groups = _rows(nondominance["groups"])
    assert nondominance["status"] == "inconclusive"
    assert groups[0]["status"] == "inconclusive"
    assert _rows(groups[0]["candidates"]) == []
    assert _rows(groups[0]["inconclusive_candidates"])[0]["run_id"] == "partial"


def test_factorial_design_metadata_has_strict_version_and_distinct_axes() -> None:
    FactorialDesign.model_validate(_FACTORIAL_DESIGN)
    with pytest.raises(ValidationError, match="version must be integer 1"):
        FactorialDesign.model_validate({**_FACTORIAL_DESIGN, "version": True})
    malformed = {
        **_FACTORIAL_DESIGN,
        "factors": [
            {"axis": "width", "control": "4x", "treatment": "1x"},
            {"axis": "width", "control": "2x", "treatment": "1x"},
        ],
    }
    with pytest.raises(ValidationError, match="distinct axes"):
        FactorialDesign.model_validate(malformed)
