"""Probe follow-ups: closed-book recall, text-level item groups, reliability
diagrams, lm-eval overrides and fact recall as a comparable metric."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sparselab.data.withheld_facts import diagnostic_manifest
from sparselab.lab_records import write_sealed
from sparselab.probes import compare as compare_mod
from sparselab.probes import lm_eval_adapter, metrics, runner, scoring
from sparselab.probes.points import collect_points, metric_points
from sparselab.probes.suite import BY_ID, parametric_items


class _Words:
    """Whitespace tokenizer; ``offset`` shifts every id (a different tokenizer)."""

    def __init__(self, offset: int = 0) -> None:
        self.offset = offset
        self.vocab: dict[str, int] = {"<eos>": 0}

    def encode(self, text: str, add_special_tokens: bool = False) -> Any:
        ids = [
            self.vocab.setdefault(w, len(self.vocab)) + self.offset
            for w in text.split()
        ]
        return type("Encoding", (), {"ids": ids})()

    def token_to_id(self, token: str) -> int | None:
        return self.vocab.get(token)


class _Loaded:
    def __init__(self, tokenizer: Any, max_seq_len: int = 256) -> None:
        self.tokenizer = tokenizer
        self.config = type(
            "Config", (), {"model": type("Model", (), {"max_seq_len": max_seq_len})()}
        )()


@pytest.fixture
def uniform(monkeypatch: pytest.MonkeyPatch) -> None:
    vocab = 5000
    monkeypatch.setattr(
        scoring,
        "log_probs",
        lambda loaded, ids: np.full((len(ids), vocab), -math.log(vocab)),
    )


def _measure(probe: str, loaded: Any) -> dict[str, Any]:
    return runner._measure(BY_ID[probe], loaded, runner.Arm(load=lambda: loaded), {})


# --- Reliability diagram ------------------------------------------------------


def test_reliability_bins_are_the_data_behind_ece() -> None:
    conf = [0.9, 0.9, 0.1, 0.1]
    hit = [True, False, False, False]
    bins = metrics.reliability_bins(conf, hit, bins=10)
    assert [(b["low"], b["high"]) for b in bins] == [(0.0, 0.1), (0.8, 0.9)]
    assert [b["accuracy"] for b in bins] == [0.0, 0.5]
    assert [b["share"] for b in bins] == [0.5, 0.5]
    assert metrics.expected_calibration_error(conf, hit, 10) == pytest.approx(
        sum(b["share"] * abs(b["accuracy"] - b["confidence"]) for b in bins)
    )


def test_calibration_row_keeps_both_reliability_diagrams() -> None:
    spec = BY_ID["calibration"]
    t = {"value": 0.1, "bins": 15, "positions": 4, "reliability": [{"share": 1.0}]}
    b = {**t, "reliability": [{"share": 0.5}, {"share": 0.5}]}
    row = runner._judge(spec, t, b, {"validation": True, "tokenizer": True}, {})
    assert row["details"]["reliability"] == t["reliability"]
    assert row["details"]["baseline_reliability"] == b["reliability"]


# --- Closed-book (parametric) fact recall -------------------------------------


def test_parametric_items_follow_the_withheld_facts_manifest() -> None:
    manifest = diagnostic_manifest(0)
    trained = parametric_items("heldout")
    control = parametric_items("heldout", control=True)
    # Verdict items: the manifest's training facts, asked closed-book with its
    # canonical prompt (the fact itself is never in the context).
    assert [f"{i['question']} {i['answer']}." for i in trained] == manifest[
        "training_statements"
    ]
    assert all(i["answer"] not in i["question"] for i in trained)
    # Control: exactly the never-trained held-out cases.
    assert [(i["question"], i["answer"]) for i in control] == [
        (c["prompt"], c["expected_value"]) for c in manifest["held_out_cases"]
    ]
    assert {i["question"] for i in trained}.isdisjoint(i["question"] for i in control)
    dev = parametric_items("dev")
    assert [i["answer"] for i in dev] == [i["answer"] for i in trained]
    assert {i["question"] for i in dev}.isdisjoint(i["question"] for i in trained)
    assert all(len(i["candidates"]) == 4 for i in trained + control)


@pytest.mark.usefixtures("uniform")
def test_parametric_recall_of_a_uniform_model_is_chance_with_a_chance_control() -> None:
    out = _measure("parametric_recall", _Loaded(_Words()))
    assert out["value"] == pytest.approx(0.25)
    assert out["control_accuracy"] == pytest.approx(0.25)
    assert out["chance"] == pytest.approx(0.25)
    assert out["manifest_sha256"] == diagnostic_manifest(0)["sha256"]
    assert len(out["items"]) == len(out["credits"]) == len(out["picked"]) == 6


# --- Text-level item groups -----------------------------------------------------


@pytest.mark.usefixtures("uniform")
def test_recall_item_group_is_tokenizer_independent_but_needle_is_not() -> None:
    small = _Loaded(_Words(), max_seq_len=64)
    other = _Loaded(_Words(offset=100), max_seq_len=256)  # another tokenizer
    for probe in ("fact_recall", "parametric_recall"):
        assert (
            _measure(probe, small)["item_group"] == _measure(probe, other)["item_group"]
        )
    # Needle items are sized in the model's own tokens: a different context
    # renders different items, so they never share a group.
    assert (
        _measure("needle", small)["item_group"]
        != _measure("needle", other)["item_group"]
    )


def _side(credits: list[float], picked: list[str], group: str = "g" * 64) -> dict:
    return {
        "value": float(np.mean(credits)),
        "credits": credits,
        "picked": picked,
        "dev_credits": [],
        "fractions": [None] * len(credits),
        "tokens": [None] * len(credits),
        "chance": 0.25,
        "item_group": group,
        "items": [{"prompt": f"q{i}", "answer": "red"} for i in range(len(credits))],
    }


def test_ranking_rows_keep_each_items_pick_and_refuse_other_items() -> None:
    spec = BY_ID["fact_recall"]
    ok = {"validation": True, "tokenizer": False}  # text-level: tokenizers may differ
    t = _side([1.0, 0.0, 1.0], ["red", "blue", "red"])
    b = _side([0.0, 0.0, 1.0], ["gold", "blue", "red"])
    row = runner._judge(spec, t, b, ok, {})
    assert row["status"] != "not_comparable" and row["delta"] == pytest.approx(1 / 3)
    first = row["details"]["items"][0]
    assert first == {
        "prompt": "q0",
        "answer": "red",
        "picked": "red",
        "credit": 1.0,
        "baseline_picked": "gold",
        "baseline_credit": 0.0,
    }
    other = runner._judge(
        spec, t, _side([1.0] * 3, ["red"] * 3, group="h" * 64), ok, {}
    )
    assert other["status"] == "not_comparable" and "different items" in other["note"]


# --- lm-eval overrides ------------------------------------------------------------


def test_lm_eval_overrides_make_a_new_benchmark_never_a_new_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runner.lm_eval_spec() is BY_ID["lm_eval"]
    spec = runner.lm_eval_spec(["piqa", "winogrande"], 200)
    assert spec.params["tasks"] == ["piqa", "winogrande"]
    assert spec.params["limit"] == 200
    assert spec.params["chance"] == {"piqa": 0.5, "winogrande": None}
    with pytest.raises(ValueError):
        runner.lm_eval_spec(["piqa", "piqa"])
    with pytest.raises(ValueError):
        runner.lm_eval_spec(limit=0)
    seen: dict[str, Any] = {}

    def fake(loaded: Any, tasks: list[str], limit: int) -> dict[str, Any]:
        seen.update(tasks=tasks, limit=limit)
        raise lm_eval_adapter.LmEvalUnavailable("not installed")

    monkeypatch.setattr(lm_eval_adapter, "run_lm_eval", fake)
    out = runner._measure_safely(spec, object(), runner.Arm(load=object), {})
    assert seen == {"tasks": ["piqa", "winogrande"], "limit": 200}
    assert out["status"] == "unavailable"


def test_probe_cli_rejects_bad_lm_eval_options_before_any_work(tmp_path: Path) -> None:
    from sparselab.probes.cli import run_probe

    with pytest.raises(ValueError, match="at least 1"):
        run_probe("missing-run", None, lab_dir=tmp_path, lm_eval_limit=0)
    assert not (tmp_path / "probes").exists()


# --- Fact recall as a comparable metric (points, compare) ------------------------


def _record(lab: Path, probe_id: str, row: dict[str, Any], sha: str) -> None:
    target = {
        "run_id": probe_id,
        "checkpoint_sha256": sha,
        "step": 1,
        "tokens_seen": 10,
        "parameters": 100,
        "active_parameters": 50,
        "eval_group": None,
    }
    record = {
        "format": "sparselab-probe-v1",
        "probe_id": probe_id,
        "created_at": "2026-10-10T12:00:00+00:00",
        "target": target,
        "baseline": None,
        "probes": [row],
        "verdict": {"status": "info"},
    }
    path = lab / "probes" / probe_id / "probe.json"
    path.parent.mkdir(parents=True)
    write_sealed(path, record)


def test_fact_recall_is_a_point_metric_paired_only_within_one_item_group(
    tmp_path: Path,
) -> None:
    spec = BY_ID["fact_recall"]
    ok = {"validation": True, "tokenizer": True}
    good = runner._judge(spec, _side([1.0, 1.0, 1.0, 0.0], ["red"] * 4), None, ok, {})
    weak = runner._judge(spec, _side([0.0, 1.0, 0.0, 0.0], ["red"] * 4), None, ok, {})
    alien = runner._judge(
        spec, _side([1.0, 0.0, 0.0, 0.0], ["red"] * 4, group="h" * 64), None, ok, {}
    )
    _record(tmp_path, "probe-good", good, "a" * 64)
    _record(tmp_path, "probe-weak", weak, "b" * 64)
    _record(tmp_path, "probe-alien", alien, "c" * 64)
    points = metric_points(collect_points(tmp_path, references=False), "fact_recall")
    assert {p["source"]: p["value"] for p in points} == {
        "probe-good": 0.75,
        "probe-weak": 0.25,
        "probe-alien": 0.25,
    }
    report = compare_mod.compare(
        "probe-good", ["probe-weak", "probe-alien"], lab_dir=tmp_path
    )
    pairs = {
        c["other"]: next(p for p in c["pairs"] if p["metric"] == "fact_recall")
        for c in report["comparisons"]
    }
    weak_pair = pairs["probe-weak@1"]
    assert weak_pair["delta"] == pytest.approx(0.5)
    assert weak_pair["se"] is not None  # paired over the same items
    assert pairs["probe-alien@1"]["status"] == "not_comparable"
    assert "fact recall" in compare_mod.render(report)
