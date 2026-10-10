"""Probe battery: metrics, verdicts, fast-fail ordering, schema, CLI and `try`."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from sparselab.probes import lm_eval_adapter, metrics, runner
from sparselab.probes.render import meter, render, sparkline
from sparselab.probes.suite import (
    BY_ID,
    PROBES,
    SUITE_VERSION,
    TIERS,
    fact_items,
    needle_split,
    ordered,
    prompts,
    suite_identity,
)
from sparselab.probes.verdict import ACTIONS, decide, judge, overfit_guard

ROOT = Path(__file__).resolve().parents[1]

# Any change to probe declarations or items must bump SUITE_VERSION and this pin,
# so historical results are never silently compared across different suites.
PINNED_SUITE = {
    1: "68a1bba35e4dca50ba4cf45d49fd466c0a7c764ef2a1f02227753f930c4545f3",
}

RESULT_KEYS = {
    "format",
    "created_at",
    "suite",
    "tier",
    "tiers_run",
    "fast_fail",
    "target",
    "baseline",
    "comparable",
    "protocol",
    "probes",
    "guard",
    "verdict",
    "seconds",
}
ROW_KEYS = {
    "id",
    "title",
    "tier",
    "cost",
    "metric",
    "higher_is_better",
    "hard",
    "thresholds",
    "suggests",
    "explains",
    "reference",
    "status",
    "value",
    "baseline_value",
    "delta",
    "regression",
    "delta_se",
    "within_noise",
    "improved",
    "note",
    "details",
    "seconds",
}
IDENTITY_KEYS = {
    "run_id",
    "run_dir",
    "checkpoint",
    "checkpoint_sha256",
    "step",
    "tokens_seen",
    "parameters",
    "parameter_bytes",
    "tokenizer_sha256",
    "validation_sha256",
    "max_seq_len",
}
VERDICT_KEYS = {"status", "action", "next_tier", "reasons", "suggestion"}


# --- Metrics (known answers) -------------------------------------------------


def test_distinct_n_and_seq_rep_known_answers() -> None:
    assert metrics.distinct_n([[1, 2, 3, 4]], 1) == 1.0
    assert metrics.distinct_n([[1, 1, 1, 1]], 1) == 0.25
    # Pooled across sequences: bigrams (1,2),(2,1),(1,2) + (1,2) -> 2 unique of 4.
    assert metrics.distinct_n([[1, 2, 1, 2], [1, 2]], 2) == 0.5
    assert metrics.distinct_n([[1]], 2) is None
    assert metrics.seq_rep_n([1, 2, 3, 4, 5], 4) == 0.0
    # A pure loop of period 1: 7 four-grams, one unique.
    assert metrics.seq_rep_n([7] * 10, 4) == pytest.approx(1 - 1 / 7)
    assert metrics.seq_rep_n([1, 2, 3], 4) is None
    assert metrics.mean_seq_rep_n([[1, 2, 3, 4, 5], [7] * 10, [1]], 4) == (
        pytest.approx((0 + 6 / 7) / 2)
    )
    with pytest.raises(ValueError):
        metrics.ngrams([1, 2], 0)


def test_ece_known_answers() -> None:
    assert metrics.expected_calibration_error([1.0, 1.0], [True, True]) == 0.0
    # Two bins of two: confident and right half the time.
    value = metrics.expected_calibration_error(
        [0.9, 0.9, 0.1, 0.1], [True, False, False, False], bins=10
    )
    assert value == pytest.approx(0.5 * abs(0.5 - 0.9) + 0.5 * abs(0.0 - 0.1))
    # Bin edges are right-closed: 0.2 joins (0.1, 0.2] with 10 bins.
    edge = metrics.expected_calibration_error([0.2, 0.15], [False, False], bins=10)
    assert edge == pytest.approx(0.175)
    with pytest.raises(ValueError):
        metrics.expected_calibration_error([1.5], [True])
    with pytest.raises(ValueError):
        metrics.expected_calibration_error([], [])


def test_divergences_and_agreement_known_answers() -> None:
    p = np.log(np.array([[0.5, 0.5], [0.9, 0.1]]))
    q = np.log(np.array([[0.5, 0.5], [0.1, 0.9]]))
    kl = metrics.kl_divergence(p, q)
    assert kl[0] == pytest.approx(0.0)
    assert kl[1] == pytest.approx(0.9 * math.log(9) + 0.1 * math.log(1 / 9))
    js = metrics.js_divergence(p, q)
    assert js[0] == pytest.approx(0.0)
    assert 0 < js[1] < math.log(2)
    disjoint = metrics.js_divergence(
        np.log(np.array([1.0, 1e-300])), np.log(np.array([1e-300, 1.0]))
    )
    assert disjoint == pytest.approx(math.log(2))
    assert metrics.top1_agreement(p[1:], q[1:]) == 0.0
    assert metrics.top1_agreement(p[1:], p[1:]) == 1.0
    logits = np.array([[1.0, 2.0, 3.0]])
    assert np.exp(metrics.log_softmax(logits)).sum() == pytest.approx(1.0)


def test_standard_errors_known_answers() -> None:
    mean, se = metrics.paired_mean_and_se([1.0, 3.0])
    assert mean == 2.0 and se == pytest.approx(1.0)
    assert metrics.paired_mean_and_se([5.0]) == (5.0, None)
    assert metrics.binomial_se(0.5, 100) == pytest.approx(0.05)
    assert metrics.accuracy_delta_se(0.5, 0.5, 50) == pytest.approx(0.1)
    assert metrics.accuracy_delta_se(0.5, 0.5, 0) is None


# --- Suite identity and splits ----------------------------------------------


def test_suite_identity_is_pinned_and_versioned() -> None:
    identity = suite_identity()
    assert identity["name"] == "sparselab-probe-battery"
    assert identity["version"] == SUITE_VERSION
    assert identity["sha256"] == PINNED_SUITE[SUITE_VERSION], (
        "probe declarations or items changed: bump SUITE_VERSION and the pin"
    )
    assert identity["verdict_split"] == "heldout"
    assert set(identity["splits"]) == {"dev", "heldout"}


def test_heldout_items_are_disjoint_from_dev_items() -> None:
    dev, held = fact_items("dev"), fact_items("heldout")
    assert {i["question"] for i in dev}.isdisjoint({i["question"] for i in held})
    assert {i["context"] for i in dev}.isdisjoint({i["context"] for i in held})
    assert {i["answer"] for i in dev}.isdisjoint({i["answer"] for i in held})
    assert set(needle_split("dev")["needles"]).isdisjoint(
        needle_split("heldout")["needles"]
    )
    assert set(prompts("dev")).isdisjoint(prompts("heldout"))
    for item in dev + held:
        assert item["answer"] in item["candidates"]
    for item in held:
        # Reworded: held-out questions never repeat the context's phrasing.
        assert item["context"].split(" is ")[0] not in item["question"]


def test_every_probe_declares_tier_cost_thresholds_and_hint() -> None:
    assert {p.tier for p in PROBES} == set(TIERS)
    for spec in PROBES:
        assert spec.cost >= 1 and spec.suggests and spec.explains and spec.reference
        assert spec.mode in {"delta_rel", "delta_abs", "value_min"}
        assert spec.fail is None or spec.fail >= spec.warn or spec.mode == "value_min"


def test_ordering_is_cheapest_first_and_tiered() -> None:
    fast = [s.id for s in ordered("fast")]
    assert fast[0] == "heldout_loss" and {BY_ID[i].tier for i in fast} == {"fast"}
    full = ordered("full")
    keys = [(TIERS.index(s.tier), s.cost) for s in full]
    assert keys == sorted(keys)
    assert full[-1].id == "lm_eval"
    with pytest.raises(ValueError):
        ordered("huge")


# --- Verdict logic ------------------------------------------------------------


def test_judge_relative_loss_thresholds_and_noise() -> None:
    spec = BY_ID["heldout_loss"]
    assert judge(spec, 4.0, 4.0)["status"] == "pass"
    worse = judge(spec, 4.06, 4.0)
    assert worse["regression"] == pytest.approx(0.015) and worse["status"] == "warn"
    assert judge(spec, 4.2, 4.0)["status"] == "fail"
    noisy = judge(spec, 4.2, 4.0, se=0.15)
    assert noisy["status"] == "pass" and noisy["within_noise"] is True
    better = judge(spec, 3.8, 4.0, se=0.01)
    assert better["improved"] is True and better["status"] == "pass"
    assert judge(spec, float("nan"), 4.0)["status"] == "fail"
    assert judge(spec, 4.0, None)["status"] == "info"


def test_judge_higher_is_better_and_value_min() -> None:
    recall = BY_ID["fact_recall"]
    assert judge(recall, 0.5, 0.7)["status"] == "fail"
    assert judge(recall, 0.6, 0.7)["status"] == "warn"
    assert judge(recall, 0.9, 0.7)["improved"] is True
    agreement = BY_ID["token_agreement"]
    assert judge(agreement, 0.9, None)["status"] == "pass"
    assert judge(agreement, 0.2, None)["status"] == "warn"
    loops = judge(BY_ID["repetition"], 0.99, None)
    assert loops["status"] == "warn"


def _result_row(probe_id: str, status: str, **extra: Any) -> dict[str, Any]:
    return runner._row(BY_ID[probe_id], status=status, **extra)


def test_decide_actions_cover_the_documented_table() -> None:
    def verdict(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        options = {
            "has_baseline": True,
            "tiers_run": ["fast"],
            "requested_tier": "fast",
            "guard": None,
            "specs": BY_ID,
            **kw,
        }
        out = decide(rows, **options)
        assert set(out) == VERDICT_KEYS and out["action"] in ACTIONS
        return out

    improved = _result_row("heldout_loss", "pass", improved=True)
    assert verdict([improved])["action"] == "escalate"
    assert verdict([improved])["next_tier"] == "standard"
    assert (
        verdict([improved], tiers_run=["fast", "standard", "full"])["action"]
        == "longer_run"
    )
    flat = _result_row("heldout_loss", "pass")
    assert verdict([flat])["action"] == "tweak"
    hard = verdict([_result_row("heldout_loss", "fail")])
    assert hard["action"] == "abandon" and hard["status"] == "fail"
    soft = verdict([improved, _result_row("calibration", "fail")])
    assert soft["action"] == "tweak" and soft["status"] == "fail"
    warned = verdict([improved, _result_row("calibration", "warn")])
    assert warned["status"] == "warn" and warned["action"] == "escalate"
    guarded = verdict([improved], guard={"overfit_suspected": True})
    assert guarded["action"] == "tweak" and "held-out" in guarded["suggestion"]
    assert verdict([improved], has_baseline=False)["action"] == "compare"


def test_overfit_guard_flags_dev_only_gains_beyond_noise() -> None:
    held = {"fact_recall": {"value": 0.5, "baseline_value": 0.5}}
    big = overfit_guard({"fact_recall": {"value": 1.0, "baseline_value": 0.5}}, held)
    assert big["overfit_suspected"] is True
    shared = overfit_guard(
        {"fact_recall": {"value": 1.0, "baseline_value": 0.5}},
        {"fact_recall": {"value": 0.9, "baseline_value": 0.5}},
    )
    assert shared["overfit_suspected"] is False
    # One flipped item out of four is noise, not overfitting.
    tiny = overfit_guard(
        {"fact_recall": {"value": 0.5, "baseline_value": 0.25, "n": 4}}, held
    )
    assert tiny["overfit_suspected"] is False
    assert tiny["verdict_split"] == "heldout"


# --- Fast-fail ordering (scripted probes, no model) ----------------------------


class _FakeSession:
    def __init__(self, loaded: Any) -> None:
        self.loaded = loaded
        self.identity = {key: None for key in IDENTITY_KEYS} | {"run_id": loaded}

    def close(self) -> None:
        pass


class _Calls(list):
    script: dict[str, str]


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> _Calls:
    calls = _Calls()
    script: dict[str, str] = {}
    calls.script = script
    monkeypatch.setattr(runner, "_Session", _FakeSession)
    monkeypatch.setattr(runner, "_validation_identity", lambda loaded: {"v": 1})
    monkeypatch.setattr(runner, "_tokenizer_digest", lambda loaded: "t")

    def fake(spec, target, base, protocol, comparable, dev):
        calls.append(spec.id)
        status = script.get(spec.id, "pass")
        return runner._row(spec, status=status, value=1.0, baseline_value=1.0)

    monkeypatch.setattr(runner, "_run_probe", fake)
    return calls


PROTOCOL = {"seq_len": 32, "batch_size": 4, "max_batches": 1}


def test_battery_runs_cheapest_first(scripted: Any) -> None:
    result = runner.run_battery("cand", "base", tier="standard", protocol=PROTOCOL)
    assert scripted == [s.id for s in ordered("standard")]
    assert result["tiers_run"] == ["fast", "standard"]
    assert result["fast_fail"]["stopped"] is False
    assert set(result) == RESULT_KEYS


def test_hard_fail_stops_the_battery(scripted: Any) -> None:
    scripted.script["heldout_loss"] = "fail"
    result = runner.run_battery("cand", "base", tier="full", protocol=PROTOCOL)
    assert scripted == ["heldout_loss"]
    assert result["fast_fail"] == {
        "stopped": True,
        "at": "heldout_loss",
        "reason": "fast-fail: hard failure in heldout_loss",
    }
    assert all(r["status"] == "skipped" for r in result["probes"][1:])
    assert result["verdict"]["action"] == "abandon"


def test_soft_fail_finishes_the_tier_but_does_not_escalate(scripted: Any) -> None:
    scripted.script["calibration"] = "fail"
    result = runner.run_battery("cand", "base", tier="standard", protocol=PROTOCOL)
    assert scripted == [s.id for s in ordered("fast")]
    assert result["tiers_run"] == ["fast"]
    assert "not escalating" in result["fast_fail"]["reason"]
    off = runner.run_battery(
        "cand", "base", tier="standard", protocol=PROTOCOL, fast_fail=False
    )
    assert off["tiers_run"] == ["fast", "standard"]


def test_probe_exception_is_an_error_row_not_a_crash(
    scripted: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(spec, *args):
        if spec.id == "calibration":
            raise RuntimeError("kaput")
        return runner._row(spec, status="pass", value=1.0, baseline_value=1.0)

    monkeypatch.setattr(runner, "_run_probe", boom)
    states: list[str] = []
    result = runner.run_battery(
        "cand",
        "base",
        tier="fast",
        protocol=PROTOCOL,
        progress=lambda s: states.append(s["state"]),
    )
    row = next(r for r in result["probes"] if r["id"] == "calibration")
    assert row["status"] == "error" and "kaput" in row["note"]
    assert states[-1] == "done" and "running" in states


def test_seal_detects_tampering() -> None:
    sealed = runner.seal({"a": 1})
    assert runner.verify(sealed)
    assert not runner.verify({**sealed, "a": 2})


# --- Rendering ---------------------------------------------------------------


def _fake_result(**verdict: Any) -> dict[str, Any]:
    loss = runner._row(BY_ID["heldout_loss"])
    loss.update(
        judge(BY_ID["heldout_loss"], 3.9, 4.0, se=0.01),
        details={"perplexity": math.exp(3.9)},
    )
    calibration = runner._row(BY_ID["calibration"])
    calibration.update(judge(BY_ID["calibration"], 0.2, 0.1))
    skipped = runner._row(BY_ID["needle"], note="skipped: fast-fail")
    who = {key: None for key in IDENTITY_KEYS} | {
        "run_id": "cand",
        "parameters": 43_200,
        "tokens_seen": 1536,
    }
    return {
        "format": runner.RESULT_FORMAT,
        "suite": suite_identity(),
        "tier": "standard",
        "tiers_run": ["fast"],
        "fast_fail": {"stopped": False, "at": None, "reason": None},
        "target": who,
        "baseline": who | {"run_id": "base"},
        "probes": [loss, calibration, skipped],
        "guard": overfit_guard({}, {}),
        "verdict": {
            "status": "fail",
            "action": "tweak",
            "next_tier": None,
            "reasons": [],
            "suggestion": "Try again.",
            **verdict,
        },
        "seconds": 0.5,
    }


def test_render_is_plain_without_color_and_shows_verdict_and_hints() -> None:
    text = render(_fake_result(), color=False)
    assert "\x1b[" not in text
    assert text.startswith("PROBE BATTERY  sparselab-probe-battery v1")
    assert "✔ PASS" in text and "✖ FAIL" in text and "⊘ SKIPPED" in text
    assert "ppl 49.4" in text
    assert "↳ " + BY_ID["calibration"].suggests in text
    assert "verdict ✖ FAIL  next → TWEAK" in text
    assert "43.2k params" in text
    colored = render(_fake_result(), color=True)
    assert "\x1b[" in colored


def test_meter_and_sparkline() -> None:
    better = {"regression": -0.03, "thresholds": {"fail": 0.03}, "status": "pass"}
    assert meter(better, False) == "◀◀◀◀◀│·····"
    worse = {"regression": 0.018, "thresholds": {"fail": 0.03}, "status": "warn"}
    assert meter(worse, False) == "·····│▶▶▶··"
    assert meter({"regression": None, "thresholds": {}}, False).strip() == ""
    assert sparkline([0.0, 1.0]) == "▁█"


# --- lm-eval adapter (fake model) -------------------------------------------


class _FakeLoaded:
    class config:
        class model:
            max_seq_len = 4


def _uniform(vocab: int, calls: list[list[int]]):
    def log_probs(loaded: Any, ids: list[int]) -> np.ndarray:
        calls.append(list(ids))
        assert len(ids) <= loaded.config.model.max_seq_len
        return np.full((len(ids), vocab), -math.log(vocab))

    return log_probs


def test_rolling_logprob_scores_every_token_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[int]] = []
    monkeypatch.setattr(lm_eval_adapter, "_log_probs", _uniform(10, calls))
    monkeypatch.setattr(lm_eval_adapter, "_eos", lambda loaded: 0)
    monkeypatch.setattr(
        lm_eval_adapter, "_encode", lambda loaded, text: [int(c) for c in text]
    )
    total = lm_eval_adapter.rolling_logprob(_FakeLoaded(), "123456789")
    assert total == pytest.approx(9 * -math.log(10))
    assert calls[0] == [0, 1, 2, 3]  # windows never exceed max_seq_len


def test_continuation_logprob_and_greedy_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    def peaked(loaded: Any, ids: list[int]) -> np.ndarray:
        rows = np.full((len(ids), 10), -20.0)
        for i, token in enumerate(ids):
            rows[i, (token + 1) % 10] = 0.0  # always predicts token + 1
        return rows

    monkeypatch.setattr(lm_eval_adapter, "_log_probs", peaked)
    monkeypatch.setattr(lm_eval_adapter, "_eos", lambda loaded: 0)
    monkeypatch.setattr(
        lm_eval_adapter, "_encode", lambda loaded, text: [int(c) for c in text]
    )
    logprob, greedy = lm_eval_adapter.continuation_logprob(_FakeLoaded(), "12", "34")
    assert logprob == pytest.approx(0.0) and greedy is True
    logprob, greedy = lm_eval_adapter.continuation_logprob(_FakeLoaded(), "12", "35")
    assert logprob == pytest.approx(-20.0) and greedy is False


def test_lm_eval_summary_prefers_acc_then_acc_norm() -> None:
    summary = lm_eval_adapter.summarize_results(
        {
            "results": {
                "piqa": {"acc,none": 0.6, "acc_norm,none": 0.55},
                "hellaswag": {"acc_norm,none": 0.3},
                "lambada_openai": {"acc,none": 0.0, "perplexity,none": 900.0},
            }
        },
        ["piqa", "hellaswag", "lambada_openai", "arc_easy"],
    )
    assert summary["tasks"]["piqa"]["acc"] == 0.6
    assert summary["tasks"]["hellaswag"]["acc"] == 0.3
    assert summary["tasks"]["arc_easy"]["acc"] is None
    assert summary["mean_accuracy"] == pytest.approx(0.3)


# --- End to end: try runs the fast tier; `probe` and `report` CLIs -----------


def _write_inputs(root: Path) -> Path:
    from sparselab.config.loading import load_tokenizer_config
    from sparselab.data.tokenizer import train_tokenizer

    tokenizer = yaml.safe_load((ROOT / "configs/tokenizer_smoke.yaml").read_text())
    tokenizer["output_dir"] = str(root / "tokenizer")
    tokenizer["dataset"]["cache_dir"] = str(root / "data")
    (root / "tokenizer.yaml").write_text(yaml.safe_dump(tokenizer, sort_keys=False))
    config = yaml.safe_load((ROOT / "configs/smoke_cpu.yaml").read_text())
    config["tokenizer"]["path"] = str(root / "tokenizer" / "tokenizer.json")
    config["dataset"]["cache_dir"] = str(root / "data")
    config["logging"]["root_dir"] = str(root / "unused-runs")
    config["training"]["max_steps"] = 8
    config["training"]["max_tokens"] = 1024
    config["checkpoint"]["every_steps"] = 4
    config["evaluation"]["every_steps"] = 4
    baseline = root / "baseline.yaml"
    baseline.write_text(yaml.safe_dump(config, sort_keys=False))
    train_tokenizer(load_tokenizer_config(root / "tokenizer.yaml"))
    return baseline


@pytest.fixture(scope="module")
def tried(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    from sparselab.lab_mode import run_try

    root = tmp_path_factory.mktemp("probe-lab")
    baseline = _write_inputs(root)
    delta = root / "wider.yaml"
    delta.write_text(
        yaml.safe_dump({"question": "wider?", "set": {"model.ffn_dim": 128}})
    )
    record, _ = run_try(delta, baseline, work_dir=root / "work")
    return root, record


def _cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    root: Path,
    *args: str,
) -> str:
    from sparselab.cli.main import main

    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(root / "work"))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys, "argv", ["sparselab", *args])
    capsys.readouterr()
    main()
    return capsys.readouterr().out


def test_try_runs_the_fast_tier_and_records_it(
    tried: tuple[Path, dict[str, Any]],
) -> None:
    root, record = tried
    probe = record["probe"]
    assert probe["format"] == runner.RESULT_FORMAT
    assert probe["tier"] == "fast" and probe["tiers_run"] == ["fast"]
    assert set(probe) == RESULT_KEYS
    assert [r["id"] for r in probe["probes"]] == [s.id for s in ordered("fast")]
    for row in probe["probes"]:
        assert set(row) == ROW_KEYS
        assert row["status"] in {"pass", "warn", "fail", "skipped", "error"}
    assert set(probe["target"]) == IDENTITY_KEYS
    assert probe["target"]["run_id"] == record["arms"]["candidate"]["run_id"]
    assert probe["baseline"]["run_id"] == record["arms"]["baseline"]["run_id"]
    assert set(probe["verdict"]) == VERDICT_KEYS
    loss = probe["probes"][0]
    # The probe's loss mirrors the lab try's native held-out loss exactly.
    assert loss["value"] == pytest.approx(
        record["arms"]["candidate"]["heldout"]["loss"], rel=1e-9
    )
    for key, value in probe["protocol"].items():
        assert record["eval_protocol"][key] == value
    progress = json.loads(
        (root / "work/lab/tries" / record["try_id"] / "probe-progress.json").read_text()
    )
    assert progress["state"] == "done"


def test_probe_cli_json_and_report(
    tried: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, record = tried
    candidate = record["arms"]["candidate"]["run_id"]
    baseline = record["arms"]["baseline"]["run_id"]
    out = _cli(
        monkeypatch,
        capsys,
        root,
        "probe",
        candidate,
        "--vs",
        baseline,
        "--tier",
        "standard",
        "--json",
    )
    result = json.loads(out)
    assert set(result) >= RESULT_KEYS | {"probe_id", "result_sha256", "record"}
    assert result["tiers_run"][0] == "fast"
    ids = [r["id"] for r in result["probes"]]
    assert ids == [s.id for s in ordered("standard")]
    assert Path(result["record"]).is_file()

    text = _cli(monkeypatch, capsys, root, "report", result["probe_id"])
    assert text.startswith("PROBE BATTERY")
    assert "verdict " in text and "next → " in text and "\x1b[" not in text

    summary = _cli(monkeypatch, capsys, root, "report", record["try_id"])
    assert "PROBE BATTERY" in summary and "tier fast" in summary


def test_probe_cli_without_baseline_reports_absolute_values(
    tried: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, record = tried
    run_dir = root / "work/lab/runs" / record["arms"]["candidate"]["run_id"]
    text = _cli(monkeypatch, capsys, root, "probe", str(run_dir))
    assert "next → COMPARE" in text
    assert "--vs BASELINE" in text


def test_dashboard_history_reads_tries_and_probes(
    tried: tuple[Path, dict[str, Any]],
) -> None:
    from sparselab.dashboard.probe_data import (
        history_rows,
        live_batteries,
        load_entries,
        pareto_frontier,
    )

    root, record = tried
    entries = load_entries(root / "work/lab")
    assert any(e.source == "try" and e.key == record["try_id"] for e in entries)
    rows = history_rows(entries)
    assert all(r["verdict"] for r in rows)
    assert live_batteries(root / "work/lab") == []  # finished batteries are not live
    assert pareto_frontier([(1, 3), (2, 2), (3, 3), (0.5, 4)]) == [3, 0, 1]
