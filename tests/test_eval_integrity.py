"""Evaluation integrity: the held-back final split, NOT_COMPARABLE precedence,
eval-noise labelling and the recall-pick investigation (PR3)."""

from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import Any

import pytest

from sparselab.probes import compare as compare_mod
from sparselab.probes import runner, scoring
from sparselab.probes import suite as suite_mod
from sparselab.probes.suite import (
    BY_ID,
    FinalSplitLocked,
    fact_items,
    final_verdict_access,
    needle_split,
    parametric_items,
    prompts,
)
from sparselab.probes.verdict import (
    SEED_CAVEAT,
    decide,
    respect_try_comparison,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "sparselab"


# --- Held-back final split -----------------------------------------------------


@pytest.mark.parametrize(
    "read",
    [
        lambda: fact_items("final"),
        lambda: parametric_items("final", seed=0),
        lambda: needle_split("final"),
        lambda: prompts("final"),
        lambda: suite_mod.split_payload("final"),
        lambda: runner.recall_items("final"),
        lambda: runner.needle_items("final", [192]),
    ],
    ids=[
        "facts",
        "parametric",
        "needle",
        "prompts",
        "payload",
        "recall",
        "needle-items",
    ],
)
def test_final_split_is_locked_outside_a_final_verdict(read: Any) -> None:
    with pytest.raises(FinalSplitLocked):
        read()
    with final_verdict_access():
        assert read()
    with pytest.raises(FinalSplitLocked):
        read()  # closed again


def test_final_split_is_deterministic_and_disjoint_from_selection_splits() -> None:
    with final_verdict_access():
        final = {
            "facts": fact_items("final"),
            "needles": needle_split("final"),
            "prompts": prompts("final"),
        }
        assert final["facts"] == fact_items("final")
    for split in suite_mod.SELECTION_SPLITS:
        facts = fact_items(split)
        assert {i["answer"] for i in final["facts"]}.isdisjoint(
            i["answer"] for i in facts
        )
        assert {i["question"] for i in final["facts"]}.isdisjoint(
            i["question"] for i in facts
        )
        assert set(final["needles"]["needles"]).isdisjoint(
            needle_split(split)["needles"]
        )
        assert set(final["needles"]["filler"]).isdisjoint(needle_split(split)["filler"])
        assert set(final["prompts"]).isdisjoint(prompts(split))
    for item in final["facts"]:
        assert item["answer"] in item["candidates"]


def test_selection_paths_never_open_the_final_split() -> None:
    """Only the probe CLI's --final path may open the held-back split.

    `final_verdict_access` is entered only in suite.py (digest) and the
    run_battery wrapper; `final=` is passed only by the probe CLI. try/lab
    mode, iteration and the dashboard never reference either.
    """
    openers, passers = set(), set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = str(path.relative_to(SRC))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "final_verdict_access":
                openers.add(rel)
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "final":
                        passers.add(rel)
    assert openers == {"probes/suite.py", "probes/runner.py"}
    assert passers == {"probes/cli.py", "cli/main.py"}


def test_try_probe_battery_is_a_selection_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sparselab import lab_mode

    seen: dict[str, Any] = {}

    def fake_battery(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        assert suite_mod.verdict_split() == "heldout"
        return {"status": "ok"}

    monkeypatch.setattr(runner, "run_battery", fake_battery)
    record = {
        "arms": {
            arm: {"run_id": f"run-{arm}", "checkpoint_sha256": "x"}
            for arm in ("baseline", "candidate")
        },
        "comparison": {"verdict": "CANDIDATE_LOWER_LOSS"},
    }
    from sparselab.lab_context import LabContext

    lab_mode._probe_arms(
        record,
        tmp_path,
        {},
        "standard",
        tmp_path,
        None,
        LabContext(tmp_path / "CANCEL"),
        {},
    )
    assert "final" not in seen
    assert seen["comparison"] == record["comparison"]


def test_final_verdict_scores_the_held_back_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scoring, "continuation_score", lambda loaded, prefix, cont: (0.0, 1, False)
    )
    loaded = None  # scoring is stubbed; fact recall needs no model here
    spec = BY_ID["fact_recall"]
    arm = runner.Arm(load=lambda: loaded)
    heldout = runner._measure(spec, loaded, arm, {})
    with final_verdict_access():
        final = runner._measure(spec, loaded, arm, {})
    assert heldout["item_group"] != final["item_group"]
    with final_verdict_access():
        expected = {i["answer"] for i in fact_items("final")}
    assert {i["answer"] for i in final["items"]} == expected
    with pytest.raises(ValueError, match="standalone"):
        runner.run_battery(arm, final=True, comparison={"verdict": "TIE"})


# --- NOT_COMPARABLE takes precedence -------------------------------------------


def _loss_row(**extra: Any) -> dict[str, Any]:
    return {"id": "heldout_loss", "status": "pass", "improved": True, **extra}


NOT_COMPARABLE = {"verdict": "NOT_COMPARABLE", "failed": ["same_training_data"]}


def test_not_comparable_try_never_escalates() -> None:
    kwargs = {
        "has_baseline": True,
        "tiers_run": ["fast"],
        "requested_tier": "fast",
        "guard": None,
        "specs": BY_ID,
    }
    promising = decide([_loss_row()], **kwargs)
    assert promising["action"] == "escalate"
    blocked = decide([_loss_row()], comparison=NOT_COMPARABLE, **kwargs)
    assert blocked["status"] == "incomplete" and blocked["action"] == "rerun"
    assert "Promising" not in blocked["suggestion"]
    assert "same_training_data" in blocked["reasons"][0]
    # A comparable try is unaffected.
    fine = decide(
        [_loss_row()], comparison={"verdict": "CANDIDATE_LOWER_LOSS"}, **kwargs
    )
    assert fine == promising
    # Older records: the shown verdict follows the try comparison too.
    assert respect_try_comparison(promising, NOT_COMPARABLE)["action"] == "rerun"
    assert respect_try_comparison(promising, None) is promising


def test_dashboard_activity_shows_not_comparable_tries_as_not_escalating(
    tmp_path: Path,
) -> None:
    from sparselab.dashboard import lab_data

    record = {
        "try_id": "t1",
        "created_at": "2026-10-10T00:00:00Z",
        "comparison": NOT_COMPARABLE,
        "probe": {"verdict": {"status": "pass", "action": "escalate"}},
    }
    snap = lab_data.LabSnapshot(
        lab_dir=tmp_path,
        entries=[],
        tries=[(tmp_path / "try.json", record)],
        probes=[],
        rejected=[],
        points=[],
    )
    (row,) = lab_data.activity(snap)
    assert row["action"] == "rerun" and row["verdict"] == "incomplete"


# --- Error bars: within-run eval noise, paired seeds ---------------------------


def test_single_seed_win_recommends_paired_seeds() -> None:
    out = decide(
        [_loss_row()],
        has_baseline=True,
        tiers_run=["fast", "standard", "full"],
        requested_tier="full",
        guard=None,
        specs=BY_ID,
    )
    assert out["action"] == "longer_run"
    assert out["suggestion"].endswith(SEED_CAVEAT)
    assert "within-run eval noise" in " ".join(out["reasons"])
    assert "paired seeds" in SEED_CAVEAT
    assert "within-run eval noise" in BY_ID["heldout_loss"].explains


def test_rendered_loss_error_bar_is_labelled_eval_noise() -> None:
    from sparselab.probes.render import _delta

    row = {"id": "heldout_loss", "delta": -0.01, "delta_se": 0.002, "thresholds": {}}
    assert "(eval noise, 1 seed)" in _delta(row)
    assert "eval noise" not in _delta({**row, "id": "fact_recall"})


# --- Recall picks: prior, not candidate order ----------------------------------


def _scored_by(monkeypatch: pytest.MonkeyPatch, score: Any) -> None:
    monkeypatch.setattr(
        scoring,
        "continuation_score",
        lambda loaded, prefix, cont: (score(prefix, cont.strip()), 1, False),
    )


def _shuffled(items: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for item in items:
        candidates = list(item["candidates"])
        rng.shuffle(candidates)
        out.append({**item, "candidates": candidates})
    return out


def test_ties_are_shown_as_ties_and_earn_chance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flat model is never reported as picking the first candidate."""
    _scored_by(monkeypatch, lambda prefix, candidate: -1.0)
    items = runner.recall_items("heldout")
    for order in (items, _shuffled(items, 1)):
        scores = scoring.ranking_scores(None, order)
        picks = [scoring.top_candidate(s) for s in scores]
        assert all(p.startswith("tie: ") for p in picks)
        assert scoring.ranking_credit(None, order) == pytest.approx(
            [1 / len(items[0]["candidates"])] * len(items)
        )


def test_context_free_prior_gives_constant_picks_at_chance_whatever_the_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PR #63 demo finding, reproduced: an untrained model's near-uniform
    output still has tiny prior preferences, so it picks the same candidate
    for every question in a candidate set (not the first one). Accuracy is
    then exactly chance, and candidate order changes nothing."""
    prior = {"gold": -6.90, "falcon": -6.11}  # e.g. the demo baseline's argmax
    _scored_by(monkeypatch, lambda prefix, candidate: prior.get(candidate, -7.0))
    items = runner.recall_items("heldout")
    base = [scoring.top_candidate(s) for s in scoring.ranking_scores(None, items)]
    assert set(base) == {"gold", "falcon"}
    assert scoring.constant_pick(items, base)
    for seed in range(3):
        shuffled = _shuffled(items, seed)
        picks = [
            scoring.top_candidate(s) for s in scoring.ranking_scores(None, shuffled)
        ]
        assert picks == base
    credits = scoring.ranking_credit(None, items)
    assert sum(credits) / len(credits) == pytest.approx(1 / 4)


def test_a_model_that_reads_the_context_is_not_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scored_by(
        monkeypatch, lambda prefix, candidate: 0.0 if candidate in prefix else -1
    )
    items = runner.recall_items("heldout")
    picks = [scoring.top_candidate(s) for s in scoring.ranking_scores(None, items)]
    assert picks == [i["answer"] for i in items]
    assert not scoring.constant_pick(items, picks)


# --- One reference resolver ----------------------------------------------------


def test_resolve_point_is_the_only_reference_resolver(tmp_path: Path) -> None:
    assert not hasattr(compare_mod, "_reference_point")
    with pytest.raises(ValueError, match="no result for ref:SmolLM2-135M"):
        compare_mod.resolve_point("ref:SmolLM2-135M", tmp_path, [])
