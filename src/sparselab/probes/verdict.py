"""Judge probe values against a baseline and turn them into a next action.

Pure functions: no model code, so the policy is easy to test and to read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from sparselab.probes.suite import TIERS, ProbeSpec

STATUSES = ("pass", "warn", "fail", "info", "skipped", "not_comparable", "error")
ACTIONS = ("abandon", "tweak", "escalate", "longer_run", "compare")
NOISE_SIGMAS = 2.0
OVERFIT_GAP = 0.15
NEXT_TIER_CHECKS = {
    "standard": "reworded fact recall and needle retrieval",
    "full": "standard lm-eval tasks; needs `uv sync --extra lmeval`",
}


def judge(
    spec: ProbeSpec,
    value: float | None,
    baseline: float | None,
    *,
    se: float | None = None,
) -> dict[str, Any]:
    """Classify one probe value; positive ``regression`` means worse."""
    out: dict[str, Any] = {
        "value": value,
        "baseline_value": baseline,
        "delta": None,
        "regression": None,
        "delta_se": se,
        "within_noise": None,
        "improved": False,
    }
    if value is None or not math.isfinite(value):
        hard_bad = spec.id == "heldout_loss" and value is not None
        out["status"] = "fail" if hard_bad else "error"
        out["note"] = "non-finite value" if value is not None else "no value"
        return out
    if spec.mode == "value_min":
        if spec.fail is not None and value < spec.fail:
            out["status"] = "fail"
        elif value < spec.warn:
            out["status"] = "warn"
        else:
            out["status"] = "pass"
        return out
    if baseline is None or not math.isfinite(baseline):
        out["status"] = "info"
        out["note"] = "absolute value; pass --vs BASELINE for a verdict"
        if spec.id == "repetition" and value >= 0.95:
            out["status"] = "warn"
            out["note"] = "greedy generations are stuck in loops"
        return out
    delta = value - baseline
    regression = -delta if spec.higher_is_better else delta
    if spec.mode == "delta_rel":
        regression = regression / abs(baseline) if baseline else regression
    noise = se is not None and abs(delta) <= NOISE_SIGMAS * se
    out.update(
        {
            "delta": delta,
            "regression": regression,
            "within_noise": noise if se is not None else None,
            "improved": regression < 0 and not noise,
        }
    )
    if noise:
        out["status"] = "pass"
        out["note"] = f"within noise (|delta| <= {NOISE_SIGMAS:g} SE)"
    elif spec.fail is not None and regression > spec.fail:
        out["status"] = "fail"
    elif regression > spec.warn:
        out["status"] = "warn"
    else:
        out["status"] = "pass"
    return out


def overfit_guard(
    dev: Mapping[str, Mapping[str, float | None]],
    heldout: Mapping[str, Mapping[str, float | None]],
) -> dict[str, Any]:
    """Flag a dev-split gain the held-out split does not share.

    ``dev``/``heldout`` map probe id -> {"value", "baseline_value"} (accuracy);
    a dev row may carry ``n`` (items) so small dev splits do not cry wolf.
    """
    rows: dict[str, Any] = {}
    suspected = False
    for probe, dev_row in dev.items():
        held = heldout.get(probe) or {}
        values = (
            dev_row.get("value"),
            dev_row.get("baseline_value"),
            held.get("value"),
            held.get("baseline_value"),
        )
        if any(v is None for v in values):
            continue
        dev_gain = float(values[0]) - float(values[1])
        held_gain = float(values[2]) - float(values[3])
        n = dev_row.get("n")
        noise = (
            NOISE_SIGMAS
            * math.sqrt(
                (
                    float(values[0]) * (1 - float(values[0]))
                    + float(values[1]) * (1 - float(values[1]))
                )
                / float(n)
            )
            if n
            else 0.0
        )
        margin = max(OVERFIT_GAP, noise)
        flag = dev_gain - held_gain > margin and dev_gain > noise
        suspected = suspected or flag
        rows[probe] = {
            "dev_gain": dev_gain,
            "heldout_gain": held_gain,
            "dev_noise": noise,
            "flag": flag,
        }
    return {
        "verdict_split": "heldout",
        "overfit_suspected": suspected,
        "probes": rows,
        "rule": f"flag when the dev gain is beyond noise and exceeds the held-out "
        f"gain by more than max({OVERFIT_GAP}, {NOISE_SIGMAS:g} SE)",
    }


def decide(
    results: Sequence[Mapping[str, Any]],
    *,
    has_baseline: bool,
    tiers_run: Sequence[str],
    requested_tier: str,
    guard: Mapping[str, Any] | None,
    specs: Mapping[str, ProbeSpec],
) -> dict[str, Any]:
    """Overall verdict and a recommended next action for humans and agents."""
    reasons: list[str] = []
    by_id = {row["id"]: row for row in results}
    last = tiers_run[-1] if tiers_run else requested_tier
    next_tier = (
        TIERS[TIERS.index(last) + 1] if last in TIERS and last != TIERS[-1] else None
    )
    if not has_baseline:
        return {
            "status": "info",
            "action": "compare",
            "next_tier": None,
            "reasons": ["no baseline: values are absolute"],
            "suggestion": "Re-run with --vs BASELINE to get pass/warn/fail and "
            "a recommended action.",
        }
    hard = [r for r in results if r["status"] == "fail" and specs[r["id"]].hard]
    fails = [r for r in results if r["status"] == "fail"]
    warns = [r for r in results if r["status"] == "warn"]
    if hard:
        first = hard[0]
        return {
            "status": "fail",
            "action": "abandon",
            "next_tier": None,
            "reasons": [f"hard fail: {r['id']}" for r in hard],
            "suggestion": specs[first["id"]].suggests,
        }
    if fails:
        first = fails[0]
        return {
            "status": "fail",
            "action": "tweak",
            "next_tier": None,
            "reasons": [f"fail: {r['id']}" for r in fails],
            "suggestion": specs[first["id"]].suggests,
        }
    if guard and guard.get("overfit_suspected"):
        return {
            "status": "warn",
            "action": "tweak",
            "next_tier": None,
            "reasons": ["dev split improved but held-out did not"],
            "suggestion": "You may be fitting the probe items: change the idea, "
            "not the prompts. Verdicts only use the held-out split.",
        }
    loss = by_id.get("heldout_loss")
    status = "warn" if warns else "pass"
    reasons.extend(f"warn: {r['id']}" for r in warns)
    if loss is None or loss["status"] in {"not_comparable", "error", "skipped"}:
        return {
            "status": "incomplete",
            "action": "tweak",
            "next_tier": None,
            "reasons": reasons + ["held-out loss is not comparable"],
            "suggestion": "Make the arms share validation data and tokenizer "
            "(or fix the error) before trusting any verdict.",
        }
    if loss.get("improved"):
        reasons.append("held-out loss improved beyond noise")
        if next_tier is not None:
            return {
                "status": status,
                "action": "escalate",
                "next_tier": next_tier,
                "reasons": reasons,
                "suggestion": f"Promising: run `--tier {next_tier}` ("
                + NEXT_TIER_CHECKS.get(next_tier, "the next tier")
                + ") before a longer run.",
            }
        return {
            "status": status,
            "action": "longer_run",
            "next_tier": None,
            "reasons": reasons,
            "suggestion": "Every tier holds up: schedule a longer run (more "
            "tokens or seeds) to confirm the gain.",
        }
    reasons.append("no held-out loss gain beyond noise")
    return {
        "status": status,
        "action": "tweak",
        "next_tier": None,
        "reasons": reasons,
        "suggestion": "No measurable gain at this budget: try a bolder change, "
        "or a longer run only if the idea needs more tokens to show.",
    }
