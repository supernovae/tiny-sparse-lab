"""Judge probe values against a baseline and turn them into a next action.

Pure functions: no model code, so the policy is easy to test and to read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from sparselab.probes.suite import TIERS, ProbeSpec

STATUSES = (
    "pass",
    "warn",
    "fail",
    "info",
    "skipped",
    "unavailable",
    "not_comparable",
    "error",
)
# A row in one of these states is a check that should have run and did not.
MISSING_STATUSES = frozenset({"unavailable", "not_comparable", "error"})
# Battery stops that leave requested checks unrun (fast-fail stops do not).
INCOMPLETE_STOPS = frozenset({"cancelled", "resources", "oom", "interrupted"})
ACTIONS = ("abandon", "tweak", "rerun", "escalate", "longer_run", "compare")
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
    a dev row may carry ``se`` (paired SE of the dev difference) so small dev
    splits do not cry wolf.
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
        noise = NOISE_SIGMAS * float(dev_row.get("se") or 0.0)
        flag = dev_gain - held_gain > max(OVERFIT_GAP, noise) and dev_gain > noise
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


def missing_evidence(
    results: Sequence[Mapping[str, Any]], stop: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Checks that should have produced evidence and did not, with why."""
    missing = [
        {"id": r["id"], "status": r["status"], "note": r.get("note")}
        for r in results
        if r["status"] in MISSING_STATUSES
    ]
    if stop and stop.get("kind") in INCOMPLETE_STOPS:
        unrun = [r["id"] for r in results if r["status"] == "skipped"]
        if unrun:
            missing.append(
                {
                    "id": ",".join(unrun),
                    "status": f"stopped:{stop['kind']}",
                    "note": stop.get("reason"),
                }
            )
    return missing


NUMERICAL_SUGGESTION = (
    "Numerical failure: the candidate's loss is NaN/inf. Lower the learning "
    "rate or lengthen warmup, check precision and initialization, and look for "
    "a division or log of zero before trying again."
)


def numerical_failure(row: Mapping[str, Any]) -> str | None:
    """The numerical-failure message of a probe row, if it has one."""
    return (row.get("details") or {}).get("numerical_failure")


def _verdict(
    status: str,
    action: str,
    reasons: list[str],
    suggestion: str,
    missing: list[dict[str, Any]],
    next_tier: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "action": action,
        "next_tier": next_tier,
        "reasons": reasons,
        "missing": missing,
        "suggestion": suggestion,
    }


def decide(
    results: Sequence[Mapping[str, Any]],
    *,
    has_baseline: bool,
    tiers_run: Sequence[str],
    requested_tier: str,
    guard: Mapping[str, Any] | None,
    specs: Mapping[str, ProbeSpec],
    stop: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Overall verdict and a recommended next action for humans and agents.

    Missing evidence never reads as success: an unavailable optional tier, a
    probe error, a non-comparable probe or a battery stopped by cancellation,
    resources or OOM makes the battery ``incomplete`` (action ``rerun``) unless
    a failure already decides it.
    """
    missing = missing_evidence(results, stop)
    by_id = {row["id"]: row for row in results}
    numerical = [r for r in results if numerical_failure(r)]
    if numerical:
        # A NaN/inf loss is a decisive outcome with or without a baseline.
        return _verdict(
            "fail",
            "abandon",
            [
                f"hard fail: {r['id']} (numerical failure: {numerical_failure(r)})"
                for r in numerical
            ],
            NUMERICAL_SUGGESTION,
            missing,
        )
    last = tiers_run[-1] if tiers_run else requested_tier
    next_tier = (
        TIERS[TIERS.index(last) + 1] if last in TIERS and last != TIERS[-1] else None
    )
    if not has_baseline:
        return _verdict(
            "info",
            "compare",
            ["no baseline: values are absolute"],
            "Re-run with --vs BASELINE to get pass/warn/fail and a recommended action.",
            missing,
        )
    hard = [r for r in results if r["status"] == "fail" and specs[r["id"]].hard]
    fails = [r for r in results if r["status"] == "fail"]
    warns = [r for r in results if r["status"] == "warn"]
    if hard:
        return _verdict(
            "fail",
            "abandon",
            [f"hard fail: {r['id']}" for r in hard],
            specs[hard[0]["id"]].suggests,
            missing,
        )
    if fails:
        return _verdict(
            "fail",
            "tweak",
            [f"fail: {r['id']}" for r in fails],
            specs[fails[0]["id"]].suggests,
            missing,
        )
    if missing:
        named = "; ".join(f"{m['id']} ({m['status']}: {m['note']})" for m in missing)
        return _verdict(
            "incomplete",
            "rerun",
            [f"missing evidence: {m['id']} ({m['status']})" for m in missing],
            f"Not enough evidence for a verdict. Fix and re-run: {named}.",
            missing,
        )
    if guard and guard.get("overfit_suspected"):
        return _verdict(
            "warn",
            "tweak",
            ["dev split improved but held-out did not"],
            "You may be fitting the probe items: change the idea, not the "
            "prompts. Verdicts only use the held-out split.",
            missing,
        )
    status = "warn" if warns else "pass"
    reasons = [f"warn: {r['id']}" for r in warns]
    loss = by_id.get("heldout_loss")
    if loss is not None and loss.get("improved"):
        reasons.append("held-out loss improved beyond noise")
        if next_tier is not None:
            return _verdict(
                status,
                "escalate",
                reasons,
                f"Promising: run `--tier {next_tier}` ("
                + NEXT_TIER_CHECKS.get(next_tier, "the next tier")
                + ") before a longer run.",
                missing,
                next_tier,
            )
        return _verdict(
            status,
            "longer_run",
            reasons,
            "Every tier holds up: schedule a longer run (more tokens or seeds) "
            "to confirm the gain. Probes screen; they do not prove usefulness.",
            missing,
        )
    reasons.append("no held-out loss gain beyond noise")
    return _verdict(
        status,
        "tweak",
        reasons,
        "No measurable gain at this budget: try a bolder change, or a longer "
        "run only if the idea needs more tokens to show.",
        missing,
    )
