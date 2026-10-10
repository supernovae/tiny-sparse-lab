"""Run the probe battery, cheapest probes first, one loaded arm at a time.

Each arm (baseline, candidate) is loaded, measured for the current tier and
released before the other is loaded, so a battery never holds two models in
memory. Measurements are cached per arm (validation is scored once and shared
by held-out loss and calibration; inside ``sparselab try`` it is the try's own
scoring pass). A :class:`~sparselab.lab_context.LabContext` is checked before
every probe: a cancel sentinel or a violated resource envelope stops the
battery at that safe point, and an out-of-memory error stops further probe work
instead of becoming an ordinary error row. A stopped battery keeps everything
measured so far and is ``incomplete``.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sparselab.lab_context import (
    LabCancelled,
    LabContext,
    LabSignal,
    is_out_of_memory,
    release_memory,
)
from sparselab.lab_records import PROBE_FORMAT, canonical, write_json_atomic
from sparselab.probes import metrics, scoring
from sparselab.probes.suite import (
    BY_ID,
    ProbeSpec,
    fact_items,
    needle_split,
    ordered,
    prompts,
    suite_identity,
    tiers_through,
)
from sparselab.probes.verdict import (
    INCOMPLETE_STOPS,
    MISSING_STATUSES,
    decide,
    judge,
    numerical_failure,
    overfit_guard,
)

RESULT_FORMAT = PROBE_FORMAT
NEAR_IDENTICAL_JS = 0.01
Progress = Callable[[dict[str, Any]], None]


class ProbeUnsupported(ValueError):
    """The checkpoint cannot run this probe (e.g. a non-PyTorch engine)."""


class _NotComparable(Exception):
    pass


@dataclass
class Arm:
    """One side of a comparison: how to load it and what is already measured."""

    load: Callable[[], Any]
    cache: dict[str, Any] = field(default_factory=dict)


def as_arm(value: Any) -> Arm | None:
    if value is None or isinstance(value, Arm):
        return value
    return Arm(load=lambda: value)


def _require_torch_engine(loaded: Any) -> None:
    if loaded.engine is not None:
        raise ProbeUnsupported("probes need the PyTorch engine (MLX not supported)")
    if getattr(loaded.model, "semantic_memories", None):
        raise ProbeUnsupported("probes do not support attached semantic packs")


def _identity(loaded: Any) -> dict[str, Any]:
    identity = loaded.identity
    parameters = list(loaded.model.parameters())
    return {
        "run_id": identity.get("run_id"),
        "run_dir": str(loaded.run),
        "checkpoint": identity.get("checkpoint_relative_path"),
        "checkpoint_sha256": identity.get("checkpoint_sha256"),
        "step": identity.get("step"),
        "tokens_seen": identity.get("tokens_seen"),
        "parameters": sum(p.numel() for p in parameters),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
        "tokenizer_sha256": identity.get("tokenizer_sha256"),
        "validation_sha256": (identity.get("data_sha256") or {}).get("validation"),
        "max_seq_len": loaded.config.model.max_seq_len,
    }


def _validation_identity(loaded: Any) -> dict[str, Any]:
    from sparselab.lab_mode import _data_identity, _supervision_identity

    data = _data_identity(loaded.run)
    return {
        "validation_sha256": data["validation_sha256"],
        "tokenizer_sha256": data["tokenizer_sha256"],
        **_supervision_identity(loaded.run),
    }


def _tokenizer_digest(loaded: Any) -> str:
    return hashlib.sha256(loaded.tokenizer.to_str().encode()).hexdigest()


def _describe(loaded: Any, arm: Arm) -> None:
    """Identity facts recorded the first time an arm is opened."""
    if "identity" not in arm.cache:
        arm.cache["identity"] = _identity(loaded)
        arm.cache["validation_identity"] = _validation_identity(loaded)
        arm.cache["tokenizer_digest"] = _tokenizer_digest(loaded)
        arm.cache["config"] = loaded.config


# --- Per-arm measurements ----------------------------------------------------


def validation(loaded: Any, arm: Arm, protocol: Mapping[str, int]) -> dict[str, Any]:
    """Native held-out scoring with the probe observer, computed once per arm."""
    cached = arm.cache.get("validation")
    if cached is not None and cached.get("protocol") == dict(protocol):
        return cached
    if protocol["seq_len"] > loaded.config.model.max_seq_len:
        raise _NotComparable("baseline eval window exceeds this model's max_seq_len")
    from sparselab.lab_mode import _with_eval_protocol

    stats = scoring.ValidationStats()
    try:
        native = _with_eval_protocol(loaded, protocol).evaluate(observer=stats)
    except FloatingPointError as error:
        # NaN/inf loss from the native evaluator: a numerical failure, kept
        # as an outcome (not a probe error) so the battery can hard-fail on it.
        arm.cache["validation"] = {
            "numerical_failure": str(error),
            "protocol": dict(protocol),
        }
        return arm.cache["validation"]
    arm.cache["validation"] = {**stats.result(native), "protocol": dict(protocol)}
    return arm.cache["validation"]


def generations(loaded: Any, max_new_tokens: int) -> list[dict[str, Any]]:
    from sparselab.evaluation.generation import generate_with_token_ids

    budget = max(1, min(max_new_tokens, loaded.config.model.max_seq_len // 2))
    rows = []
    for prompt in prompts("heldout"):
        text, ids = generate_with_token_ids(
            loaded.model,
            loaded.tokenizer,
            prompt,
            loaded.config.model.max_seq_len,
            budget,
            loaded.device,
            temperature=0.0,
        )
        rows.append(
            {"prompt": prompt, "continuation": text.removeprefix(prompt), "ids": ids}
        )
    return rows


def recall_items(split: str) -> list[dict[str, Any]]:
    return [
        {
            "prefix": f"{item['context']} {item['question']}",
            "answer": item["answer"],
            "candidates": item["candidates"],
        }
        for item in fact_items(split)
    ]


def needle_items(loaded: Any, split: str, fractions: list[float]) -> list[dict]:
    spec = needle_split(split)
    max_len = loaded.config.model.max_seq_len
    items = []
    for fraction in fractions:
        target = max(8, int(fraction * max_len))
        for offset, word in enumerate(spec["needles"]):
            filler = spec["filler"]
            # The needle opens the context; filler pushes it further back.
            parts = [spec["statement"].format(w=word)]
            k = offset
            while True:
                candidate = parts + [filler[k % len(filler)]]
                text = " ".join([*candidate, spec["question"], word])
                if len(scoring.encode(loaded, text)) > target:
                    break
                parts = candidate
                k += 1
            prefix = " ".join([*parts, spec["question"]])
            items.append(
                {
                    "prefix": prefix,
                    "answer": word,
                    "candidates": spec["needles"],
                    "fraction": fraction,
                    "tokens": len(scoring.encode(loaded, prefix)),
                }
            )
    return items


def _measure(
    spec: ProbeSpec, loaded: Any, arm: Arm, protocol: Mapping[str, int]
) -> dict[str, Any]:
    """Everything one arm contributes to SPEC; judged later against the other."""
    if spec.id in {"heldout_loss", "calibration"}:
        val = validation(loaded, arm, protocol)
        if "numerical_failure" in val:
            if spec.id == "heldout_loss":
                return {
                    "value": math.nan,
                    "numerical_failure": val["numerical_failure"],
                }
            return {
                "status": "error",
                "note": f"numerical failure: {val['numerical_failure']}",
            }
        if spec.id == "heldout_loss":
            return {
                "value": val["loss"],
                "window_sums": val["window_sums"],
                "window_counts": val["window_counts"],
                "valid_targets": val["valid_targets"],
                "windows": val["windows"],
                "top1_accuracy": val["accuracy"],
                "ms_per_token": val["ms_per_token"],
            }
        bins = int(spec.params.get("bins", 15))
        return {
            "value": metrics.expected_calibration_error(
                val["confidences"], val["correct"], bins
            ),
            "positions": int(val["correct"].size),
            "bins": bins,
        }
    if spec.id == "token_agreement":
        rows = []
        for prompt in prompts("heldout"):
            ids = scoring.encode(loaded, prompt)[-loaded.config.model.max_seq_len :]
            rows.append(scoring.log_probs(loaded, ids).astype(np.float32))
        return {"log_probs": rows}
    if spec.id == "repetition":
        rows = generations(loaded, int(spec.params.get("max_new_tokens", 32)))
        ids = [r["ids"] for r in rows]
        return {
            "value": metrics.mean_seq_rep_n(ids, 4),
            "seq_rep_4": metrics.mean_seq_rep_n(ids, 4),
            "distinct_1": metrics.distinct_n(ids, 1),
            "distinct_2": metrics.distinct_n(ids, 2),
            "samples": [
                {"prompt": r["prompt"], "continuation": r["continuation"]}
                for r in rows[:2]
            ],
        }
    if spec.id in {"fact_recall", "needle"}:

        def items(split: str) -> list[dict[str, Any]]:
            if spec.id == "fact_recall":
                return recall_items(split)
            return needle_items(loaded, split, list(spec.params["length_fractions"]))

        held = items("heldout")
        credits = scoring.ranking_credit(loaded, held)
        dev = scoring.ranking_credit(loaded, items("dev"))
        return {
            "value": float(np.mean(credits)) if credits else None,
            "credits": credits,
            "dev_credits": dev,
            "fractions": [i.get("fraction") for i in held],
            "tokens": [i.get("tokens") for i in held],
            "chance": 1 / len(held[0]["candidates"]) if held else None,
        }
    if spec.id == "lm_eval":
        from sparselab.probes.lm_eval_adapter import LmEvalUnavailable, run_lm_eval

        try:
            out = run_lm_eval(loaded, spec.params["tasks"], spec.params["limit"])
        except LmEvalUnavailable as error:
            raise ProbeUnsupported(str(error)) from None
        return {"value": out["mean_accuracy"], **out}
    raise ValueError(f"unknown probe {spec.id}")


def _measure_safely(
    spec: ProbeSpec, loaded: Any, arm: Arm, protocol: Mapping[str, int]
) -> dict[str, Any]:
    """Probe failures become evidence gaps; OOM and stops propagate."""
    key = f"probe:{spec.id}"
    if key in arm.cache:
        return arm.cache[key]
    try:
        measured = _measure(spec, loaded, arm, protocol)
    except _NotComparable as error:
        measured = {"status": "not_comparable", "note": str(error)}
    except ProbeUnsupported as error:
        measured = {"status": "unavailable", "note": str(error)}
    except Exception as error:
        if is_out_of_memory(error):
            raise
        measured = {"status": "error", "note": f"{type(error).__name__}: {error}"}
    arm.cache[key] = measured
    return measured


# --- Judging ------------------------------------------------------------------


def _row(spec: ProbeSpec, **extra: Any) -> dict[str, Any]:
    return {
        "id": spec.id,
        "title": spec.title,
        "tier": spec.tier,
        "cost": spec.cost,
        "metric": spec.metric,
        "higher_is_better": spec.higher_is_better,
        "hard": spec.hard,
        "thresholds": {"mode": spec.mode, "warn": spec.warn, "fail": spec.fail},
        "uncertainty": spec.uncertainty,
        "suggests": spec.suggests,
        "explains": spec.explains,
        "reference": spec.reference,
        "status": "skipped",
        "value": None,
        "baseline_value": None,
        "delta": None,
        "regression": None,
        "delta_se": None,
        "within_noise": None,
        "improved": False,
        "note": None,
        "details": {},
        "seconds": None,
        **extra,
    }


def _paired(t: list[float], b: list[float]) -> float | None:
    if len(t) != len(b) or len(t) < 2:
        return None
    return metrics.paired_mean_and_se(np.asarray(t) - np.asarray(b))[1]


def _judge(
    spec: ProbeSpec,
    t: Mapping[str, Any],
    b: Mapping[str, Any] | None,
    comparable: Mapping[str, Any],
    dev: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row = _row(spec)
    for name, side in (("candidate", t), ("baseline", b)):
        if side is not None and "status" in side:
            note = f"{name}: {side['note']}"
            return {**row, "status": side["status"], "note": note}
    if (
        b is not None
        and spec.id in {"heldout_loss", "calibration"}
        and not comparable["validation"]
    ):
        return {
            **row,
            "status": "not_comparable",
            "note": comparable["validation_reason"],
        }
    if (
        b is not None
        and spec.id in {"token_agreement", "fact_recall", "needle"}
        and not comparable["tokenizer"]
    ):
        return {**row, "status": "not_comparable", "note": "tokenizers differ"}
    if spec.id == "heldout_loss" and (b or {}).get("numerical_failure"):
        # The reference is broken: nothing can be judged against it.
        note = f"baseline: numerical failure: {b['numerical_failure']}"
        return {**row, "status": "error", "note": note}
    if spec.id == "heldout_loss" and t.get("numerical_failure"):
        row.update(judge(spec, math.nan, b["value"] if b else None))
        row.update(
            status="fail",
            note=f"numerical failure: {t['numerical_failure']}",
            details={"numerical_failure": t["numerical_failure"]},
        )
        return row
    if spec.id == "heldout_loss":
        se = (
            metrics.ratio_difference_se(
                t["window_sums"],
                t["window_counts"],
                b["window_sums"],
                b["window_counts"],
            )
            if b is not None
            else None
        )
        row.update(judge(spec, t["value"], b["value"] if b else None, se=se))
        row["details"] = {
            "perplexity": math.exp(t["value"]) if t["value"] < 700 else None,
            "baseline_perplexity": math.exp(b["value"])
            if b and b["value"] < 700
            else None,
            "valid_targets": t["valid_targets"],
            "windows": t["windows"],
            "top1_accuracy": t["top1_accuracy"],
            "baseline_top1_accuracy": b["top1_accuracy"] if b else None,
            "ms_per_token": t["ms_per_token"],
        }
        return row
    if spec.id == "calibration":
        row.update(judge(spec, t["value"], b["value"] if b else None))
        row["details"] = {"bins": t["bins"], "positions": t["positions"]}
        return row
    if spec.id == "token_agreement":
        if b is None:
            return {**row, "status": "info", "note": "needs a baseline"}
        log_p = np.concatenate(b["log_probs"]).astype(np.float64)
        log_q = np.concatenate(t["log_probs"]).astype(np.float64)
        stats = {
            "top1_agreement": metrics.top1_agreement(log_p, log_q),
            "kl_base_to_candidate": float(metrics.kl_divergence(log_p, log_q).mean()),
            "js": float(metrics.js_divergence(log_p, log_q).mean()),
            "positions": int(log_p.shape[0]),
        }
        row.update(judge(spec, stats["top1_agreement"], None))
        if stats["js"] < NEAR_IDENTICAL_JS and row["status"] != "pass":
            # Argmax ties between near-identical (often near-uniform) predictions
            # are arbitrary; low agreement then says nothing.
            row.update(
                status="pass",
                note=f"near-identical distributions (JS < {NEAR_IDENTICAL_JS}); "
                "argmax agreement is not informative",
            )
        row["details"] = {**stats, "informative": stats["js"] >= NEAR_IDENTICAL_JS}
        return row
    if spec.id == "repetition":
        row.update(judge(spec, t["value"], b["value"] if b else None))
        keep = ("seq_rep_4", "distinct_1", "distinct_2")
        row["details"] = {
            **{k: t[k] for k in keep},
            "baseline": {k: b[k] for k in keep} if b else None,
            "samples": t["samples"],
            "baseline_samples": b["samples"] if b else [],
        }
        return row
    if spec.id in {"fact_recall", "needle"}:
        se = _paired(t["credits"], b["credits"]) if b else None
        row.update(judge(spec, t["value"], b["value"] if b else None, se=se))
        details: dict[str, Any] = {
            "n": len(t["credits"]),
            "chance": t["chance"],
            "credit": "fractional_ties",
            "dev_accuracy": float(np.mean(t["dev_credits"]))
            if t["dev_credits"]
            else None,
            "baseline_dev_accuracy": float(np.mean(b["dev_credits"]))
            if b and b["dev_credits"]
            else None,
        }
        if spec.id == "needle":
            by_length: dict[str, dict[str, Any]] = {}
            for fraction in spec.params["length_fractions"]:
                picks = [i for i, f in enumerate(t["fractions"]) if f == fraction]
                by_length[str(fraction)] = {
                    "tokens": max((t["tokens"][i] for i in picks), default=0),
                    "accuracy": float(np.mean([t["credits"][i] for i in picks]))
                    if picks
                    else None,
                    "baseline_accuracy": float(
                        np.mean([b["credits"][i] for i in picks])
                    )
                    if picks and b
                    else None,
                }
            details["by_length"] = by_length
        row["details"] = details
        if b is not None:
            dev[spec.id] = {
                "value": details["dev_accuracy"],
                "baseline_value": details["baseline_dev_accuracy"],
                "se": _paired(t["dev_credits"], b["dev_credits"]),
            }
        return row
    if spec.id == "lm_eval":
        row.update(judge(spec, t["value"], b["value"] if b else None))
        row["details"] = {
            "tasks": t["tasks"],
            "baseline_tasks": b["tasks"] if b else None,
            "limit": spec.params["limit"],
            "chance": spec.params["chance"],
            "lm_eval_version": t["version"],
            "caveat": "tiny models usually score near chance on these tasks; "
            "small differences are noise at this limit",
        }
        return row
    raise ValueError(f"unknown probe {spec.id}")


# --- Battery ------------------------------------------------------------------


def _eval_group(validation_identity: Mapping[str, Any], protocol: Mapping) -> str:
    """Results with the same group share data, tokenizer, mask and protocol."""
    body = {"validation": dict(validation_identity), "protocol": dict(protocol)}
    return hashlib.sha256(canonical(body)).hexdigest()


def run_battery(
    target: Any,
    baseline: Any | None = None,
    *,
    tier: str = "fast",
    fast_fail: bool = True,
    protocol: Mapping[str, int] | None = None,
    progress: Progress | None = None,
    context: LabContext | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Run the battery through TIER and return one result (never raises on probes).

    TARGET/BASELINE are :class:`Arm` loaders or already loaded runs. Arms are
    opened one at a time per tier: the baseline first, then the candidate,
    which is judged probe by probe so a hard failure stops it immediately.
    """
    from sparselab.lab_mode import eval_protocol

    started_at = (now or (lambda: datetime.now(UTC)))()
    started = time.monotonic()
    tiers_through(tier)
    suite = suite_identity()
    specs = ordered(tier)
    t_arm = as_arm(target)
    b_arm = as_arm(baseline)
    assert t_arm is not None
    results: list[dict[str, Any]] = []
    dev: dict[str, dict[str, Any]] = {}
    tiers_run: list[str] = []
    stop: dict[str, Any] = {"stopped": False, "kind": None, "at": None, "reason": None}
    state: dict[str, Any] = {"protocol": dict(protocol) if protocol else None}

    def report(phase: str, arm: str | None, current: str | None) -> None:
        state["current"] = current
        if progress is None:
            return
        progress(
            {
                "state": phase,
                "suite": suite["sha256"],
                "tier": tier,
                "arm": arm,
                "current": current,
                "done": [r["id"] for r in results],
                "statuses": {r["id"]: r["status"] for r in results},
                "total": len(specs),
                "target": (t_arm.cache.get("identity") or {}).get("run_id"),
                "started_at": started_at.isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )

    def checkpoint() -> None:
        if context is not None:
            context.checkpoint()

    def open_arm(
        arm: Arm,
        name: str,
        work: Callable[[Any, list[ProbeSpec]], None],
        tier_specs: list[ProbeSpec],
    ) -> None:
        if context is not None:
            context.enter(name, "probing")
        checkpoint()
        loaded = arm.load()
        try:
            _require_torch_engine(loaded)
            _describe(loaded, arm)
            if state["protocol"] is None:
                state["protocol"] = eval_protocol(loaded.config)
            was_training = loaded.model.training
            loaded.model.eval()
            try:
                work(loaded, tier_specs)
            finally:
                loaded.model.train(was_training)
        finally:
            del loaded
            release_memory()

    def comparable() -> dict[str, Any]:
        out = {"validation": True, "validation_reason": None, "tokenizer": True}
        if b_arm is None or "identity" not in b_arm.cache:
            return out
        t_id = t_arm.cache["validation_identity"]
        b_id = b_arm.cache["validation_identity"]
        if t_id != b_id:
            out["validation"] = False
            out["validation_reason"] = "validation identity differs: " + ", ".join(
                sorted(k for k in t_id if t_id.get(k) != b_id.get(k))
            )
        out["tokenizer"] = (
            t_arm.cache["tokenizer_digest"] == b_arm.cache["tokenizer_digest"]
        )
        return out

    def stop_battery(kind: str, at: str | None, reason: str) -> None:
        stop.update(stopped=True, kind=kind, at=at, reason=reason)

    def measure_baseline(loaded: Any, tier_specs: list[ProbeSpec]) -> None:
        assert b_arm is not None
        for spec in tier_specs:
            checkpoint()
            report("running", "baseline", spec.id)
            _measure_safely(spec, loaded, b_arm, state["protocol"])

    def measure_candidate(loaded: Any, tier_specs: list[ProbeSpec]) -> None:
        for spec in tier_specs:
            checkpoint()
            report("running", "candidate", spec.id)
            tick = time.monotonic()
            measured = _measure_safely(spec, loaded, t_arm, state["protocol"])
            base = b_arm.cache.get(f"probe:{spec.id}") if b_arm is not None else None
            row = _judge(spec, measured, base, comparable(), dev)
            row["seconds"] = round(time.monotonic() - tick, 3)
            results.append(row)
            if fast_fail and row["status"] == "fail" and spec.hard:
                what = "numerical failure" if numerical_failure(row) else "hard failure"
                stop_battery("fast_fail", spec.id, f"fast-fail: {what} in {spec.id}")
                return

    try:
        for tier_name in tiers_through(tier):
            tier_specs = [s for s in specs if s.tier == tier_name]
            if stop["stopped"]:
                break
            if (
                tiers_run
                and fast_fail
                and b_arm is not None
                and any(r["status"] == "fail" for r in results)
            ):
                stop_battery(
                    "not_promising",
                    tier_specs[0].id,
                    f"not promising after the {tiers_run[-1]} tier; not escalating",
                )
                break
            if (
                tiers_run
                and fast_fail
                and any(r["status"] in MISSING_STATUSES for r in results)
            ):
                stop_battery(
                    "missing_evidence",
                    tier_specs[0].id,
                    f"missing evidence in the {tiers_run[-1]} tier; not escalating",
                )
                break
            tiers_run.append(tier_name)
            if b_arm is not None:
                open_arm(b_arm, "baseline", measure_baseline, tier_specs)
            open_arm(t_arm, "candidate", measure_candidate, tier_specs)
    except LabCancelled as error:
        stop_battery(error.kind, state.get("current"), error.reason)
    except (LabSignal, KeyboardInterrupt) as error:
        # Ctrl-C/SIGTERM: keep everything measured so far and finalize the
        # result; the caller re-raises after publishing it.
        release_memory()
        name = error.name if isinstance(error, LabSignal) else "keyboard_interrupt"
        stop_battery("interrupted", state.get("current"), f"interrupted: {name}")
    except Exception as error:
        if not is_out_of_memory(error):
            raise
        release_memory()
        stop_battery(
            "oom",
            state.get("current"),
            f"out of memory: {type(error).__name__}: {error}",
        )
    if context is not None:
        # From here on signals are noted, never raised, so the partial result
        # and its progress state are finalized; callers check context.signals.
        context.finalizing = True
    done = {r["id"] for r in results}
    for spec in specs:
        if spec.id not in done:
            note = f"skipped: {stop['reason']}" if stop["stopped"] else "skipped"
            results.append(_row(spec, note=note))
    order = {spec.id: i for i, spec in enumerate(specs)}
    results.sort(key=lambda r: order[r["id"]])
    guard = overfit_guard(
        dev,
        {
            r["id"]: {"value": r["value"], "baseline_value": r["baseline_value"]}
            for r in results
            if r["id"] in dev
        },
    )
    verdict = decide(
        results,
        has_baseline=b_arm is not None,
        tiers_run=tiers_run,
        requested_tier=tier,
        guard=guard,
        specs=BY_ID,
        stop=stop,
    )
    protocol_used = state["protocol"] or {}

    def identity(arm: Arm | None) -> dict[str, Any] | None:
        if arm is None or "identity" not in arm.cache:
            return None
        return {
            **arm.cache["identity"],
            "eval_group": _eval_group(arm.cache["validation_identity"], protocol_used),
        }

    result = {
        "format": RESULT_FORMAT,
        "created_at": started_at.isoformat(),
        "suite": suite,
        "tier": tier,
        "tiers_run": tiers_run,
        "stop": stop,
        "target": identity(t_arm),
        "baseline": identity(b_arm),
        "comparable": comparable() if "identity" in t_arm.cache else None,
        "protocol": dict(protocol_used),
        "probes": results,
        "guard": guard,
        "verdict": verdict,
        "seconds": round(time.monotonic() - started, 3),
    }
    report(
        "stopped" if stop["kind"] in INCOMPLETE_STOPS else "done",
        None,
        None,
    )
    return result


def progress_writer(path: Path) -> Progress:
    """Live progress for the dashboard: an atomically replaced JSON file."""

    def write(state: dict[str, Any]) -> None:
        try:
            write_json_atomic(path, state)
        except OSError:
            pass

    return write
