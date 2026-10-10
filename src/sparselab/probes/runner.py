"""Run the probe battery on loaded checkpoints, cheapest probes first."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sparselab.probes import metrics
from sparselab.probes.suite import (
    BY_ID,
    TIERS,
    ProbeSpec,
    fact_items,
    needle_split,
    ordered,
    prompts,
    suite_identity,
    tiers_through,
)
from sparselab.probes.verdict import decide, judge, overfit_guard

RESULT_FORMAT = "sparselab-probe-v1"
NEAR_IDENTICAL_JS = 0.01
Progress = Callable[[dict[str, Any]], None]


class ProbeUnsupported(ValueError):
    """The checkpoint cannot run this probe (e.g. a non-PyTorch engine)."""


# --- Model access ------------------------------------------------------------


def _torch():
    import torch

    return torch


def _require_torch_engine(loaded: Any) -> None:
    if loaded.engine is not None:
        raise ProbeUnsupported("probes need the PyTorch engine (MLX not supported)")
    if getattr(loaded.model, "semantic_memories", None):
        raise ProbeUnsupported("probes do not support attached semantic packs")


def _byte_inputs(loaded: Any, ids: list[int]) -> dict[str, Any]:
    config = loaded.config.model
    if config.memory not in {"byte", "portable"}:
        return {}
    from sparselab.evaluation.generation import _addresses_from_ids

    torch = _torch()
    addresses = _addresses_from_ids(
        loaded.tokenizer, ids, config.memory_table_size, config.memory_ngram_size
    )
    return {
        "byte_addresses": torch.tensor(
            [addresses], dtype=torch.long, device=loaded.device
        )
    }


def _log_probs(loaded: Any, ids: list[int]) -> np.ndarray:
    """Log-softmax over the vocabulary at every position of IDS -> [T, V]."""
    torch = _torch()
    ids = ids[-loaded.config.model.max_seq_len :]
    x = torch.tensor([ids], dtype=torch.long, device=loaded.device)
    with torch.inference_mode():
        logits = loaded.model(x, **_byte_inputs(loaded, ids))[0].float()
        return torch.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float64)


def _encode(loaded: Any, text: str) -> list[int]:
    return list(loaded.tokenizer.encode(text, add_special_tokens=False).ids)


def _answer_score(loaded: Any, prefix: str, answer: str) -> float:
    """Mean log-probability of `` answer`` after PREFIX (context kept on the right)."""
    prefix_ids = _encode(loaded, prefix)
    full = _encode(loaded, f"{prefix} {answer}")
    if full[: len(prefix_ids)] != prefix_ids or len(full) <= len(prefix_ids):
        # Tokenization merged across the boundary; score the differing suffix.
        common = 0
        while common < min(len(full), len(prefix_ids)) and (
            full[common] == prefix_ids[common]
        ):
            common += 1
        prefix_ids = full[: max(common, 1)]
    answer_count = len(full) - len(prefix_ids)
    limit = loaded.config.model.max_seq_len + 1
    window = full[-limit:]
    log_probs = _log_probs(loaded, window[:-1])
    targets = window[1:][-answer_count:]
    rows = log_probs[-answer_count:]
    return float(np.mean(rows[np.arange(answer_count), targets]))


class _Session:
    """Per-checkpoint state with eval mode and cached shared computations."""

    def __init__(self, loaded: Any) -> None:
        _require_torch_engine(loaded)
        self.loaded = loaded
        self.cache: dict[str, Any] = {}
        self._was_training = loaded.model.training
        loaded.model.eval()

    def close(self) -> None:
        self.loaded.model.train(self._was_training)

    @property
    def identity(self) -> dict[str, Any]:
        loaded = self.loaded
        identity = loaded.identity
        parameters = sum(p.numel() for p in loaded.model.parameters())
        parameter_bytes = sum(
            p.numel() * p.element_size() for p in loaded.model.parameters()
        )
        return {
            "run_id": identity.get("run_id"),
            "run_dir": str(loaded.run),
            "checkpoint": identity.get("checkpoint_relative_path"),
            "checkpoint_sha256": identity.get("checkpoint_sha256"),
            "step": identity.get("step"),
            "tokens_seen": identity.get("tokens_seen"),
            "parameters": parameters,
            "parameter_bytes": parameter_bytes,
            "tokenizer_sha256": identity.get("tokenizer_sha256"),
            "validation_sha256": (identity.get("data_sha256") or {}).get("validation"),
            "max_seq_len": loaded.config.model.max_seq_len,
        }


# --- Probe computations ------------------------------------------------------


def validation_pass(loaded: Any, protocol: Mapping[str, int]) -> dict[str, Any]:
    """Per-window loss sums plus top-1 confidence/correctness, native semantics.

    Mirrors ``evaluation.language_model.evaluate`` (same windows, masks and
    byte addresses) so the mean equals the native held-out loss.
    """
    from sparselab.lab_mode import _with_eval_protocol

    torch = _torch()
    from sparselab.data.allocation import OWNER_HYBRID, OWNER_LEXICAL

    scored = _with_eval_protocol(loaded, protocol)
    dataset = scored.validation_dataset()
    limit = min(len(dataset), protocol["batch_size"] * protocol["max_batches"])
    sums: list[float] = []
    counts: list[int] = []
    confidences: list[np.ndarray] = []
    correct: list[np.ndarray] = []
    started = time.monotonic()
    tokens = 0
    with torch.inference_mode():
        for index in range(limit):
            inputs, targets, addresses, owners, queries, masks = (
                dataset.numpy_microblock(index)
            )
            if queries is not None or masks is not None:
                raise ProbeUnsupported("semantic-query validation is not supported")
            x = torch.from_numpy(np.asarray(inputs)[None]).to(loaded.device)
            y = torch.from_numpy(np.asarray(targets)[None]).to(loaded.device)
            kwargs: dict[str, Any] = {}
            if addresses is not None:
                kwargs["byte_addresses"] = torch.from_numpy(
                    np.asarray(addresses)[None]
                ).to(loaded.device)
            if owners is not None:
                owner = np.asarray(owners)[None]
                kwargs["memory_mask"] = torch.from_numpy(
                    (owner == OWNER_LEXICAL) | (owner == OWNER_HYBRID)
                ).to(loaded.device)
            logits = loaded.model(x, **kwargs)[0].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            valid = y[0] != -100
            target = y[0].clamp(min=0)
            nll = -log_probs.gather(1, target[:, None])[:, 0]
            sums.append(float(nll[valid].sum()))
            counts.append(int(valid.sum()))
            top = log_probs.max(dim=-1)
            confidences.append(top.values.exp()[valid].cpu().numpy())
            correct.append((top.indices == target)[valid].cpu().numpy())
            tokens += int(x.numel())
    elapsed = time.monotonic() - started
    total = sum(counts)
    if total == 0:
        raise ValueError("validation contains no valid labels")
    return {
        "loss": sum(sums) / total,
        "window_losses": [s / c if c else float("nan") for s, c in zip(sums, counts)],
        "window_sums": sums,
        "window_counts": counts,
        "valid_targets": total,
        "windows": limit,
        "confidences": np.concatenate(confidences),
        "correct": np.concatenate(correct),
        "accuracy": float(np.concatenate(correct).mean()),
        "ms_per_token": 1000.0 * elapsed / max(tokens, 1),
    }


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


def ranking_accuracy(loaded: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
    hits = []
    for item in items:
        scores = {
            c: _answer_score(loaded, item["prefix"], c) for c in item["candidates"]
        }
        best = max(scores, key=lambda c: (scores[c], c == item["answer"]))
        hits.append(best == item["answer"])
    accuracy = float(np.mean(hits)) if hits else None
    return {"accuracy": accuracy, "n": len(hits), "hits": [bool(h) for h in hits]}


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
                if len(_encode(loaded, text)) > target:
                    break
                parts = candidate
                k += 1
            items.append(
                {
                    "prefix": " ".join([*parts, spec["question"]]),
                    "answer": word,
                    "candidates": spec["needles"],
                    "fraction": fraction,
                    "tokens": len(
                        _encode(loaded, " ".join([*parts, spec["question"]]))
                    ),
                }
            )
    return items


def _divergence(target: _Session, base: _Session) -> dict[str, Any]:
    log_p_rows, log_q_rows = [], []
    for prompt in prompts("heldout"):
        ids = _encode(target.loaded, prompt)
        log_p_rows.append(_log_probs(base.loaded, ids))
        log_q_rows.append(_log_probs(target.loaded, ids))
    log_p = np.concatenate(log_p_rows)
    log_q = np.concatenate(log_q_rows)
    return {
        "top1_agreement": metrics.top1_agreement(log_p, log_q),
        "kl_base_to_candidate": float(metrics.kl_divergence(log_p, log_q).mean()),
        "js": float(metrics.js_divergence(log_p, log_q).mean()),
        "positions": int(log_p.shape[0]),
    }


# --- Battery ----------------------------------------------------------------


def _tokenizer_digest(loaded: Any) -> str:
    return hashlib.sha256(loaded.tokenizer.to_str().encode()).hexdigest()


def _validation_identity(loaded: Any) -> dict[str, Any]:
    from sparselab.lab_mode import _data_identity, _supervision_identity

    data = _data_identity(loaded.run)
    return {
        "validation_sha256": data["validation_sha256"],
        "tokenizer_sha256": data["tokenizer_sha256"],
        **_supervision_identity(loaded.run),
    }


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


def _run_probe(
    spec: ProbeSpec,
    target: _Session,
    base: _Session | None,
    protocol: Mapping[str, int] | None,
    comparable: Mapping[str, Any],
    dev: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    row = _row(spec)
    if spec.id in {"heldout_loss", "calibration"}:
        if base is not None and not comparable["validation"]:
            row.update(status="not_comparable", note=comparable["validation_reason"])
            return row
        assert protocol is not None
        if protocol["seq_len"] > target.loaded.config.model.max_seq_len:
            row.update(
                status="not_comparable",
                note="baseline eval window exceeds the candidate's max_seq_len",
            )
            return row
        t = target.cache.setdefault("val", validation_pass(target.loaded, protocol))
        b = (
            base.cache.setdefault("val", validation_pass(base.loaded, protocol))
            if base is not None
            else None
        )
        if spec.id == "heldout_loss":
            se = None
            if b is not None:
                diffs = np.asarray(t["window_losses"]) - np.asarray(b["window_losses"])
                _, se = metrics.paired_mean_and_se(diffs[np.isfinite(diffs)])
            row.update(judge(spec, t["loss"], b["loss"] if b else None, se=se))
            row["details"] = {
                "perplexity": math.exp(t["loss"]) if t["loss"] < 700 else None,
                "baseline_perplexity": math.exp(b["loss"])
                if b and b["loss"] < 700
                else None,
                "valid_targets": t["valid_targets"],
                "windows": t["windows"],
                "protocol": dict(protocol),
                "top1_accuracy": t["accuracy"],
                "baseline_top1_accuracy": b["accuracy"] if b else None,
                "ms_per_token": t["ms_per_token"],
            }
        else:
            bins = int(spec.params.get("bins", 15))
            value = metrics.expected_calibration_error(
                t["confidences"], t["correct"], bins
            )
            base_value = (
                metrics.expected_calibration_error(b["confidences"], b["correct"], bins)
                if b
                else None
            )
            row.update(judge(spec, value, base_value))
            row["details"] = {"bins": bins, "positions": int(t["correct"].size)}
        return row
    if spec.id == "token_agreement":
        if base is None:
            row.update(status="info", note="needs a baseline")
            return row
        if not comparable["tokenizer"]:
            row.update(status="not_comparable", note="tokenizers differ")
            return row
        stats = _divergence(target, base)
        row.update(judge(spec, stats["top1_agreement"], None))
        if stats["js"] < NEAR_IDENTICAL_JS and row["status"] != "pass":
            # Argmax ties between near-identical (often near-uniform) predictions
            # are arbitrary; low agreement then says nothing.
            row.update(
                status="pass",
                note=f"near-identical distributions (JS < {NEAR_IDENTICAL_JS}); "
                "argmax agreement is not informative",
            )
        row["details"] = {
            **stats,
            "informative": stats["js"] >= NEAR_IDENTICAL_JS,
        }
        return row
    if spec.id == "repetition":
        budget = int(spec.params.get("max_new_tokens", 32))
        t_rows = generations(target.loaded, budget)
        b_rows = generations(base.loaded, budget) if base is not None else None

        def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
            ids = [r["ids"] for r in rows]
            return {
                "seq_rep_4": metrics.mean_seq_rep_n(ids, 4),
                "distinct_1": metrics.distinct_n(ids, 1),
                "distinct_2": metrics.distinct_n(ids, 2),
            }

        t_sum = summary(t_rows)
        b_sum = summary(b_rows) if b_rows is not None else None
        row.update(
            judge(spec, t_sum["seq_rep_4"], b_sum["seq_rep_4"] if b_sum else None)
        )
        row["details"] = {
            **t_sum,
            "baseline": b_sum,
            "samples": [
                {"prompt": r["prompt"], "continuation": r["continuation"]}
                for r in t_rows[:2]
            ],
            "baseline_samples": [
                {"prompt": r["prompt"], "continuation": r["continuation"]}
                for r in (b_rows or [])[:2]
            ],
        }
        return row
    if spec.id in {"fact_recall", "needle"}:
        if base is not None and not comparable["tokenizer"]:
            row.update(status="not_comparable", note="tokenizers differ")
            return row

        def items(session: _Session, split: str) -> list[dict[str, Any]]:
            if spec.id == "fact_recall":
                return recall_items(split)
            return needle_items(
                session.loaded, split, list(spec.params["length_fractions"])
            )

        held_t = ranking_accuracy(target.loaded, items(target, "heldout"))
        held_b = (
            ranking_accuracy(base.loaded, items(target, "heldout")) if base else None
        )
        dev_t = ranking_accuracy(target.loaded, items(target, "dev"))
        dev_b = ranking_accuracy(base.loaded, items(target, "dev")) if base else None
        n = held_t["n"]
        se = (
            metrics.accuracy_delta_se(held_t["accuracy"], held_b["accuracy"], n)
            if held_b
            else None
        )
        row.update(
            judge(
                spec, held_t["accuracy"], held_b["accuracy"] if held_b else None, se=se
            )
        )
        candidates = items(target, "heldout")[0]["candidates"] if n else []
        details: dict[str, Any] = {
            "n": n,
            "chance": 1 / len(candidates) if candidates else None,
            "dev_accuracy": dev_t["accuracy"],
            "baseline_dev_accuracy": dev_b["accuracy"] if dev_b else None,
        }
        if spec.id == "needle":
            held_items = items(target, "heldout")
            by_length: dict[str, dict[str, Any]] = {}
            for fraction in spec.params["length_fractions"]:
                picks = [
                    i for i, it in enumerate(held_items) if it["fraction"] == fraction
                ]
                tokens = max(held_items[i]["tokens"] for i in picks) if picks else 0
                by_length[str(fraction)] = {
                    "tokens": tokens,
                    "accuracy": float(np.mean([held_t["hits"][i] for i in picks]))
                    if picks
                    else None,
                    "baseline_accuracy": float(
                        np.mean([held_b["hits"][i] for i in picks])
                    )
                    if picks and held_b
                    else None,
                }
            details["by_length"] = by_length
        row["details"] = details
        dev[spec.id] = {
            "value": dev_t["accuracy"],
            "baseline_value": dev_b["accuracy"] if dev_b else None,
            "n": dev_t["n"],
        }
        return row
    if spec.id == "lm_eval":
        from sparselab.probes.lm_eval_adapter import LmEvalUnavailable, run_lm_eval

        try:
            t = run_lm_eval(target.loaded, spec.params["tasks"], spec.params["limit"])
            b = (
                run_lm_eval(base.loaded, spec.params["tasks"], spec.params["limit"])
                if base
                else None
            )
        except LmEvalUnavailable as error:
            row.update(status="skipped", note=str(error))
            return row
        value = t["mean_accuracy"]
        row.update(judge(spec, value, b["mean_accuracy"] if b else None))
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


def run_battery(
    target: Any,
    baseline: Any | None = None,
    *,
    tier: str = "fast",
    fast_fail: bool = True,
    protocol: Mapping[str, int] | None = None,
    progress: Progress | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Score TARGET (and BASELINE) with the suite through TIER; fast-fail."""
    from sparselab.lab_mode import eval_protocol

    started_at = (now or (lambda: datetime.now(UTC)))()
    started = time.monotonic()
    tiers_through(tier)
    suite = suite_identity()
    specs = ordered(tier)
    t_session = _Session(target)
    b_session = _Session(baseline) if baseline is not None else None
    try:
        if protocol is None:
            protocol = eval_protocol((baseline or target).config)
        comparable = {"validation": True, "validation_reason": None, "tokenizer": True}
        if b_session is not None:
            t_id = _validation_identity(target)
            b_id = _validation_identity(baseline)
            if t_id != b_id:
                differing = sorted(k for k in t_id if t_id[k] != b_id[k])
                comparable.update(
                    validation=False,
                    validation_reason="validation identity differs: "
                    + ", ".join(differing),
                )
            comparable["tokenizer"] = _tokenizer_digest(target) == _tokenizer_digest(
                baseline
            )
        results: list[dict[str, Any]] = []
        dev: dict[str, dict[str, float | None]] = {}
        stopped: dict[str, Any] = {"stopped": False, "at": None, "reason": None}
        tiers_run: list[str] = []

        def report(state: str, current: str | None) -> None:
            if progress is not None:
                progress(
                    {
                        "state": state,
                        "suite": suite["sha256"],
                        "tier": tier,
                        "current": current,
                        "done": [r["id"] for r in results],
                        "statuses": {r["id"]: r["status"] for r in results},
                        "total": len(specs),
                        "target": t_session.identity["run_id"],
                        "started_at": started_at.isoformat(),
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                )

        for spec in specs:
            if stopped["stopped"]:
                results.append(_row(spec, note=f"skipped: {stopped['reason']}"))
                continue
            if spec.tier not in tiers_run:
                previous = [r for r in results if r["tier"] in tiers_run]
                if (
                    fast_fail
                    and b_session is not None
                    and any(r["status"] == "fail" for r in previous)
                ):
                    stopped.update(
                        stopped=True,
                        at=spec.id,
                        reason=f"not promising after the {tiers_run[-1]} tier; "
                        "not escalating",
                    )
                    results.append(_row(spec, note=f"skipped: {stopped['reason']}"))
                    continue
                tiers_run.append(spec.tier)
            report("running", spec.id)
            tick = time.monotonic()
            try:
                row = _run_probe(spec, t_session, b_session, protocol, comparable, dev)
            except ProbeUnsupported as error:
                row = _row(spec, status="skipped", note=str(error))
            except Exception as error:  # noqa: BLE001 - one probe never sinks the battery
                row = _row(
                    spec, status="error", note=f"{type(error).__name__}: {error}"
                )
            row["seconds"] = round(time.monotonic() - tick, 3)
            results.append(row)
            if fast_fail and row["status"] == "fail" and spec.hard:
                stopped.update(
                    stopped=True,
                    at=spec.id,
                    reason=f"fast-fail: hard failure in {spec.id}",
                )
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
            has_baseline=b_session is not None,
            tiers_run=tiers_run,
            requested_tier=tier,
            guard=guard,
            specs=BY_ID,
        )
        result = {
            "format": RESULT_FORMAT,
            "created_at": started_at.isoformat(),
            "suite": suite,
            "tier": tier,
            "tiers_run": tiers_run,
            "fast_fail": stopped,
            "target": t_session.identity,
            "baseline": b_session.identity if b_session is not None else None,
            "comparable": comparable,
            "protocol": dict(protocol),
            "probes": results,
            "guard": guard,
            "verdict": verdict,
            "seconds": round(time.monotonic() - started, 3),
        }
        report("done", None)
        return result
    finally:
        t_session.close()
        if b_session is not None:
            b_session.close()


# --- Records -----------------------------------------------------------------


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()


def seal(result: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in result.items() if k != "result_sha256"}
    return {**body, "result_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def verify(result: Mapping[str, Any]) -> bool:
    return seal(dict(result))["result_sha256"] == result.get("result_sha256")


def write_json_atomic(path: Path, value: Any, *, exclusive: bool = False) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def progress_writer(path: Path) -> Progress:
    """Live progress for the dashboard: an atomically replaced JSON file."""

    def write(state: dict[str, Any]) -> None:
        try:
            write_json_atomic(path, state)
        except OSError:
            pass

    return write


def tier_names() -> tuple[str, ...]:
    return TIERS
