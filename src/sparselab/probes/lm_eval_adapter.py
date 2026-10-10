"""Optional lm-evaluation-harness backend for the ``full`` probe tier.

Install with ``uv sync --extra lmeval`` (``lm-eval`` is never a hard
dependency; without it the probe is ``unavailable`` and the battery is
``incomplete``). ``SparseLabLM`` adapts a loaded SparseLab checkpoint to the
harness's ``LM`` interface, so task definitions, prompts and metric
definitions are the harness's own. All scoring goes through
:mod:`sparselab.probes.scoring`: context/continuation boundaries follow the
harness, and continuations longer than the model's context are scored
completely with windowed left context (never truncated).

Honest scale note: at tiny scale (a few million parameters, thousands of
tokens) these tasks mostly score at chance (hellaswag/arc_easy ~0.25,
piqa ~0.5, lambada ~0); the tier exists so larger lab runs sit on a known
public scale, not to separate two smoke checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from sparselab.probes import scoring


class LmEvalUnavailable(RuntimeError):
    pass


def greedy_until(loaded: Any, context: str, gen_kwargs: dict[str, Any]) -> str:
    from sparselab.evaluation.generation import generate

    until = gen_kwargs.get("until") or []
    if isinstance(until, str):
        until = [until]
    max_new = int(gen_kwargs.get("max_gen_toks", 32))
    text = generate(
        loaded.model,
        loaded.tokenizer,
        context,
        loaded.config.model.max_seq_len,
        max_new,
        loaded.device,
        temperature=0.0,
        stop_sequences=tuple(until),
    )
    out = text.removeprefix(context)
    for stop in until:
        if stop and stop in out:
            out = out[: out.index(stop)]
    return out


def _harness() -> Any:
    try:
        import lm_eval
        from lm_eval.api.model import LM
    except ImportError as error:
        raise LmEvalUnavailable(
            "lm-evaluation-harness is not installed: uv sync --extra lmeval"
        ) from error
    return lm_eval, LM


def build_lm(loaded: Any) -> Any:
    """Wrap LOADED as an ``lm_eval.api.model.LM`` instance."""
    _, base = _harness()

    class SparseLabLM(base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.loaded = loaded

        def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
            out = []
            for request in requests:
                total, _, greedy = scoring.continuation_score(
                    self.loaded, *request.args
                )
                out.append((total, greedy))
            return out

        def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
            return [scoring.rolling_score(self.loaded, r.args[0]) for r in requests]

        def generate_until(self, requests: list[Any]) -> list[str]:
            return [greedy_until(self.loaded, *r.args) for r in requests]

    return SparseLabLM()


# Bump when how SparseLab scores a harness request changes (context and
# continuation boundaries, windowing, end-of-text handling): results scored
# under another protocol land in another benchmark group.
SCORING_PROTOCOL = "sparselab-lmeval-scoring-v1"


def item_identity(item: dict[str, Any]) -> str:
    """Digest of what was actually scored for one item: the document, every
    rendered request (context and continuation(s)) and the gold target."""
    body = {
        "doc_hash": item.get("doc_hash"),
        "arguments": item.get("arguments"),
        "target": str(item.get("target")),
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str, ensure_ascii=False).encode()
    ).hexdigest()


def summarize_results(results: dict[str, Any], tasks: Sequence[str]) -> dict[str, Any]:
    """Each task's harness accuracy (acc, else acc_norm), per-item outcomes and
    the benchmark group that decides which results may be compared."""
    rows: dict[str, Any] = {}
    samples = results.get("samples") or {}
    for task in tasks:
        metrics = (results.get("results") or {}).get(task) or {}
        metric = "acc" if "acc,none" in metrics else "acc_norm"
        logged = sorted(samples.get(task) or [], key=lambda item: item["doc_id"])
        rows[task] = {
            "acc": metrics.get(f"{metric},none"),
            "acc_metric": metric if f"{metric},none" in metrics else None,
            "acc_norm": metrics.get("acc_norm,none"),
            "perplexity": metrics.get("perplexity,none"),
            "items": [float(item[metric]) for item in logged if metric in item],
            "items_sha256": hashlib.sha256(
                "\n".join(item_identity(item) for item in logged).encode()
            ).hexdigest()
            if logged
            else None,
        }
    values = [row["acc"] for row in rows.values() if row["acc"] is not None]
    return {
        "tasks": rows,
        "mean_accuracy": float(np.mean(values)) if values else None,
        "benchmark": benchmark_identity(results, tasks, rows),
    }


def benchmark_identity(
    results: dict[str, Any], tasks: Sequence[str], rows: dict[str, Any]
) -> dict[str, Any]:
    """What must match for two lm-eval results to be compared: tasks, task
    versions, shots, the metric per task and the exact scored items."""
    versions = results.get("versions") or {}
    shots = results.get("n-shot") or {}
    return {
        "harness": "lm-evaluation-harness",
        "scoring_protocol": SCORING_PROTOCOL,
        "tasks": list(tasks),
        "task_versions": {t: versions.get(t) for t in tasks},
        "num_fewshot": {t: shots.get(t, 0) for t in tasks},
        "metric": {t: rows[t]["acc_metric"] for t in tasks},
        "items_sha256": {t: rows[t]["items_sha256"] for t in tasks},
    }


def missing_tasks(
    summary: dict[str, Any], tasks: Sequence[str], limit: int
) -> list[str]:
    """Tasks without an accuracy or without all LIMIT items scored."""
    rows = summary.get("tasks") or {}
    gaps = [
        task
        for task in tasks
        if (rows.get(task) or {}).get("acc") is None
        or len((rows.get(task) or {}).get("items") or []) < limit
    ]
    if not gaps and not summary.get("benchmark_group"):
        return list(tasks)
    return gaps


def benchmark_group(summary: dict[str, Any]) -> str | None:
    """Comparison group of an lm-eval summary (None when incomplete)."""
    from sparselab.lab_records import comparison_group

    identity = summary.get("benchmark")
    if not identity or any(v is None for v in identity["items_sha256"].values()):
        return None
    return comparison_group(identity)


def run_lm_eval(loaded: Any, tasks: Sequence[str], limit: int) -> dict[str, Any]:
    lm_eval, _ = _harness()
    from lm_eval import simple_evaluate

    os.environ.setdefault("TQDM_DISABLE", "1")
    results = simple_evaluate(
        model=build_lm(loaded),
        tasks=list(tasks),
        limit=limit,
        bootstrap_iters=0,
        log_samples=True,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )
    summary = summarize_results(results or {}, tasks)
    summary["version"] = getattr(lm_eval, "__version__", None)
    missing = [task for task, row in summary["tasks"].items() if row["acc"] is None]
    if missing:
        # A task without its metric is missing evidence, not a low score.
        raise RuntimeError(f"lm-eval returned no accuracy for: {', '.join(missing)}")
    summary["benchmark"]["limit"] = limit
    summary["benchmark_group"] = benchmark_group(summary)
    return summary
