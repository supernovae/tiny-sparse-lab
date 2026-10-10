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


def summarize_results(results: dict[str, Any], tasks: Sequence[str]) -> dict[str, Any]:
    """Pick each task's harness accuracy (acc, else acc_norm) and average."""
    rows: dict[str, Any] = {}
    for task in tasks:
        metrics = (results.get("results") or {}).get(task) or {}
        acc = metrics.get("acc,none", metrics.get("acc_norm,none"))
        rows[task] = {
            "acc": acc,
            "acc_norm": metrics.get("acc_norm,none"),
            "perplexity": metrics.get("perplexity,none"),
        }
    values = [row["acc"] for row in rows.values() if row["acc"] is not None]
    return {"tasks": rows, "mean_accuracy": float(np.mean(values)) if values else None}


def run_lm_eval(loaded: Any, tasks: Sequence[str], limit: int) -> dict[str, Any]:
    lm_eval, _ = _harness()
    from lm_eval import simple_evaluate

    os.environ.setdefault("TQDM_DISABLE", "1")
    results = simple_evaluate(
        model=build_lm(loaded),
        tasks=list(tasks),
        limit=limit,
        bootstrap_iters=0,
        log_samples=False,
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
    return summary
