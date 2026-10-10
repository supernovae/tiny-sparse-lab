"""Optional lm-evaluation-harness backend for the ``full`` probe tier.

Install with ``uv sync --extra lmeval`` (``lm-eval`` is never a hard
dependency). ``SparseLabLM`` adapts a loaded SparseLab checkpoint to the
harness's ``LM`` interface, so task definitions, prompts and metric
definitions are the harness's own. The scoring helpers below are plain
functions so they are tested without the harness installed.

Honest scale note: at tiny scale (a few million parameters, thousands of
tokens) these tasks mostly score at chance (hellaswag/arc_easy ~0.25,
piqa ~0.5, lambada ~0); the tier exists so larger lab runs sit on a known
public scale, not to separate two smoke checkpoints.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class LmEvalUnavailable(RuntimeError):
    pass


def _encode(loaded: Any, text: str) -> list[int]:
    return list(loaded.tokenizer.encode(text, add_special_tokens=False).ids)


def _eos(loaded: Any) -> int:
    eos = loaded.tokenizer.token_to_id("<eos>")
    return int(eos) if eos is not None else 0


def _log_probs(loaded: Any, ids: Sequence[int]) -> np.ndarray:
    from sparselab.probes.runner import _log_probs as native

    return native(loaded, list(ids))


def continuation_logprob(
    loaded: Any, context: str, continuation: str
) -> tuple[float, bool]:
    """Sum log P(continuation | context) and whether greedy decoding yields it."""
    context_ids = _encode(loaded, context) if context else [_eos(loaded)]
    whole = _encode(loaded, context + continuation)
    if context and whole[: len(context_ids)] == context_ids:
        continuation_ids = whole[len(context_ids) :]
    else:
        continuation_ids = _encode(loaded, continuation)
    if not continuation_ids:
        return 0.0, True
    ids = [*context_ids, *continuation_ids]
    window = ids[-(loaded.config.model.max_seq_len + 1) :]
    count = min(len(continuation_ids), len(window) - 1)
    rows = _log_probs(loaded, window[:-1])[-count:]
    targets = np.asarray(window[-count:])
    logprob = float(rows[np.arange(count), targets].sum())
    greedy = bool(np.all(rows.argmax(axis=-1) == targets))
    return logprob, greedy


def rolling_logprob(loaded: Any, text: str) -> float:
    """Every token scored exactly once with maximal left context (harness rule)."""
    tokens = [_eos(loaded), *_encode(loaded, text)]
    length = loaded.config.model.max_seq_len
    total = 0.0
    start = 1  # index of the first unscored target in ``tokens``
    while start < len(tokens):
        end = min(start + length, len(tokens))  # targets tokens[start:end]
        first = max(0, end - 1 - length)  # inputs tokens[first:end-1]
        rows = _log_probs(loaded, tokens[first : end - 1])
        count = end - start
        targets = np.asarray(tokens[start:end])
        total += float(rows[-count:][np.arange(count), targets].sum())
        start = end
    return total


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
            return [continuation_logprob(self.loaded, *r.args) for r in requests]

        def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
            return [rolling_logprob(self.loaded, r.args[0]) for r in requests]

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
    import os

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
    return summary
