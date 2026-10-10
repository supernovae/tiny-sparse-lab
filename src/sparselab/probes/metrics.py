"""Pure probe metrics with established definitions (no model code here).

* distinct-n: unique n-grams / total n-grams over a pool of generations
  (Li et al., 2016, "A Diversity-Promoting Objective Function").
* seq-rep-n: 1 - unique n-grams / total n-grams within one continuation,
  averaged over continuations (Welleck et al., 2019, "Neural Text Generation
  with Unlikelihood Training").
* ECE: expected calibration error with equal-width confidence bins over the
  top-1 prediction (Guo et al., 2017; 15 bins by default).
* KL(P_base || P_cand) and Jensen-Shannon divergence in nats (JS <= ln 2).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def ngrams(tokens: Sequence[int], n: int) -> list[tuple[int, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(sequences: Iterable[Sequence[int]], n: int) -> float | None:
    """Pooled distinct-n; None when no sequence has n tokens."""
    pooled: list[tuple[int, ...]] = []
    for sequence in sequences:
        pooled.extend(ngrams(sequence, n))
    return len(set(pooled)) / len(pooled) if pooled else None


def seq_rep_n(sequence: Sequence[int], n: int = 4) -> float | None:
    """Within-sequence repeated n-gram rate; 0 = no repeats, ->1 = a loop."""
    grams = ngrams(sequence, n)
    return 1.0 - len(set(grams)) / len(grams) if grams else None


def mean_seq_rep_n(sequences: Iterable[Sequence[int]], n: int = 4) -> float | None:
    values = [v for v in (seq_rep_n(s, n) for s in sequences) if v is not None]
    return float(np.mean(values)) if values else None


def expected_calibration_error(
    confidences: Sequence[float] | np.ndarray,
    correct: Sequence[bool] | np.ndarray,
    bins: int = 15,
) -> float:
    """ECE = sum_b |B_b|/N * |acc(B_b) - conf(B_b)| over equal-width bins."""
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    if conf.shape != hit.shape or conf.ndim != 1 or conf.size == 0:
        raise ValueError("confidences and correctness must be equal nonempty 1-D")
    if np.any((conf < 0) | (conf > 1)):
        raise ValueError("confidences must lie in [0, 1]")
    # Bin b covers (b/B, (b+1)/B]; confidence 0 joins the first bin.
    index = np.clip(np.ceil(conf * bins).astype(np.int64) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        members = index == b
        if members.any():
            total += members.mean() * abs(hit[members].mean() - conf[members].mean())
    return float(total)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def kl_divergence(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    """Per-position KL(P || Q) in nats from log-probabilities [..., vocab]."""
    p = np.exp(log_p)
    return (p * (log_p - log_q)).sum(axis=-1)


def js_divergence(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    """Per-position Jensen-Shannon divergence in nats, bounded by ln 2."""
    p, q = np.exp(log_p), np.exp(log_q)
    log_m = np.log(np.clip(0.5 * (p + q), 1e-300, None))
    return 0.5 * (p * (log_p - log_m)).sum(axis=-1) + 0.5 * (q * (log_q - log_m)).sum(
        axis=-1
    )


def top1_agreement(log_p: np.ndarray, log_q: np.ndarray) -> float:
    return float(np.mean(log_p.argmax(axis=-1) == log_q.argmax(axis=-1)))


def paired_mean_and_se(
    values: Sequence[float] | np.ndarray,
) -> tuple[float, float | None]:
    """Mean and standard error of paired per-item differences (None for n < 2)."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("no values")
    mean = float(data.mean())
    if data.size < 2:
        return mean, None
    return mean, float(data.std(ddof=1) / math.sqrt(data.size))


def ratio_difference_se(
    sums_a: Sequence[float],
    counts_a: Sequence[int],
    sums_b: Sequence[float],
    counts_b: Sequence[int],
) -> float | None:
    """SE of (sum S_a / sum N_a) - (sum S_b / sum N_b) over paired windows.

    Each window is a cluster; the linearized (delta-method) ratio estimator
    weights windows by their scored tokens, matching a token-weighted mean
    loss rather than an unweighted mean of per-window losses.
    """
    s_a, n_a = np.asarray(sums_a, float), np.asarray(counts_a, float)
    s_b, n_b = np.asarray(sums_b, float), np.asarray(counts_b, float)
    if not (s_a.shape == n_a.shape == s_b.shape == n_b.shape) or s_a.size < 2:
        return None
    total_a, total_b = n_a.sum(), n_b.sum()
    if total_a <= 0 or total_b <= 0:
        return None
    mean_a, mean_b = s_a.sum() / total_a, s_b.sum() / total_b
    z = (s_a - mean_a * n_a) / total_a - (s_b - mean_b * n_b) / total_b
    windows = z.size
    return float(math.sqrt(windows / (windows - 1) * float((z**2).sum())))
