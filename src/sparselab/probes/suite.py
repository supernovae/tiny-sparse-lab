"""The versioned probe suite: probe declarations, items and their identity.

Every probe declares its cost tier, how it is judged against a baseline
(pass / warn / fail thresholds), whether a failure is *hard* (stop the battery:
fast-fail) and what a failure suggests. Prompt items come in two disjoint splits:

* ``dev``: free to read and iterate against;
* ``heldout``: the only split verdicts use. Agents must not tune against its
  items; the battery also scores ``dev`` and flags a dev gain that the held-out
  split does not share (``guard.overfit_suspected``).

The suite digest covers declarations, thresholds and both splits, so two
results are comparable exactly when their ``suite.sha256`` matches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from sparselab.data.withheld_facts import _FACTS

SUITE_NAME = "sparselab-probe-battery"
SUITE_VERSION = 1
TIERS = ("fast", "standard", "full")
Mode = Literal["delta_rel", "delta_abs", "value_min"]


@dataclass(frozen=True)
class ProbeSpec:
    id: str
    title: str
    tier: str
    cost: int
    metric: str
    higher_is_better: bool
    mode: Mode
    warn: float
    fail: float | None
    hard: bool
    suggests: str
    explains: str
    reference: str = ""
    params: dict[str, Any] = field(default_factory=dict)


PROBES: tuple[ProbeSpec, ...] = (
    ProbeSpec(
        id="heldout_loss",
        title="Held-out loss",
        tier="fast",
        cost=1,
        metric="loss",
        higher_is_better=False,
        mode="delta_rel",
        warn=0.005,
        fail=0.03,
        hard=True,
        suggests="The change raises held-out loss at this budget: revert it or "
        "retune LR/warmup before spending a longer run.",
        explains="Mean next-token cross-entropy on the run's validation split, "
        "scored under one shared window/batch protocol for both arms. Lower is "
        "better; perplexity = exp(loss). Judged on the relative change, with a "
        "paired standard error across the same windows.",
        reference="token-level cross-entropy; paired windows",
        params={"max_windows": 64},
    ),
    ProbeSpec(
        id="calibration",
        title="Calibration (ECE)",
        tier="fast",
        cost=1,
        metric="ece",
        higher_is_better=False,
        mode="delta_abs",
        warn=0.02,
        fail=0.05,
        hard=False,
        suggests="Confidence no longer tracks accuracy: check the LR schedule "
        "end, label smoothing or an output-scale change.",
        explains="Expected calibration error of the top-1 next-token "
        "prediction: how far the model's confidence is from how often it is "
        "right. 0 is perfectly calibrated.",
        reference="Guo et al. 2017, 15 equal-width bins",
        params={"bins": 15},
    ),
    ProbeSpec(
        id="token_agreement",
        title="Top-1 agreement",
        tier="fast",
        cost=1,
        metric="top1_agreement",
        higher_is_better=True,
        mode="value_min",
        warn=0.5,
        fail=None,
        hard=False,
        suggests="Large behavior shift vs the baseline: if loss improved, "
        "escalate to see what moved; if loss is flat it may be init/seed noise.",
        explains="On fixed held-out prompts, how often both models pick the "
        "same next token, plus KL(baseline || candidate) and Jensen-Shannon "
        "divergence. Not good or bad by itself: it tells you how different "
        "the candidate's predictions are.",
        reference="argmax agreement; KL and JS in nats",
    ),
    ProbeSpec(
        id="repetition",
        title="Degeneration",
        tier="fast",
        cost=2,
        metric="seq_rep_4",
        higher_is_better=False,
        mode="delta_abs",
        warn=0.05,
        fail=0.2,
        hard=True,
        suggests="Greedy generations loop more than the baseline: look for "
        "duplicated data, a too-high LR or a positional change; read samples.",
        explains="Greedy continuations of fixed prompts, scored by seq-rep-4 "
        "(share of repeated 4-grams within a continuation; 1 = stuck in a "
        "loop) and distinct-1/2 (vocabulary variety across continuations).",
        reference="seq-rep-n: Welleck et al. 2019; distinct-n: Li et al. 2016",
        params={"max_new_tokens": 32},
    ),
    ProbeSpec(
        id="fact_recall",
        title="Fact recall (reworded)",
        tier="standard",
        cost=2,
        metric="accuracy",
        higher_is_better=True,
        mode="delta_abs",
        warn=0.05,
        fail=0.15,
        hard=False,
        suggests="Worse at using a stated fact when asked in different words: "
        "check context handling (attention/positions) or memory wiring.",
        explains="A fact is stated, then asked with a reworded held-out "
        "question; the model ranks candidate answers by log-probability. "
        "Accuracy is top-1; chance is 1/candidates.",
        reference="candidate ranking by mean answer log-probability",
    ),
    ProbeSpec(
        id="needle",
        title="Needle in context",
        tier="standard",
        cost=3,
        metric="accuracy",
        higher_is_better=True,
        mode="delta_abs",
        warn=0.05,
        fail=0.15,
        hard=False,
        suggests="Retrieval from earlier context degraded: suspect attention "
        "span, positional encoding or sequence-length changes.",
        explains="A code word is hidden early in filler text of a few lengths "
        "up to the model's context; at the end the model must recall it. "
        "Scored by candidate ranking at each length.",
        reference="needle-in-a-haystack, candidate ranking",
        params={"length_fractions": [0.5, 0.75, 0.95]},
    ),
    ProbeSpec(
        id="lm_eval",
        title="Standard tasks (lm-eval)",
        tier="full",
        cost=4,
        metric="accuracy",
        higher_is_better=True,
        mode="delta_abs",
        warn=0.02,
        fail=0.05,
        hard=False,
        suggests="Lower on standard zero-shot tasks; at tiny scale most tasks "
        "sit near chance, so treat small moves as noise.",
        explains="A tiny zero-shot slice of public tasks through "
        "EleutherAI's lm-evaluation-harness (optional `lmeval` extra). "
        "Metric definitions are the harness's own.",
        reference="lm-evaluation-harness acc (zero-shot)",
        params={
            "tasks": ["lambada_openai", "hellaswag", "arc_easy", "piqa"],
            "limit": 50,
            "chance": {
                "lambada_openai": 0.0,
                "hellaswag": 0.25,
                "arc_easy": 0.25,
                "piqa": 0.5,
            },
        },
    ),
)
BY_ID = {spec.id: spec for spec in PROBES}

# --- Items -----------------------------------------------------------------

_DEV_PROMPTS = (
    "Once upon a time, a little dog found a ball in the park.",
    "The girl looked at the sky and saw a big red balloon.",
    "Tom and his mom went to the shop to buy some bread.",
    "The cat sat by the window and watched the rain fall.",
)
_HELDOUT_PROMPTS = (
    "One day, a small bird flew over the river and sang a song.",
    "Lily had a blue box. She opened it and found a tiny key.",
    "The boy wanted to play, but his friend was very sleepy.",
    "In the garden there was a tall tree with green leaves.",
    "Sam was happy because it was his birthday today.",
    "The old man walked slowly to the bench and sat down.",
    "A little fish swam in the pond and met a frog.",
    "Mia helped her dad wash the car in the sun.",
)

_DEV_FACTS = (
    ("Pip", "favorite fruit", "apple"),
    ("Rex", "favorite fruit", "pear"),
    ("Zoe", "favorite fruit", "plum"),
    ("Kai", "favorite fruit", "lime"),
    ("Ivy", "favorite toy", "ball"),
    ("Leo", "favorite toy", "doll"),
    ("Nia", "favorite toy", "yo-yo"),
    ("Oz", "favorite toy", "top"),
)
# Held-out facts reuse the withheld-facts diagnostic assets.
_HELDOUT_FACTS = tuple((f.subject, f.relation, f.value) for f in _FACTS)
_DEV_TEMPLATES = (
    "Question: what is {s}'s {r}? Answer:",
    "Tell me {s}'s {r}:",
)
_HELDOUT_TEMPLATES = (
    "Someone asked about the {r} of {s}. It is",
    "If you ask {s} for their {r}, they say",
)

_DEV_NEEDLES = ("lamp", "door", "cup", "shoe")
_HELDOUT_NEEDLES = ("moon", "star", "boat", "cake", "drum", "kite")
_DEV_FILLER = (
    "The sun was warm.",
    "A dog ran by.",
    "The grass was green.",
)
_HELDOUT_FILLER = (
    "The day was calm.",
    "A cat slept on the mat.",
    "Birds sang in the tree.",
    "The road was long.",
)


def fact_items(split: str) -> list[dict[str, Any]]:
    facts = _HELDOUT_FACTS if split == "heldout" else _DEV_FACTS
    templates = _HELDOUT_TEMPLATES if split == "heldout" else _DEV_TEMPLATES
    items = []
    for subject, relation, value in facts:
        candidates = sorted({v for s, r, v in facts if r == relation})
        for template in templates:
            items.append(
                {
                    "context": f"{subject}'s {relation} is {value}.",
                    "question": template.format(s=subject, r=relation),
                    "answer": value,
                    "candidates": candidates,
                }
            )
    return items


def needle_split(split: str) -> dict[str, Any]:
    return {
        "needles": list(_HELDOUT_NEEDLES if split == "heldout" else _DEV_NEEDLES),
        "filler": list(_HELDOUT_FILLER if split == "heldout" else _DEV_FILLER),
        "statement": "Code: {w}.",
        "question": "Code:",
    }


def prompts(split: str) -> list[str]:
    return list(_HELDOUT_PROMPTS if split == "heldout" else _DEV_PROMPTS)


def split_payload(split: str) -> dict[str, Any]:
    return {
        "prompts": prompts(split),
        "facts": fact_items(split),
        "needle": needle_split(split),
    }


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def suite_identity() -> dict[str, Any]:
    """Name, version and digests of declarations and both item splits."""
    splits = {name: _sha(split_payload(name)) for name in ("dev", "heldout")}
    declarations = [asdict(spec) for spec in PROBES]
    return {
        "name": SUITE_NAME,
        "version": SUITE_VERSION,
        "sha256": _sha(
            {
                "name": SUITE_NAME,
                "version": SUITE_VERSION,
                "probes": declarations,
                "splits": splits,
            }
        ),
        "splits": splits,
        "verdict_split": "heldout",
    }


def tiers_through(tier: str) -> tuple[str, ...]:
    if tier not in TIERS:
        raise ValueError(f"unknown probe tier {tier!r}; choose {', '.join(TIERS)}")
    return TIERS[: TIERS.index(tier) + 1]


def ordered(tier: str) -> list[ProbeSpec]:
    """Probes to run through TIER, cheapest tier and cost first."""
    allowed = tiers_through(tier)
    return sorted(
        (spec for spec in PROBES if spec.tier in allowed),
        key=lambda spec: (TIERS.index(spec.tier), spec.cost, PROBES.index(spec)),
    )
