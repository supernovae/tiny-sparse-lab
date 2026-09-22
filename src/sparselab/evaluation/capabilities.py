"""Versioned, narrow capability cards for controlled local experiments."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from sparselab.config.models import RunConfig
from sparselab.data.engram_recall import recall_case
from sparselab.evaluation.generation import generate


@dataclass(frozen=True)
class CapabilityCase:
    identifier: str
    prompt: str
    expected: str


@dataclass(frozen=True)
class CapabilityCard:
    name: str
    version: int
    hypothesis: str
    scorer: str
    cases: tuple[CapabilityCase, ...]

    @property
    def digest(self) -> str:
        payload = asdict(self)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def capability_card(name: str) -> CapabilityCard:
    if name != "engram-recall-v1":
        raise ValueError(f"unknown capability card: {name}")
    cases = tuple(
        CapabilityCase(f"recall-{index:02d}", *recall_case(42, 2_000_000 + index))
        for index in range(16)
    )
    return CapabilityCard(
        name=name,
        version=1,
        hypothesis=(
            "At matched controls, an Engram variant may improve exact recall of "
            "fixed associations across held-out record wording."
        ),
        scorer="first_normalized_word_exact_v1",
        cases=cases,
    )


def _first_word(value: str) -> str:
    words = re.findall(r"[a-z]+", value.casefold())
    return words[0] if words else ""


def evaluate_capability(
    card: CapabilityCard,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    max_seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    results = []
    for case in card.cases:
        completion = generate(
            model, tokenizer, case.prompt, max_seq_len, 8, device)
        response = completion.removeprefix(case.prompt).strip()
        results.append(
            {
                "id": case.identifier,
                "expected": case.expected,
                "response": response,
                "passed": _first_word(response) == case.expected,
            }
        )
    passed = sum(int(item["passed"]) for item in results)
    return {
        "format": "capability_result_v1",
        "card": card.name,
        "card_digest": card.digest,
        "hypothesis": card.hypothesis,
        "scorer": card.scorer,
        "case_count": len(results),
        "passed": passed,
        "score": passed / len(results),
        "results": results,
    }


def write_capability_result(run: Path, result: dict[str, Any]) -> Path:
    evaluations = run / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    path = evaluations / f"capability-{result['card']}-{result['card_digest'][:12]}.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return path


def comparison_controls(config: RunConfig) -> dict[str, Any]:
    model = config.model.model_dump(mode="json")
    for key in (
        "memory",
        "memory_table_size",
        "memory_ngram_size",
        "memory_dim",
        "memory_package_path",
        "memory_ngram_orders",
        "memory_hash_heads",
    ):
        model.pop(key)
    return {
        "model": model,
        "attention": config.attention.model_dump(mode="json"),
        "dataset": config.dataset.model_dump(mode="json"),
        "tokenizer": config.tokenizer.model_dump(mode="json"),
        "training": config.training.model_dump(mode="json"),
        "optimizer": config.optimizer.model_dump(mode="json"),
        "runtime": config.runtime.model_dump(mode="json"),
    }


def control_differences(base: RunConfig, variant: RunConfig) -> tuple[str, ...]:
    baseline, candidate = comparison_controls(base), comparison_controls(variant)
    return tuple(
        key
        for key in baseline
        if baseline[key] != candidate[key]
    )
