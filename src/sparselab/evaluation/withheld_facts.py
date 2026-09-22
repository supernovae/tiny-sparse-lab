"""Strict held-out completion evaluation for verified fact manifests."""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

from sparselab.data.withheld_facts import verify_manifest
from sparselab.evaluation.generation import generate


def evaluate_withheld_facts(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    manifest_path: Path,
    *,
    max_seq_len: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, object]:
    """Generate each verified held-out completion and score exact continuations."""
    manifest = verify_manifest(manifest_path)
    held_out_cases = manifest["held_out_cases"]
    if not isinstance(held_out_cases, list):
        raise TypeError("verified manifest has invalid held-out cases")
    cases: list[dict[str, object]] = []
    for case in held_out_cases:
        assert isinstance(case, dict)
        prompt = case["prompt"]
        expected = case["expected_value"]
        assert isinstance(prompt, str)
        assert isinstance(expected, str)
        completion = generate(
            model, tokenizer, prompt, max_seq_len, max_new_tokens, device
        )
        predicted = completion.removeprefix(prompt).strip()
        cases.append(
            {
                "exact_match": predicted == expected,
                "expected_value": expected,
                "prediction": predicted,
                "prompt": prompt,
            }
        )
    exact_match_count = sum(bool(case["exact_match"]) for case in cases)
    return {
        "case_count": len(cases),
        "cases": cases,
        "exact_match_count": exact_match_count,
        "exact_match_rate": exact_match_count / len(cases),
        "manifest_sha256": manifest["sha256"],
    }
