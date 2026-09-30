"""Scientific independence checks for proposed DevMind cards, not model scores."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

FREEZER = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "experiments/research/devmind-pretrain-v1/freeze_cards.py"
    )
)


def _world(seed: int, split: str, cause: str) -> dict:
    generator = "platform_fault_v1"
    world_id = f"{generator}:{seed}"
    facts = {"cause": cause, "fixture_id": world_id}
    return {
        "split": split,
        "generator_id": generator,
        "generator_world_id": world_id,
        "world_state": facts,
        "oracle_receipt": {
            "world_facts": facts.copy(),
            "judgment": "inspect",
            "evidence": cause,
            "next_diagnostic": "inspect the evidence",
        },
        "template_family_id": f"template_{split}",
    }


def test_seed_id_does_not_hide_world_leakage() -> None:
    worlds = [_world(0, "train", "DNS"), _world(1000, "test", "DNS")]
    with pytest.raises(ValueError, match="world facts cross splits"):
        FREEZER["check_release_lineage"]([], worlds, range(1000, 1001))


def test_more_seed_ids_do_not_inflate_independent_cases() -> None:
    worlds = [_world(seed, "test", "DNS") for seed in range(1000, 1300)]
    cases = FREEZER["scenario_cases"]("next_diagnostic", worlds)
    assert len(cases) == 1


def test_multiple_sections_of_one_source_file_are_one_parent() -> None:
    parent = FREEZER["source_parent"]
    first = {
        "source_id": "technical_test",
        "source_revision": "0123456789abcdef0123456789abcdef01234567",
        "source_location": "docs/networking.md#lines=1-20",
    }
    second = {**first, "source_location": "docs/networking.md#lines=21-40"}
    assert parent(first) == parent(second)
