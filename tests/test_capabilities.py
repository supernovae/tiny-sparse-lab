from __future__ import annotations

from pathlib import Path

from sparselab.config.loading import load_config
from sparselab.evaluation.capabilities import capability_card, control_differences


def test_engram_recall_card_is_versioned_and_held_out() -> None:
    card = capability_card("engram-recall-v1")

    assert card.version == 1
    assert len(card.cases) == 16
    assert len({case.prompt for case in card.cases}) == len(card.cases)
    assert all(case.expected for case in card.cases)
    assert len(card.digest) == 64


def test_matched_recall_configs_differ_only_by_memory() -> None:
    dense = load_config(Path("configs/capability_recall_dense_cpu.yaml"))
    ngram = load_config(Path("configs/capability_recall_ngram_cpu.yaml"))

    assert control_differences(dense, ngram) == ()
