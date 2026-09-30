"""PILOT cases are causal-world diagnostics, never promotion evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sparselab.corpus.pipeline import _scenario_v2

FREEZERS = Path(__file__).resolve().parents[1] / "experiments/research/devmind-pretrain-v1"
sys.path.insert(0, str(FREEZERS))
spec = importlib.util.spec_from_file_location("freeze_pilot_cards", FREEZERS / "freeze_pilot_cards.py")
assert spec and spec.loader
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def scenario(seed: int, generator: str = "filesystem_judgment_v2") -> dict:
    split = "train" if seed < 800 else "validation" if seed < 900 else "test"
    row = _scenario_v2(generator, seed, f"{generator}_{split}", f"{generator}_template_{split}", "pilot")
    row["split"] = split
    return row


def test_pilot_rejects_repeated_causal_state_and_seed_leakage() -> None:
    first, second = scenario(900), scenario(901)
    assert pilot.causal_hash(first) != pilot.causal_hash(second)
    second["world_state"] = dict(first["world_state"])
    second["oracle_receipt"]["world_facts"] = second["world_state"]
    with pytest.raises(ValueError, match="repeated causal world facts"):
        pilot.check_worlds([first, second])
    invalid = scenario(900)
    invalid["split"] = "train"
    with pytest.raises(ValueError, match="split partition"):
        pilot.check_worlds([invalid])
    invalid = scenario(900)
    invalid["world_state"]["fixture_id"] = "synthetic-identity"
    invalid["oracle_receipt"]["world_facts"] = invalid["world_state"]
    with pytest.raises(ValueError, match="identity cannot count"):
        pilot.check_worlds([invalid])


def test_pilot_refuses_source_parent_leakage() -> None:
    doc = {
        "source_id": "technical", "source_revision": "a" * 40,
        "source_location": "docs/example.md#lines=1-4",
        "content_sha256": "b" * 64, "source_family": "technical_train",
        "split": "train",
    }
    test_doc = {**doc, "source_location": "docs/example.md#lines=5-8", "content_sha256": "c" * 64, "source_family": "technical_test", "split": "test"}
    with pytest.raises(ValueError, match="source parent or content crosses splits"):
        pilot.check_release_lineage([doc, test_doc], [scenario(900)])


def test_pilot_receipt_discloses_effective_n_correlations_and_no_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = tmp_path / "release-id"
    release.mkdir()
    (release / "manifest.json").write_text(json.dumps({"release_id": release.name}))
    for name in ("documents", "generations"):
        (release / f"{name}.jsonl").write_text("")
    rows = [scenario(seed, generator) for generator in pilot.V2_GENERATORS for seed in (0, 800, 900, 901)]
    (release / "scenarios.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(pilot, "verify_release", lambda _: {"release_id": release.name})
    output = tmp_path / "pilot-cards"
    receipt = pilot.freeze(release, output)
    assert receipt["promotion_eligible"] is False
    assert receipt["cards"]["verification"]["effective_n"] == 2
    assert receipt["cards"]["verification"]["underpowered_for_promotion"] is True
    assert "operational_risk" in receipt["cards"]["abstention_unknown"]["shared_world_with"]
    assert receipt["cards"]["source_grounded_qa"]["effective_n"] == 0
    assert receipt["test_worlds"]["platform_fault_v2"] == {
        "distinct_causal_states": 2, "fewer_than_100": True
    }
    assert "PILOT diagnostic only" in json.loads((output / "verification.json").read_text())["limitations"]
    assert (output / "verification.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        pilot.freeze(release, output)
