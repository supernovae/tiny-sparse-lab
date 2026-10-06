"""Campaign binds native verified direct inputs without inventing a Forge release."""

from types import SimpleNamespace

import pytest
import yaml
from test_experiment_direct_inputs import inputs

from sparselab.campaign.engine import CampaignEngine
from sparselab.experiments.direct_inputs import bind_direct_inputs
from sparselab.experiments.lock import publish_lock, resolve_plan


@pytest.mark.parametrize("source", ["synthetic", "tinystories"])
@pytest.mark.parametrize("mode", ["lock", "reference"])
def test_campaign_verified_direct_inputs(tmp_path, monkeypatch, source, mode):
    _, prepared, run, template = inputs(tmp_path, monkeypatch, source)
    declaration = tmp_path / "experiment.yaml"
    plan = bind_direct_inputs(run, template, prepared, declaration)
    lock = publish_lock(resolve_plan(plan, declaration), tmp_path / "locked")
    original = lock.read_bytes()
    stages = [
        {
            "id": name,
            "kind": "artifact_reference",
            "scope": "tokenizer" if name == "tokenizer" else "model",
            "artifact": plan.artifacts[plan.inputs[kind]].model_dump(mode="json"),
        }
        for name, kind in (("tokenizer", "tokenizer"), ("prepared", "prepared_data"))
    ]
    stages.append(
        {
            "id": "plan",
            "kind": "experiment_plan",
            "scope": "model",
            "requires": ["tokenizer", "prepared"],
            "tokenizer": "tokenizer",
            "prepared": "prepared",
            "source": "experiment.yaml",
            "mode": mode,
            **({"lock": str(lock)} if mode == "reference" else {}),
        }
    )
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        yaml.safe_dump({"campaign_version": 1, "id": "direct", "stages": stages})
    )

    def no_download(*_args, **_kwargs):
        pytest.fail("existing inputs must not download")

    monkeypatch.setattr("sparselab.data.datasets.load_dataset", no_download)
    monkeypatch.setattr("datasets.load_dataset", no_download)
    engine = CampaignEngine(campaign, tmp_path / "work")
    result = engine.apply(allow_uncommitted_declaration=True)
    rows = {row["id"]: row for row in result["stages"]}
    if source == "tinystories" and mode == "lock":
        assert rows["plan"]["state"] == "FAILED"
        assert "retired for new execution" in rows["plan"]["reason"]
        assert lock.read_bytes() == original
        return
    assert rows["plan"]["state"] == "COMPLETE", rows["plan"].get("reason")
    assert lock.read_bytes() == original
    receipt = next((engine.store.root / "receipts/plan").glob("*.json"))
    retained = receipt.read_bytes()
    array = prepared / "train.npy"
    array.write_bytes(array.read_bytes() + b"tampered")
    # Fresh verification rejects changed bytes while retaining immutable evidence.
    with pytest.raises(ValueError):
        engine._lock(rows, "plan")
    assert receipt.read_bytes() == retained
    assert lock.read_bytes() == original


@pytest.mark.parametrize("source", ["synthetic", "tinystories", "snapshot"])
def test_direct_contract_does_not_require_corpus_after_lock_verification(source):
    """Future snapshot routing at the Campaign boundary, not snapshot validation."""
    engine, stage, lock = contract_fixture(source)
    engine._check_lock(stage, {}, lock)


def contract_fixture(source, *, corpus=None, forge=False):
    artifacts = {
        kind: {"kind": kind, "identifier": kind, "sha256": "a" * 64}
        for kind in ("tokenizer", "prepared_data")
    }
    if forge:
        artifacts["release"] = {"kind": "corpus_release", "sha256": "b" * 64}
    engine = object.__new__(CampaignEngine)
    engine.plan = SimpleNamespace(stages=[])
    engine._upstream = lambda _rows, name: {"outputs": [artifacts[name]]}
    stage = SimpleNamespace(
        id="plan", tokenizer="tokenizer", prepared="prepared_data", corpus=corpus
    )
    dataset = SimpleNamespace(
        source=source, corpus_release_path=None, corpus_export_path=None
    )
    lock = SimpleNamespace(
        cells=[
            SimpleNamespace(
                id="single",
                config=SimpleNamespace(dataset=dataset),
                artifacts=artifacts,
            )
        ]
    )
    return engine, stage, lock


@pytest.mark.parametrize("source,forge", [("local_text", False), ("snapshot", True)])
def test_forge_closure_still_requires_campaign_corpus(source, forge):
    engine, stage, lock = contract_fixture(source, forge=forge)
    with pytest.raises(ValueError, match="Forge locked cell requires"):
        engine._check_lock(stage, {}, lock)


def test_synthetic_cannot_claim_campaign_corpus():
    engine, stage, lock = contract_fixture("synthetic", corpus="release")
    with pytest.raises(ValueError, match="synthetic cell cannot claim"):
        engine._check_lock(stage, {}, lock)
