"""Offline native snapshot Campaign lifecycle, including continuation and contrast.

Only acquisition is substituted. Tokenizer, packing, locks, workers, checkpoint
ingestion, heldout evaluation and descriptive generation use native operations.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from dataset_fixtures import offline_hub_identity

from sparselab.campaign.engine import CampaignEngine
from sparselab.config.loading import load_config
from sparselab.config.models import DatasetConfig, RunConfig, TokenizerTrainConfig
from sparselab.data import sources
from sparselab.experiments.lock import open_lock
from sparselab.training.manifest import sha256_file


def write_yaml(path, value):
    path.write_text(yaml.safe_dump(value))


def rows(result):
    return {row["id"]: row for row in result["stages"]}


@pytest.fixture
def snapshot_campaign(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    revision = "a" * 40
    monkeypatch.setattr(
        sources,
        "_hub_identity",
        offline_hub_identity,
    )

    def stream(source, split, cache_dir, **_options):
        return iter(
            {"body": f"{split} story {i}: the small fox explored a quiet forest."}
            for i in range(8 if split == "train" else 4)
        )

    monkeypatch.setattr(sources, "_stream", stream)
    source = tmp_path / "source.yaml"
    write_yaml(
        source,
        {
            "repo_id": "fixture/campaign-stories",
            "revision": revision,
            "config": "default",
            "splits": {"train": "training", "validation": "heldout"},
            "text_field": "body",
            "attribution": "Offline Campaign fixture",
            "license": "MIT",
            "selection": {"mode": "exhaustion"},
            "resources": {
                "max_source_records": 100,
                "max_text_bytes": 100000,
                "max_record_bytes": 10000,
                "max_work_bytes": 10000000,
                "min_free_bytes": 0,
            },
        },
    )
    lock = sources.lock_source(source, tmp_path / "source.lock.json", tmp_path / "hub")
    snapshot = tmp_path / "snapshot"
    dataset = DatasetConfig(
        source="snapshot",
        revision=revision,
        license="MIT",
        cache_dir=tmp_path / "cache",
        train_path=snapshot / "train.jsonl",
        validation_path=snapshot / "validation.jsonl",
        source_manifest_path=snapshot / "manifest.json",
        train_max_documents=8,
        validation_max_documents=4,
        train_max_tokens=1024,
        validation_max_tokens=512,
    )
    tokenizer = TokenizerTrainConfig(
        schema_version=1,
        vocab_size=260,
        min_frequency=1,
        max_documents=8,
        output_dir=tmp_path / "tokenizer",
        dataset=dataset,
    )
    write_yaml(tmp_path / "tokenizer.yaml", tokenizer.model_dump(mode="json"))
    base = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    )
    raw = base.model_dump(mode="json")
    raw["dataset"] = dataset.model_dump(mode="json")
    raw["tokenizer"]["path"] = str(tokenizer.output_dir / "tokenizer.json")
    raw["model"]["vocab_size"] = 260
    raw["training"].update(max_steps=2, max_tokens=64)
    raw["optimizer"].update(warmup_steps=0, decay_steps=2)
    raw["checkpoint"]["every_steps"] = 1
    raw["evaluation"].update(every_steps=1, max_batches=1)
    raw["logging"]["root_dir"] = str(tmp_path / "runs")
    config = RunConfig.model_validate(raw)
    write_yaml(tmp_path / "run.yaml", config.model_dump(mode="json"))
    write_yaml(
        tmp_path / "suite.yaml",
        {
            "evaluation_suite_version": 1,
            "id": "fixed-heldout",
            "evaluations": [{"id": "heldout", "role": "gate", "kind": "heldout_lm"}],
        },
    )
    write_yaml(
        tmp_path / "panel.yaml",
        {
            "generation_panel_version": 1,
            "id": "fixed-descriptive",
            "role": "descriptive_not_quality_gate",
            "prompts": ["The fox"],
            "decoder": {"temperature": 0, "top_k": 0, "max_new_tokens": 2, "seed": 7},
            "checkpoint_selection": "same immutable generation as heldout evaluation",
        },
    )
    write_yaml(
        tmp_path / "template.yaml",
        {
            "plan_version": 1,
            "id": "snapshot-lifecycle",
            "base_run": "run.yaml",
            "evaluation_suite": "suite.yaml",
            "execution": {"backend": "cpu"},
            "phases": [
                {"id": "baseline", "transition": "fresh"},
                {
                    "id": "child",
                    "transition": "extend_budget",
                    "parent": "baseline",
                    "selector": "terminal",
                    "at_step": 2,
                    "set": {"training.max_steps": 4, "training.max_tokens": 128},
                },
                {"id": "width", "transition": "fresh", "set": {"model.hidden_dim": 32}},
            ],
        },
    )
    stages = [
        {
            "id": "snapshot",
            "kind": "dataset_snapshot",
            "scope": "corpus",
            "lock": str(lock),
            "output": str(snapshot),
            "cache_dir": str(tmp_path / "hub"),
        },
        {
            "id": "tokenizer",
            "kind": "tokenizer_train",
            "scope": "tokenizer",
            "config": "tokenizer.yaml",
            "snapshot": "snapshot",
            "requires": ["snapshot"],
        },
        {
            "id": "prepared",
            "kind": "data_prepare",
            "scope": "model",
            "config": "run.yaml",
            "snapshot": "snapshot",
            "tokenizer": "tokenizer",
            "requires": ["snapshot", "tokenizer"],
        },
        {
            "id": "plan",
            "kind": "experiment_plan",
            "scope": "model",
            "mode": "bind",
            "source": "template.yaml",
            "prepared": "prepared",
            "tokenizer": "tokenizer",
            "requires": ["prepared", "tokenizer"],
        },
        {
            "id": "runtime",
            "kind": "runtime_acceptance",
            "scope": "runtime",
            "plan": "plan",
            "requires": ["plan"],
        },
        {
            "id": "gate",
            "kind": "approval",
            "scope": "model",
            "requires": ["runtime"],
            "bind": ["snapshot", "tokenizer", "prepared", "plan", "runtime"],
        },
    ]
    for phase in ("baseline", "child", "width"):
        requires = ["plan", "runtime", "gate"]
        if phase == "child":
            requires.append("collect-baseline")
        stages.extend(
            [
                {
                    "id": f"run-{phase}",
                    "kind": "experiment_run",
                    "scope": "model",
                    "requires": requires,
                    "plan": "plan",
                    "runtime": "runtime",
                    "cell": f"{phase}:single",
                },
                {
                    "id": f"collect-{phase}",
                    "kind": "experiment_collect",
                    "scope": "evaluation",
                    "requires": ["plan", f"run-{phase}"],
                    "plan": "plan",
                    "run": f"run-{phase}",
                },
                {
                    "id": f"eval-{phase}",
                    "kind": "evaluation",
                    "scope": "evaluation",
                    "requires": [f"collect-{phase}"],
                    "collect": f"collect-{phase}",
                    "suite": "suite.yaml",
                },
                {
                    "id": f"panel-{phase}",
                    "kind": "generation_panel",
                    "scope": "evaluation",
                    "requires": [f"collect-{phase}", f"eval-{phase}", "runtime"],
                    "collect": f"collect-{phase}",
                    "evaluation": f"eval-{phase}",
                    "runtime": "runtime",
                    "panel": "panel.yaml",
                },
            ]
        )
    campaign = tmp_path / "campaign.yaml"
    write_yaml(
        campaign, {"campaign_version": 1, "id": "snapshot-lifecycle", "stages": stages}
    )
    return campaign, stream


def test_native_snapshot_baseline_continuation_width_collect_eval_panel(
    snapshot_campaign, tmp_path
):
    from sparselab.evaluation.panel import verify_panel_result
    from sparselab.evaluation.suite import verify_evaluation_index

    campaign, _ = snapshot_campaign
    retained = {}

    def after_commit(stage_id, state):
        if stage_id == "collect-baseline":
            current = rows(state)
            generation = (
                Path(current["run-baseline"]["availability"]["path"])
                / "checkpoints"
                / current[stage_id]["measurements"]["generation"]
            )
            retained.update(
                {
                    path: sha256_file(path)
                    for path in generation.rglob("*")
                    if path.is_file()
                }
            )

    engine = CampaignEngine(campaign, tmp_path / "work", after_commit=after_commit)
    waiting = rows(engine.apply(allow_uncommitted_declaration=True))
    assert waiting["gate"]["state"] == "AWAITING_APPROVAL", waiting
    lock_path = Path(waiting["plan"]["availability"]["path"])
    lock_bytes = lock_path.read_bytes()
    locked = open_lock(lock_path)
    assert {cell.id for cell in locked.cells} == {
        "baseline:single",
        "child:single",
        "width:single",
    }
    engine.approve("gate", note="offline wiring acceptance only")
    result = rows(
        engine.apply(
            execute_runs=True, max_wait_seconds=120, allow_uncommitted_declaration=True
        )
    )
    assert all(row["state"] == "COMPLETE" for row in result.values()), result
    assert retained and all(
        sha256_file(path) == digest for path, digest in retained.items()
    )
    assert lock_path.read_bytes() == lock_bytes
    for phase in ("baseline", "child", "width"):
        selected = result[f"collect-{phase}"]["measurements"]
        evaluation = verify_evaluation_index(
            Path(result[f"eval-{phase}"]["availability"]["path"])
        )
        panel = verify_panel_result(
            Path(result[f"panel-{phase}"]["availability"]["path"])
        )
        assert (
            evaluation["checkpoint_sha256"]
            == panel["checkpoint_sha256"]
            == selected["sha256"]
        )
        assert panel["role"] == "descriptive_not_quality_gate"
    baseline = result["run-baseline"]["outputs"][0]["identifier"]
    child_manifest = json.loads(
        (
            Path(result["run-child"]["availability"]["path"]) / "manifest.json"
        ).read_text()
    )
    assert child_manifest["parent_run_id"] == baseline
    assert rows(engine.apply(resume=True, allow_uncommitted_declaration=True)) == result


def test_snapshot_acquisition_resume_keeps_declaration_and_prior_attempt(
    snapshot_campaign, tmp_path, monkeypatch
):
    campaign, original_stream = snapshot_campaign
    declaration = yaml.safe_load(campaign.read_text())
    declaration["stages"] = declaration["stages"][:1]
    write_yaml(campaign, declaration)
    before = campaign.read_bytes()

    def interrupted(source, split, cache_dir, **_options):
        yield next(original_stream(source, split, cache_dir))
        raise OSError("fixture acquisition interruption")

    monkeypatch.setattr(sources, "_stream", interrupted)
    engine = CampaignEngine(campaign, tmp_path / "work")
    failed = rows(engine.apply(allow_uncommitted_declaration=True))["snapshot"]
    assert failed["state"] == "INTERRUPTED"
    assert not list((engine.store.root / "receipts/snapshot").glob("*.json"))
    assert (
        engine.apply(allow_uncommitted_declaration=True)["next_action"]["action"]
        == "resume"
    )
    monkeypatch.setattr(sources, "_stream", original_stream)
    completed = rows(engine.apply(resume=True, allow_uncommitted_declaration=True))[
        "snapshot"
    ]
    assert completed["state"] == "COMPLETE", completed
    assert completed["attempts"][0]["state"] == "INTERRUPTED"
    assert campaign.read_bytes() == before
    # Existing output can be reused from another Campaign state root, without acquisition.
    monkeypatch.setattr(
        sources, "snapshot_source", lambda *a, **kw: pytest.fail("reacquisition")
    )
    reused = rows(
        CampaignEngine(campaign, tmp_path / "other-work").apply(
            allow_uncommitted_declaration=True
        )
    )
    assert reused["snapshot"]["outputs"] == completed["outputs"]


@pytest.mark.parametrize("source", ["tinystories", "local_stories"])
def test_campaign_runtime_rejects_new_legacy_execution(source):
    engine = object.__new__(CampaignEngine)
    engine.plan = SimpleNamespace(stages=[])
    cell = SimpleNamespace(
        id="single", config=SimpleNamespace(dataset=SimpleNamespace(source=source))
    )
    engine._lock = lambda *_args: SimpleNamespace(cells=[cell])
    stage = SimpleNamespace(id="runtime", kind="runtime_acceptance", plan="plan")
    with pytest.raises(ValueError, match="retired for new execution"):
        CampaignEngine.dispatch.__wrapped__(
            engine, stage, {"runtime": {"stage_input_sha256": "a" * 64}}, 0
        )


@pytest.mark.parametrize("kind", ["tokenizer_train", "data_prepare"])
def test_preparation_rejects_legacy_before_compute(kind, monkeypatch):
    from sparselab.campaign import preparation

    config = SimpleNamespace(dataset=SimpleNamespace(source="tinystories"))
    monkeypatch.setattr(preparation, "config_for", lambda *args: config)
    monkeypatch.setattr(
        "sparselab.data.tokenizer.train_tokenizer",
        lambda *a, **kw: pytest.fail("legacy training"),
    )
    monkeypatch.setattr(
        "sparselab.data.packing.prepare_data",
        lambda *a, **kw: pytest.fail("legacy packing"),
    )
    with pytest.raises(ValueError, match="retired for new execution"):
        preparation.dispatch(SimpleNamespace(), SimpleNamespace(kind=kind), {})


def test_snapshot_upstream_mismatch_blocks_tokenizer_before_training(
    snapshot_campaign, tmp_path, monkeypatch
):
    campaign, _ = snapshot_campaign
    document = yaml.safe_load((tmp_path / "tokenizer.yaml").read_text())
    document["dataset"]["source_manifest_path"] = str(tmp_path / "other/manifest.json")
    write_yaml(tmp_path / "tokenizer.yaml", document)
    monkeypatch.setattr(
        "sparselab.data.tokenizer.train_tokenizer",
        lambda *a, **kw: pytest.fail("mismatched snapshot trained"),
    )
    result = rows(
        CampaignEngine(campaign, tmp_path / "work").apply(
            allow_uncommitted_declaration=True
        )
    )
    assert result["snapshot"]["state"] == "COMPLETE"
    assert result["tokenizer"]["state"] == "FAILED"
    assert "upstream snapshot" in result["tokenizer"]["reason"]
    assert result["prepared"]["state"] == "BLOCKED"


def test_completed_snapshot_tampering_preserves_receipt_and_rejects_reuse(
    snapshot_campaign, tmp_path
):
    campaign, _ = snapshot_campaign
    document = yaml.safe_load(campaign.read_text())
    document["stages"] = document["stages"][:1]
    write_yaml(campaign, document)
    engine = CampaignEngine(campaign, tmp_path / "work")
    result = rows(engine.apply(allow_uncommitted_declaration=True))
    assert result["snapshot"]["state"] == "COMPLETE"
    receipt = next((engine.store.root / "receipts/snapshot").glob("*.json"))
    retained = receipt.read_bytes()
    train = tmp_path / "snapshot/train.jsonl"
    train_bytes = train.read_bytes()
    train.write_bytes(train_bytes + b"tampered")
    with pytest.raises(ValueError):
        engine.inspect("next")
    assert receipt.read_bytes() == retained
    train.write_bytes(train_bytes)
    recovered = rows(engine.inspect("status"))
    assert recovered["snapshot"]["state"] == "COMPLETE"


def test_projection_rechecks_snapshot_after_commit_before_downstream_compute(
    snapshot_campaign, tmp_path
):
    campaign, _ = snapshot_campaign
    document = yaml.safe_load(campaign.read_text())
    document["stages"] = document["stages"][:2]
    write_yaml(campaign, document)
    retained = {}

    def after_commit(stage_id, state):
        if stage_id == "snapshot":
            receipt = next((engine.store.root / "receipts/snapshot").glob("*.json"))
            retained[receipt] = receipt.read_bytes()
            train = tmp_path / "snapshot/train.jsonl"
            train.write_bytes(train.read_bytes() + b"tampered")

    engine = CampaignEngine(campaign, tmp_path / "work", after_commit=after_commit)
    with pytest.raises(ValueError):
        engine.apply(allow_uncommitted_declaration=True)
    assert retained and all(
        path.read_bytes() == original for path, original in retained.items()
    )
    assert not (tmp_path / "tokenizer/tokenizer.json").exists()
