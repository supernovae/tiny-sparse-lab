"""Authored Forge provenance survives sealed pilots and worker continuations."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from sparselab.experiments.lock import publish_lock, resolve_plan
from sparselab.experiments.plan import ExperimentPlan
from sparselab.experiments.prepare import prepare_plan
from sparselab.staging import (
    _inventory,
    _read_sealed,
    _seal,
    materialize_prepared_inputs,
    stage,
    verify_prepared_inputs,
)
from sparselab.training.checkpoints import CheckpointManager
from sparselab.training.manifest import canonical_json, sha256_file
from sparselab.training.metrics import ExperimentStore
from sparselab.training.trainer import train
from sparselab.workers import artifacts
from sparselab.workers.bundles import (
    install_dispatch_bundle,
    materialize_dispatch_bundle,
    verify_portable_corpus_binding,
)
from sparselab.workers.controller import Controller
from sparselab.workers.execution import (
    _continuation_checkpoint,
    _continuation_run,
    _final_artifacts,
    _receipt_payload,
)
from sparselab.workers.models import AttemptReceipt, WorkerDefinition


@pytest.fixture
def sealed_forge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    source = tmp_path / "source"
    source.mkdir()
    shutil.copytree("corpora/devmind-sample-v0", source / "project")
    base = yaml.safe_load(Path("configs/runtime_smoke_cpu.yaml").read_text())
    base["training"].update(max_steps=4, max_tokens=128)
    base["optimizer"]["warmup_steps"] = 0
    base["staging"].update(smoke_steps=1, warmup_steps=2)
    base["checkpoint"]["every_steps"] = 1
    base["evaluation"].update(every_steps=2, max_batches=1)
    base["logging"]["root_dir"] = str(tmp_path / "worker" / "runs")
    (source / "run.yaml").write_text(yaml.safe_dump(base))
    plan = ExperimentPlan.model_validate(
        {
            "plan_version": 1,
            "id": "forge-offline-pilots",
            "base_run": "run.yaml",
            "corpus_variants": [
                {"id": "tiny", "project": "project/corpus.yaml", "vocab_size": 300}
            ],
            "axes": [
                {
                    "name": "corpus",
                    "choices": [
                        {"label": "tiny", "set": {"inputs.corpus_variant": "tiny"}}
                    ],
                }
            ],
            "execution": {"backend": "cpu"},
        }
    )
    declaration = source / "plan.json"
    declaration.write_text(plan.model_dump_json())
    prepared = prepare_plan(plan, declaration, source / "workspace")
    locked = resolve_plan(plan, declaration, prepared=prepared)
    publish_lock(locked, source / "workspace")
    config = locked.cells[0].config
    inputs = materialize_prepared_inputs(config, tmp_path / "sealed-inputs")
    source.rename(tmp_path / "unavailable-original-source")
    assert not config.dataset.corpus_export_path.exists()
    assert not config.tokenizer.path.exists()
    return config, inputs


def _install(controller, spec, worker, destination):
    bundle_path, bundle = controller._dispatch_bundle(spec)
    attachments = {
        f"assets/{item.sha256}": bundle_path / item.relative_path
        for item in bundle.files
    }
    install_dispatch_bundle(worker.root, bundle, attachments, check_only=False)
    materialize_dispatch_bundle(worker.root, bundle.digest(), destination)
    return bundle


def test_authored_forge_pilots_ingestion_and_child_resume(
    sealed_forge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, inputs = sealed_forge
    # This real pilot is the first consumer after all original inputs disappear.
    local_stage = stage(
        config, tmp_path / "local-stage", through="warmup", prepared_inputs=inputs
    )
    expected = verify_portable_corpus_binding(config, local_stage / "assets")
    controller = Controller(tmp_path / "controller")
    submission = controller.submit(config, stage_bundle=local_stage)
    attempt = controller.store.attempt_by_run(submission.run_id)
    spec = controller._model("ExperimentSpec", attempt["spec"])
    worker = WorkerDefinition(
        worker_id="forge-worker",
        name="forge-worker",
        transport="local",
        python=Path(sys.executable),
        root=tmp_path / "worker",
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    materialized = tmp_path / "worker-inputs"
    bundle = _install(controller, spec, worker, materialized)
    worker_stage = stage(
        config,
        tmp_path / "worker-stage",
        through="warmup",
        prepared_inputs=materialized,
    )
    train(
        config,
        run_id=submission.run_id,
        stage_bundle=worker_stage,
        worker_id=worker.worker_id,
        experiment_id=submission.experiment_id,
        attempt_id=submission.attempt_id,
        dispatch_metadata={
            "spec_digest": spec.digest(),
            "bundle_digest": bundle.digest(),
        },
        stop_after_step=1,
    )
    run = config.logging.root_dir / submission.run_id
    assert verify_portable_corpus_binding(config, run) == expected
    controller.store.metrics.import_records(
        ExperimentStore(config.logging.root_dir).export_records()["records"]
    )
    raw = _receipt_payload(
        worker,
        {
            "attempt_id": submission.attempt_id,
            "run_id": submission.run_id,
            "experiment_id": submission.experiment_id,
            "spec_digest": spec.digest(),
            "bundle_digest": bundle.digest(),
        },
    )
    raw.update(state="INTERRUPTED", artifacts=_final_artifacts(worker, raw))
    receipt = AttemptReceipt.model_validate(raw)

    def local_transfer(_worker, _receipt, item, destination, **kwargs):
        shutil.copyfile(run / item.relative_path.removeprefix("run/"), destination)

    monkeypatch.setattr(artifacts, "_download", local_transfer)
    controller.store.assign(submission.attempt_id, worker.worker_id)
    controller._reconcile_receipt(attempt, raw)
    artifacts.ingest_attempt_artifacts(
        worker,
        receipt,
        controller.root,
        spec=spec,
        bundle=bundle,
        records=controller.store.metrics,
    )
    controller.store.mark_ingestion_complete(submission.attempt_id)
    parent = controller.root / submission.run_id
    assert verify_portable_corpus_binding(config, parent) == expected
    for name in (*expected["files"], "binding.json"):
        assert (parent / "corpus" / name).read_bytes() == (
            materialized / "assets/corpus" / name
        ).read_bytes()
    child = controller.resume(submission.run_id, worker=worker.name)
    child_attempt = controller.store.attempt_by_run(child.run_id)
    child_spec = controller._model("ExperimentSpec", child_attempt["spec"])
    child_inputs = tmp_path / "child-inputs"
    child_bundle = _install(controller, child_spec, worker, child_inputs)
    child_stage = stage(
        config, tmp_path / "child-stage", through="warmup", prepared_inputs=child_inputs
    )
    embedded = _continuation_run(child_inputs, "RESUMED")
    train(
        config,
        run_id=child.run_id,
        stage_bundle=child_stage,
        resume=_continuation_checkpoint(embedded),
        worker_id=worker.worker_id,
        experiment_id=child.experiment_id,
        attempt_id=child.attempt_id,
        dispatch_metadata={
            "spec_digest": child_spec.digest(),
            "bundle_digest": child_bundle.digest(),
        },
    )
    child_run = config.logging.root_dir / child.run_id
    assert verify_portable_corpus_binding(config, child_run) == expected
    child_manifest = json.loads((child_run / "manifest.json").read_bytes())
    assert child_manifest["continuation_kind"] == "RESUMED"
    assert child_manifest["parent_run_id"] == submission.run_id
    assert (
        child_manifest["checkpoint_sha256"]
        == child_bundle.continuation.checkpoint_sha256
    )
    checkpoint = CheckpointManager(child_run).load(
        child_run / "checkpoints/latest.json"
    )
    assert checkpoint.step == 4
    assert checkpoint.tokens_seen == 128


@pytest.mark.parametrize(
    "member",
    [
        "binding.json",
        "manifest.json",
        "report.json",
        "license-report.json",
        "audit.json",
        "export.json",
        "tokenizer.json",
        "tokenizer_manifest.json",
    ],
)
@pytest.mark.parametrize("mutation", ["change", "remove"])
def test_sealed_forge_provenance_tamper(
    sealed_forge, member: str, mutation: str
) -> None:
    config, inputs = sealed_forge
    assets = inputs / "assets"
    path = (
        assets / member
        if member.startswith("tokenizer")
        else assets / "corpus" / member
    )
    if mutation == "change":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        path.unlink()
    with pytest.raises(ValueError):
        verify_prepared_inputs(inputs, config)


def test_relocated_forge_rejects_rebound_export_identity(sealed_forge) -> None:
    config, inputs = sealed_forge
    assets = inputs / "assets"
    export_path = assets / "corpus/export.json"
    export = json.loads(export_path.read_bytes())
    export["release_manifest_sha256"] = "0" * 64
    export_path.write_bytes(canonical_json(export) + b"\n")
    binding_path = assets / "corpus/binding.json"
    binding = json.loads(binding_path.read_bytes())
    binding["files"]["export.json"] = sha256_file(export_path)
    binding["corpus_export"]["export_sha256"] = sha256_file(export_path)
    binding_path.write_bytes(canonical_json(binding) + b"\n")
    with pytest.raises(ValueError, match="inner identity"):
        verify_portable_corpus_binding(config, assets)
    # Re-sign the outer inventory to exercise semantic verification rather than
    # stopping at the first changed file hash.
    inventory = _read_sealed(inputs / "inputs.json")
    inventory.pop("sha256")
    inventory["artifacts"] = _inventory(assets)
    _seal(inputs / "inputs.json", inventory)
    with pytest.raises(ValueError, match="inner identity"):
        verify_prepared_inputs(inputs, config)
