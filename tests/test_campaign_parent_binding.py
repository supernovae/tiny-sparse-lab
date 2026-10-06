"""Campaign distinguishes run-manifest and checkpoint continuation identities."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab.campaign.engine import CampaignEngine
from sparselab.training.manifest import MANIFEST_VERSION, canonical_json


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "run_id",
        "manifest_digest",
        "manifest_size",
        "manifest_path",
        "checkpoint_parent",
        "checkpoint_digest",
        "tampered_run",
        "missing_identity",
    ],
)
def test_continuation_parent_uses_run_manifest_identity(tmp_path, damage):
    run = tmp_path / "parent"
    generation = run / "checkpoints/step_00000002_gen_000003"
    generation.mkdir(parents=True)
    # A supported historical run-manifest shape needs no executable-source fixture.
    manifest = {"manifest_version": MANIFEST_VERSION, "run_id": "parent"}
    run_digest = hashlib.sha256(canonical_json(manifest)).hexdigest()
    manifest_path = run / "manifest.json"
    manifest_path.write_bytes(
        canonical_json({**manifest, "sha256": run_digest}) + b"\n"
    )
    checkpoint_digest = "c" * 64
    checkpoint = {"sha256": checkpoint_digest, "manifest_sha256": run_digest}
    # Generation manifests deliberately do not contain run_id.
    identity = SimpleNamespace(
        relative_path="continuation/parent_manifest.json",
        sha256=run_digest,
        size_bytes=manifest_path.stat().st_size,
    )
    continuation = SimpleNamespace(
        parent_run_id="parent",
        artifact_identity=identity,
        checkpoint_sha256=checkpoint_digest,
    )
    if damage == "run_id":
        continuation.parent_run_id = "another-parent"
    elif damage == "manifest_digest":
        identity.sha256 = checkpoint_digest
    elif damage == "manifest_size":
        identity.size_bytes += 1
    elif damage == "manifest_path":
        identity.relative_path = "continuation/wrong.json"
    elif damage == "checkpoint_parent":
        checkpoint["manifest_sha256"] = "d" * 64
    elif damage == "checkpoint_digest":
        continuation.checkpoint_sha256 = "e" * 64
    elif damage == "tampered_run":
        manifest_path.write_text(
            json.dumps({**manifest, "run_id": "tampered", "sha256": run_digest})
        )
    elif damage == "missing_identity":
        continuation.artifact_identity = None
    (generation / "manifest.json").write_text(json.dumps(checkpoint))
    if damage is None:
        CampaignEngine._check_continuation_parent(generation, continuation)
    else:
        with pytest.raises(
            ValueError, match="manifest hash mismatch|controller parent"
        ):
            CampaignEngine._check_continuation_parent(generation, continuation)


@pytest.mark.parametrize("transition", ["resume", "extend_budget", "promote"])
def test_existing_child_attempt_reconciles_distinct_parent_identities(
    tmp_path, monkeypatch, transition
):
    from sparselab.config.loading import load_config
    from sparselab.training.manifest import config_sha256
    from sparselab.workers.models import ExperimentSpec, current_required_versions

    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    )
    config_digest = config_sha256(config.model_dump(mode="json"))
    parent = tmp_path / "parent"
    generation = parent / "checkpoints/step_00000002_gen_000003"
    generation.mkdir(parents=True)
    manifest = {"manifest_version": MANIFEST_VERSION, "run_id": "parent"}
    run_digest = hashlib.sha256(canonical_json(manifest)).hexdigest()
    parent_manifest = parent / "manifest.json"
    parent_manifest.write_bytes(
        canonical_json({**manifest, "sha256": run_digest}) + b"\n"
    )
    checkpoint_digest = "c" * 64
    (generation / "manifest.json").write_text(
        json.dumps(
            {
                "sha256": checkpoint_digest,
                "manifest_sha256": run_digest,
            }
        )
    )
    spec = ExperimentSpec(
        experiment_id="child",
        config=config,
        config_sha256=config_digest,
        bound_worker="cpu",
        required_versions=current_required_versions(),
        source_identity_sha256="a" * 64,
        dispatch_bundle_digest="b" * 64,
        plan={
            "plan_id": "fixture",
            "plan_sha256": "d" * 64,
            "scientific_sha256": "e" * 64,
            "cell_id": "child:single",
            "phase_id": "child",
            "coordinate": {},
            "config_sha256": config_digest,
            "parent_checkpoint_sha256": checkpoint_digest,
        },
        continuation={
            "kind": "PROMOTED" if transition == "promote" else "RESUMED",
            "parent_run_id": "parent",
            "checkpoint_sha256": checkpoint_digest,
            "budget_extension": transition == "extend_budget",
            "artifact_identity": {
                "relative_path": "continuation/parent_manifest.json",
                "sha256": run_digest,
                "size_bytes": parent_manifest.stat().st_size,
            },
        },
    )
    request = {"plan": spec.plan.model_dump(mode="json"), transition: generation}
    # Request construction authenticates the generation upstream. Exercise the
    # actual Campaign reconciliation path with a schema-validated worker spec.
    monkeypatch.setattr(
        "sparselab.experiments.cli.locked_cell_request", lambda *a, **kw: request
    )
    engine = object.__new__(CampaignEngine)
    engine._upstream = lambda *a: {"availability": {}}
    attempt = {"spec": spec.model_dump(mode="json"), "run_id": "child"}
    controller = SimpleNamespace(
        root=tmp_path / "controller", list_experiments=lambda: [attempt]
    )
    lock = SimpleNamespace(
        plan_sha256=spec.plan.plan_sha256,
        cells=[
            SimpleNamespace(
                id="child:single", config=config, config_sha256=config_digest
            )
        ],
    )
    stage = SimpleNamespace(cell="child:single", runtime="runtime")
    assert engine._attempts(controller, lock, stage, {}) == [attempt]


@pytest.mark.parametrize(
    "damage", [None, "missing_run", "nested_run", "not_ingested", "ambiguous"]
)
def test_locked_child_request_uses_one_ingested_controller_attempt(
    tmp_path, monkeypatch, damage
):
    from sparselab.experiments.cli import locked_cell_request

    parent = {
        "spec": {"plan": {"plan_sha256": "a" * 64, "cell_id": "baseline:single"}},
        "run_id": "parent-run",
        "status": "COMPLETE",
        "ingestion_status": "COMPLETE",
    }
    if damage in {"missing_run", "nested_run"}:
        parent.pop("run_id")
        if damage == "nested_run":
            parent["latest_attempt"] = {"run_id": "untrusted-nested-run"}
    elif damage == "not_ingested":
        parent["ingestion_status"] = "PENDING"
    attempts = [parent, dict(parent)] if damage == "ambiguous" else [parent]
    controller = SimpleNamespace(
        root=tmp_path / "controller", list_experiments=lambda: attempts
    )
    phase = SimpleNamespace(
        id="child",
        parent="baseline",
        selector="terminal",
        at_step=2,
        transition="extend_budget",
    )
    cell = SimpleNamespace(
        id="child:single",
        phase="child",
        coordinate={},
        config=object(),
        config_sha256="b" * 64,
    )
    locked = SimpleNamespace(
        id="fixture", phases=[phase], plan_sha256="a" * 64, scientific_sha256="c" * 64
    )
    binding = tmp_path / "binding.json"
    binding.write_text("{}")
    checkpoint = controller.root / "parent-run/checkpoints/step_00000002_gen_000003"
    calls = []

    def bind(workspace, **kwargs):
        calls.append(kwargs)
        assert kwargs["parent_run"] == controller.root / "parent-run"
        assert kwargs["read_only"] is True
        assert kwargs["full_state"] is True
        return binding, {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": "d" * 64,
        }

    monkeypatch.setattr("sparselab.experiments.binding.bind_generation", bind)
    if damage is not None:
        with pytest.raises(ValueError, match="ingested parent"):
            locked_cell_request(
                locked, cell, "cpu", tmp_path, controller, read_only=True
            )
        assert not calls
    else:
        request = locked_cell_request(
            locked, cell, "cpu", tmp_path, controller, read_only=True
        )
        assert request["extend_budget"] == checkpoint
        assert request["plan"]["parent_checkpoint_sha256"] == "d" * 64
        assert len(calls) == 1
