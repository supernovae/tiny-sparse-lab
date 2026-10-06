from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab.config.loading import load_config
from sparselab.iteration import _classification, check_iteration


@pytest.mark.parametrize(
    ("gates", "campaign", "locked", "pilot", "runtime", "expected"),
    [
        ([], None, False, True, True, "SAFE_TO_PREPARE"),
        ([], None, True, True, True, "NEEDS_PILOT"),
        ([], None, True, False, True, "NEEDS_PILOT"),
        ([], None, True, False, False, "READY_TO_DISPATCH"),
        (
            [{"code": "PARENT_INVALID", "reason": "tampered"}],
            None,
            True,
            False,
            False,
            "BLOCKED",
        ),
        (
            [{"code": "SOURCE_INCOMPATIBLE", "reason": "foreign"}],
            {"stages": [{"state": "RUNNING"}]},
            True,
            False,
            False,
            "BLOCKED",
        ),
        (
            [],
            {"next_action": {"action": "approve"}},
            True,
            False,
            False,
            "NEEDS_APPROVAL",
        ),
        ([], {"stages": [{"state": "RUNNING"}]}, True, False, False, "RUNNING"),
        (
            [],
            {
                "next_action": {"action": "apply", "stage": "child"},
                "stages": [{"id": "child", "kind": "experiment_run", "state": "READY"}],
            },
            True,
            True,
            False,
            "NEEDS_PILOT",
        ),
    ],
)
def test_typed_gates(gates, campaign, locked, pilot, runtime, expected):
    assert (
        _classification(
            gates,
            campaign=campaign,
            locked=locked,
            pilot_needed=pilot,
            runtime_needed=runtime,
        )[0]
        == expected
    )


def test_exact_parent_required_without_following_pointer(tmp_path: Path):
    from sparselab.iteration import _parent

    result, snapshot = _parent(tmp_path / "latest.json", {})
    assert not result["valid"]
    assert snapshot is None
    assert result["errors"][0]["field"] == "parent"


def test_lock_inspection_reports_actual_delta_and_source_mismatch(
    monkeypatch, tmp_path: Path
):
    from sparselab import iteration

    source = tmp_path / "lock.json"
    source.write_text(json.dumps({"lock_version": 1}))
    base = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    )
    target = base.model_copy(
        update={
            "training": base.training.model_copy(
                update={"max_steps": base.training.max_steps + 1}
            )
        }
    )
    phase = SimpleNamespace(
        id="child", transition="resume", checkpoint="input", parent=None, at_step=None
    )
    cell = SimpleNamespace(
        id="child:single",
        phase="child",
        config=target,
        artifacts={"parent_checkpoint": {"sha256": "a" * 64}},
    )
    lock = SimpleNamespace(
        id="declared",
        plan_sha256="b" * 64,
        scientific_sha256="c" * 64,
        availability={
            "declaration_source": None,
            "artifacts": {"input": str(tmp_path / "step_1_gen_1")},
        },
        inputs={},
        artifacts={"input": {"sha256": "a" * 64}},
        cells=[cell],
        phases=[phase],
        retention={},
    )
    snapshot = SimpleNamespace(
        config=base.model_dump(mode="json"),
        step=base.training.max_steps,
        tokens_seen=base.training.max_tokens,
        source_identity_sha256="foreign",
        checkpoint_sha256="a" * 64,
    )

    class Store:
        def diagnostics(self):
            return {"hits": 1, "misses": 0, "recorded": 0, "events": []}

    store = Store()
    monkeypatch.setattr(
        iteration,
        "verification_options",
        lambda *a, **k: {"proof_store": store, "verification_mode": "verified_reuse"},
    )
    monkeypatch.setattr(iteration, "open_lock", lambda *a, **k: lock)
    from sparselab.experiments import source_compatibility

    # Even a reviewed historical lock mapping cannot authorize optimizer drift.
    monkeypatch.setattr(
        source_compatibility,
        "source_identities_compatible",
        lambda saved, current: True,
    )
    monkeypatch.setattr(
        iteration,
        "_parent",
        lambda *a, **k: (
            {"valid": True, "full_state": True, "path": str(tmp_path / "step_1_gen_1")},
            snapshot,
        ),
    )
    monkeypatch.setattr(
        iteration,
        "_storage",
        lambda *a, **k: {"estimate": {}, "capacity": [], "location_warnings": []},
    )
    report = check_iteration(source, tmp_path / "step_1_gen_1", work_dir=tmp_path)
    assert report["state"] == "BLOCKED"
    assert report["parent"]["authenticated_source_mapping"] is True
    assert report["parent"]["source_compatible"] is False
    assert report["verification"]["diagnostics"]["hits"] == 1
    assert any(row["path"] == "training.max_steps" for row in report["delta"])
    codes = {gate["code"] for gate in report["missing_gates"]}
    assert {
        "SOURCE_INCOMPATIBLE",
        "CONTINUATION_INCOMPATIBLE",
        "DECLARATION_SOURCE_UNAVAILABLE",
    } <= codes


def test_parent_tokenizer_accepts_relocated_identical_bytes_only(
    monkeypatch, tmp_path: Path
):
    from hashlib import sha256

    from sparselab.iteration import _parent_tokenizer
    from sparselab.training import manifest

    root = tmp_path / "run"
    root.mkdir()
    member = root / "tokenizer.json"
    member.write_bytes(b"tokenizer")
    record = {
        "relative_path": "tokenizer.json",
        "size_bytes": member.stat().st_size,
        "sha256": sha256(member.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(manifest, "read_manifest", lambda path: {"artifacts": [record]})
    generation = root / "checkpoints" / "step_1_gen_1"
    assert _parent_tokenizer(generation) == member
    member.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        _parent_tokenizer(generation)


def test_parent_generation_must_match_native_ingested_selection(
    monkeypatch, tmp_path: Path
):
    from sparselab import iteration
    from sparselab.experiments import binding

    run = tmp_path / "ingested-run"
    supplied = tmp_path / "other-run" / "checkpoints" / "step_4_gen_1"
    phase = SimpleNamespace(selector="terminal", at_step=4, transition="extend_budget")
    calls = []

    def select(selected_run, *, selector, at_step, full_state):
        calls.append((selected_run, selector, at_step, full_state))
        return {
            "checkpoint_path": str(run / "checkpoints" / "step_4_gen_1"),
            "checkpoint_sha256": "a" * 64,
        }

    monkeypatch.setattr(binding, "select_generation", select)
    parent_report = {}
    gates = []
    iteration._selected_parent_gate(
        run, phase, supplied, "a" * 64, parent_report, gates
    )
    assert calls == [(run, "terminal", 4, True)]
    assert gates[0]["code"] == "PARENT_BINDING_MISMATCH"
    gates.clear()
    iteration._selected_parent_gate(
        run,
        phase,
        run / "checkpoints" / "step_4_gen_1",
        "b" * 64,
        parent_report,
        gates,
    )
    assert gates[0]["code"] == "PARENT_BINDING_MISMATCH"


def test_missing_workroot_is_unchanged_by_iteration_check(monkeypatch, tmp_path: Path):
    from sparselab import iteration

    work = tmp_path / "work"
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(
        iteration,
        "_parent",
        lambda *a, **k: (
            {
                "path": "missing",
                "valid": False,
                "errors": [{"field": "checkpoint", "reason": "missing"}],
            },
            None,
        ),
    )
    check_iteration(missing, tmp_path / "step_1_gen_1", work_dir=work, cold_verify=True)
    assert not work.exists()


def test_campaign_inspection_does_not_create_state(monkeypatch, tmp_path: Path):
    from sparselab import iteration
    from sparselab.campaign import engine as campaign_engine

    source = tmp_path / "campaign.json"
    source.write_text('{"campaign_version": 1}')
    work = tmp_path / "work"
    monkeypatch.setattr(
        iteration,
        "load_campaign",
        lambda source: SimpleNamespace(
            id="campaign",
            stages=(
                SimpleNamespace(
                    id="lock",
                    kind="experiment_plan",
                    mode="reference",
                    lock="absent.json",
                ),
            ),
        ),
    )

    class Engine:
        def __init__(self, source, root):
            assert root == work

        def inspect(self, verb):
            assert verb == "explain"
            assert self._verification_options["proof_store"] is None
            return {
                "stages": [{"id": "lock", "kind": "experiment_plan", "state": "READY"}],
                "next_action": {"action": "apply", "stage": "lock"},
            }

    monkeypatch.setattr(campaign_engine, "CampaignEngine", Engine)
    monkeypatch.setattr(
        iteration,
        "_parent",
        lambda *a, **k: (
            {
                "path": "absent",
                "valid": False,
                "errors": [{"field": "checkpoint", "reason": "missing"}],
            },
            None,
        ),
    )
    report = check_iteration(
        source, tmp_path / "step_1_gen_1", work_dir=work, cold_verify=True
    )
    assert report["state"] == "BLOCKED"
    assert not work.exists()


def test_fresh_declaration_without_parent_uses_native_inputs_and_cli(
    monkeypatch, tmp_path: Path, capsys
):
    import argparse

    import yaml
    from test_experiment_direct_inputs import inputs

    from sparselab.experiments.direct_inputs import bind_direct_inputs
    from sparselab.iteration_cli import register_parser

    config, prepared, run, template = inputs(tmp_path, monkeypatch)
    authored = yaml.safe_load(template.read_text())
    authored["phases"][0]["set"] = {"training.max_steps": config.training.max_steps + 1}
    template.write_text(yaml.safe_dump(authored))
    declaration = tmp_path / "fresh.yaml"
    bind_direct_inputs(run, template, prepared, declaration)
    work = tmp_path / "iteration-work"

    def inventory():
        return {
            str(path.relative_to(tmp_path)): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                path.stat().st_ctime_ns,
            )
            for path in tmp_path.rglob("*")
            if path.is_file()
        }

    before = inventory()
    report = check_iteration(declaration, work_dir=work, cold_verify=True)
    assert report["state"] == "SAFE_TO_PREPARE"
    assert report["parent"] == {
        "path": None,
        "status": "NOT_APPLICABLE",
        "valid": None,
        "full_state": None,
        "terminal": None,
        "compatible": None,
    }
    assert report["transition"] == "fresh"
    assert report["declared_delta"] == ["training.max_steps"]
    assert {row["path"] for row in report["delta"]} == {
        "training.max_steps",
        "effective.optimizer_decay_horizon",
        "effective.target_token_exposures",
    }
    assert report["artifacts"]["verification_status"] == "VERIFIED"
    assert report["verification"]["mode"] == "cold"
    assert inventory() == before
    assert not work.exists()

    parser = argparse.ArgumentParser()
    register_parser(parser.add_subparsers(dest="command", required=True))
    parser.set_defaults(work_dir=work)
    args = parser.parse_args(
        ["iteration", "check", str(declaration), "--json", "--cold-verify"]
    )
    args.handler(args)
    assert json.loads(capsys.readouterr().out) == report
    assert inventory() == before


def test_real_budget_iteration_is_read_only_and_rejects_tampered_parent(
    monkeypatch, tmp_path: Path
):
    import yaml
    from test_experiment_direct_inputs import inputs

    from sparselab.experiments.direct_inputs import bind_direct_inputs
    from sparselab.training.checkpoints import CheckpointManager
    from sparselab.training.trainer import train

    config, prepared, run, template = inputs(tmp_path, monkeypatch)
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))
    raw = config.model_dump(mode="json")
    raw["training"].update(max_steps=4, max_tokens=128)
    raw["optimizer"]["decay_steps"] = 4
    config = config.model_validate(raw)
    run.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    train(config, run_id="parent")
    manager = CheckpointManager(config.logging.root_dir / "parent")
    generation = manager._resolve(manager.root / "latest.json")
    parent = manager.load(generation)
    authored = {
        "plan_version": 1,
        "id": "read-only-child",
        "base_run": str(run),
        "artifacts": {
            "parent": {
                "kind": "checkpoint",
                "version": 2,
                "producer": "sparselab",
                "identifier": generation.name,
                "sha256": parent.checkpoint_sha256,
                "path": str(generation),
                "state": "full",
            },
        },
        "phases": [
            {
                "id": "child",
                "transition": "extend_budget",
                "checkpoint": "parent",
                "set": {"training.max_steps": 8, "training.max_tokens": 256},
            }
        ],
    }
    template.write_text(yaml.safe_dump(authored))
    declaration = tmp_path / "child.yaml"
    bind_direct_inputs(run, template, prepared, declaration)

    def inventory():
        return {
            str(path.relative_to(tmp_path)): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                path.stat().st_ctime_ns,
            )
            for path in tmp_path.rglob("*")
            if path.is_file()
        }

    missing_parent = check_iteration(declaration, work_dir=tmp_path, cold_verify=True)
    assert missing_parent["state"] == "BLOCKED"
    assert missing_parent["parent"]["status"] == "NOT_SUPPLIED"
    assert missing_parent["parent"]["path"] is None
    assert missing_parent["parent"]["compatible"] is None
    assert missing_parent["declared_delta"] == []
    assert missing_parent["delta"] == []
    assert "PARENT_REQUIRED" in {
        gate["code"] for gate in missing_parent["missing_gates"]
    }
    assert missing_parent["next_command"].startswith("sparselab --work-dir ")
    before = inventory()
    for cold in (False, True):
        report = check_iteration(
            declaration, generation, work_dir=tmp_path, cold_verify=cold
        )
        assert report["format"] == "sparselab-iteration-check-v1"
        assert report["state"] == "SAFE_TO_PREPARE"
        assert report["parent"]["compatible"] and report["parent"]["full_state"]
        assert report["declared_delta"] == ["training.max_steps", "training.max_tokens"]
        assert inventory() == before
    state = generation / "training_state.pt"
    state.write_bytes(state.read_bytes() + b"tamper")
    blocked = check_iteration(declaration, generation, work_dir=tmp_path)
    assert blocked["state"] == "BLOCKED"
    assert any(gate["code"] == "PARENT_INVALID" for gate in blocked["missing_gates"])


@pytest.mark.parametrize(
    ("status", "ingestion_status", "pending"),
    [
        ("COMPLETE", "COMPLETE", False),
        ("COMPLETE", "PENDING", True),
        ("RUNNING", "PENDING", True),
    ],
)
def test_controller_ingestion_read_is_non_mutating(
    monkeypatch,
    tmp_path: Path,
    status: str,
    ingestion_status: str,
    pending: bool,
):
    from sparselab import iteration
    from sparselab.workers import controller as controller_module

    source = tmp_path / "lock.json"
    source.write_text('{"lock_version": 1}')
    work = tmp_path / "work"
    database = work / "experiments" / "declared" / "controller" / "experiments.sqlite3"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"existing-controller-state")
    base = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    )
    phase = SimpleNamespace(
        id="child",
        transition="resume",
        checkpoint=None,
        parent="parent",
        selector="terminal",
        at_step=base.training.max_steps,
    )
    cell = SimpleNamespace(id="child:single", phase="child", config=base, artifacts={})
    lock = SimpleNamespace(
        id="declared",
        plan_sha256="b" * 64,
        scientific_sha256="c" * 64,
        availability={"declaration_source": None},
        inputs={},
        artifacts={},
        cells=[cell],
        phases=[phase],
        retention={},
    )
    snapshot = SimpleNamespace(
        config=base.model_dump(mode="json"),
        step=base.training.max_steps,
        tokens_seen=base.training.max_tokens,
        source_identity_sha256="foreign",
        checkpoint_sha256="a" * 64,
    )
    row = {
        "spec": {"plan": {"plan_sha256": "b" * 64, "cell_id": "parent:single"}},
        "run_id": "parent-run",
        "status": status,
        "ingestion_status": ingestion_status,
    }

    class Controller:
        def __init__(self, root, *, read_only, **kwargs):
            assert read_only is True
            assert root == database.parent

        def list_experiments(self):
            return [
                {
                    **row,
                    "run_id": "failed-parent",
                    "status": "FAILED",
                    "ingestion_status": "ERROR",
                },
                row,
            ]

    monkeypatch.setattr(controller_module, "Controller", Controller)
    from sparselab.experiments import binding

    monkeypatch.setattr(
        binding,
        "select_generation",
        lambda run, **kwargs: {
            "checkpoint_path": str(tmp_path / "step_1_gen_1"),
            "checkpoint_sha256": "a" * 64,
            "run_id": "parent-run",
        },
    )
    monkeypatch.setattr(iteration, "open_lock", lambda *a, **k: lock)
    monkeypatch.setattr(
        iteration,
        "_parent",
        lambda *a, **k: (
            {"path": "parent", "valid": True, "full_state": True},
            snapshot,
        ),
    )
    monkeypatch.setattr(
        iteration,
        "_storage",
        lambda *a, **k: {
            "estimate": {},
            "capacity": [],
            "location_warnings": [],
        },
    )
    before = {
        item.relative_to(work): item.read_bytes()
        for item in work.rglob("*")
        if item.is_file()
    }
    report = check_iteration(
        source, tmp_path / "step_1_gen_1", work_dir=work, cold_verify=True
    )
    after = {
        item.relative_to(work): item.read_bytes()
        for item in work.rglob("*")
        if item.is_file()
    }
    assert after == before
    assert (
        "PARENT_INGESTION_PENDING" in {item["code"] for item in report["missing_gates"]}
    ) == pending


def test_changed_authored_declaration_blocks_verified_lock(monkeypatch, tmp_path: Path):
    from sparselab import iteration
    from sparselab.recovery import provenance

    source = tmp_path / "lock.json"
    source.write_text('{"lock_version": 1}')
    authored = tmp_path / "plan.yaml"
    authored.write_text("changed: true")
    lock = SimpleNamespace(
        id="declared",
        plan_sha256="b" * 64,
        scientific_sha256="c" * 64,
        availability={
            "declaration_source": str(authored),
            "declaration_hashes": {"plan.yaml": "a" * 64},
        },
        inputs={},
        artifacts={},
        cells=[],
        phases=[],
        retention={},
    )
    monkeypatch.setattr(iteration, "open_lock", lambda *a, **k: lock)
    monkeypatch.setattr(provenance, "declaration_paths", lambda *a: [authored])
    monkeypatch.setattr(provenance, "repository_root", lambda *a: None)
    monkeypatch.setattr(
        iteration,
        "_parent",
        lambda *a, **k: (
            {
                "path": "missing",
                "valid": False,
                "errors": [{"field": "parent", "reason": "missing"}],
            },
            None,
        ),
    )
    report = check_iteration(
        source, tmp_path / "step_1_gen_1", work_dir=tmp_path / "work", cold_verify=True
    )
    assert "DECLARATION_IDENTITY_CHANGED" in {
        item["code"] for item in report["missing_gates"]
    }
    assert report["state"] == "BLOCKED"


def test_campaign_next_cell_rejects_single_other_parent_match(
    monkeypatch, tmp_path: Path
):
    from sparselab import iteration
    from sparselab.campaign import engine as campaign_engine

    source = tmp_path / "campaign.json"
    source.write_text('{"campaign_version": 1}')
    base = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    )
    phase = SimpleNamespace(
        id="other",
        transition="resume",
        parent=None,
        checkpoint="external",
        selector=None,
        at_step=None,
    )
    cell = SimpleNamespace(
        id="other:single",
        phase="other",
        config=base,
        artifacts={},
    )
    lock = SimpleNamespace(
        id="declared",
        plan_sha256="b" * 64,
        scientific_sha256="c" * 64,
        availability={"artifacts": {"external": str(tmp_path / "step_1_gen_1")}},
        inputs={},
        artifacts={"external": {"sha256": "a" * 64}},
        cells=[cell],
        phases=[phase],
        retention={},
    )
    stages = (
        SimpleNamespace(
            id="plan",
            kind="experiment_plan",
            mode="reference",
            lock="lock.json",
            source="authored.yaml",
        ),
        SimpleNamespace(
            id="child", kind="experiment_run", cell="child:single", runtime="runtime"
        ),
    )
    monkeypatch.setattr(
        iteration,
        "load_campaign",
        lambda path: SimpleNamespace(id="campaign", stages=stages),
    )
    monkeypatch.setattr(iteration, "open_lock", lambda *a, **k: lock)
    monkeypatch.setattr(iteration, "_source_closure_matches", lambda *a, **k: True)
    monkeypatch.setattr(
        iteration,
        "load_plan",
        lambda *a: SimpleNamespace(phases=()),
    )
    monkeypatch.setattr(
        iteration,
        "_storage",
        lambda *a, **k: {"estimate": {}, "capacity": [], "location_warnings": []},
    )
    monkeypatch.setattr(
        iteration,
        "_parent",
        lambda *a, **k: (
            {"path": str(tmp_path / "step_1_gen_1"), "valid": True, "full_state": True},
            SimpleNamespace(
                config=base.model_dump(mode="json"),
                step=base.training.max_steps,
                tokens_seen=base.training.max_tokens,
                source_identity_sha256="foreign",
                checkpoint_sha256="a" * 64,
            ),
        ),
    )

    class Engine:
        def __init__(self, source, work_dir):
            pass

        def inspect(self, command):
            return {
                "stages": [
                    {
                        "id": "plan",
                        "kind": "experiment_plan",
                        "state": "COMPLETE",
                        "availability": {"path": str(tmp_path / "lock.json")},
                    },
                    {"id": "child", "kind": "experiment_run", "state": "READY"},
                ],
                "next_action": {"action": "apply", "stage": "child"},
            }

    monkeypatch.setattr(campaign_engine, "CampaignEngine", Engine)
    report = check_iteration(
        source,
        tmp_path / "step_1_gen_1",
        work_dir=tmp_path / "work",
        cold_verify=True,
    )
    assert report["state"] == "BLOCKED"
    assert "PARENT_BINDING_REQUIRED" in {
        gate["code"] for gate in report["missing_gates"]
    }
    assert report["cell"] is None
