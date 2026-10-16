"""Retention guard regressions; only the final test executes tiny synthetic CPU updates."""

import json
import sqlite3
from types import SimpleNamespace

import pytest
from test_milestone_retention import (
    config,
    generations,
    manager,
    policy,
    report,
    snapshot,
)

from sparselab.config.models import RunConfig
from sparselab.training import retention


@pytest.mark.parametrize(
    "case, message",
    [
        ("names", "names must be unique"),
        ("periodic", "keep_periodic"),
        ("step", "terminal"),
        ("targets", "terminal"),
    ],
)
def test_config_cross_guards(case, message):
    raw = config(policy()).model_dump(mode="json")
    block = raw["checkpoint"]["retention"]
    if case == "names":
        block["milestones"][1]["name"] = "early"
    elif case == "periodic":
        raw["checkpoint"]["keep_periodic"] = True
    elif case == "step":
        raw["training"]["max_steps"] += 1
    else:
        raw["training"]["max_tokens"] += 1
    with pytest.raises(ValueError, match=message):
        RunConfig.model_validate(raw)


def parent_refs(root):
    parent = root / "parent"
    manager(parent, policy()).save(snapshot(2, policy()))
    return parent, retention.inherited_milestones(parent, policy(), 2, 16)


def test_changed_inherited_binding_refuses(tmp_path):
    _, refs = parent_refs(tmp_path)
    child = tmp_path / "child"
    manager(child, policy(), inherited_milestones=refs).save(snapshot(3, policy()))
    before = generations(child)
    refs[0]["roles"] = ["milestone:crossed"]
    with pytest.raises(ValueError, match="selections cannot change"):
        manager(child, policy(), inherited_milestones=refs).reconcile()
    assert generations(child) == before


@pytest.mark.parametrize("case", ["absent", "malformed", "digest"])
def test_parent_selection_evidence_refuses(tmp_path, case):
    parent, refs = parent_refs(tmp_path)
    path = parent / "checkpoints/retention.json"
    if case == "absent":
        path.unlink()
        with pytest.raises(ValueError, match="lacks verified parent"):
            retention.inherited_milestones(parent, policy(), 2, 16)
    elif case == "malformed":
        path.write_text("not json")
        with pytest.raises(ValueError):
            retention.inherited_milestones(parent, policy(), 2, 16)
    else:
        refs[0]["manifest_sha256"] = "b" * 64
        with pytest.raises(ValueError, match="missing or unverified"):
            manager(tmp_path / "child", policy(), inherited_milestones=refs).save(
                snapshot(3, policy())
            )


@pytest.mark.parametrize("inherited", [False, True])
def test_admission_exact_free_and_peak_boundaries(tmp_path, monkeypatch, inherited):
    root = tmp_path / "root"
    root.mkdir()
    extra = 12000 if inherited else 0
    monkeypatch.setattr(
        retention, "inherited_rows", lambda path: [{"full_directory_bytes": extra}]
    )
    retain = policy(reserve_free_bytes=1234)
    required = retain.max_peak_bytes - extra + retain.reserve_free_bytes
    monkeypatch.setattr(
        retention.shutil, "disk_usage", lambda path: SimpleNamespace(free=required - 1)
    )
    with pytest.raises(ValueError, match="insufficient free"):
        retention.admit_write(root, retain)
    monkeypatch.setattr(
        retention.shutil, "disk_usage", lambda path: SimpleNamespace(free=required)
    )
    assert retention.admit_write(root, retain)["required_free_bytes"] == required
    (root / "occupied").write_bytes(
        b"x" * (retain.max_peak_bytes - retain.max_generation_bytes - extra)
    )
    assert (
        retention.admit_write(root, retain)["projected_peak_bytes"]
        == retain.max_peak_bytes
    )
    with (root / "occupied").open("ab") as output:
        output.write(b"x")
    with pytest.raises(ValueError, match="cannot fit another"):
        retention.admit_write(root, retain)


def test_inherited_bytes_count_in_observed_peak_and_generation_bound(tmp_path):
    _, refs = parent_refs(tmp_path)
    child = tmp_path / "child"
    writer = manager(child, policy(), inherited_milestones=refs)
    writer.reconcile()
    inherited = retention.inherited_rows(writer.root)
    size = inherited[0]["full_directory_bytes"]
    assert size > 1
    root_bytes = retention.footprint(writer.root)
    (writer.root / "occupied").write_bytes(
        b"x" * (policy().max_peak_bytes - root_bytes - size + 1)
    )
    with pytest.raises(ValueError, match="observed checkpoint bytes exceed"):
        retention.plan(writer.root, [], policy())
    (writer.root / "occupied").unlink()
    with pytest.raises(ValueError, match="inherited checkpoint exceeds"):
        retention.plan(writer.root, [], policy(max_generation_bytes=size - 1))


@pytest.mark.parametrize("inherited", [False, True])
def test_staged_peak_refusal_preserves_previous_and_removes_stage(
    tmp_path, monkeypatch, inherited
):
    refs = parent_refs(tmp_path)[1] if inherited else []
    child = tmp_path / "child"
    writer = manager(child, policy(), inherited_milestones=refs)
    writer.save(snapshot(3, policy()))
    before = generations(child)
    original = writer._save_weights
    extra = sum(
        row["full_directory_bytes"] for row in retention.inherited_rows(writer.root)
    )

    def grow_after_admission(destination, *args):
        result = original(destination, *args)
        # Concurrent unrelated growth after admission; own generation remains small.
        target = policy().max_peak_bytes - extra + 1
        occupied = target - retention.footprint(writer.root)
        (writer.root / "unowned-growth").write_bytes(b"x" * occupied)
        return result

    monkeypatch.setattr(writer, "_save_weights", grow_after_admission)
    with pytest.raises(ValueError, match="staged checkpoint exceeds"):
        writer.save(snapshot(4, policy()))
    assert generations(child) == before
    assert not [
        p for p in writer.root.iterdir() if p.is_dir() and p.name.startswith(".step_")
    ]
    assert (writer.root / "unowned-growth").exists()


def test_interrupted_unselected_prune_stops_reconcile_and_save(tmp_path):
    retain = policy()
    writer = manager(tmp_path, retain)
    for step in range(3):
        writer.save(snapshot(step, retain))

    def crash(point):
        if point == "after_retention_publication_before_prune":
            doomed = report(tmp_path)["prune"]
            if doomed:
                (writer.root / doomed[0] / "training_state.pt").unlink()
                raise RuntimeError("interrupted prune fixture")

    with pytest.raises(RuntimeError, match="interrupted prune"):
        manager(tmp_path, retain, fault_injector=crash).save(snapshot(3, retain))
    before = generations(tmp_path)
    for action in (
        lambda: manager(tmp_path).reconcile(),
        lambda: manager(tmp_path, retain).save(snapshot(4, retain)),
    ):
        with pytest.raises(
            ValueError, match="refuses unverified checkpoint generations"
        ):
            action()
        assert generations(tmp_path) == before


def test_reconcile_does_not_reuse_previous_save_admission(tmp_path):
    writer = manager(tmp_path, policy())
    writer.save(snapshot(2, policy()))
    assert report(tmp_path)["write_admission"] is not None
    writer.reconcile()
    assert report(tmp_path)["write_admission"] is None


def test_forecast_exact_events_and_capacity():
    from sparselab.experiments.storage import checkpoint_generation_bounds

    raw = config(policy()).model_dump(mode="json")
    raw["checkpoint"]["every_steps"] = 4
    raw["evaluation"]["every_steps"] = 2
    bounds = checkpoint_generation_bounds(RunConfig.model_validate(raw))
    assert bounds["scheduled_steps"] == 2
    assert bounds["validation_steps"] == 4
    assert (
        bounds["upper"] == 11
    )  # initial, terminal, 3 milestones, 2 cadence, 4 validation
    assert bounds["retained_upper"] == 7
    assert bounds["peak_generations"] == 8


def test_native_trainer_policy_inheritance_and_tracking(tmp_path):
    from test_training import config as tiny_config

    from sparselab.training.trainer import train

    raw = tiny_config(tmp_path).model_dump(mode="json")
    raw["checkpoint"].update(
        keep_periodic=False,
        retention=policy(
            terminal_step=12,
            terminal_targets=384,
            milestones=[
                {"name": "early", "step": 3, "committed_targets": 96},
                {"name": "crossed", "step": 6, "committed_targets": 193},
                {"name": "final", "step": 12, "committed_targets": 384},
            ],
            max_generation_bytes=1000000,
            max_retained_bytes=7000000,
            max_peak_bytes=9000000,
        ).model_dump(mode="json"),
    )
    cfg = RunConfig.model_validate(raw)
    train(cfg, run_id="full")
    train(cfg, run_id="parent", stop_after_step=5)
    parent = cfg.logging.root_dir / "parent"
    before = {
        str(p.relative_to(parent)): p.read_bytes()
        for p in parent.rglob("*")
        if p.is_file()
    }
    train(cfg, run_id="child", resume=parent / "checkpoints/latest.json")
    assert {
        str(p.relative_to(parent)): p.read_bytes()
        for p in parent.rglob("*")
        if p.is_file()
    } == before
    child = cfg.logging.root_dir / "child"
    refs = json.loads((child / "checkpoints/retention-inherited.json").read_text())[
        "references"
    ]
    assert len(refs) == 1 and refs[0]["step"] == 3
    assert refs[0]["roles"] == ["milestone:early"]
    for name in ("full", "child"):
        evidence = report(cfg.logging.root_dir / name)
        assert evidence["milestones_complete"] and evidence["terminal_reached"]
        selected = {
            role: row["step"] for row in evidence["retained"] for role in row["roles"]
        }
        assert {
            key: selected[key]
            for key in ("milestone:early", "milestone:crossed", "milestone:final")
        } == {"milestone:early": 3, "milestone:crossed": 7, "milestone:final": 12}
    databases = list(cfg.logging.root_dir.rglob("experiments.sqlite3"))
    assert len(databases) == 1
    with sqlite3.connect(databases[0]) as db:
        rows = db.execute(
            "SELECT run_id, step, payload_json FROM events WHERE kind = 'checkpoint_retention'"
        ).fetchall()
    for name in ("full", "parent", "child"):
        events = [
            json.loads(payload) for run_id, step, payload in rows if run_id == name
        ]
        assert events
        assert all(event["status"] == "verified" for event in events)
    assert any(run_id == "child" and step == 12 for run_id, step, payload in rows)
