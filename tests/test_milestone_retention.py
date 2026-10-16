"""Synthetic tensors and counters only: no model construction, forward or optimizer update."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sparselab.config.loading import load_config
from sparselab.config.models import CheckpointRetention, RunConfig
from sparselab.model.inspection import named_tensor_inventory
from sparselab.training import retention
from sparselab.training.checkpoints import CheckpointManager, TrainingSnapshot
from sparselab.training.optimizer import learning_rate_for_step, schedule_payload
from sparselab.training.trainer import _checkpoint_due


def policy(**updates):
    return CheckpointRetention.model_validate(
        {
            "milestones": [
                {"name": "early", "step": 2, "committed_targets": 16},
                {"name": "crossed", "step": 4, "committed_targets": 25},
                {"name": "final", "step": 10, "committed_targets": 80},
            ],
            "terminal_step": 10,
            "terminal_targets": 80,
            "max_generations": 7,
            "max_generation_bytes": 100_000,
            "max_retained_bytes": 700_000,
            "max_peak_bytes": 900_000,
            "reserve_free_bytes": 0,
            **updates,
        }
    )


def config(retain):
    raw = load_config(Path("configs/smoke_cpu.yaml")).model_dump(mode="json")
    raw["model"].update(
        vocab_size=260, hidden_dim=8, num_layers=1, num_heads=2, ffn_dim=16
    )
    raw["training"].update(max_steps=10, max_tokens=80)
    raw["optimizer"]["warmup_steps"] = 0
    raw["checkpoint"].update(
        keep_periodic=False, every_steps=100, retention=retain.model_dump(mode="json")
    )
    return RunConfig.model_validate(raw)


def snapshot(step, retain, tokens=None):
    resolved = config(retain)
    inventory = named_tensor_inventory(resolved.model, resolved.attention)
    weights = {
        name: torch.zeros(spec.shape)
        for name, spec in inventory.items()
        if spec.alias_of is None
    }
    for name, spec in inventory.items():
        if spec.alias_of is not None:
            weights[name] = weights[spec.alias_of]
    names = {
        i: name
        for i, (name, spec) in enumerate(
            pair
            for pair in inventory.items()
            if pair[1].trainable and pair[1].alias_of is None
        )
    }
    settings = resolved.optimizer
    rate = (
        learning_rate_for_step(step, 10, 0, settings.peak, settings.floor)
        if step
        else settings.peak
    )
    groups = [
        {
            "params": [
                i
                for i, name in names.items()
                if (len(inventory[name].shape) >= 2) == decay
            ],
            "lr": rate,
            "betas": settings.betas,
            "eps": settings.eps,
            "weight_decay": settings.weight_decay if decay else 0.0,
        }
        for decay in (True, False)
    ]
    rng = np.random.RandomState(7).get_state()
    return TrainingSnapshot(
        model=weights,
        optimizer={"state": {}, "param_groups": groups},
        schedule=schedule_payload(resolved, step),
        step=step,
        tokens_seen=step * 8 if tokens is None else tokens,
        cursor=(0, step),
        config=resolved.model_dump(mode="json"),
        run_id="fake",
        optimizer_parameter_names=names,
        source_identity_sha256="a" * 64,
        rng={
            "python": random.Random(7).getstate(),
            "numpy_kind": rng[0],
            "numpy_keys": torch.from_numpy(rng[1].copy()),
            "numpy_pos": rng[2],
            "numpy_has_gauss": rng[3],
            "numpy_cached_gaussian": rng[4],
            "torch": torch.get_rng_state(),
            "device_type": "cpu",
            "device_index": 0,
            "device_rng": None,
        },
    )


def manager(root, retain=None, **kwargs):
    return CheckpointManager(root, keep_periodic=False, retention=retain, **kwargs)


def report(root):
    return json.loads((root / "checkpoints/retention.json").read_text())


def generations(root):
    return {
        p.name: {f.name: f.read_bytes() for f in p.iterdir()}
        for p in (root / "checkpoints").glob("step_*_gen_*")
    }


def test_exact_crossed_coincident_roles_resume_and_full_state_readback(tmp_path):
    retain = policy()
    writer = manager(tmp_path, retain)
    milestones = {}
    for step in range(11):
        if step == 6:
            writer = manager(tmp_path)  # Persisted policy protects a resumed writer.
        record = writer.save(snapshot(step, retain), 0.5 if step == 1 else 2.0)
        if step in (2, 4):
            milestones[step] = record.manifest_sha256
    evidence = report(tmp_path)
    assert evidence["milestones_complete"] and evidence["terminal_reached"]
    rows = {row["step"]: row for row in evidence["retained"]}
    assert set(rows) == {1, 2, 4, 9, 10}
    assert rows[10]["roles"] == ["latest", "milestone:final", "terminal"]
    assert rows[1]["roles"] == ["best"]
    assert rows[9]["roles"] == ["previous"]
    assert rows[4]["tokens_seen"] == 32 and rows[4]["roles"] == ["milestone:crossed"]
    for step, row in rows.items():
        path = Path(row["verified_location"])
        assert writer.verify(path).valid
        loaded = writer.load(path)
        assert loaded.step == step and loaded.tokens_seen == step * 8
        assert row["resume_level"] == "full"
        assert row["source_identity_sha256"] == "a" * 64
        if step in milestones:
            assert row["manifest_sha256"] == milestones[step]
    assert len(generations(tmp_path)) == 5
    assert evidence["retained_bytes"] == sum(
        row["full_directory_bytes"] for row in rows.values()
    )


def test_cadence_crosses_both_boundaries_once():
    cfg = config(policy())
    prior = {"step": 3, "tokens": 24}
    assert not _checkpoint_due(cfg, 3, 28, 0, prior)
    assert not _checkpoint_due(cfg, 4, 24, 0, prior)
    assert _checkpoint_due(cfg, 4, 32, 0, prior)
    assert not _checkpoint_due(cfg, 5, 40, 0, {"step": 4, "tokens": 32})


@pytest.mark.parametrize(
    "change",
    [
        {"max_generations": 2},
        {"max_retained_bytes": 1},
        {"max_peak_bytes": 1},
        {"max_generation_bytes": True},
        {"milestones": [{"name": "late", "step": 11, "committed_targets": 1}]},
    ],
)
def test_infeasible_plan_refuses(change):
    with pytest.raises(ValueError):
        policy(**change)


@pytest.mark.parametrize(
    "point",
    [
        "before_finalized_rename",
        "after_finalized_rename",
        "before_projection_publication",
        "after_retention_publication_before_prune",
    ],
)
def test_interrupted_publication_preserves_recovery_and_milestones(tmp_path, point):
    retain = policy()
    writer = manager(tmp_path, retain)
    for step in range(4):
        writer.save(snapshot(step, retain))
    before = generations(tmp_path)

    def crash(at):
        if at == point:
            raise RuntimeError("fake publication interruption")

    broken = manager(tmp_path, retain, fault_injector=crash)
    with pytest.raises(RuntimeError, match="fake publication"):
        broken.save(snapshot(4, retain))
    after = generations(tmp_path)
    assert all(after[name] == value for name, value in before.items())
    recovered = manager(tmp_path)
    assert recovered.reconcile().record.step in (3, 4)
    assert any(
        "milestone:early" in row["roles"] for row in report(tmp_path)["retained"]
    )


def test_storage_refusal_before_serialization_preserves_files(tmp_path, monkeypatch):
    retain = policy()
    writer = manager(tmp_path, retain)
    writer.save(snapshot(0, retain))
    before = generations(tmp_path)
    monkeypatch.setattr(
        retention.shutil, "disk_usage", lambda path: SimpleNamespace(free=1)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("serialized despite insufficient staging capacity")

    monkeypatch.setattr(writer, "_save_weights", forbidden)
    with pytest.raises(ValueError, match="insufficient free"):
        writer.save(snapshot(1, retain))
    assert generations(tmp_path) == before
    assert (
        json.loads((tmp_path / "checkpoints/retention-failure.json").read_text())[
            "status"
        ]
        == "refused"
    )


def test_oversized_staging_never_replaces_recovery(tmp_path, monkeypatch):
    retain = policy()
    writer = manager(tmp_path, retain)
    writer.save(snapshot(0, retain))
    before = generations(tmp_path)
    original = writer._save_weights

    def oversized(destination, *args):
        result = original(destination, *args)
        (destination / "oversized-fixture").write_bytes(
            b"x" * retain.max_generation_bytes
        )
        return result

    monkeypatch.setattr(writer, "_save_weights", oversized)
    with pytest.raises(ValueError, match="per-generation bytes"):
        writer.save(snapshot(1, retain))
    assert generations(tmp_path) == before


def test_missing_required_checkpoint_and_policy_change_refuse(tmp_path):
    retain = policy()
    writer = manager(tmp_path, retain)
    record = writer.save(snapshot(2, retain))
    with pytest.raises(ValueError, match="cannot change"):
        manager(tmp_path, policy(reserve_free_bytes=1)).reconcile()
    generation = tmp_path / "checkpoints" / record.relative_path
    generation.rename(tmp_path / "outside")
    with pytest.raises(ValueError, match="missing or unverified"):
        manager(tmp_path).reconcile()


def test_default_serialization_and_storage_bounds():
    from sparselab.experiments.storage import checkpoint_generation_bounds

    baseline = load_config(Path("configs/smoke_cpu.yaml"))
    assert "retention" not in baseline.checkpoint.model_dump(mode="json")
    cfg = config(policy())
    bounds = checkpoint_generation_bounds(cfg)
    assert bounds["retained_upper"] <= 7
    assert bounds["peak_generations"] <= 8


@pytest.mark.parametrize("failure", ["tamper", "symlink", "peak"])
def test_invalid_or_overfull_inventory_preserves_every_generation(tmp_path, failure):
    retain = policy()
    writer = manager(tmp_path, retain)
    record = writer.save(snapshot(2, retain))
    checkpoint = tmp_path / "checkpoints"
    if failure == "tamper":
        (checkpoint / record.relative_path / "training_state.pt").write_bytes(
            b"tampered"
        )
    elif failure == "symlink":
        (checkpoint / "foreign").symlink_to(tmp_path / "outside")
    else:
        (checkpoint / "unowned-space-fixture").write_bytes(b"x" * retain.max_peak_bytes)
    before = generations(tmp_path)
    with pytest.raises(ValueError, match="unverified|unsafe|peak"):
        manager(tmp_path).reconcile()
    assert generations(tmp_path) == before


def test_explicit_plan_cannot_be_retrofitted_to_historical_checkpoints(tmp_path):
    snap = snapshot(0, policy())
    del snap.config["checkpoint"]["retention"]
    plain = CheckpointManager(tmp_path)
    plain.save(snap)
    before = generations(tmp_path)
    with pytest.raises(ValueError, match="before the first checkpoint"):
        manager(tmp_path, policy()).reconcile()
    assert generations(tmp_path) == before


def test_terminal_and_best_survive_later_duplicate_generation(tmp_path):
    retain = policy()
    writer = manager(tmp_path, retain)
    first = writer.save(snapshot(10, retain), 0.5)
    writer.save(snapshot(10, retain), 1.0)
    writer.save(snapshot(10, retain), 1.0)
    rows = report(tmp_path)["retained"]
    original = next(
        row for row in rows if row["manifest_sha256"] == first.manifest_sha256
    )
    assert "terminal" in original["roles"] and "best" in original["roles"]
    assert "milestone:final" in original["roles"]
    assert len(rows) == 3


def test_declared_plan_forecast_uses_actual_root_and_finite_headroom(tmp_path):
    retain = policy()
    writer = manager(tmp_path, retain)
    writer.save(snapshot(0, retain))
    before = retention.footprint(tmp_path / "checkpoints")
    writer.save(snapshot(1, retain))
    evidence = report(tmp_path)
    admission = evidence["write_admission"]
    assert admission["observed_root_bytes"] >= before
    assert (
        admission["projected_peak_bytes"]
        == admission["observed_root_bytes"] + retain.max_generation_bytes
    )
    assert evidence["planned_retained_bytes_upper"] == 700_000
    assert (
        evidence["storage_semantics"]
        == "admission_and_postwrite_checks_not_filesystem_quota"
    )


def test_retention_directory_durability_failure_prevents_pruning(tmp_path, monkeypatch):
    import os
    import stat

    retain = policy()
    writer = manager(tmp_path, retain)
    for step in range(4):
        writer.save(snapshot(step, retain))
    before = generations(tmp_path)
    original_publish = writer._publish_retention
    original_sync = os.fsync

    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("fixture directory sync failure")
        return original_sync(fd)

    def publish(path, payload):
        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", fail_directory)
            original_publish(path, payload)

    monkeypatch.setattr(writer, "_publish_retention", publish)
    with pytest.raises(OSError, match="directory sync"):
        writer.save(snapshot(4, retain))
    after = generations(tmp_path)
    assert all(after[name] == value for name, value in before.items())


def test_policy_mismatch_refuses_before_any_reconciliation_prune(tmp_path, monkeypatch):
    retain = policy()
    writer = manager(tmp_path, retain)
    writer.save(snapshot(0, retain))
    before = generations(tmp_path)
    changed = snapshot(1, retain)
    del changed.config["checkpoint"]["retention"]

    def forbidden():
        pytest.fail("reconciled before rejecting changed policy")

    monkeypatch.setattr(writer, "reconcile", forbidden)
    with pytest.raises(ValueError, match="policies differ"):
        writer.save(changed)
    assert generations(tmp_path) == before


def test_terminal_with_unreached_target_threshold_is_explicit(tmp_path):
    retain = policy()
    writer = manager(tmp_path, retain)
    writer.save(snapshot(10, retain, tokens=24))
    evidence = report(tmp_path)
    assert evidence["terminal_reached"]
    assert not evidence["milestones_complete"]
    assert [row["name"] for row in evidence["pending_milestones"]] == [
        "crossed",
        "final",
    ]
    assert "terminal" in evidence["retained"][0]["roles"]


def test_native_child_resume_inherits_exact_milestones_without_copying(tmp_path):
    retain = policy()
    parent = tmp_path / "parent"
    writer = manager(parent, retain)
    for step in range(6):
        writer.save(snapshot(step, retain), 2.0)
    before = generations(parent)
    references = retention.inherited_milestones(parent, retain, 5, 40)
    child = tmp_path / "child"
    resumed = manager(child, retain, inherited_milestones=references)
    for step in range(5, 11):
        resumed.save(snapshot(step, retain), 2.0)
    evidence = report(child)
    inherited = [row for row in evidence["retained"] if not row["local"]]
    assert {row["step"] for row in inherited} == {2, 4}
    for row in inherited:
        assert Path(row["verified_location"]).parent == parent / "checkpoints"
        assert resumed.load(Path(row["verified_location"])).step == row["step"]
    assert not any(
        "milestone:early" in row["roles"]
        for row in evidence["retained"]
        if row["local"]
    )
    assert generations(parent) == before
    assert not list((child / "checkpoints").glob("step_00000002*"))
    assert manager(child).reconcile().record.step == 10
    assert len(report(child)["retained"]) == 5


@pytest.mark.parametrize("case", ["changed-boundary", "past-selection", "missing-file"])
def test_child_resume_refuses_unavailable_or_changed_parent_milestone(tmp_path, case):
    retain = policy()
    parent = tmp_path / "parent"
    writer = manager(parent, retain)
    writer.save(
        snapshot(3, retain)
    )  # First local observation crossed the early boundary.
    if case == "changed-boundary":
        changed = retain.model_dump(mode="json")
        changed["milestones"][0]["committed_targets"] = 15
        with pytest.raises(ValueError, match="differs from parent"):
            retention.inherited_milestones(
                parent, CheckpointRetention.model_validate(changed), 3, 24
            )
    elif case == "past-selection":
        with pytest.raises(ValueError, match="before selected"):
            retention.inherited_milestones(parent, retain, 2, 16)
    else:
        refs = retention.inherited_milestones(parent, retain, 3, 24)
        target = Path(refs[0]["verified_location"]) / "training_state.pt"
        target.write_bytes(b"corrupted inherited fixture")
        child = tmp_path / "child"
        with pytest.raises(ValueError, match="inherited milestone"):
            manager(child, retain, inherited_milestones=refs).save(snapshot(3, retain))
        assert not generations(child)


def test_startup_storage_includes_declared_peak_before_execution(tmp_path):
    import math

    from sparselab.experiments.storage import checkpoint_generation_bounds
    from sparselab.model.inspection import inspection_report
    from sparselab.workspace_preflight import training_storage_checks

    cfg = config(policy())
    raw = cfg.model_dump(mode="json")
    del raw["checkpoint"]["retention"]
    baseline = RunConfig.model_validate(raw)
    prior = sum(
        row.projected_bytes
        for row in training_storage_checks(baseline, work_dir=tmp_path)
    )
    current = sum(
        row.projected_bytes for row in training_storage_checks(cfg, work_dir=tmp_path)
    )
    old_checkpoint = checkpoint_generation_bounds(baseline)[
        "peak_generations"
    ] * math.ceil(1.25 * int(inspection_report(cfg)["estimated_checkpoint_bytes"]))
    assert current - prior == cfg.checkpoint.retention.max_peak_bytes - old_checkpoint


def test_inherited_selection_counters_must_match_verified_checkpoint(tmp_path):
    retain = policy()
    parent = tmp_path / "parent"
    writer = manager(parent, retain)
    writer.save(snapshot(3, retain))
    refs = retention.inherited_milestones(parent, retain, 3, 24)
    refs[0]["step"] = 2
    with pytest.raises(ValueError, match="counters differ"):
        manager(tmp_path / "child", retain, inherited_milestones=refs).save(
            snapshot(3, retain)
        )
