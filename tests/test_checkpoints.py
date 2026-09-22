from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from sparselab.config.loading import load_config
from sparselab.model.transformer import DenseLM
from sparselab.training.checkpoints import (
    CheckpointManager,
    TrainingSnapshot,
    load_checkpoint,
    save_checkpoint,
)
from sparselab.training.manifest import canonical_json
from sparselab.training.optimizer import learning_rate_for_step, make_optimizer
from sparselab.training.trainer import _safe_rng_state


def state() -> dict[str, object]:
    return {
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {},
        "schedule": {"max_steps": 2},
        "step": 1,
        "tokens_seen": 4,
        "cursor": (0, 1),
        "config": {"schema_version": 1},
    }


def test_checkpoint_requires_matching_manifest_hash(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, state())
    loaded = load_checkpoint(path)
    assert loaded["step"] == 1
    assert loaded["schedule"] == {"max_steps": 2}
    record = json.loads(path.with_suffix(".json").read_text())
    latest = json.loads((path.parent / "latest.json").read_text())
    assert record["format_version"] == 1
    assert latest == record
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(path)


def test_checkpoint_requires_manifest(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(state(), path)
    with pytest.raises(ValueError, match="manifest missing"):
        load_checkpoint(path)


def test_pointer_digest_must_bind_selected_generation(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(_snapshot(1))
    pointer = tmp_path / "checkpoints/latest.json"
    payload = json.loads(pointer.read_text())
    payload["manifest_sha256"] = "0" * 64
    pointer.write_text(json.dumps(payload))
    report = manager.verify(pointer)
    assert not report.valid
    assert any(error["field"] == "pointer" for error in report.errors)


def test_checkpoint_verification_rejects_tampered_weight_file(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    record = manager.save(_snapshot(1))
    generation = tmp_path / "checkpoints" / record.relative_path
    weights = next(generation.glob("*.safetensors"))
    weights.write_bytes(weights.read_bytes() + b"tampered")
    assert not manager.verify(generation).valid


def test_checkpoint_allows_symlinked_ancestor_but_not_member(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    manager = CheckpointManager(alias / "run")
    snapshot = _snapshot(1)
    record = manager.save(snapshot)
    generation = alias / "run/checkpoints" / record.relative_path
    loaded = manager.load(generation)
    assert torch.equal(
        loaded.model["embedding.weight"], snapshot.model["embedding.weight"]
    )
    weights = next(generation.glob("*.safetensors"))
    outside = real / "outside.safetensors"
    weights.replace(outside)
    weights.symlink_to(outside)
    assert not manager.verify(generation).valid


def _snapshot(step: int, loss: float | None = None) -> TrainingSnapshot:
    config = load_config(Path("configs/smoke_cpu.yaml"))
    model = DenseLM(config.model, config.attention)
    settings = config.optimizer
    optimizer = make_optimizer(
        model, settings.peak, settings.weight_decay, settings.betas, settings.eps
    )
    inputs = torch.arange(8).reshape(1, 8)
    for update in range(1, step + 1):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_for_step(
                update,
                config.training.max_steps,
                settings.warmup_steps,
                settings.peak,
                settings.floor,
            )
        model(inputs).square().mean().backward()
        optimizer.step()
    optimizer_state = optimizer.state_dict()
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return TrainingSnapshot(
        model=model.state_dict(),
        optimizer=optimizer_state,
        schedule={
            "kind": "warmup_cosine_v1",
            "completed_updates": step,
            "max_steps": config.training.max_steps,
            "warmup_steps": settings.warmup_steps,
            "peak": settings.peak,
            "floor": settings.floor,
        },
        step=step,
        tokens_seen=step * 8,
        cursor=(0, step),
        config=config.model_dump(mode="json"),
        run_id="tiny",
        validation_loss=loss,
        rng=_safe_rng_state(torch.device("cpu"), "cpu"),
        optimizer_parameter_names={
            stored_id: names[id(parameter)]
            for stored, live in zip(
                optimizer_state["param_groups"], optimizer.param_groups, strict=True
            )
            for stored_id, parameter in zip(
                stored["params"], live["params"], strict=True
            )
        },
    )


def test_reconcile_repairs_pointers_after_finalized_generations(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    first = manager.save(_snapshot(2, 0.5), 0.5)
    second = manager.save(_snapshot(10, 0.5), 0.5)
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "latest.json").write_text('{"relative_path":"missing"}')
    (checkpoints / "best.json").write_text('{"relative_path":"missing"}')
    recovered = manager.reconcile()
    assert recovered.record is not None
    assert recovered.record.relative_path == second.relative_path
    # Equal finite losses use the earliest generation, not lexical pointer order.
    assert (
        json.loads((checkpoints / "best.json").read_text())["relative_path"]
        == first.relative_path
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "counter",
        "moment_shape",
        "missing_moment",
        "missing_parameter",
        "schedule",
        "rng",
        "config",
    ],
)
def test_full_verify_rejects_tampered_native_state(
    tmp_path: Path, mutation: str
) -> None:
    manager = CheckpointManager(tmp_path)
    record = manager.save(_snapshot(1, 0.25), 0.25)
    generation = tmp_path / "checkpoints" / record.relative_path
    state_path = generation / "training_state.pt"
    state = torch.load(state_path, weights_only=True)
    parameter_id = next(iter(state["optimizer"]["state"]))
    if mutation == "counter":
        state["step"] = 99
    elif mutation == "moment_shape":
        state["optimizer"]["state"][parameter_id]["exp_avg"] = torch.zeros(1)
    elif mutation == "missing_moment":
        state["optimizer"]["state"][parameter_id].pop("exp_avg_sq")
    elif mutation == "missing_parameter":
        state["optimizer"]["state"].pop(parameter_id)
    elif mutation == "schedule":
        state["schedule"]["peak"] *= 2
    elif mutation == "rng":
        state["rng"]["python"] = (3, (), None)
    elif mutation == "config":
        state["config"]["model"]["hidden_dim"] *= 2
    torch.save(state, state_path)
    manifest = json.loads((generation / "manifest.json").read_text())
    for entry in manifest["files"]:
        if entry["name"] == "training_state.pt":
            entry["sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
            entry["bytes"] = state_path.stat().st_size
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    manifest["sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    (generation / "manifest.json").write_text(json.dumps(manifest))
    report = manager.verify(generation)
    assert not report.valid
    assert manager.verify(generation, require_training_state=False).valid


def test_checkpoint_reader_does_not_create_missing_run(tmp_path: Path) -> None:
    run = tmp_path / "missing"
    report = CheckpointManager(run).verify(run / "checkpoints/latest.json")
    assert not report.valid
    assert not run.exists()


def test_writer_lease_excludes_another_writer_and_recovery(tmp_path: Path) -> None:
    first, second = CheckpointManager(tmp_path), CheckpointManager(tmp_path)
    with first.writer_lease():
        record = first.save(_snapshot(0))
        with pytest.raises(RuntimeError, match="lease is held"):
            second.save(_snapshot(0))
        with pytest.raises(RuntimeError, match="lease is held"):
            second.reconcile()
    recovered = second.reconcile()
    assert recovered.record is not None
    assert recovered.record.manifest_sha256 == record.manifest_sha256


def test_retention_keeps_best_latest_and_recovery_predecessor(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, keep_periodic=False)
    with manager.writer_lease():
        oldest = manager.save(_snapshot(0), 1.0)
        best = manager.save(_snapshot(1), 0.5)
        previous = manager.save(_snapshot(2), 0.75)
        latest = manager.save(_snapshot(3), 0.8)
    checkpoints = tmp_path / "checkpoints"
    assert not (checkpoints / oldest.relative_path).exists()
    for record in (best, previous, latest):
        assert manager.verify(checkpoints / record.relative_path).valid
    weights = next((checkpoints / latest.relative_path).glob("*.safetensors"))
    weights.write_bytes(weights.read_bytes() + b"corrupt")
    recovered = manager.reconcile()
    assert recovered.record is not None
    assert recovered.record.manifest_sha256 == previous.manifest_sha256
    assert recovered.rejected
    assert (
        json.loads((checkpoints / "best.json").read_text())["manifest_sha256"]
        == best.manifest_sha256
    )
