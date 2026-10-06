"""Bitwise, fail-closed training-state canonicalization tests."""

from __future__ import annotations

import copy
import struct
from dataclasses import replace
from typing import ClassVar

import numpy as np
import pytest
import torch

from sparselab.engines.base import CanonicalTensor
from sparselab.training.checkpoints import (
    CheckpointManager,
    LineageBest,
    TrainingSnapshot,
)
from sparselab.training.state_digest import (
    canonical_training_state,
    compare_training_states,
)


def snapshot() -> TrainingSnapshot:
    return TrainingSnapshot(
        model={"w": torch.tensor([1.0, -0.0], dtype=torch.float32)},
        optimizer={
            "state": {0: {"exp_avg": torch.tensor([0.25])}},
            "param_groups": [{"params": [0], "lr": 0.1}],
        },
        schedule={"step": 1, "rate": 0.1},
        step=1,
        tokens_seen=8,
        cursor=(0, 1),
        config={
            "checkpoint": {"every_steps": 5, "every_minutes": None},
            "training": {"max_steps": 10},
            "logging": {"root_dir": "/tmp/run"},
        },
        run_id="one",
        rng={
            "torch": torch.tensor([0, 1], dtype=torch.uint8),
            "python": (3, (1, 2), None),
        },
        scaler={"scale": 32.0},
        optimizer_parameter_names={0: "w"},
        cadence={"step": 1, "tokens": 8},
        validation_loss=2.0,
        lineage_best=LineageBest("one", "a" * 64, 1, 2.0),
    )


def observation() -> dict:
    return {
        "step": 1,
        "tokens_seen": 8,
        "learning_rate": 0.1,
        "loss": 1.25,
        "gradient_norm": 0.5,
        "overflow": False,
        "retry_state": {"attempts": 0},
    }


def digest(state: TrainingSnapshot, update: dict | None = None) -> dict:
    return canonical_training_state(
        state, observation=observation() if update is None else update
    )


def differences(left: dict, right: dict) -> list[str]:
    return compare_training_states(left, right)["differences"]


def test_semantic_state_changes_are_component_local_and_bitwise() -> None:
    base = snapshot()
    original = digest(base)
    changes = {
        "model": lambda s: s.model["w"].view(torch.int32)[0].bitwise_xor_(1),
        "optimizer": lambda s: s.optimizer["state"][0]["exp_avg"].add_(0.5),
        "rng": lambda s: s.rng["torch"][0].bitwise_xor_(1),
        "schedule": lambda s: s.schedule.update(rate=0.2),
        "cursor": lambda s: setattr(s, "cursor", (1, 0)),
        "scaler": lambda s: s.scaler.update(scale=16.0),
        "cadence": lambda s: s.cadence.update(step=0),
        "optimizer_parameter_names": lambda s: s.optimizer_parameter_names.update(
            {0: "other"}
        ),
        "config": lambda s: s.config["training"].update(max_steps=11),
        "lineage_best": lambda s: setattr(
            s, "lineage_best", LineageBest("other", "b" * 64, 1, 1.9)
        ),
    }
    for component, mutate in changes.items():
        changed = copy.deepcopy(base)
        mutate(changed)
        assert component in differences(original, digest(changed)), component
    altered = observation()
    altered["gradient_norm"] = -0.0
    assert differences(original, digest(base, altered)) == ["observation"]
    altered = observation()
    altered["learning_rate"] = struct.unpack(
        ">d", (struct.unpack(">Q", struct.pack(">d", 0.1))[0] + 1).to_bytes(8, "big")
    )[0]
    assert differences(original, digest(base, altered)) == ["observation"]


def test_ordering_and_operational_identity_do_not_change_digest() -> None:
    first = snapshot()
    first.model["buffer"] = torch.tensor([7], dtype=torch.int64)
    second = copy.deepcopy(first)
    second.model = dict(reversed(list(second.model.items())))
    second.optimizer = dict(reversed(list(second.optimizer.items())))
    second.run_id = "another"
    second.manifest_sha256 = "f" * 64
    second.checkpoint_sha256 = "e" * 64
    second.source_identity_sha256 = "d" * 64
    second.cumulative_wall_seconds = 91.0
    second.cumulative_update_seconds = 81.0
    second.config["logging"]["root_dir"] = "/another/run"
    first.cadence["minutes"] = 19.0
    second.cadence["minutes"] = 98.0
    second.lineage_best = replace(
        second.lineage_best, parent_run_id="another", checkpoint_digest="f" * 64
    )
    assert compare_training_states(digest(first), digest(second)) == {
        "result": "BITWISE_EQUIVALENT",
        "differences": [],
    }


def test_tensor_dtype_shape_signed_zero_and_trainability() -> None:
    base = snapshot()
    state = digest(base)
    changed = copy.deepcopy(base)
    changed.model["w"][1] = 0.0
    assert "model" in differences(state, digest(changed))
    changed.model["w"] = changed.model["w"].to(torch.float64)
    assert "model" in differences(state, digest(changed))
    changed = copy.deepcopy(base)
    changed.model["w"] = changed.model["w"].reshape(2, 1)
    assert "model" in differences(state, digest(changed))
    changed = copy.deepcopy(base)
    changed.tensor_trainability = {"w": False}
    assert "model" in differences(state, digest(changed))


def test_canonical_exported_weight_source_matches_loaded_model() -> None:
    class Weights:
        aliases: ClassVar[dict[str, str]] = {"alias": "w"}

        def tensors(self):
            yield CanonicalTensor("w", np.array([1.0, -0.0], dtype=np.float32), True)

    base = snapshot()
    base.model["alias"] = base.model["w"]
    base.optimizer_parameter_names[1] = "alias"
    exported = copy.deepcopy(base)
    exported.model = {}
    exported.weight_source = Weights()
    assert differences(digest(base), digest(exported)) == []
    assert "model" in differences(
        digest(base),
        digest(
            replace(
                exported,
                weight_source=type(
                    "ChangedWeights",
                    (),
                    {
                        "aliases": {"alias": "w"},
                        "tensors": lambda self: iter(
                            (
                                CanonicalTensor(
                                    "w", np.array([1.0, 0.0], dtype=np.float32), True
                                ),
                            )
                        ),
                    },
                )(),
            )
        ),
    )


def test_rejects_nonfinite_observation_unknown_types_and_time_cadence() -> None:
    state = snapshot()
    changed = observation()
    changed["retry_state"] = {"attempts": float("nan")}
    with pytest.raises(ValueError):
        digest(state, changed)
    state.optimizer["unexpected"] = object()
    with pytest.raises(TypeError):
        digest(state)
    state = snapshot()
    state.cadence["minutes"] = 0.5
    assert differences(digest(state), digest(snapshot())) == []
    state.config["checkpoint"]["every_minutes"] = 2.0
    with pytest.raises(ValueError):
        digest(state)
    state = snapshot()
    state.config["checkpoint"]["every_minutes"] = 2.0
    with pytest.raises(ValueError):
        digest(state)
    state = snapshot()
    state.cadence["unknown"] = 1.0
    with pytest.raises(ValueError):
        digest(state)
    state = snapshot()
    state.unrecognized = True
    with pytest.raises(ValueError):
        digest(state)
    changed = observation()
    changed["retry_state"] = {"unknown": object()}
    with pytest.raises(TypeError):
        digest(snapshot(), changed)
    with pytest.raises(ValueError):
        digest(snapshot(), {**observation(), "unknown": True})
    with pytest.raises(ValueError):
        compare_training_states(
            digest(snapshot()), {**digest(snapshot()), "sha256": "0" * 64}
        )


def test_native_checkpoint_load_is_digestible(tmp_path) -> None:
    # Full native checkpoint, rather than a fabricated serialized state.
    from test_checkpoints import _snapshot

    manager = CheckpointManager(tmp_path)
    native = _snapshot(1)
    record = manager.save(native)
    loaded = manager.load(manager.root / record.relative_path)
    observed = observation()
    observed["tokens_seen"] = loaded.tokens_seen
    assert (
        compare_training_states(
            canonical_training_state(loaded, observation=observed),
            canonical_training_state(loaded, observation=observed),
        )["result"]
        == "BITWISE_EQUIVALENT"
    )
