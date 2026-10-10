"""Zero-model policy and mocked canonical train CLI coverage."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sparselab.training.spot_safety import (
    SpotSafetyPolicy,
    check_spot_capacity,
    load_spot_policy,
)
from sparselab.workspace_preflight import StorageCheck


def policy(**overrides):
    return SpotSafetyPolicy(
        **(
            {
                "observation_reference": "retained://fixture-observations",
                "checkpoint_write_seconds": 10.0,
                "boundary_seconds": 5.0,
                "restart_seconds": 20.0,
                "interruption_notice_seconds": 15.0,
                "max_recovery_seconds": 100.0,
                "checkpoint_bytes": 100,
                "checkpoint_inodes": 10,
                "reserve_bytes": 50,
                "reserve_inodes": 5,
            }
            | overrides
        )
    )


@pytest.mark.parametrize(
    "field",
    [
        "checkpoint_write_seconds",
        "boundary_seconds",
        "restart_seconds",
        "interruption_notice_seconds",
        "max_recovery_seconds",
    ],
)
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, True])
def test_unavailable_or_invalid_observations(field, value):
    with pytest.raises(ValueError):
        policy(**{field: value})


def test_infeasible_policy():
    with pytest.raises(ValueError, match="notice"):
        policy(interruption_notice_seconds=14.9)
    with pytest.raises(ValueError, match="recovery"):
        policy(max_recovery_seconds=35.0)
    with pytest.raises(ValueError):
        policy(checkpoint_bytes=True)


def test_capacity_exact_and_negative(monkeypatch, tmp_path):
    from sparselab.training import spot_safety

    observed = StorageCheck(
        path=str(tmp_path),
        filesystem_path=str(tmp_path),
        available_bytes=250,
        available_inodes=25,
        projected_bytes=200,
        projected_inodes=20,
        reserve_bytes=50,
        reserve_inodes=5,
        status="adequate",
    )
    monkeypatch.setattr(spot_safety, "check_storage", lambda *a, **k: observed)
    decision = check_spot_capacity(policy(), tmp_path)
    assert decision["every_seconds"] == 65
    assert decision["required_free_bytes"] == 250
    for changed in [
        replace(observed, available_inodes=None),
        replace(observed, available_bytes=249, status="insufficient"),
        replace(observed, available_inodes=24, status="insufficient"),
    ]:
        observed = changed
        with pytest.raises(ValueError):
            check_spot_capacity(policy(), tmp_path)


def test_native_cadence_preserves_config():
    from sparselab.config import load_config
    from sparselab.training.trainer import _checkpoint_due, _checkpoint_watermarks

    config = load_config(Path("configs/smoke_cpu.yaml"))
    config = config.model_copy(
        update={
            "checkpoint": config.checkpoint.model_copy(
                update={
                    "every_steps": 1000,
                    "every_tokens": None,
                    "every_minutes": None,
                    "steps": (),
                }
            )
        }
    )
    before = config.model_dump(mode="json")
    assert not policy().due(64, 0)
    assert policy().due(65, 0)
    assert not policy().due(66, 65)
    original = {"step": 0.0, "tokens": 0.0, "minutes": 0.0}
    retained = _checkpoint_watermarks(900, 900, 65, original, operational_only=True)
    assert retained == original
    assert _checkpoint_due(config, 1000, 1000, 66, retained)
    assert not _checkpoint_due(config, 1, 1, 65, {})
    advanced = _checkpoint_watermarks(1000, 1000, 66, retained, operational_only=False)
    assert advanced == {"step": 1000.0, "tokens": 1000.0, "minutes": 66}
    assert config.model_dump(mode="json") == before


def test_train_cli_loads_separate_policy(monkeypatch, tmp_path):
    from sparselab.cli import main
    from sparselab.config import load_config

    config = load_config(Path("configs/smoke_cpu.yaml"))
    config = config.model_copy(
        update={"logging": config.logging.model_copy(update={"root_dir": tmp_path})}
    )
    before = config.model_dump(mode="json")
    path = tmp_path / "spot.json"
    path.write_text(policy().model_dump_json())
    captured = {}
    monkeypatch.setattr(main, "load_config", lambda _: config)

    def fake_train(config, **kwargs):
        captured.update(kwargs)
        return "zero-model-fixture"

    monkeypatch.setattr(main, "train", fake_train)
    args = main.build_parser().parse_args(
        ["train", "fixture.yaml", "--spot-policy", str(path)]
    )
    args.runtime_authorization = None
    args.resource_envelope_value = None
    args.handler(args)
    assert captured["spot_policy"] == load_spot_policy(path)
    assert config.model_dump(mode="json") == before
    path.write_text(json.dumps({"checkpoint_write_seconds": None}))
    with pytest.raises(ValueError):
        args.handler(args)


def test_read_only_plan_cli(monkeypatch, tmp_path, capsys):
    from sparselab.cli import main

    path = tmp_path / "spot.json"
    path.write_text(policy().model_dump_json())
    destination = tmp_path / "not-created"
    args = main.build_parser().parse_args(
        [
            "checkpoint",
            "plan-spot",
            str(path),
            "--workspace",
            str(destination),
        ]
    )
    assert main._read_only_command(args)
    args.handler(args)
    decision = json.loads(capsys.readouterr().out)
    assert decision["every_seconds"] == 65
    assert decision["policy"] == policy().model_dump(mode="json")
    assert not destination.exists()


def test_python_train_forwards_policy_without_model(monkeypatch):
    from sparselab.config import load_config
    from sparselab.training import trainer

    config = load_config(Path("configs/smoke_cpu.yaml"))
    captured = {}

    def fake_impl(config, **kwargs):
        captured.update(kwargs)
        return "zero-model"

    monkeypatch.setattr(trainer, "require_model_runtime_allowed", lambda: None)
    monkeypatch.setattr(trainer, "require_authorization", lambda *args: None)
    monkeypatch.setattr(trainer, "_train_impl", fake_impl)
    monkeypatch.setattr(trainer, "_attempt_contract_suppresses_triage", lambda: True)
    assert trainer.train(config, spot_policy=policy()) == "zero-model"
    assert captured["spot_policy"] == policy()
