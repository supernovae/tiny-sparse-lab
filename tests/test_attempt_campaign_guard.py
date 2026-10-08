"""Offline Campaign contract guard checks; no Campaign apply or worker execution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab.campaign.engine import CampaignEngine
from sparselab.cli.main import build_parser
from sparselab.training.attempt_budget import AttemptBudget, AttemptBudgetError


def _engine(monkeypatch: pytest.MonkeyPatch) -> CampaignEngine:
    engine = CampaignEngine.__new__(CampaignEngine)
    engine.stages = {
        "run": SimpleNamespace(
            id="run", kind="experiment_run", plan="plan", cell="cell", runtime="runtime"
        ),
        "runtime": SimpleNamespace(worker=None),
    }
    lock = SimpleNamespace(
        execution={"attempt_contract": {"sha256": "a" * 64}},
        scientific_sha256="b" * 64,
        cells=(
            SimpleNamespace(
                id="cell",
                config=SimpleNamespace(
                    training=SimpleNamespace(max_steps=2, max_tokens=7)
                ),
            ),
        ),
    )
    monkeypatch.setattr(CampaignEngine, "_lock", lambda *_args: lock)
    return engine


def test_contract_guard_refuses_unwrapped_or_undercharged_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(monkeypatch)
    from sparselab.training import attempt_budget

    monkeypatch.setattr(
        attempt_budget.AttemptBudget,
        "status",
        lambda _self: {
            "version": 2,
            "contract_sha256": "a" * 64,
            "content_identity_sha256": "b" * 64,
            "reservations": [
                {
                    "label": "phase",
                    "updates": 1,
                    "target_positions": 7,
                    "generation_calls": 0,
                    "generated_tokens": 0,
                    "actual_updates": None,
                }
            ],
        },
    )
    stage = engine.stages["run"]
    with pytest.raises(ValueError, match="only-stage"):
        engine._require_attempt_contract(stage, {}, only_stage=None)
    with pytest.raises(ValueError, match="active v2 ledger"):
        engine._require_attempt_contract(stage, {}, only_stage="run")
    for key, value in {
        "SPARSELAB_ATTEMPT_BUDGET_LEDGER": str(tmp_path / "ledger.sqlite"),
        "SPARSELAB_ATTEMPT_PHASE_LABEL": "phase",
        "SPARSELAB_ATTEMPT_CONTRACT_SHA256": "a" * 64,
        "SPARSELAB_ATTEMPT_CONTENT_IDENTITY_SHA256": "b" * 64,
        "SPARSELAB_ATTEMPT_ACTIVITY": "train",
    }.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="exceeds attempt reservation"):
        engine._require_attempt_contract(stage, {}, only_stage="run")
    engine.stages["runtime"].worker = "possibly-remote"
    with pytest.raises(ValueError, match="owned local runtime"):
        engine._require_attempt_contract(stage, {}, only_stage="run")


def test_campaign_only_stage_is_an_explicit_apply_option() -> None:
    parsed = build_parser().parse_args(
        ["campaign", "apply", "campaign.yaml", "--only-stage", "run"]
    )
    assert parsed.only_stage == "run"
    assert parsed.execute_runs is False


def test_campaign_preflight_resolves_collect_to_locked_run_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.campaign import engine as campaign_engine

    class FakeEngine:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.stages = {
                "run": SimpleNamespace(id="run", kind="experiment_run", plan="plan"),
                "collect": SimpleNamespace(run="run"),
                "panel": SimpleNamespace(
                    id="panel",
                    kind="generation_panel",
                    collect="collect",
                    panel="panel.yaml",
                ),
            }

        def inspect(self, command: str) -> dict:
            assert command == "next"
            return {
                "next_action": {"stage": "panel", "action": "apply"},
                "stages": [{"id": "panel"}],
            }

        def _lock(self, _rows: dict, plan: str) -> SimpleNamespace:
            assert plan == "plan"
            return SimpleNamespace(
                scientific_sha256="a" * 64,
                execution={"attempt_contract": {"sha256": "b" * 64}},
            )

        def _path(self, name: str) -> Path:
            return tmp_path / name

    monkeypatch.setattr(campaign_engine, "CampaignEngine", FakeEngine)
    from sparselab.evaluation import panel

    monkeypatch.setattr(
        panel,
        "load_panel",
        lambda _path: SimpleNamespace(
            prompts=("one", "two"), decoder=SimpleNamespace(max_new_tokens=3)
        ),
    )
    assert AttemptBudget._preflight_campaign_receipt(
        tmp_path / "campaign.yaml",
        tmp_path,
        "panel",
        expected_kind="generation_panel",
        content_identity_sha256="a" * 64,
        contract_sha256="b" * 64,
    ) == (2, 6)


def test_native_command_preflight_rejects_unreserved_panel_and_wrong_stage(
    tmp_path: Path,
) -> None:
    with pytest.raises(AttemptBudgetError, match="exceeds reserved declaration"):
        AttemptBudget._preflight_native_command(
            [
                "sparselab",
                "campaign",
                "apply",
                str(tmp_path / "campaign.yaml"),
                "--only-stage",
                "panel",
            ],
            native_receipt_kind="campaign_panel",
            native_receipt_path=tmp_path / "campaign.yaml",
            campaign_stage="panel",
            stage_limits=(2, 6),
            reserved=(0, 0, 1, 6),
            parent_checkpoint_path=None,
            content_identity_sha256="a" * 64,
        )
    with pytest.raises(AttemptBudgetError, match="phase binding"):
        AttemptBudget._preflight_native_command(
            [
                "sparselab",
                "campaign",
                "apply",
                str(tmp_path / "campaign.yaml"),
                "--only-stage",
                "other",
            ],
            native_receipt_kind="campaign_panel",
            native_receipt_path=tmp_path / "campaign.yaml",
            campaign_stage="panel",
            stage_limits=(2, 6),
            reserved=(0, 0, 2, 6),
            parent_checkpoint_path=None,
            content_identity_sha256="a" * 64,
        )


def test_direct_train_preflight_pins_config_and_full_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.config import loading

    config = tmp_path / "config.yaml"
    config.write_text("fixed test config")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    monkeypatch.setattr(
        loading,
        "load_config",
        lambda _path: SimpleNamespace(
            training=SimpleNamespace(max_steps=3, max_tokens=77)
        ),
    )
    kwargs = {
        "native_receipt_kind": "train",
        "native_receipt_path": tmp_path / "runs" / "bounded",
        "campaign_stage": None,
        "stage_limits": None,
        "parent_checkpoint_path": None,
        "content_identity_sha256": digest,
    }
    command = [
        "sparselab",
        "train",
        str(config),
        "--run-id",
        "bounded",
        "--runs-dir",
        str(tmp_path / "runs"),
    ]
    AttemptBudget._preflight_native_command(command, reserved=(3, 77, 0, 0), **kwargs)
    with pytest.raises(AttemptBudgetError, match="config differs"):
        AttemptBudget._preflight_native_command(
            command,
            reserved=(3, 77, 0, 0),
            **{**kwargs, "content_identity_sha256": "a" * 64},
        )
    with pytest.raises(AttemptBudgetError, match="exceeds phase reservation"):
        AttemptBudget._preflight_native_command(
            command, reserved=(2, 77, 0, 0), **kwargs
        )
