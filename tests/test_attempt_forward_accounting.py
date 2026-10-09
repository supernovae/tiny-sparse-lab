"""Zero-update, no-generation tests for optional shared forward-input charges."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparselab.evaluation import fixed_slices
from sparselab.evaluation.generation_request import generate_result
from sparselab.training.attempt_budget import (
    AttemptBudget,
    AttemptBudgetError,
    AttemptContract,
)


def _ledger(
    tmp_path: Path,
    *,
    forward: bool = True,
    fixed: int = 55_000,
    total: int = 170_000,
    batches: int = 11,
    validation: int = 11_264,
    calls: int = 16,
    tokens: int = 1024,
) -> tuple[AttemptBudget, str, bytes]:
    identity = "a" * 64
    declaration: dict[str, object] = {
        "contract_version": 1,
        "max_optimizer_updates": 0,
        "max_actual_target_positions": 0,
        "max_generation_calls": calls,
        "max_generated_tokens": tokens,
        "max_wall_seconds": 120.0,
        "content_identity_sha256": identity,
        "monitor_policy_sha256": "b" * 64,
        "workspace_baseline_sha256": "c" * 64,
    }
    if forward:
        declaration.update(
            max_fixed_profile_forward_positions=fixed,
            max_nontraining_forward_positions=total,
            max_operational_validation_batches=batches,
            max_operational_validation_forward_positions=validation,
        )
    raw = json.dumps(declaration, sort_keys=True, separators=(",", ":")).encode()
    contract = tmp_path / "contract.json"
    contract.write_bytes(raw)
    budget = AttemptBudget.create_contract(
        tmp_path / "attempt.sqlite",
        contract_path=contract,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )
    return budget, identity, raw


def _environment(monkeypatch, budget: AttemptBudget, identity: str) -> None:
    monkeypatch.setenv("SPARSELAB_ATTEMPT_BUDGET_LEDGER", str(budget.path))
    monkeypatch.setenv("SPARSELAB_ATTEMPT_CONTENT_IDENTITY_SHA256", identity)
    monkeypatch.setenv("SPARSELAB_ATTEMPT_PHASE_LABEL", "evaluation-a")


def test_absent_fields_preserve_legacy_contract_dump_and_status(
    tmp_path: Path, monkeypatch
) -> None:
    budget, _, raw = _ledger(tmp_path, forward=False)
    _environment(monkeypatch, budget, "a" * 64)
    parsed = AttemptContract.model_validate_json(raw)
    assert parsed.model_dump(mode="json") == json.loads(raw)
    assert budget.status()["contract_sha256"] == hashlib.sha256(raw).hexdigest()
    assert "charged_nontraining_forward_positions" not in budget.status()
    assert not AttemptBudget.forward_allocation_active_from_environment()
    with pytest.raises(AttemptBudgetError, match="no forward-input allocation"):
        budget.reserve_forward_positions(
            "old", kind="fixed_profile", positions=1, content_identity_sha256="a" * 64
        )


def test_forward_limits_are_shared_persistent_and_nonrefundable(tmp_path: Path) -> None:
    budget, identity, _ = _ledger(
        tmp_path, fixed=257, total=260, batches=1, validation=3
    )
    budget.reserve_forward_positions(
        "first", kind="fixed_profile", positions=257, content_identity_sha256=identity
    )
    reopened = AttemptBudget(budget.path)
    with pytest.raises(AttemptBudgetError, match="fixed-profile"):
        reopened.reserve_forward_positions(
            "failed-second",
            kind="fixed_profile",
            positions=1,
            content_identity_sha256=identity,
        )
    reopened.reserve_forward_positions(
        "validation",
        kind="operational_validation",
        positions=3,
        content_identity_sha256=identity,
    )
    with pytest.raises(AttemptBudgetError, match="nontraining"):
        reopened.reserve_forward_positions(
            "generation",
            kind="generation",
            positions=1,
            content_identity_sha256=identity,
        )
    with pytest.raises(AttemptBudgetError, match="duplicate forward-input"):
        reopened.reserve_forward_positions(
            "first", kind="fixed_profile", positions=1, content_identity_sha256=identity
        )
    status = reopened.status()
    assert status["charged_fixed_profile_forward_positions"] == 257
    assert status["charged_nontraining_forward_positions"] == 260
    assert status["charged_operational_validation_batches"] == 1


def test_operational_validation_batch_cap_prevents_extra_forward(
    tmp_path: Path,
) -> None:
    budget, identity, _ = _ledger(tmp_path)
    for index in range(11):
        budget.reserve_forward_positions(
            f"validation-{index}",
            kind="operational_validation",
            positions=1024,
            content_identity_sha256=identity,
        )
    with pytest.raises(AttemptBudgetError, match="operational validation"):
        budget.reserve_forward_positions(
            "validation-12",
            kind="operational_validation",
            positions=1,
            content_identity_sha256=identity,
        )
    assert budget.status()["charged_operational_validation_forward_positions"] == 11_264


def test_generation_request_charges_existing_vector_and_forward_budget(
    tmp_path: Path, monkeypatch
) -> None:
    budget, identity, _ = _ledger(tmp_path, calls=1, tokens=64)
    _environment(monkeypatch, budget, identity)
    AttemptBudget.reserve_generation_from_environment(
        "continuation-1", requested_tokens=64, forward_positions=6112
    )
    with pytest.raises(AttemptBudgetError, match="generation calls"):
        AttemptBudget.reserve_generation_from_environment(
            "continuation-2", requested_tokens=64, forward_positions=6112
        )
    status = AttemptBudget(budget.path).status()
    assert status["charged_generation_calls"] == 1
    assert status["charged_generated_tokens"] == 64
    assert status["charged_nontraining_forward_positions"] == 6112


def test_declared_full_allocation_closes_and_repeated_sweeps_need_new_phase(
    tmp_path: Path, monkeypatch
) -> None:
    budget, identity, _ = _ledger(tmp_path)
    _environment(monkeypatch, budget, identity)
    for sweep in range(11):
        monkeypatch.setenv("SPARSELAB_ATTEMPT_PHASE_LABEL", f"validation-{sweep}")
        budget.reserve_forward_from_environment(
            "fixed:item-1", kind="fixed_profile", positions=3084
        )
    with pytest.raises(AttemptBudgetError, match="duplicate forward-input"):
        budget.reserve_forward_from_environment(
            "fixed:item-1", kind="fixed_profile", positions=1
        )
    monkeypatch.setenv("SPARSELAB_ATTEMPT_PHASE_LABEL", "other-fixed")
    budget.reserve_forward_from_environment(
        "remaining", kind="fixed_profile", positions=55_000 - 11 * 3084
    )
    for index in range(11):
        budget.reserve_forward_positions(
            f"operational-{index}",
            kind="operational_validation",
            positions=1024,
            content_identity_sha256=identity,
        )
    monkeypatch.setenv("SPARSELAB_ATTEMPT_PHASE_LABEL", "continuations")
    for index in range(16):
        budget.reserve_generation_from_environment(
            f"prompt-{index}", requested_tokens=64, forward_positions=6112
        )
    status = budget.status()
    assert status["charged_fixed_profile_forward_positions"] == 55_000
    assert status["charged_operational_validation_forward_positions"] == 11_264
    assert status["charged_nontraining_forward_positions"] == 164_056
    assert status["charged_generation_calls"] == 16
    assert status["charged_generated_tokens"] == 1024
    assert status["max_nontraining_forward_positions"] - 164_056 == 5944


def test_failed_forward_charge_keeps_prior_generation_reservation(
    tmp_path: Path, monkeypatch
) -> None:
    budget, identity, _ = _ledger(tmp_path, fixed=20, total=20, validation=11)
    _environment(monkeypatch, budget, identity)
    with pytest.raises(AttemptBudgetError, match="nontraining"):
        budget.reserve_generation_from_environment(
            "failed", requested_tokens=64, forward_positions=6112
        )
    status = AttemptBudget(budget.path).status()
    assert status["charged_generation_calls"] == 1
    assert status["charged_generated_tokens"] == 64
    assert status["charged_nontraining_forward_positions"] == 0


def test_corrupt_forward_reservation_fails_closed(tmp_path: Path) -> None:
    budget, identity, _ = _ledger(tmp_path)
    budget.reserve_forward_positions(
        "one", kind="fixed_profile", positions=1, content_identity_sha256=identity
    )
    with sqlite3.connect(budget.path) as connection:
        connection.execute("UPDATE forward_reservations SET positions = -1")
    with pytest.raises(AttemptBudgetError, match="invalid forward-input reservations"):
        budget.status()


def test_fixed_scoring_exhausts_shared_cap_before_second_mock_forward(
    tmp_path: Path, monkeypatch
) -> None:
    budget, identity, _ = _ledger(tmp_path, fixed=257)
    _environment(monkeypatch, budget, identity)
    rows = [
        {
            "id": f"validation-general_prose-{index}",
            "split": "validation",
            "stratum": "general_prose",
            "document_id": f"doc-{index}",
            "start_token": 0,
        }
        for index in range(12)
    ]
    bound = fixed_slices.BoundFixedSlices(
        {"loss_slices": rows},
        "d" * 64,
        None,
        {f"doc-{index}": [1] * 257 for index in range(12)},
        {},
        {},
    )
    calls: list[int] = []
    monkeypatch.setattr(
        fixed_slices,
        "score_window",
        lambda _model, ids, **_kw: calls.append(len(ids)) or 1.0,
    )

    class NoModel:
        training = True

        def eval(self) -> None:
            self.training = False

        def train(self, state: bool) -> None:
            self.training = state

    with pytest.raises(AttemptBudgetError, match="fixed-profile"):
        fixed_slices.score_fixed_slices(
            bound, NoModel(), torch.device("cpu"), "validation", 3084
        )
    assert calls == [257]
    assert budget.status()["charged_fixed_profile_forward_positions"] == 257


def test_generation_exhaustion_rejects_before_model_access(
    tmp_path: Path, monkeypatch
) -> None:
    budget, identity, _ = _ledger(tmp_path, calls=0, tokens=0)
    _environment(monkeypatch, budget, identity)

    class Tokenizer:
        def encode(self, _prompt, *, add_special_tokens):
            assert add_special_tokens is False
            return SimpleNamespace(ids=[4] * 64)

    with pytest.raises(AttemptBudgetError, match="generation calls"):
        generate_result(
            SimpleNamespace(config=SimpleNamespace(memory="none")),
            Tokenizer(),
            "frozen prompt",
            128,
            64,
            torch.device("cpu"),
            strict_context=True,
            accounting_label="first",
        )
    assert budget.status()["charged_nontraining_forward_positions"] == 0
