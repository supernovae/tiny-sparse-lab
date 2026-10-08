"""A bounded test campaign charges retries before any optimizer update."""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from sparselab.training import attempt_budget as module
from sparselab.training.attempt_budget import AttemptBudget, AttemptBudgetError


def test_updates_persist_across_attempts_and_never_refund(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite"
    first = AttemptBudget.create(path, max_updates=5, max_wall_seconds=60)
    assert first.reserve("original test", 3) == 2
    retry = AttemptBudget(path)
    assert retry.reserve("failed test retry", 2) == 0
    with pytest.raises(AttemptBudgetError, match="shared update limit"):
        retry.reserve("resumed run", 1)
    assert retry.status()["charged_updates"] == 5
    assert [row["updates"] for row in retry.status()["reservations"]] == [3, 2]
    with pytest.raises(FileExistsError):
        AttemptBudget.create(path, max_updates=100, max_wall_seconds=60)


def test_deadline_and_clock_rollback_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1_000_000_000_000]
    monkeypatch.setattr(module.time, "time_ns", lambda: now[0])
    budget = AttemptBudget.create(
        tmp_path / "budget.sqlite", max_updates=5, max_wall_seconds=10
    )
    now[0] += 4_000_000_000
    assert budget.reserve("before deadline", 1) == 4
    now[0] -= 1
    with pytest.raises(AttemptBudgetError, match="clock moved backwards"):
        budget.reserve("rolled-back clock", 1)
    now[0] += 6_000_000_001
    with pytest.raises(AttemptBudgetError, match="wall-time limit"):
        budget.reserve("late retry", 1)
    assert budget.status()["charged_updates"] == 1


def test_missing_budget_and_invalid_bounds_refuse_execution(tmp_path: Path) -> None:
    with pytest.raises(AttemptBudgetError, match="missing"):
        AttemptBudget(tmp_path / "missing.sqlite").reserve("test", 1)
    with pytest.raises(ValueError, match="positive integer"):
        AttemptBudget.create(
            tmp_path / "invalid.sqlite", max_updates=0, max_wall_seconds=10
        )
    assert not (tmp_path / "invalid.sqlite").exists()


def test_tampered_reservation_total_fails_closed(tmp_path: Path) -> None:
    budget = AttemptBudget.create(
        tmp_path / "budget.sqlite", max_updates=5, max_wall_seconds=60
    )
    budget.reserve("first", 2)
    with sqlite3.connect(budget.path) as connection:
        connection.execute("UPDATE budget SET used_updates = 0 WHERE id = 1")
    with pytest.raises(AttemptBudgetError, match="reservation total disagrees"):
        budget.reserve("retry", 1)


def test_command_does_not_spawn_past_update_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = AttemptBudget.create(
        tmp_path / "budget.sqlite", max_updates=1, max_wall_seconds=60
    )
    budget.reserve("first test", 1)

    def unexpected_spawn(*args: object, **kwargs: object) -> None:
        pytest.fail("command spawned after the shared update cap")

    monkeypatch.setattr(module.subprocess, "Popen", unexpected_spawn)
    with pytest.raises(AttemptBudgetError, match="shared update limit"):
        budget.run(["retry"], reserve_updates=1)


def test_command_inherits_budget_and_is_killed_at_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = AttemptBudget.create(
        tmp_path / "budget.sqlite", max_updates=2, max_wall_seconds=60
    )
    calls: list[object] = []

    class FakeProcess:
        pid = 42

        def wait(self, *, timeout: float | None = None) -> int:
            calls.append(("wait", timeout))
            if (42, module.signal.SIGKILL) in calls:
                return -9
            raise subprocess.TimeoutExpired(["dummy"], timeout)

    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda command, **kwargs: (
            calls.append(("spawn", command, kwargs)),
            FakeProcess(),
        )[1],
    )
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    with pytest.raises(AttemptBudgetError, match="wall-time limit"):
        budget.run(["dummy"], reserve_updates=1)
    assert budget.status()["charged_updates"] == 1
    assert calls[0][2]["env"]["SPARSELAB_ATTEMPT_BUDGET_LEDGER"] == str(budget.path)
    assert calls[1][0] == "wait"
    assert calls[2] == (42, module.signal.SIGKILL)


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit, OSError])
def test_command_cleanup_on_interrupted_or_failed_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    budget = AttemptBudget.create(
        tmp_path / "budget.sqlite", max_updates=2, max_wall_seconds=60
    )
    error = failure("interrupted supervisor")
    calls = []

    class FakeProcess:
        pid = 42

        def wait(self, *, timeout: float | None = None) -> int:
            calls.append(("wait", timeout))
            if timeout is not None:
                raise error
            return -9

    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: FakeProcess())
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    with pytest.raises(failure) as raised:
        budget.run(["dummy"], reserve_updates=1)
    assert raised.value is error
    assert calls[1:] == [(42, module.signal.SIGKILL), ("wait", None)]
    assert budget.status()["charged_updates"] == 1
