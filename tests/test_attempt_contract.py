"""Inspected zero-update v2 contract tests; subprocesses perform no model work."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from sparselab.operational_monitor import capture_workspace_baseline
from sparselab.training import attempt_budget as module
from sparselab.training.attempt_budget import AttemptBudget, AttemptBudgetError


def _fixture(
    tmp_path: Path,
    *,
    updates: int = 0,
    seconds: float = 10,
    added_bytes: int | None = None,
) -> tuple[AttemptBudget, dict[str, Path | str]]:
    root = tmp_path / "root"
    root.mkdir()
    baseline_path = tmp_path / "baseline.json"
    baseline = capture_workspace_baseline(root, baseline_path)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "monitor_policy_version: 1\n"
        + (
            f"max_added_workspace_bytes: {added_bytes}\n"
            if added_bytes is not None
            else ""
        )
    )
    identity = "a" * 64
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "max_optimizer_updates": updates,
                "max_actual_target_positions": 1024,
                "max_generation_calls": 4,
                "max_generated_tokens": 256,
                "max_wall_seconds": seconds,
                "content_identity_sha256": identity,
                "monitor_policy_sha256": hashlib.sha256(
                    policy_path.read_bytes()
                ).hexdigest(),
                "workspace_baseline_sha256": baseline.sha256,
            },
            sort_keys=True,
        )
    )
    sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    budget = AttemptBudget.create_contract(
        tmp_path / "ledger.sqlite", contract_path=contract_path, expected_sha256=sha
    )
    return budget, {
        "root": root,
        "baseline": baseline_path,
        "policy": policy_path,
        "contract": contract_path,
        "identity": identity,
        "sha": sha,
    }


def _run_args(paths: dict[str, Path | str], completion: Path) -> dict[str, object]:
    return {
        "activity": "evaluate",
        "label": "evaluation",
        "content_identity_sha256": paths["identity"],
        "monitor_policy_path": paths["policy"],
        "workspace_baseline_path": paths["baseline"],
        "workspace_root": paths["root"],
        "completion": completion,
        "generation_calls": 1,
        "generated_tokens": 64,
        "grace_seconds": 0.05,
    }


def test_zero_update_vector_reservations_persist_without_refund(tmp_path: Path) -> None:
    budget, paths = _fixture(tmp_path)
    assert (
        budget.reserve_vector(
            "first",
            generation_calls=2,
            generated_tokens=128,
            content_identity_sha256=paths["identity"],
        )["generation_calls"]
        == 2
    )
    reopened = AttemptBudget(budget.path)
    assert (
        reopened.reserve_vector(
            "retry",
            generation_calls=2,
            generated_tokens=128,
            content_identity_sha256=paths["identity"],
        )["generated_tokens"]
        == 0
    )
    with pytest.raises(AttemptBudgetError, match="shared generation calls limit"):
        reopened.reserve_vector(
            "third",
            generation_calls=1,
            content_identity_sha256=paths["identity"],
        )
    with pytest.raises(AttemptBudgetError, match="duplicate phase"):
        reopened.reserve_vector("first", content_identity_sha256=paths["identity"])
    status = reopened.status()
    assert status["version"] == 2
    assert status["charged_updates"] == 0
    assert status["charged_generation_calls"] == 4
    assert status["charged_generated_tokens"] == 256
    assert len(status["reservations"]) == 2


def test_vector_failure_is_atomic_and_actual_targets_are_charged(
    tmp_path: Path,
) -> None:
    budget, paths = _fixture(tmp_path, updates=2)
    budget.reserve_vector(
        "first",
        updates=1,
        target_positions=700,
        generation_calls=1,
        generated_tokens=64,
        content_identity_sha256=paths["identity"],
    )
    with pytest.raises(AttemptBudgetError, match="actual target positions limit"):
        budget.reserve_vector(
            "too much",
            updates=1,
            target_positions=325,
            generation_calls=1,
            generated_tokens=64,
            content_identity_sha256=paths["identity"],
        )
    assert budget.status()["charged_updates"] == 1
    assert budget.status()["charged_generation_calls"] == 1
    assert budget.status()["charged_actual_target_positions"] == 700
    budget.record_actual(
        "first",
        updates=1,
        target_positions=699,
        generation_calls=1,
        generated_tokens=60,
        content_identity_sha256=paths["identity"],
    )
    budget.record_actual(
        "first",
        updates=1,
        target_positions=699,
        generation_calls=1,
        generated_tokens=60,
        content_identity_sha256=paths["identity"],
    )
    with pytest.raises(AttemptBudgetError, match="differ from recorded"):
        budget.record_actual(
            "first",
            updates=1,
            target_positions=698,
            generation_calls=1,
            generated_tokens=60,
            content_identity_sha256=paths["identity"],
        )
    with pytest.raises(AttemptBudgetError, match="exceeds phase reservation"):
        budget.record_actual(
            "first",
            updates=2,
            target_positions=699,
            generation_calls=1,
            generated_tokens=60,
            content_identity_sha256=paths["identity"],
        )
    assert budget.status()["charged_actual_target_positions"] == 700
    assert budget.status()["reservations"][0]["actual_target_positions"] == 699


def test_identity_drift_and_tampered_counters_fail_closed(tmp_path: Path) -> None:
    budget, paths = _fixture(tmp_path)
    with pytest.raises(AttemptBudgetError, match="content identity"):
        budget.reserve_vector("wrong", content_identity_sha256="b" * 64)
    paths["contract"].write_text(paths["contract"].read_text() + "\n")
    with pytest.raises(AttemptBudgetError, match="digest mismatch"):
        budget.reserve_vector("changed", content_identity_sha256=paths["identity"])
    paths["contract"].write_text(paths["contract"].read_text().rstrip("\n"))
    budget.reserve_vector(
        "first", generation_calls=1, content_identity_sha256=paths["identity"]
    )
    with sqlite3.connect(budget.path) as connection:
        connection.execute("UPDATE budget SET used_calls = 0 WHERE id = 1")
    with pytest.raises(AttemptBudgetError, match="reservation total"):
        budget.reserve_vector("second", content_identity_sha256=paths["identity"])


def test_tampered_deadline_and_legacy_bypass_fail_closed(tmp_path: Path) -> None:
    budget, paths = _fixture(tmp_path)
    with pytest.raises(AttemptBudgetError, match="reserve_vector"):
        budget.reserve("bypass", 0)
    with pytest.raises(AttemptBudgetError, match="run_contract"):
        budget.run(["unmonitored"], reserve_updates=0)
    with sqlite3.connect(budget.path) as connection:
        connection.execute(
            "UPDATE budget SET deadline_ns = deadline_ns + 1 WHERE id = 1"
        )
    with pytest.raises(AttemptBudgetError, match="deadline disagrees"):
        budget.reserve_vector("late", content_identity_sha256=paths["identity"])


def test_zero_update_training_and_warmup_are_refused_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget, paths = _fixture(tmp_path)
    monkeypatch.setattr(
        module.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("spawned")
    )
    with pytest.raises(AttemptBudgetError, match="zero-update"):
        budget.run_contract(
            ["sparselab", "train"],
            **(
                _run_args(paths, paths["root"] / "train.json")
                | {"activity": "train", "updates": 1}
            ),
        )
    with pytest.raises(AttemptBudgetError, match="zero-update"):
        budget.run_contract(
            ["sparselab", "stage", "--through", "warmup"],
            **_run_args(paths, paths["root"] / "warmup.json"),
        )
    with pytest.raises(AttemptBudgetError, match="zero-update"):
        budget.run_contract(
            ["bash", "-c", "sparselab train"],
            **_run_args(paths, paths["root"] / "wrapped.json"),
        )
    assert budget.status()["reservations"] == []


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper ownership")
def test_owned_runner_cleans_detached_worker_and_preserves_sentinel(
    tmp_path: Path,
) -> None:
    budget, paths = _fixture(tmp_path, updates=1)
    worker_pid = tmp_path / "worker.pid"
    worker = (
        "import os,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(worker_pid)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {worker!r}], start_new_session=True); "
        "time.sleep(0.1)"
    )
    sentinel = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        completion = paths["root"] / "owned.json"
        rc = budget.run_contract(
            [sys.executable, "-c", parent], **_run_args(paths, completion)
        )
        assert rc != 0
        assert worker_pid.exists()
        assert not _alive(int(worker_pid.read_text()))
        assert sentinel.poll() is None
        assert json.loads(completion.read_text())["living_descendants"] == 0
        assert budget.status()["charged_generation_calls"] == 1
    finally:
        sentinel.kill()
        sentinel.wait(timeout=2)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper ownership")
def test_runtime_deadline_keeps_charge_and_zero_survivors(tmp_path: Path) -> None:
    budget, paths = _fixture(tmp_path, updates=1, seconds=0.5)
    completion = paths["root"] / "owned.json"
    with pytest.raises(AttemptBudgetError, match="wall-time limit"):
        budget.run_contract(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            **_run_args(paths, completion),
        )
    assert budget.status()["charged_generation_calls"] == 1
    assert json.loads(completion.read_text())["living_descendants"] == 0
    assert (paths["root"] / "owned.json.attempt.json").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper ownership")
def test_outer_receipts_crossing_cap_cannot_return_success(tmp_path: Path) -> None:
    def run(
        name: str, cap: int
    ) -> tuple[int | None, dict[str, object], dict[str, object]]:
        base = tmp_path / name
        base.mkdir()
        budget, paths = _fixture(base, updates=1, added_bytes=cap)
        completion = paths["root"] / "owned.json"
        rc: int | None
        try:
            rc = budget.run_contract(
                [sys.executable, "-c", "import time;time.sleep(0.1)"],
                **_run_args(paths, completion),
            )
        except AttemptBudgetError:
            rc = None
        native = json.loads(
            (paths["root"] / "owned.json.monitor" / "completion.json").read_text()
        )
        final = json.loads((paths["root"] / "owned.json.attempt.json").read_text())
        return rc, native, final

    rc, native, final = run("probe", 100000)
    assert rc == 0
    assert native["status"] == "COMPLETE"
    assert final["status"] == "WITHIN_CAP"
    assert (
        final["final_added_workspace_bytes"]
        > native["final_added_workspace_bytes"] + 100
    )
    cap = (
        native["final_added_workspace_bytes"] + final["final_added_workspace_bytes"]
    ) // 2
    stopped, native_bounded, final_bounded = run("limit", cap)
    assert stopped is None
    assert native_bounded["status"] == "COMPLETE"
    assert final_bounded["status"] == "VIOLATED"
    assert final_bounded["final_added_workspace_bytes"] > cap


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper ownership")
def test_policy_change_at_child_boundary_prevents_native_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget, paths = _fixture(tmp_path, updates=1)
    completion = paths["root"] / "owned.json"
    original = subprocess.Popen

    def changed_policy_then_spawn(*args: object, **kwargs: object) -> subprocess.Popen:
        paths["policy"].write_text(
            "monitor_policy_version: 1\nmax_tree_rss_bytes: 999999999\n"
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", changed_policy_then_spawn)
    with pytest.raises(
        AttemptBudgetError, match="native monitor completion unverified"
    ):
        budget.run_contract(
            [sys.executable, "-c", "pass"], **_run_args(paths, completion)
        )
    assert budget.status()["charged_generation_calls"] == 1
    assert json.loads(completion.read_text())["living_descendants"] == 0
    assert not (paths["root"] / "owned.json.monitor").exists()
    assert (paths["root"] / "owned.json.attempt.json").exists()
