"""Inspected zero-update v2 contract tests; subprocesses perform no model work."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from sparselab.operational_monitor import capture_workspace_baseline
from sparselab.training import attempt_budget as module
from sparselab.training.attempt_budget import AttemptBudget, AttemptBudgetError
from sparselab.training.attempt_commands import AttemptCommand, phase_output_paths

_PHASE_MAP = (
    Path(__file__).resolve().parents[1]
    / "experiments/research/kernel-memory-lab/card05-base-50m/preparation-phase-paths-v1.json"
)


def _fixture(
    tmp_path: Path,
    *,
    updates: int = 0,
    seconds: float = 10,
    added_bytes: int | None = None,
    nested: bool = False,
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
                **(
                    {
                        "preparation_monitor_policy_sha256": hashlib.sha256(
                            policy_path.read_bytes()
                        ).hexdigest()
                    }
                    if nested
                    else {}
                ),
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


def test_public_dispatch_rejects_b16_paths_before_reservation(tmp_path: Path) -> None:
    """The real CLI, ledger and both monitors exercise zero-model render work."""
    budget, paths = _fixture(tmp_path, seconds=60, nested=True)
    root = paths["root"]
    assert isinstance(root, Path)
    for name in ("prep", "receipts", "logs"):
        (root / name).mkdir()
    template = root / "prep" / "template.json"
    template.write_text('{"fixture":true}\n')
    alias = root / "prep" / "template-alias.json"
    alias.symlink_to(template)
    phase = phase_output_paths(root, "admission-draft", "admission-draft.json")

    def dispatch(
        output: Path, completion: Path, *, label: str
    ) -> subprocess.CompletedProcess[str]:
        leaf = [
            sys.executable,
            "-m",
            "sparselab",
            "--work-dir",
            str(root),
            "corpus",
            "render-declaration",
            "--template",
            str(template),
            "--values-json",
            "{}",
            "--output",
            str(output),
        ]
        guarded = [
            sys.executable,
            "-m",
            "sparselab",
            "--work-dir",
            str(root),
            "monitor",
            "--policy",
            str(paths["policy"]),
            "--log-dir",
            str(phase["inner_monitor"]),
            "--workspace",
            str(root),
            "--baseline",
            str(paths["baseline"]),
            "--",
            *leaf,
        ]
        command = [
            sys.executable,
            "-m",
            "sparselab",
            "--work-dir",
            str(root),
            "attempt",
            "run",
            "--ledger",
            str(budget.path),
            "--label",
            label,
            "--activity",
            "inspect",
            "--content-identity-sha256",
            str(paths["identity"]),
            "--policy",
            str(paths["policy"]),
            "--baseline",
            str(paths["baseline"]),
            "--workspace",
            str(root),
            "--completion",
            str(completion),
            "--updates",
            "0",
            "--target-positions",
            "0",
            "--generation-calls",
            "0",
            "--generated-tokens",
            "0",
            "--receipt-kind",
            "none",
            "--",
            *guarded,
        ]
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            env=os.environ.copy(),
            check=False,
        )

    for output, completion in (
        (phase["leaf"], phase["leaf"]),
        (
            phase["completion"].with_name(phase["completion"].name + ".attempt.json"),
            phase["completion"],
        ),
        (
            phase["completion"].with_name(phase["completion"].name + ".monitor"),
            phase["completion"],
        ),
        (phase["inner_monitor"], phase["completion"]),
        (template, phase["completion"]),
        (alias, phase["completion"]),
    ):
        result = dispatch(output, completion, label="collision")
        assert result.returncode != 0
        assert "attempt output" in result.stderr
        assert not phase["leaf"].exists()
        assert not phase["completion"].exists()
        assert not phase["inner_monitor"].exists()
        assert budget.status()["reservations"] == []
    result = dispatch(phase["leaf"], phase["completion"], label="admission-draft")
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert phase["leaf"].read_bytes() == template.read_bytes()
    assert json.loads(phase["completion"].read_text())["living_descendants"] == 0
    assert (phase["inner_monitor"] / "completion.json").exists()
    assert len(budget.status()["reservations"]) == 1


def test_complete_preparation_phase_paths_are_generated_and_disjoint(
    tmp_path: Path,
) -> None:
    packet = json.loads(_PHASE_MAP.read_text())
    assert packet["format"] == "sparselab-preparation-phase-paths-v1"
    phases = packet["phases"]
    assert len(phases) == 31
    assert len({phase for phase, _ in phases}) == len(phases)
    required = {
        "verify-snapshots",
        "admission-draft",
        "admission-inspection",
        "family-freeze",
        "release-acceptance",
        "measure-accepted-supply",
        "materialize-mixture",
        "publish-prepared-bundle",
        "cold-verify-prepared-bundle",
    }
    assert required.issubset({phase for phase, _ in phases})
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    outputs: set[Path] = set()
    for label, leaf_name in phases:
        rendered = phase_output_paths(attempt, label, leaf_name)
        assert rendered["completion"].parent == attempt / "receipts"
        assert rendered["inner_monitor"].parent == attempt / "logs"
        if leaf_name is not None:
            assert rendered["leaf"].parent == attempt / "prep"
        completion = rendered["completion"]
        derived = (
            completion.with_name(completion.name + ".ready"),
            completion.with_name(completion.name + ".attempt.json"),
            completion.with_name(completion.name + ".monitor"),
        )
        for path in (*rendered.values(), *derived):
            assert all(
                path != old
                and not path.is_relative_to(old)
                and not old.is_relative_to(path)
                for old in outputs
            )
            outputs.add(path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sparselab",
            "attempt",
            "phase-paths",
            "--attempt-root",
            str(attempt),
            "--label",
            "admission-draft",
            "--leaf-name",
            "admission-draft.json",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        key: str(value)
        for key, value in phase_output_paths(
            attempt, "admission-draft", "admission-draft.json"
        ).items()
    }


def _run_args(paths: dict[str, Path | str], completion: Path) -> dict[str, object]:
    return {
        "activity": "inspect",
        "label": "evaluation",
        "content_identity_sha256": paths["identity"],
        "monitor_policy_path": paths["policy"],
        "workspace_baseline_path": paths["baseline"],
        "workspace_root": paths["root"],
        "completion": completion,
        "generation_calls": 0,
        "generated_tokens": 0,
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
    budget._record_verified_actual(
        "first",
        updates=1,
        target_positions=699,
        generation_calls=1,
        generated_tokens=60,
        content_identity_sha256=paths["identity"],
    )
    budget._record_verified_actual(
        "first",
        updates=1,
        target_positions=699,
        generation_calls=1,
        generated_tokens=60,
        content_identity_sha256=paths["identity"],
    )
    with pytest.raises(AttemptBudgetError, match="differ from recorded"):
        budget._record_verified_actual(
            "first",
            updates=1,
            target_positions=698,
            generation_calls=1,
            generated_tokens=60,
            content_identity_sha256=paths["identity"],
        )
    with pytest.raises(AttemptBudgetError, match="exceeds phase reservation"):
        budget._record_verified_actual(
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
            [
                "sparselab",
                "train",
                "config.yaml",
                "--run-id",
                "run",
                "--runs-dir",
                str(paths["root"]),
            ],
            **(
                _run_args(paths, paths["root"] / "train.json")
                | {"activity": "train", "updates": 1}
            ),
        )
    with pytest.raises(AttemptBudgetError, match="stage must stop at validation"):
        budget.run_contract(
            [
                "sparselab",
                "stage",
                "config.yaml",
                "--through",
                "warmup",
                "--output",
                str(paths["root"] / "stage"),
            ],
            **(
                _run_args(paths, paths["root"] / "warmup.json") | {"activity": "warmup"}
            ),
        )
    with pytest.raises(AttemptBudgetError, match="direct native sparselab"):
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


def _allow_harmless_supervisor_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep process-tree tests isolated from the production native allowlist."""
    original = module.classify_attempt_command

    def classify(command: list[str]) -> AttemptCommand:
        if command[:2] == [sys.executable, "-c"]:
            return AttemptCommand(
                ("inspect", "fixture"), "inspect", "inspection", None, {}
            )
        return original(command)

    monkeypatch.setattr(module, "classify_attempt_command", classify)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper ownership")
def test_owned_runner_cleans_detached_worker_and_preserves_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_harmless_supervisor_fixture(monkeypatch)
    monkeypatch.setattr(
        AttemptBudget, "_approved_counter_free_command", staticmethod(lambda _c: True)
    )
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
        assert budget.status()["charged_generation_calls"] == 0
    finally:
        sentinel.kill()
        sentinel.wait(timeout=2)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper ownership")
def test_runtime_deadline_keeps_charge_and_zero_survivors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_harmless_supervisor_fixture(monkeypatch)
    monkeypatch.setattr(
        AttemptBudget, "_approved_counter_free_command", staticmethod(lambda _c: True)
    )
    budget, paths = _fixture(tmp_path, updates=1, seconds=0.5)
    completion = paths["root"] / "owned.json"
    with pytest.raises(AttemptBudgetError, match="wall-time limit"):
        budget.run_contract(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            **_run_args(paths, completion),
        )
    assert len(budget.status()["reservations"]) == 1
    assert json.loads(completion.read_text())["living_descendants"] == 0
    assert (paths["root"] / "owned.json.attempt.json").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper ownership")
def test_outer_receipts_crossing_cap_cannot_return_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_harmless_supervisor_fixture(monkeypatch)
    monkeypatch.setattr(
        AttemptBudget, "_approved_counter_free_command", staticmethod(lambda _c: True)
    )

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
    _allow_harmless_supervisor_fixture(monkeypatch)
    monkeypatch.setattr(
        AttemptBudget, "_approved_counter_free_command", staticmethod(lambda _c: True)
    )
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
    assert len(budget.status()["reservations"]) == 1
    assert json.loads(completion.read_text())["living_descendants"] == 0
    assert not (paths["root"] / "owned.json.monitor").exists()
    assert (paths["root"] / "owned.json.attempt.json").exists()
