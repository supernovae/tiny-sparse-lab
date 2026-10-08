"""Zero-model-work checks for the opt-in KML hosted qualification guard."""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from sparselab.training.attempt_budget import AttemptBudget

ROOT = Path(__file__).resolve().parents[1] / "tools/kernel-memory-lab/ci-guard"


def test_pre_setup_guard_bootstrap_parses_as_python_312() -> None:
    ast.parse((ROOT / "run_job.py").read_text(), feature_version=(3, 12))


def _module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _budget(tmp_path: Path, kind: str) -> dict[str, str]:
    init = _module("kml_ci_init", "init_budget.py")
    (tmp_path / "storage-baseline.json").write_text("{}\n")
    return init.initialize(tmp_path, kind, "a" * 40, time.time_ns() + 120_000_000_000)


def _child(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, **env, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_shared_child_generation_precharge_and_nested_wrapper(tmp_path: Path) -> None:
    env = _budget(tmp_path, "serving")
    script = """
import sitecustomize, types
module = types.SimpleNamespace()
module._next_token = lambda: 7
sitecustomize._patch_sampler(module)
def inner(*args, **kwargs):
    return [module._next_token(), module._next_token()]
module.generate_with_token_ids = inner
sitecustomize._patch_generation(module, 'generate_with_token_ids')
def outer(*args, **kwargs):
    return module.generate_with_token_ids(*args, **kwargs)
module.generate_result = outer
sitecustomize._patch_generation(module, 'generate_result')
assert module.generate_result(None, None, 'prompt', 8, 2, None) == [7, 7]
"""
    parent = (
        "import subprocess,sys\n"
        + f"code={script!r}\n"
        + "exec(code)\nsubprocess.run([sys.executable, '-c', code], check=True)\n"
    )
    result = _child(parent, env)
    assert result.returncode == 0, result.stderr
    status = AttemptBudget(Path(env["KML_CI_BUDGET_LEDGER"])).status()
    assert (status["charged_generation_calls"], status["charged_generated_tokens"]) == (
        2,
        4,
    )
    actual = [
        json.loads(path.read_text()) for path in (tmp_path / "events").glob("*.json")
    ]
    assert sum(row.get("actual_sampled_tokens", 0) for row in actual) == 4


def test_zero_quota_refuses_generation_before_double_runs(tmp_path: Path) -> None:
    env = _budget(tmp_path, "zero")
    result = _child(
        """
import sitecustomize, types
module = types.SimpleNamespace(generate_with_token_ids=lambda *a, **k: (_ for _ in ()).throw(AssertionError('ran')))
sitecustomize._patch_generation(module, 'generate_with_token_ids')
module.generate_with_token_ids(None, None, 'prompt', 8, 1, None)
""",
        env,
    )
    assert result.returncode != 0
    assert "shared generation calls limit" in result.stderr
    assert "AssertionError: ran" not in result.stderr
    assert (
        AttemptBudget(Path(env["KML_CI_BUDGET_LEDGER"])).status()[
            "charged_generation_calls"
        ]
        == 0
    )


def test_failed_decode_retains_charge_and_unmetered_sampler_is_refused(
    tmp_path: Path,
) -> None:
    env = _budget(tmp_path, "serving")
    result = _child(
        """
import sitecustomize, types
module = types.SimpleNamespace(_next_token=lambda: 1, generate_with_token_ids=lambda *a, **k: (_ for _ in ()).throw(ValueError('decoder failed')))
sitecustomize._patch_sampler(module)
sitecustomize._patch_generation(module, 'generate_with_token_ids')
try:
    module.generate_with_token_ids(None,None,'prompt',8,2,None)
except ValueError as error:
    assert str(error) == 'decoder failed'
else:
    raise AssertionError('failure was hidden')
try:
    module._next_token()
except RuntimeError as error:
    assert 'unmetered CI token sampling' in str(error)
else:
    raise AssertionError('unmetered sampling was allowed')
""",
        env,
    )
    assert result.returncode == 0, result.stderr
    status = AttemptBudget(Path(env["KML_CI_BUDGET_LEDGER"])).status()
    assert (status["charged_generation_calls"], status["charged_generated_tokens"]) == (
        1,
        2,
    )
    event = next((tmp_path / "events").glob("generation-*.json"))
    assert json.loads(event.read_text())["completed"] is False


def test_update_double_and_unmetered_optimizer_refusal(tmp_path: Path) -> None:
    env = _budget(tmp_path, "serving")
    result = _child(
        """
import sitecustomize, types
class Engine:
    def train_update(self, microbatches, update_index, valid_targets):
        return types.SimpleNamespace(status='SKIPPED')
module = types.SimpleNamespace(PyTorchEngine=Engine)
sitecustomize._patch_engine(module)
token = sitecustomize._update_context.set(('original', 32))
try:
    assert Engine().train_update([], 1, 32).status == 'SKIPPED'
finally:
    sitecustomize._update_context.reset(token)
import torch
parameter = torch.nn.Parameter(torch.zeros(1))
optimizer = torch.optim.AdamW([parameter])
try:
    optimizer.step()
except RuntimeError as error:
    assert 'unmetered CI optimizer.step' in str(error)
else:
    raise AssertionError('optimizer escaped guard')
""",
        env,
    )
    assert result.returncode == 0, result.stderr
    status = AttemptBudget(Path(env["KML_CI_BUDGET_LEDGER"])).status()
    assert (status["charged_updates"], status["charged_actual_target_positions"]) == (
        1,
        32,
    )


def test_optimizer_step_is_single_and_inside_reserved_update(tmp_path: Path) -> None:
    env = _budget(tmp_path, "serving")
    result = _child(
        """
import sitecustomize, types
class Optimizer:
    def step(self):
        return 'stepped'
module = types.SimpleNamespace(Optimizer=Optimizer)
sitecustomize._patch_optimizer(module)
class Engine:
    def train_update(self, microbatches, update_index, valid_targets):
        assert Optimizer().step() == 'stepped'
        try:
            Optimizer().step()
        except RuntimeError as error:
            assert 'unmetered CI optimizer.step' in str(error)
        else:
            raise AssertionError('second optimizer step escaped')
        return types.SimpleNamespace(status='APPLIED')
engine_module = types.SimpleNamespace(PyTorchEngine=Engine)
sitecustomize._patch_engine(engine_module)
token = sitecustomize._update_context.set(('original', 32))
try:
    assert Engine().train_update([], 1, 32).status == 'APPLIED'
finally:
    sitecustomize._update_context.reset(token)
try:
    Optimizer().step()
except RuntimeError as error:
    assert 'unmetered CI optimizer.step' in str(error)
else:
    raise AssertionError('step outside engine escaped')
""",
        env,
    )
    assert result.returncode == 0, result.stderr
    status = AttemptBudget(Path(env["KML_CI_BUDGET_LEDGER"])).status()
    assert (status["charged_updates"], status["charged_actual_target_positions"]) == (
        1,
        32,
    )


def test_import_hook_rejects_fixture_drift_before_model_init(tmp_path: Path) -> None:
    env = _budget(tmp_path, "serving")
    result = _child(
        """
from types import SimpleNamespace
from sparselab.training.trainer import train
from sparselab.evaluation.generation import generate_with_token_ids
from sparselab.evaluation.generation_request import generate_result
from sparselab.evaluation import generation_request
assert train._kml_ci_guarded
assert generate_with_token_ids._kml_ci_guarded
assert generate_result._kml_ci_guarded
assert generation_request._next_token._kml_ci_guarded
config = SimpleNamespace(training=SimpleNamespace(max_steps=12, seq_len=16, micro_batch_size=2, gradient_accumulation=1, max_tokens=384), runtime=SimpleNamespace(backend='cpu', precision='fp32'))
try:
    train(config, run_id='original')
except RuntimeError as error:
    assert 'unreviewed CI training fixture' in str(error)
else:
    raise AssertionError('drifted fixture reached trainer')
""",
        env,
    )
    assert result.returncode == 0, result.stderr
    assert (
        AttemptBudget(Path(env["KML_CI_BUDGET_LEDGER"])).status()["charged_updates"]
        == 0
    )


def test_supervisor_kills_detached_term_ignoring_child_and_preserves_sentinel(
    tmp_path: Path,
) -> None:
    if sys.platform != "linux":
        pytest.skip("Linux subreaper branch is exercised on Linux")
    runner = _module("kml_ci_runner", "run_job.py")
    root, workspace = tmp_path / "output", tmp_path / "checkout"
    root.mkdir()
    workspace.mkdir()
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True
    )
    child_record = root / "child.pid"
    script = (
        "import os, signal, subprocess, sys, time; "
        "p=subprocess.Popen([sys.executable,'-c','import signal,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)'],start_new_session=True); "
        f"open({str(child_record)!r},'w').write(str(p.pid)); "
        "time.sleep(0.5)"
    )
    try:
        with pytest.raises(RuntimeError, match="leader exited with owned descendants"):
            runner.run_phase(
                [sys.executable, "-c", script],
                cwd=workspace,
                root=root,
                workspace=workspace,
                environment=os.environ.copy(),
                deadline_ns=time.time_ns() + 25_000_000_000,
                name="detached",
            )
        assert child_record.exists()
        child_pid = int(child_record.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert sentinel.poll() is None
        receipt = json.loads((root / "detached.monitor.json").read_text())
        assert receipt["living_descendants"] == 0
    finally:
        sentinel.send_signal(signal.SIGKILL)
        sentinel.wait(timeout=5)


def test_sampler_fails_closed_on_real_io_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    original = os.scandir
    child = tmp_path / "child"
    child.mkdir()

    def forbidden(path):
        if os.fspath(path) == os.fspath(child):
            raise PermissionError("blocked child")
        return original(path)

    monkeypatch.setattr(os, "scandir", forbidden)
    with pytest.raises(PermissionError, match="blocked child"):
        runner.sample_tree(tmp_path)


def test_sampler_does_not_swallow_io_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    child = tmp_path / "child"
    child.mkdir()
    original = os.scandir

    def broken(path):
        if os.fspath(path) == os.fspath(child):
            raise OSError(5, "fixture I/O fault")
        return original(path)

    monkeypatch.setattr(os, "scandir", broken)
    with pytest.raises(OSError, match="fixture I/O fault"):
        runner.sample_tree(tmp_path)


def test_sampler_large_tree_keeps_live_sqlite_and_hardlink_count(
    tmp_path: Path,
) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    for number in range(10_000):
        (tmp_path / f"row-{number:05d}").write_bytes(b"x")
    (tmp_path / "attempt.sqlite").write_bytes(b"ledger")
    os.link(tmp_path / "attempt.sqlite", tmp_path / "attempt-copy.sqlite")
    progress: dict = {}
    byte_count, inodes = runner.sample_tree(tmp_path, progress=progress)
    assert inodes == 10_003
    assert byte_count >= 10_006
    assert progress["vanished_entries"] == 0


def test_sampler_skips_only_vanished_entry_and_counts_live_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    removed = tmp_path / "a-removed"
    removed.write_bytes(b"old")
    live = tmp_path / "z-attempt.sqlite"
    live.write_bytes(b"live-ledger")
    original = os.scandir

    def transient(path):
        if os.fspath(path) == os.fspath(tmp_path):
            with original(path) as listing:
                entries = list(listing)
            removed.unlink()
            return nullcontext(iter(entries))
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", transient)
        progress: dict = {}
        byte_count, inodes = runner.sample_tree(tmp_path, progress=progress)
    assert inodes == 2
    assert byte_count >= len(b"live-ledger")
    assert progress["vanished_entries"] == 1


def test_sampler_deadline_reports_root_and_last_path(tmp_path: Path) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    progress: dict = {}
    with pytest.raises(TimeoutError, match="deadline exhausted") as failure:
        runner.sample_declared_root(
            tmp_path,
            phase="startup-ambient-baseline",
            deadline=time.monotonic() - 1,
            required=True,
            progress=progress,
        )
    receipt = runner.sample_failure(progress, failure.value)
    assert receipt["root"] == str(tmp_path)
    assert receipt["phase"] == "startup-ambient-baseline"
    assert receipt["last_inspected_path"] == str(tmp_path)
    assert receipt["elapsed_seconds"] >= 0
    assert receipt["exception"].startswith("TimeoutError:")


def test_sampler_interrupts_blocked_read_and_restores_signal_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    previous = signal.getsignal(signal.SIGALRM)
    original = os.scandir

    def blocked(path):
        if os.fspath(path) == os.fspath(tmp_path):
            time.sleep(10)
        return original(path)

    monkeypatch.setattr(os, "scandir", blocked)
    with pytest.raises(TimeoutError, match="deadline exhausted"):
        runner.sample_tree(tmp_path, seconds=0.05)
    assert signal.getsignal(signal.SIGALRM) is previous
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_optional_root_stat_is_inside_sampling_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    original = Path.lstat

    def blocked(path):
        if path == tmp_path:
            time.sleep(10)
        return original(path)

    monkeypatch.setattr(Path, "lstat", blocked)
    with pytest.raises(TimeoutError, match="deadline exhausted"):
        runner.sample_optional_root(tmp_path, required=True, seconds=0.05)
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_sampler_refuses_occupied_timer_and_nonmain_thread(tmp_path: Path) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    previous = signal.getsignal(signal.SIGALRM)
    signal.setitimer(signal.ITIMER_REAL, 5)
    try:
        with pytest.raises(RuntimeError, match="occupied timer"):
            runner.sample_tree(tmp_path)
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    with (
        ThreadPoolExecutor(max_workers=1) as pool,
        pytest.raises(RuntimeError, match="main thread"),
    ):
        pool.submit(runner.sample_tree, tmp_path).result(timeout=5)


def test_supervisor_mac_path_cleans_same_group_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    monkeypatch.setattr(runner.sys, "platform", "darwin")
    root, workspace = tmp_path / "output", tmp_path / "checkout"
    root.mkdir()
    workspace.mkdir()
    child_record = root / "child.pid"
    script = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)']); "
        f"open({str(child_record)!r},'w').write(str(p.pid)); "
        "time.sleep(0.5)"
    )
    with pytest.raises(RuntimeError, match="leader exited with owned descendants"):
        runner.run_phase(
            [sys.executable, "-c", script],
            cwd=workspace,
            root=root,
            workspace=workspace,
            environment=os.environ.copy(),
            deadline_ns=time.time_ns() + 25_000_000_000,
            name="mac-branch",
        )
    child_pid = int(child_record.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_supervisor_storage_cap_stops_worker_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    root, workspace = tmp_path / "output", tmp_path / "checkout"
    root.mkdir()
    workspace.mkdir()
    monkeypatch.setattr(runner, "LIMIT_DISK", 4096)
    pid_file = root / "worker.pid"
    script = (
        "import os,time; "
        f"open({str(pid_file)!r},'w').write(str(os.getpid())); "
        f"open({str(root / 'payload')!r},'w').write('x'*8192); "
        "time.sleep(20)"
    )
    with pytest.raises(RuntimeError, match="resource cap"):
        runner.run_phase(
            [sys.executable, "-c", script],
            cwd=workspace,
            root=root,
            workspace=workspace,
            environment=os.environ.copy(),
            deadline_ns=time.time_ns() + 25_000_000_000,
            name="disk-cap",
        )
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_supervisor_deadline_reserves_term_to_kill(tmp_path: Path) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    root, workspace = tmp_path / "output", tmp_path / "checkout"
    root.mkdir()
    workspace.mkdir()
    pid_file = root / "worker.pid"
    script = (
        "import os,signal,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"open({str(pid_file)!r},'w').write(str(os.getpid())); "
        "time.sleep(20)"
    )
    with pytest.raises((RuntimeError, TimeoutError), match="deadline|reserve"):
        runner.run_phase(
            [sys.executable, "-c", script],
            cwd=workspace,
            root=root,
            workspace=workspace,
            environment=os.environ.copy(),
            deadline_ns=time.time_ns() + 16_000_000_000,
            name="deadline",
        )
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_shutdown_reserve_rejects_phase_before_start(tmp_path: Path) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    root, workspace = tmp_path / "output", tmp_path / "checkout"
    root.mkdir()
    workspace.mkdir()
    with pytest.raises(TimeoutError, match="shutdown reserve"):
        runner.run_phase(
            [sys.executable, "-c", "raise AssertionError('started')"],
            cwd=workspace,
            root=root,
            workspace=workspace,
            environment=os.environ.copy(),
            deadline_ns=time.time_ns() + 1_000_000_000,
            name="reserve",
        )
    assert not (root / "reserve.stdout.log").exists()


def test_workflow_dispatch_is_pinned_guarded_and_full_cpu_is_skipped() -> None:
    workflow = yaml.safe_load(
        (ROOT.parents[2] / ".github/workflows/ci.yml").read_text()
    )
    trigger = workflow.get("on", workflow.get(True))
    inputs = trigger["workflow_dispatch"]["inputs"]
    assert inputs["serving_only"]["default"] is False
    assert "qualification_sha" in inputs
    jobs = workflow["jobs"]
    for name in ("lint", "fast", "evidence-macos-arm64", "serving-smoke"):
        text = json.dumps(jobs[name]["steps"])
        assert "--start-watch" in text
        assert "--stop-watch" in text
        assert "qualification_sha" in text
        assert "run_job.py --job" in text
        assert "upload-artifact@v4" in text
    assert (
        jobs["test"]["if"]
        == "github.event_name == 'workflow_dispatch' && !inputs.serving_only"
    )
    assert jobs["test-macos-arm64"]["if"] == jobs["test"]["if"]
    assert (
        sum(
            jobs[name]["timeout-minutes"]
            for name in ("lint", "fast", "evidence-macos-arm64")
        )
        + 2 * jobs["serving-smoke"]["timeout-minutes"]
        == 45
    )


def test_guarded_collection_has_no_model_charge_and_detects_omission(
    tmp_path: Path,
) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    env = _budget(tmp_path, "serving")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *runner.TESTS["serving"],
        ],
        cwd=ROOT.parents[2],
        env={
            **os.environ,
            **env,
            "PYTHONPATH": str(ROOT),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "KML_CI_PREFLIGHT_ONLY": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    runner.verify_collection(result.stdout, "serving")
    with pytest.raises(RuntimeError, match="node collection differs"):
        runner.verify_collection(
            result.stdout.replace("tests/test_serving.py::", "tests/omitted.py::", 1),
            "serving",
        )
    status = AttemptBudget(Path(env["KML_CI_BUDGET_LEDGER"])).status()
    assert status["charged_updates"] == status["charged_generation_calls"] == 0


def test_watchdog_survives_step_boundary_and_records_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / "file").write_text("fixture")
    subprocess.run(["git", "-C", str(checkout), "add", "file"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setenv("GITHUB_WORKSPACE", str(checkout))
    monkeypatch.setattr(
        runner, "hosted_deadline_ns", lambda: time.time_ns() + 30_000_000_000
    )
    monkeypatch.setattr(
        runner, "_runner_worker", lambda rows: (os.getpid(), rows[os.getpid()][4])
    )
    root = tmp_path / "watch"
    runner.start_watcher(root, commit)
    try:
        runner._watcher_identity(root)
        runner.stop_watcher(root)
        assert (
            json.loads((root / "watch-receipt.json").read_text())["reason"]
            == "completed"
        )
    finally:
        record = json.loads((root / "watch-pid.json").read_text())
        rows = runner.process_rows()
        if (
            record["pid"] in rows
            and rows[record["pid"]][4] == record["birth"]
            and not rows[record["pid"]][3].startswith("Z")
        ):
            os.kill(record["pid"], signal.SIGKILL)
            pytest.fail("watcher survived expected shutdown")


def test_startup_retains_failed_root_and_finalizes_without_watcher_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    (checkout / "file").write_text("fixture")
    subprocess.run(["git", "-C", str(checkout), "add", "file"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    monkeypatch.setenv("GITHUB_WORKSPACE", str(checkout))
    monkeypatch.setenv("RUNNER_TEMP", str(ambient))
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(
        runner, "hosted_deadline_ns", lambda: time.time_ns() + 60_000_000_000
    )
    monkeypatch.setattr(
        runner, "_runner_worker", lambda rows: (os.getpid(), rows[os.getpid()][4])
    )

    def fail(path, *, phase, deadline, required, progress):
        progress.update(
            root=str(path),
            phase=phase,
            last_path=str(path / "last-file"),
            started_ns=time.monotonic_ns(),
            vanished_entries=0,
        )
        raise TimeoutError("fixture traversal expired")

    monkeypatch.setattr(runner, "sample_declared_root", fail)
    root = tmp_path / "watch"
    with pytest.raises(TimeoutError, match="fixture traversal expired"):
        runner.start_watcher(root, commit)
    startup = json.loads((root / "startup.json").read_text())
    failure = json.loads((root / "startup-failure.json").read_text())
    assert startup["ambient_roots"] == [str(ambient)]
    assert failure["root"] == str(ambient)
    assert failure["phase"] == "startup-ambient-baseline"
    assert failure["last_inspected_path"] == str(ambient / "last-file")
    assert failure["elapsed_seconds"] >= 0
    assert failure["exception"] == "TimeoutError: fixture traversal expired"
    assert not (root / "watch-pid.json").exists()
    runner.stop_watcher(root)
    assert (
        json.loads((root / "finalization.json").read_text())["state"]
        == "startup-failed"
    )


def test_finalization_records_incomplete_startup_without_pid(tmp_path: Path) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    root = tmp_path / "watch"
    root.mkdir()
    (root / "startup.json").write_text('{"state":"started"}\n')
    runner.stop_watcher(root)
    assert (
        json.loads((root / "finalization.json").read_text())["state"]
        == "startup-incomplete"
    )


def test_uv_run_preserves_guard_in_python_child(tmp_path: Path) -> None:
    env = _budget(tmp_path, "zero")
    result = subprocess.run(
        [
            "uv",
            "run",
            "--locked",
            "--no-sync",
            "python",
            "-c",
            "import sitecustomize; from sparselab.training.trainer import train; assert train._kml_ci_guarded",
        ],
        cwd=ROOT.parents[2],
        env={**os.environ, **env, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_hosted_deadline_uses_workflow_creation_time_including_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    created = datetime.now(UTC) - timedelta(minutes=31)
    payload = {
        "id": 123,
        "head_sha": "a" * 40,
        "created_at": created.isoformat().replace("+00:00", "Z"),
    }
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repository")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_API_URL", "https://api.github.invalid")
    monkeypatch.setenv("KML_CI_GITHUB_TOKEN", "fixture")
    monkeypatch.setenv("KML_CI_EXPECTED_SHA", "a" * 40)
    monkeypatch.setattr(
        runner.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(payload).encode()),
    )
    assert runner.hosted_deadline_ns() < time.time_ns()


def test_optional_storage_root_disappearance_fails_closed(tmp_path: Path) -> None:
    runner = _module("kml_ci_runner", "run_job.py")
    path = tmp_path / "cache"
    path.mkdir()
    assert runner.sample_optional_root(path, required=True)[1] == 1
    path.rmdir()
    with pytest.raises(RuntimeError, match="disappeared"):
        runner.sample_optional_root(path, required=True)
