"""Admission fault injection and tiny CPU subprocesses; no models or host pilots."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sparselab import lab_job as jobs
from sparselab.campaign.state import publish_immutable
from sparselab.operational_monitor import MonitorPolicy

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native Linux runner")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def job(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    original = Path(jobs.__file__).resolve().parents[1]
    shutil.copytree(
        original, repo / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    (repo / "tests").mkdir()
    (repo / "tests/test_tiny.py").write_text(
        "def test_one():\n    assert sum(range(4)) == 6\n"
    )
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    sha = git(repo, "rev-parse", "HEAD")
    git(repo, "update-ref", "refs/remotes/origin/main", sha)
    for name in ("state", "evidence", "work"):
        (tmp_path / name).mkdir(mode=0o700)
    import platform

    import torch

    monkeypatch.setattr(jobs, "__file__", str(repo / "src/sparselab/lab_job.py"))
    admitted = jobs.Job(
        format="sparselab-lab-job-v1",
        job_id="one",
        attempt_id="authorized-one",
        source_sha=sha,
        validation_reference="synthetic fixture only",
        backend="cpu",
        host=jobs.Host(
            repository=repo,
            state=tmp_path / "state",
            evidence=tmp_path / "evidence",
            work=tmp_path / "work",
            lock=tmp_path / "lock",
            blocked=tmp_path / "blocked",
            python=Path(sys.executable),
        ),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        runtime_sha256=jobs.runtime_identity()["runtime_sha256"],
        workload=jobs.PytestWorkload(
            kind="pytest",
            profile="cpu-regression",
            pytest_version=pytest.__version__,
            nodes=["tests/test_tiny.py::test_one"],
        ),
        policy=MonitorPolicy(
            monitor_policy_version=1,
            max_wall_seconds=20,
            max_tree_rss_bytes=2_000_000_000,
            interval_seconds=0.05,
        ),
        max_evidence_bytes=4_000_000,
    )

    yield admitted
    # Keep actual supervisor evidence in failing CI output, where temporary
    # fixture directories are otherwise unavailable after the runner exits.
    import json

    for completion in admitted.host.evidence.glob("*/monitor/completion.json"):
        evidence = completion.parent.parent
        if (evidence / "gates.json").exists():
            continue
        receipt = json.loads(completion.read_bytes())
        print(
            "INCOMPLETE_GATE_RECEIPT",
            json.dumps(
                {
                    key: receipt[key]
                    for key in (
                        "status",
                        "violations",
                        "elapsed_seconds",
                        "returncode",
                        "peak_tree_rss_bytes",
                        "sample_count",
                    )
                },
                sort_keys=True,
            ),
        )
        for name in ("monitor/command.stderr.log", "runtime.stderr", "runtime.stdout"):
            path = evidence / name
            if path.exists():
                print(name, path.read_text()[-4096:])


def test_end_to_end_and_duplicate_returns_verified_result(job):
    result = jobs.admit(job)
    assert result["summary"]["controller_completion"] == "PASS"
    assert result["summary"]["coordinator_acceptance"] == "PENDING"
    assert result["summary"]["cleanup_verified"] is True
    assert result["summary"]["tests"] == 1
    evidence, _, _ = jobs._paths(job)
    assert result["summary"]["evidence_bytes"] == sum(
        path.stat().st_size for path in evidence.rglob("*") if path.is_file()
    )
    assert jobs.admit(job) == result
    assert jobs.collect(job) == result
    assert len(list(job.host.state.glob("*/pending.json"))) == 1
    pending = jobs.runner._read(next(job.host.state.glob("*/pending.json")))
    assert pending["job_binding"]["manifest_sha256"] == job.identity()
    _, scratch, source = jobs._paths(job)
    assert git(source, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    plan = jobs.cleanup_preview(job, scratch)
    assert plan["preview_only"] and not plan["delete_implemented"]
    assert scratch.exists()


def test_conflicting_handoff_and_reused_authorization(job, monkeypatch):
    monkeypatch.setattr(
        jobs.runner,
        "run_controlled",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop before launch")),
    )
    with pytest.raises(RuntimeError, match="stop before launch"):
        jobs.admit(job)
    assert jobs.admit(job)["status"] == "UNKNOWN_OR_RUNNING"
    with pytest.raises(ValueError, match="conflicting job"):
        jobs.admit(job.model_copy(update={"validation_reference": "different"}))
    with pytest.raises(FileExistsError):
        jobs.admit(job.model_copy(update={"job_id": "other"}))
    assert not list(job.host.state.glob("*/pending.json"))


def test_concurrent_admission_only_one_launch(job, monkeypatch):
    # Re-enter while the first admission holds its durable claim: same guarantee
    # as a second process arriving before the native host flock has been acquired.
    seen = []

    def enter(*args, **kwargs):
        seen.append(jobs.admit(job))
        raise RuntimeError("interrupted launch window")

    monkeypatch.setattr(jobs.runner, "run_controlled", enter)
    with pytest.raises(RuntimeError, match="interrupted launch"):
        jobs.admit(job)
    assert len(seen) == 1
    assert seen[0]["status"] == "UNKNOWN_OR_RUNNING"


@pytest.mark.parametrize("damage", ["source", "unmerged", "dirty"])
def test_stale_source_refuses_before_binding(job, damage):
    if damage == "source":
        job = job.model_copy(update={"source_sha": "0" * 40})
    elif damage == "dirty":
        (job.host.repository / "untracked").touch()
    else:
        git(job.host.repository, "update-ref", "-d", "refs/remotes/origin/main")
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        jobs.admit(job)
    assert not (job.host.evidence / job.job_id).exists()


def test_stale_runtime_records_real_exit_and_skips_workload(job):
    result = jobs.admit(job.model_copy(update={"torch_version": "not-installed"}))
    gates = result["summary"]["gates"]
    assert gates[1]["executed"] and gates[1]["returncode"] != 0
    assert gates[2]["executed"] is False and gates[2]["returncode"] is None
    assert result["summary"]["controller_completion"] == "FAIL"


@pytest.mark.parametrize("damage", ["held", "partial", "blocked"])
def test_host_state_refuses_without_execution(job, damage):
    import fcntl

    with job.host.lock.open("w") as lock:
        if damage == "held":
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif damage == "blocked":
            job.host.blocked.write_text("partial")
        else:
            (job.host.state / "contract.json").write_text("{")
        with pytest.raises((ValueError, OSError)):
            jobs.admit(job)
    assert not list(job.host.state.glob("*/pending.json"))
    assert not (job.host.evidence / job.job_id / "monitor").exists()


def test_partial_handoff_remains_blocked(job):
    directory = job.host.evidence / job.job_id
    directory.mkdir()
    (directory / "manifest.json").write_text("{")
    with pytest.raises(ValueError):
        jobs.admit(job)
    assert not list(job.host.state.glob("*/pending.json"))


def test_unknown_native_launch_never_restarts(job, monkeypatch):
    monkeypatch.setattr(
        jobs.runner,
        "monitor_command",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unknown launch")),
    )
    with pytest.raises(RuntimeError, match="unknown launch"):
        jobs.admit(job)
    assert job.host.blocked.exists()
    assert len(list(job.host.state.glob("*/pending.json"))) == 1
    assert jobs.admit(job)["status"] == "UNKNOWN_OR_RUNNING"
    with pytest.raises(FileNotFoundError):
        jobs.collect(job)


def test_missing_evidence_and_failed_final_publication_never_relaunch(job, monkeypatch):
    publish = jobs.publish_immutable

    def fail(path, payload):
        if path.name == "summary.json":
            raise OSError("publication failed")
        return publish(path, payload)

    monkeypatch.setattr(jobs, "publish_immutable", fail)
    with pytest.raises(OSError, match="publication failed"):
        jobs.admit(job)
    assert jobs.status(job)["status"] == "COLLECTABLE"
    assert jobs.admit(job)["status"] == "COLLECTABLE"
    monkeypatch.setattr(jobs, "publish_immutable", publish)
    assert jobs.collect(job)["summary"]["controller_completion"] == "PASS"
    assert len(list(job.host.state.glob("*/pending.json"))) == 1
    evidence, _, _ = jobs._paths(job)
    (evidence / "pytest.xml").rename(evidence / "retained.xml")
    with pytest.raises(FileNotFoundError):
        jobs.collect(job)
    with pytest.raises(FileNotFoundError):
        jobs.admit(job)


@pytest.mark.parametrize("target", ["state", "evidence", "work", "lock", "blocked"])
def test_unsafe_cleanup_plan_refuses_without_deletion(job, monkeypatch, target):
    monkeypatch.setattr(jobs, "_record", lambda job: {})
    with pytest.raises(ValueError, match="unsafe cleanup plan"):
        jobs.cleanup_preview(job, getattr(job.host, target))
    assert job.host.state.exists() and job.host.evidence.exists()


def test_manifest_rejects_gpu_and_broad_nodes(job, tmp_path):
    for updates in (
        {"backend": "rocm"},
        {"workload": {**job.workload.model_dump(), "nodes": ["tests/"]}},
        {
            "workload": {
                **job.workload.model_dump(),
                "nodes": [job.workload.nodes[0]] * 6,
            }
        },
    ):
        with pytest.raises(ValueError):
            jobs.Job.model_validate({**job.payload(), **updates})
    path = tmp_path / "manifest.json"
    publish_immutable(path, job.payload())
    assert jobs.load_job(path) == job
    path.write_text('{"job_id":"one","job_id":"two"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        jobs.load_job(path)


def test_two_processes_share_one_durable_admission(job, tmp_path):
    manifest = tmp_path / "handoff.json"
    publish_immutable(manifest, job.payload())
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-P",
                "-B",
                "-m",
                "sparselab",
                "lab-job",
                "admit",
                str(manifest),
            ],
            env={**os.environ, "PYTHONPATH": str(job.host.repository / "src")},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    results = [process.communicate(timeout=30) for process in processes]
    assert any(process.returncode == 0 for process in processes), results
    assert len(list(job.host.state.glob("*/pending.json"))) == 1
    assert jobs.collect(job)["summary"]["tests"] == 1


@pytest.mark.parametrize(
    "body", ["import pytest; pytest.skip('unavailable')", "assert False"]
)
def test_failed_or_skipped_node_is_not_acceptance(job, body):
    (job.host.repository / "tests/test_tiny.py").write_text(
        f"def test_one():\n    {body}\n"
    )
    git(job.host.repository, "add", ".")
    git(
        job.host.repository,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "negative fixture",
    )
    sha = git(job.host.repository, "rev-parse", "HEAD")
    git(job.host.repository, "update-ref", "refs/remotes/origin/main", sha)
    job = job.model_copy(update={"source_sha": sha})
    result = jobs.admit(job)
    assert result["summary"]["controller_completion"] == "FAIL"
    assert result["summary"]["failures"] + result["summary"]["skipped"] == 1
    assert jobs.admit(job) == result


def test_unsafe_linked_scratch_cleanup_refused(job):
    jobs.admit(job)
    _, scratch, _ = jobs._paths(job)
    (scratch / "borrowed").symlink_to(job.host.state, target_is_directory=True)
    with pytest.raises(ValueError, match="linked scratch"):
        jobs.cleanup_preview(job, scratch)
    assert job.host.state.exists()


def test_public_entry_uses_manifest_paths_without_default_workdir(monkeypatch):
    from sparselab.cli import entry

    seen = []
    monkeypatch.setattr(jobs, "main", lambda argv: seen.append(argv))
    monkeypatch.setattr(sys, "argv", ["sparselab", "lab-job", "status", "/job.json"])
    entry.main()
    assert seen == [["status", "/job.json"]]
    monkeypatch.setattr(
        sys,
        "argv",
        ["sparselab", "--work-dir", "/other", "lab-job", "status", "/job.json"],
    )
    with pytest.raises(SystemExit):
        entry.main()
    assert len(seen) == 1


def test_inherited_pytest_options_and_plugins_cannot_change_pilot(job, monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "--lab70-nonexistent-option")
    monkeypatch.setenv("PYTEST_PLUGINS", "lab70_nonexistent_host_plugin")
    result = jobs.admit(job)
    assert result["summary"]["controller_completion"] == "PASS"
    assert result["summary"]["tests"] == 1
    assert result["summary"]["gates"][-1]["returncode"] == 0


def test_worktree_preparation_never_runs_host_checkout_hook(job, tmp_path):
    marker = tmp_path / "host-hook-ran"
    hook = job.host.repository / ".git/hooks/post-checkout"
    hook.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('unexpected execution')\n"
    )
    hook.chmod(0o700)
    result = jobs.admit(job)
    assert result["summary"]["controller_completion"] == "PASS"
    assert not marker.exists()
    config = subprocess.run(
        [
            "git",
            "-C",
            str(job.host.repository),
            "config",
            "--local",
            "--get",
            "core.hooksPath",
        ],
        capture_output=True,
        check=False,
    )
    assert config.returncode == 1  # No persistent hook setting was written.
