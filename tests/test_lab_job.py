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
def job_factory(tmp_path, monkeypatch):
    """Use actual admission/runner/gates; substitute only native model operations.

    The private substitution is committed into the temporary fixture repository,
    never exposed as a manifest kind, environment switch or production hook.
    Native integration calls this factory with synthetic=False.
    """

    def make(*, synthetic=True):
        repo = tmp_path / "repo"
        repo.mkdir()
        original = Path(jobs.__file__).resolve().parents[1]
        shutil.copytree(
            original, repo / "src", ignore=shutil.ignore_patterns("__pycache__")
        )
        if synthetic:
            (repo / "fixture_payload.py").write_text(
                "import sys\nassert sum(range(4)) == 6\nprint(sys.argv[1])\n"
            )
            fixture = repo / "src/sparselab/_lab_job_fixture.py"
            fixture.write_text(
                "def command(job, source, evidence, scratch, gate):\n"
                "    if gate == 'runtime':\n"
                "        from sparselab.lab_job import _runtime_probe\n"
                "        return _runtime_probe(job, source)\n"
                "    return [str(job.host.python), '-I', '-B', str(source / 'fixture_payload.py'), gate]\n"
                "def verify(job, evidence, binding):\n"
                "    return {'fixture': 'arithmetic-only'}\n"
            )
            module = repo / "src/sparselab/lab_job.py"
            # CLI dispatch imports this module; no executable test-only surface
            # is added to the product. Native verification has separate coverage.
            module.write_text(
                module.read_text()
                + "\nfrom sparselab._lab_job_fixture import command as _command, verify as verify_native_result\n"
            )
            monkeypatch.setattr(
                jobs,
                "verify_native_result",
                lambda *args: {"fixture": "arithmetic-only"},
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
        root = tmp_path / "declarations"
        root.mkdir()
        inputs = Path(__file__).resolve().parents[1] / "examples/lab-job-try"
        names = (
            "baseline.yaml",
            "tokenizer.yaml",
            "delta.yaml",
            "try.yaml",
            "bounds.yaml",
            "train.jsonl",
            "validation.jsonl",
        )
        for name in names:
            shutil.copyfile(inputs / name, root / name)
        import platform

        import torch

        monkeypatch.setattr(jobs, "__file__", str(repo / "src/sparselab/lab_job.py"))
        return jobs.Job(
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
            workload=jobs.TryWorkload(
                kind="lab_try",
                profile="tiny-cpu-lab",
                root=root,
                tokenizer="tokenizer.yaml",
                baseline="baseline.yaml",
                candidate="delta.yaml",
                declaration="try.yaml",
                files={name: jobs.sha256_file(root / name) for name in names},
            ),
            policy=MonitorPolicy(
                monitor_policy_version=1,
                max_wall_seconds=90,
                max_tree_rss_bytes=2_000_000_000,
                interval_seconds=0.05,
            ),
            max_evidence_bytes=4_000_000,
        )

    yield make
    # Keep actual supervisor evidence in failing CI output, where temporary
    # fixture directories are otherwise unavailable after the runner exits.
    import json

    for completion in tmp_path.glob("evidence/*/monitor/completion.json"):
        evidence = completion.parent.parent
        receipt = json.loads(completion.read_bytes())
        if (evidence / "gates.json").exists() and receipt["status"] not in {
            "VIOLATED",
            "MONITOR_ERROR",
        }:
            continue
        print(
            "INCOMPLETE_OR_VIOLATED_GATE_RECEIPT",
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


@pytest.fixture
def job(job_factory):
    return job_factory()


@pytest.mark.parametrize("mask", [0o000, 0o002, 0o022])
def test_end_to_end_and_duplicate_returns_verified_result(job, mask):
    previous = os.umask(mask)
    try:
        result = jobs.admit(job)
    finally:
        observed = os.umask(previous)
    assert observed == mask
    evidence, _, _ = jobs._paths(job)
    logs = evidence / "monitor"
    assert logs.stat().st_mode & 0o777 == 0o700
    assert {path.name for path in logs.iterdir()} == {
        "command.stdout.log",
        "command.stderr.log",
        "events.jsonl",
        "owned-ready.json",
        "owned-completion.json",
        "completion.json",
    }
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in logs.iterdir())
    assert result["summary"]["controller_completion"] == "PASS"
    assert result["summary"]["coordinator_acceptance"] == "PENDING"
    assert result["summary"]["cleanup_verified"] is True
    assert result["summary"]["native_workflow"] == {"fixture": "arithmetic-only"}
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
    # Deliberately unsafe fixture evidence must still refuse, without repair.
    target = logs / "command.stdout.log"
    target.chmod(0o664)
    retained = target.read_bytes(), target.stat().st_mtime_ns
    with pytest.raises(ValueError, match="unsafe runner state"):
        jobs.collect(job)
    assert target.stat().st_mode & 0o777 == 0o664
    assert (target.read_bytes(), target.stat().st_mtime_ns) == retained


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


def test_different_filesystem_refuses_before_claim_or_scratch(job, monkeypatch):
    actual_stat = Path.stat
    different_device = job.host.evidence.stat().st_dev + 1

    def stat(path, *args, **kwargs):
        result = actual_stat(path, *args, **kwargs)
        if path == job.host.work:
            fields = list(result)
            fields[2] = different_device  # os.stat_result.st_dev
            return os.stat_result(fields)
        return result

    def unexpected_launch(*args, **kwargs):
        pytest.fail("different-filesystem admission reached launch")

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(jobs.runner, "run_controlled", unexpected_launch)
    with pytest.raises(ValueError, match="must share the monitored filesystem"):
        jobs.admit(job)
    assert not list(job.host.state.glob("*/pending.json"))
    assert not (job.host.evidence / f"attempt-{job.attempt_id}.json").exists()
    assert not (job.host.evidence / job.job_id).exists()
    assert not (job.host.work / "scratch").exists()


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
    (evidence / "report.stdout").rename(evidence / "retained-report.stdout")
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


def test_manifest_rejects_gpu_pytest_and_arbitrary_commands(job, tmp_path):
    for updates in (
        {"backend": "rocm"},
        {
            "workload": {
                "kind": "pytest",
                "profile": "cpu-regression",
                "nodes": ["tests/test_tiny.py::test_one"],
            }
        },
        {"workload": {**job.workload.model_dump(), "command": ["sh", "-c", "true"]}},
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
    results = [process.communicate(timeout=120) for process in processes]
    assert any(process.returncode == 0 for process in processes), results
    assert len(list(job.host.state.glob("*/pending.json"))) == 1
    assert jobs.collect(job)["summary"]["native_workflow"] == {
        "fixture": "arithmetic-only"
    }


@pytest.mark.parametrize("failed_gate", ["tokenizer", "try"])
def test_failed_gate_and_skipped_dependencies_are_not_acceptance(job, failed_gate):
    (job.host.repository / "fixture_payload.py").write_text(
        f"import sys\nraise SystemExit(3 if sys.argv[1] == {failed_gate!r} else 0)\n"
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
    gates = result["summary"]["gates"]
    failed = next(
        index for index, gate in enumerate(gates) if gate["name"] == failed_gate
    )
    assert gates[failed]["executed"] and gates[failed]["returncode"] == 3
    assert all(
        not gate["executed"] and gate["returncode"] is None
        for gate in gates[failed + 1 :]
    )
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


@pytest.mark.parametrize(
    "prefix",
    [
        ["--hf-token-file", "/never-read/token"],
        ["--hf-token-file=/never-read/token"],
    ],
)
def test_public_entry_rejects_unused_global_token_without_reading(
    monkeypatch, capsys, prefix
):
    from sparselab.cli import entry

    # Exercise the merged LAB69 parser, not a fabricated parsed namespace.
    monkeypatch.setattr(
        sys, "argv", ["sparselab", *prefix, "lab-job", "status", "/job.json"]
    )
    monkeypatch.setattr(jobs, "main", lambda argv: pytest.fail("must not dispatch"))
    monkeypatch.setattr(
        Path, "read_bytes", lambda path: pytest.fail("must not read token")
    )
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    assert "global work/token options are unsupported" in capsys.readouterr().err
