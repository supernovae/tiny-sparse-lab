"""Bounded CPU pytest handoffs over the native host runner; no scheduler or retry."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sparselab import runner_control as runner
from sparselab.campaign.state import digest, publish_immutable, read_canonical
from sparselab.operational_monitor import MonitorCompletion, MonitorPolicy
from sparselab.training.manifest import canonical_json, sha256_file
from sparselab.workdir import ensure_scratch_dir
from sparselab.workers.transport import _strict_json


class Host(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    repository: Path
    lock: Path
    state: Path
    blocked: Path
    evidence: Path
    work: Path
    python: Path

    @field_validator("*")
    @classmethod
    def absolute(cls, value: Path) -> Path:
        if not value.is_absolute() or ".." in value.parts:
            raise ValueError("host paths must be explicit absolute paths")
        return value

    @model_validator(mode="after")
    def separate(self) -> Host:
        roots = [self.state, self.evidence, self.work, self.repository]
        for index, left in enumerate(roots):
            for right in roots[index + 1 :]:
                if left.is_relative_to(right) or right.is_relative_to(left):
                    raise ValueError(
                        "source, evidence, state and work must be separate"
                    )
        for marker in (self.lock, self.blocked):
            if any(marker.is_relative_to(root) for root in roots):
                raise ValueError("lock and blocked marker must be outside roots")
        return self


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    format: Literal["sparselab-lab-job-v1"]
    job_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    attempt_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    validation_reference: str = Field(min_length=1)
    backend: Literal["cpu"]
    host: Host
    python_version: str
    torch_version: str
    pytest_version: str
    nodes: list[str] = Field(min_length=1, max_length=5)
    policy: MonitorPolicy
    max_evidence_bytes: int = Field(strict=True, gt=0)

    @field_validator("nodes")
    @classmethod
    def explicit_nodes(cls, nodes: list[str]) -> list[str]:
        if len(set(nodes)) != len(nodes) or any(
            not re.fullmatch(r"tests/test_[a-zA-Z0-9_]+\.py::test_[a-zA-Z0-9_]+", node)
            for node in nodes
        ):
            raise ValueError(
                "select unique explicit test function nodes (maximum five)"
            )
        return nodes

    @model_validator(mode="after")
    def bounded(self) -> Job:
        if (
            self.policy.max_wall_seconds is None
            or self.policy.max_tree_rss_bytes is None
        ):
            raise ValueError("bounded job requires explicit wall and tree RSS limits")
        if (
            self.policy.max_added_workspace_bytes is not None
            or self.policy.max_added_workspace_inodes is not None
        ):
            raise ValueError(
                "use native disk floors; workspace baseline mode is not supported"
            )
        if self.policy.expected_device_uuid is not None:
            raise ValueError("GPU admission is outside this CPU adapter")
        return self

    def payload(self) -> dict:
        return self.model_dump(mode="json")

    def identity(self) -> str:
        return digest("sparselab-lab-job-v1", self.payload())


def load_job(path: Path) -> Job:
    # Reuse native wire parsing: allow formatted input, reject duplicate/nonfinite keys.
    return Job.model_validate(_strict_json(path.read_bytes()))


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _source(job: Job, root: Path, *, detached: bool) -> None:
    if _git(root, "rev-parse", "HEAD") != job.source_sha:
        raise ValueError("stale source SHA")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source workdir is not clean")
    if detached and _git(root, "rev-parse", "--abbrev-ref", "HEAD") != "HEAD":
        raise ValueError("source workdir must be detached")


def _paths(job: Job) -> tuple[Path, Path, Path]:
    evidence = job.host.evidence / job.job_id
    scratch = job.host.work / "scratch" / job.attempt_id
    return evidence, scratch, scratch / "source"


def _binding(job: Job, native_id: str) -> dict[str, str]:
    return {
        "job_id": job.job_id,
        "attempt_id": job.attempt_id,
        "manifest_sha256": job.identity(),
        "runner_run_id": native_id,
    }


def _record(job: Job) -> dict:
    evidence, _, _ = _paths(job)
    runner._safe(evidence, directory=True)
    saved = read_canonical(evidence / "manifest.json")
    if saved != job.payload():
        raise ValueError("conflicting job handoff")
    prepared = read_canonical(evidence / "prepared.json")
    if not re.fullmatch(r"[0-9a-f]{32}", prepared.get("runner_run_id", "")):
        raise ValueError("invalid native runner identity")
    if prepared != _binding(job, prepared["runner_run_id"]):
        raise ValueError("conflicting job binding")
    if runner._read(job.host.evidence / f"attempt-{job.attempt_id}.json") != prepared:
        raise ValueError("conflicting authorized attempt")
    return prepared


def status(job: Job) -> dict:
    prepared = _record(job)
    evidence, _, _ = _paths(job)
    summary = evidence / "summary.json"
    if summary.exists():
        return collect(job)
    complete = job.host.state / prepared["runner_run_id"] / "completion.json"
    return {
        **prepared,
        "status": "COLLECTABLE" if complete.exists() else "UNKNOWN_OR_RUNNING",
        "coordinator_acceptance": "PENDING",
    }


def admit(job: Job) -> dict:
    evidence, scratch, source = _paths(job)
    if os.path.lexists(evidence):
        return status(job)  # No launch branch on redelivery, including PREPARED.
    for root in (job.host.repository, job.host.state, job.host.evidence, job.host.work):
        runner._safe(root, directory=True)
        if root.resolve() != root:
            raise ValueError("host roots must be canonical, without symlink aliases")
    _source(job, job.host.repository, detached=False)
    if Path(__file__).resolve() != job.host.repository / "src/sparselab/lab_job.py":
        raise ValueError("adapter import does not match selected source")
    # No fetch and no moving target: the full selected SHA must already be merged.
    subprocess.run(
        [
            "git",
            "-C",
            str(job.host.repository),
            "merge-base",
            "--is-ancestor",
            job.source_sha,
            "refs/remotes/origin/main",
        ],
        check=True,
    )
    runner._unblocked(job.host.blocked)
    native_id = uuid.uuid4().hex
    prepared = _binding(job, native_id)
    try:
        evidence.mkdir(mode=0o700)
    except FileExistsError:
        return status(job)  # A concurrent partial publisher refuses in _record.
    runner._sync_dir(evidence.parent)
    publish_immutable(evidence / "manifest.json", job.payload())
    # Exclusive authorized-attempt index across job IDs. Never replace or reclaim.
    runner._write(job.host.evidence / f"attempt-{job.attempt_id}.json", prepared)
    publish_immutable(evidence / "prepared.json", prepared)
    ensure_scratch_dir(job.host.work)
    scratch.mkdir(mode=0o700)
    runner._sync_dir(scratch.parent)
    publish_immutable(evidence / "scratch.json", {"path": str(scratch), **prepared})
    subprocess.run(
        [
            "git",
            "-C",
            str(job.host.repository),
            "-c",
            "core.hooksPath=/dev/null",
            "worktree",
            "add",
            "--detach",
            "--quiet",
            str(source),
            job.source_sha,
        ],
        check=True,
    )
    policy_path = evidence / "policy.json"
    publish_immutable(policy_path, job.policy.model_dump(mode="json"))
    command = [
        str(job.host.python),
        "-I",
        "-B",
        "-c",
        (
            f"import sys; sys.path.insert(0, {str(source / 'src')!r}); "
            "from sparselab.lab_job import execute; "
            f"raise SystemExit(execute({str(evidence / 'manifest.json')!r}))"
        ),
    ]
    runner.run_controlled(
        command,
        job.policy,
        lock_path=job.host.lock,
        state_dir=job.host.state,
        blocked_path=job.host.blocked,
        run_id=native_id,
        job_binding=prepared,
        workspace=job.host.work,
        log_dir=evidence / "monitor",
        policy_path=policy_path,
        cwd=source,
    )
    return collect(job)


def _runtime_probe(job: Job, source: Path) -> list[str]:
    code = (
        "import sys,platform,json; "
        f"sys.path.insert(0,{str(source / 'src')!r}); "
        "import sparselab,torch,pytest; "
        "from pathlib import Path; "
        "actual=[platform.python_version(),str(torch.__version__),pytest.__version__]; "
        "print(json.dumps({'versions':actual,'import':sparselab.__file__})); "
        f"assert actual == {[job.python_version, job.torch_version, job.pytest_version]!r}, 'stale runtime'; "
        f"assert Path(sparselab.__file__).resolve() == Path({str(source / 'src/sparselab/__init__.py')!r}), 'stale import'"
    )
    return [str(job.host.python), "-I", "-B", "-c", code]


def execute(manifest: str) -> int:
    """Fixed gate payload, invoked only as the native monitor's owned child."""
    os.umask(0o077)  # Private evidence from this owned child, including pytest JUnit.
    job = load_job(Path(manifest))
    prepared = _record(job)
    evidence, scratch, source = _paths(job)
    env = dict(os.environ)
    env.update(
        dict.fromkeys(
            (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "HF_HUB_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "PYTHONDONTWRITEBYTECODE",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            ),
            "1",
        )
    )
    env.update(
        dict.fromkeys(
            ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"), ""
        )
    )
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env["SPARSELAB_WORK_DIR"] = str(scratch / "work")
    temporary = ensure_scratch_dir(scratch / "work")
    env.update(dict.fromkeys(("TMPDIR", "TEMP", "TMP"), str(temporary)))
    failed = False
    gates = []
    for name in ("source", "runtime", "pytest"):
        code = None
        detail = None
        if not failed:
            if name == "source":
                try:
                    _source(job, source, detached=True)
                    code = 0
                except (ValueError, OSError, subprocess.CalledProcessError) as error:
                    detail, code = str(error), 1
            else:
                if name == "pytest" and os.path.lexists(scratch / "pytest"):
                    raise ValueError(
                        "pytest scratch already exists; refusing implicit cleanup"
                    )
                command = (
                    _runtime_probe(job, source)
                    if name == "runtime"
                    else [
                        str(job.host.python),
                        "-I",
                        "-B",
                        "-c",
                        (
                            f"import sys; sys.path.insert(0,{str(source / 'src')!r}); "
                            "import pytest; raise SystemExit(pytest.main(sys.argv[1:]))"
                        ),
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "-o",
                        "addopts=",
                        "-m",
                        "not mps and not mlx and not cuda and not rocm and not xpu and not network",
                        "--basetemp",
                        str(scratch / "pytest"),
                        "--junitxml",
                        str(evidence / "pytest.xml"),
                        *job.nodes,
                    ]
                )
                with (
                    (evidence / f"{name}.stdout").open("xb") as out,
                    (evidence / f"{name}.stderr").open("xb") as err,
                ):
                    code = subprocess.run(
                        command,
                        cwd=source,
                        env=env,
                        stdout=out,
                        stderr=err,
                        check=False,
                    ).returncode
                    os.fsync(out.fileno())
                    os.fsync(err.fileno())
            failed = code != 0
        gate = {
            "name": name,
            "executed": code is not None,
            "returncode": code,
            "detail": detail,
        }
        publish_immutable(evidence / f"gate-{name}.json", {**prepared, **gate})
        gates.append(gate)
    junit = evidence / "pytest.xml"
    if junit.exists():
        runner._safe(junit)
        with junit.open("rb") as stream:
            os.fsync(stream.fileno())
    publish_immutable(evidence / "gates.json", {**prepared, "gates": gates})
    return int(failed)


def collect(job: Job) -> dict:
    """Verify and publish evidence only. Publication retry cannot reach admission."""
    prepared = _record(job)
    evidence, scratch, source = _paths(job)
    contract = runner._read(job.host.state / "contract.json")
    identity = runner._safe(job.host.lock)
    if (identity.st_dev, identity.st_ino) != (
        contract["lock_device"],
        contract["lock_inode"],
    ):
        raise ValueError("runner lock identity changed")
    runner._prior_runs(job.host.state, contract)
    pending = runner._read(job.host.state / prepared["runner_run_id"] / "pending.json")
    if contract["lock"] != str(job.host.lock) or contract["blocked"] != str(
        job.host.blocked
    ):
        raise ValueError("host contract changed")
    if pending.get("job_binding") != prepared or pending["log_dir"] != str(
        evidence / "monitor"
    ):
        raise ValueError("native completion binding mismatch")
    completion = MonitorCompletion.model_validate_json(
        (evidence / "monitor/completion.json").read_bytes()
    )
    gates = read_canonical(evidence / "gates.json")
    if {k: gates[k] for k in prepared} != prepared:
        raise ValueError("gate binding mismatch")
    for name, gate in zip(("source", "runtime", "pytest"), gates["gates"], strict=True):
        if gate["name"] != name or read_canonical(evidence / f"gate-{name}.json") != {
            **prepared,
            **gate,
        }:
            raise ValueError("missing or conflicting gate evidence")
    for gate in gates["gates"]:
        if gate["executed"] and gate["name"] != "source":
            for suffix in ("stdout", "stderr"):
                runner._safe(evidence / f"{gate['name']}.{suffix}")
    for name in ("command.stdout.log", "command.stderr.log", "events.jsonl"):
        runner._safe(evidence / "monitor" / name)
    passed = completion.status == "COMPLETE" and all(
        g["executed"] and g["returncode"] == 0 for g in gates["gates"]
    )
    tests = skipped = failures = 0
    if gates["gates"][-1]["executed"]:
        junit_path = evidence / "pytest.xml"
        runner._safe(junit_path)
        if junit_path.stat().st_size > job.max_evidence_bytes:
            raise ValueError("JUnit evidence exceeds manifest size cap")
        junit = ET.parse(junit_path)
        cases = list(junit.iter("testcase"))
        tests = len(cases)
        skipped = sum(case.find("skipped") is not None for case in cases)
        failures = sum(
            case.find("failure") is not None or case.find("error") is not None
            for case in cases
        )
        observed = {
            f"{case.get('classname', '').replace('.', '/')}.py::{case.get('name', '').split('[')[0]}"
            for case in cases
        }
        passed = (
            passed and set(job.nodes) <= observed and skipped == 0 and failures == 0
        )
    files = {}
    for path in sorted(evidence.rglob("*")):
        if path == evidence / "summary.json":
            continue
        if path.is_dir():
            runner._safe(path, directory=True)
            continue
        runner._safe(path)
        files[str(path.relative_to(evidence))] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    size = sum(item["bytes"] for item in files.values())
    try:
        _source(job, source, detached=True)
        source_unchanged = True
    except ValueError, OSError, subprocess.CalledProcessError:
        source_unchanged = False
    checks = {
        "source_unchanged": source_unchanged,
        "manifest": read_canonical(evidence / "manifest.json") == job.payload(),
        "size": size <= job.max_evidence_bytes,
        "time": completion.elapsed_seconds <= job.policy.max_wall_seconds,
    }
    summary = {
        **prepared,
        "controller_completion": "PASS" if passed and all(checks.values()) else "FAIL",
        "coordinator_acceptance": "PENDING",
        "cleanup_verified": True,
        "native_status": completion.status,
        "native_returncode": completion.returncode,
        "gates": gates["gates"],
        "tests": tests,
        "skipped": skipped,
        "failures": failures,
        "elapsed_seconds": completion.elapsed_seconds,
        "evidence_bytes": size,
        "checks": checks,
        "files": files,
        "scratch": str(scratch),
    }
    # Include the summary itself in the final byte check. Converge its byte-count
    # field and PASS/FAIL spelling before immutable publication.
    while True:
        total = size + len(canonical_json(summary)) + 1
        within_cap = total <= job.max_evidence_bytes
        completion_status = (
            "PASS"
            if passed and all({**checks, "size": within_cap}.values())
            else "FAIL"
        )
        if (
            summary["evidence_bytes"] == total
            and checks["size"] == within_cap
            and summary["controller_completion"] == completion_status
        ):
            break
        summary["evidence_bytes"] = total
        checks["size"] = within_cap
        summary["controller_completion"] = completion_status
    publish_immutable(evidence / "summary.json", summary)
    return {**prepared, "status": "FINALIZED", "summary": summary}


def cleanup_preview(job: Job, candidate: Path) -> dict:
    """Return a plan only. No deletion, unlinking, worktree removal or pruning."""
    prepared = _record(job)
    evidence, scratch, _ = _paths(job)
    if candidate != scratch or candidate.resolve() != scratch:
        raise ValueError("unsafe cleanup plan: only exact owned scratch is eligible")
    if read_canonical(evidence / "scratch.json") != {"path": str(scratch), **prepared}:
        raise ValueError("unsafe cleanup plan: scratch ownership mismatch")
    result = collect(job)
    for pending in job.host.state.glob("*/pending.json"):
        if Path(runner._read(pending)["log_dir"]).is_relative_to(scratch):
            raise ValueError("unsafe cleanup plan: receipt-referenced logs")
    runner._safe(scratch, directory=True)
    for path in scratch.rglob("*"):
        if path.is_symlink() or (path.is_file() and path.stat().st_nlink != 1):
            raise ValueError("unsafe cleanup plan: linked scratch content")
    return {
        "preview_only": True,
        "candidate": str(scratch),
        "delete_implemented": False,
        "retain": [
            str(job.host.evidence),
            str(job.host.state),
            str(job.host.lock),
            str(job.host.blocked),
        ],
        "job_status": result["status"],
    }


def register_job_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "lab-job", help="Admit or collect a bounded CPU test handoff"
    )
    _arguments(parser)
    parser.set_defaults(handler=_handle)


def _arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "action", choices=("admit", "status", "collect", "cleanup-preview")
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--candidate", type=Path)


def _handle(args: argparse.Namespace) -> None:
    job = load_job(args.manifest)
    if args.action == "cleanup-preview":
        if args.candidate is None:
            raise ValueError("cleanup-preview requires --candidate")
        result = cleanup_preview(job, args.candidate)
    else:
        if args.candidate is not None:
            raise ValueError("--candidate is only valid for cleanup-preview")
        result = {"admit": admit, "status": status, "collect": collect}[args.action](
            job
        )
    print(json.dumps(result, sort_keys=True))
    if result.get("summary", {}).get("controller_completion") == "FAIL":
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _arguments(parser)
    _handle(parser.parse_args(argv))


if __name__ == "__main__":
    main()
