"""Closed-execution adoption stays distinct from native cold production."""

import os
from pathlib import Path

import pytest
import test_split_inventory_reuse as reuse_fixtures

from sparselab.corpus.project import project_path
from sparselab.corpus.split_inventory_adoption import adopt_completed_inventory
from sparselab.corpus.split_inventory_cache import InventoryCache, verified_inventory
from sparselab.training.manifest import canonical_json, sha256_file

# Register the shared pytest fixture without an unused fixture import.
completed_inventory = reuse_fixtures.completed_inventory
_no_replay = reuse_fixtures._no_replay


def _evidence(fixture, mutate=None):
    root, project, inventory, report = fixture
    directory = root / "completed-evidence"
    directory.mkdir(mode=0o700)

    def put(name, value):
        path = directory / name
        raw = value if isinstance(value, bytes) else canonical_json(value)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
        return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}

    project_file = project.root / "acquire.yaml"
    inputs = [
        project_file,
        *(project_path(project.root, p) for p in project.config.sources),
        project_path(project.root, project.config.splits),
        project_path(project.root, project.config.release),
        root / "corpora" / project.config.id / "acquisition.json",
    ]
    revision, runtime_sha = "a" * 40, "b" * 64
    image = "ghcr.io/example/lab@sha256:" + "c" * 64
    runtime = {"python": "3.14.4", "executable_sha256": runtime_sha}
    argv = [
        "/fixture/python",
        "-B",
        "-m",
        "sparselab",
        "--work-dir",
        str(root),
        "corpus",
        "split-inventory",
        str(project_file),
        "--output",
        str(inventory),
        "--json",
    ]
    spec = {
        "job_id": "inventory",
        "source": {"path": "/fixture/source", "commit": revision},
        "runtime": runtime,
        "argv": argv,
        "wrap": None,
        "required_files": {str(p): sha256_file(p) for p in inputs},
    }
    job = {
        "job_id": "inventory",
        "source": dict(spec["source"]),
        "runtime": dict(runtime),
        "runtime_identity": {"executable_sha256": runtime_sha, "invocation": argv[0]},
        "launched": True,
        "reason": "exit",
        "returncode": 0,
        "process_success": True,
        "cleanup_verified": True,
        "evidence_complete": True,
        "logs_complete": True,
        "errors": [],
        "interruption_signal": None,
    }
    session_spec = {
        "session_id": "completed-session",
        "roots": {"A": str(root)},
        "execution_policy": {"image": image},
        "runtime_sha256": runtime_sha,
        "slots": [{"id": "inventory", "root": "A", "argv": list(argv)}],
    }
    session = {
        "session_id": "completed-session",
        "reason": "complete",
        "cleanup_verified": True,
        "evidence_complete": True,
        "pending_failure": None,
        "incomplete_jobs": [],
        "slots_completed": 1,
        "slots_total": 1,
    }
    outer = {
        "reason": "exit",
        "engine_exit": 0,
        "container_exit": 0,
        "oom_killed": False,
        "cleanup_verified": True,
        "logs_complete": True,
        "errors": [],
        "image": image,
    }
    output = dict(report)
    if mutate is not None:
        mutate(spec, job, session_spec, session, outer, output)
    refs = {
        "stdout": put("stdout.json", output),
        "stderr": put("stderr.log", b""),
        "events": put("events.jsonl", b'{"fixture":"completed"}\n'),
    }
    refs["job_spec"] = put("job-spec.json", spec)
    job["spec_sha256"] = refs["job_spec"]["sha256"]
    refs["job_report"] = put("job-report.json", job)
    refs["session_spec"] = put("session-spec.json", session_spec)
    outer["session_spec_sha256"] = refs["session_spec"]["sha256"]
    session["progress"] = {
        "completed_slots": 1,
        "events_sha256": refs["events"]["sha256"],
        "jobs": [
            {
                "job_id": "inventory",
                "slot": "inventory",
                "root": "A",
                "launched": True,
                "returncode": 0,
                "spec_sha256": refs["job_spec"]["sha256"],
                "report_sha256": refs["job_report"]["sha256"],
                "stdout_sha256": refs["stdout"]["sha256"],
                "stderr_sha256": refs["stderr"]["sha256"],
            }
        ],
    }
    refs["session_report"] = put("session-report.json", session)
    refs["container_report"] = put("container-report.json", outer)
    declaration = {
        "kind": "completed_split_inventory_execution",
        "producer": {
            "revision": revision,
            "runtime_sha256": runtime_sha,
            "image": image,
        },
        "project": {"path": str(project_file), "sha256": sha256_file(project_file)},
        "inventory": {"path": str(inventory), "sha256": report["sha256"]},
        "evidence": refs,
    }
    reference = put("completion.json", declaration)
    cache = InventoryCache(root)
    cache._path("inventory", cache.request(project, inventory)).unlink()
    return reference


def test_adoption_reuses_closed_execution_without_claiming_cold_production(
    completed_inventory,
):
    root, project, path, report = completed_inventory
    reference = _evidence(completed_inventory)
    adopted = adopt_completed_inventory(project, root, path, reference)
    assert adopted.provenance["mode"] == "adopted_execution"
    assert adopted.provenance["producer_revision"] == "a" * 40
    assert verified_inventory(
        project, root, path, report["sha256"], _no_replay, adopted=adopted
    ) == (report, True)
    assert (
        InventoryCache(root).completed(project, path)[2]["mode"] == "adopted_execution"
    )


def test_warm_adoption_skips_large_artifact_hashes(completed_inventory, monkeypatch):
    from sparselab.corpus import split_inventory_adoption

    root, project, path, report = completed_inventory
    reference = _evidence(completed_inventory)
    adopt_completed_inventory(project, root, path, reference)
    original = split_inventory_adoption.sha256_file
    lock = root / "corpora" / project.config.id / "acquisition.json"

    def forbid_large(target):
        if Path(target) in (path, lock):
            raise AssertionError("warm adoption reread a large artifact")
        return original(target)

    monkeypatch.setattr(split_inventory_adoption, "sha256_file", forbid_large)
    # Native verification inherits the already selected, authenticated adoption.
    work = {}
    assert verified_inventory(
        project, root, path, report["sha256"], _no_replay, work=work
    ) == (
        report,
        True,
    )
    assert work["completion_mode"] == "adopted_execution"
    assert work["adopted"].provenance["mode"] == "adopted_execution"


def test_adoption_refuses_current_project_normalizer_change(completed_inventory):
    root, project, path, _ = completed_inventory
    reference = _evidence(completed_inventory)
    normalizer = (
        "normalizer-structure-v3"
        if project.release.normalizer != "normalizer-structure-v3"
        else "normalizer-nfc-markdown-v1"
    )
    changed = project.model_copy(
        update={
            "release": project.release.model_copy(update={"normalizer": normalizer})
        }
    )
    with pytest.raises(ValueError, match="current project differs"):
        adopt_completed_inventory(changed, root, path, reference)
    assert InventoryCache(root).completed(changed, path) is None


@pytest.mark.parametrize(
    "case",
    [
        "failed",
        "incomplete",
        "cleanup",
        "revision",
        "runtime",
        "image",
        "command",
        "missing_input",
        "output",
    ],
)
def test_coherently_hashed_but_invalid_execution_is_not_adopted(
    completed_inventory, case
):
    root, project, path, _ = completed_inventory

    def mutate(spec, job, session_spec, session, outer, output):
        if case == "failed":
            job["returncode"] = 1
        elif case == "incomplete":
            job["evidence_complete"] = False
        elif case == "cleanup":
            outer["cleanup_verified"] = False
        elif case == "revision":
            job["source"]["commit"] = "d" * 40
        elif case == "runtime":
            job["runtime_identity"]["executable_sha256"] = "d" * 64
        elif case == "image":
            outer["image"] = "ghcr.io/other/lab@sha256:" + "d" * 64
        elif case == "command":
            spec["argv"] = [*spec["argv"], "--unknown"]
        elif case == "missing_input":
            spec["required_files"].pop(
                str(project_path(project.root, project.config.sources[0]))
            )
        elif case == "output":
            output["sha256"] = "d" * 64

    reference = _evidence(completed_inventory, mutate)
    with pytest.raises(ValueError):
        adopt_completed_inventory(project, root, path, reference)
    assert InventoryCache(root).completed(project, path) is None


def test_adopted_completion_refuses_later_input_mutation(completed_inventory):
    root, project, path, report = completed_inventory
    adopted = adopt_completed_inventory(
        project, root, path, _evidence(completed_inventory)
    )
    source = project_path(project.root, project.config.sources[0])
    source.write_bytes(source.read_bytes() + b"\n# changed after adoption\n")
    with pytest.raises(ValueError, match="changed during execution"):
        verified_inventory(
            project, root, path, report["sha256"], _no_replay, adopted=adopted
        )


def test_live_adoption_does_not_replay_if_optional_cache_is_unavailable(
    completed_inventory, monkeypatch
):
    root, project, path, report = completed_inventory
    reference = _evidence(completed_inventory)
    monkeypatch.setattr(
        InventoryCache, "record_completed", lambda *args, **kwargs: None
    )
    adopted = adopt_completed_inventory(project, root, path, reference)
    assert verified_inventory(
        project, root, path, report["sha256"], _no_replay, adopted=adopted
    ) == (report, True)
    with pytest.raises(TypeError, match="process-owned"):
        verified_inventory(
            project,
            root,
            path,
            report["sha256"],
            _no_replay,
            adopted={"report": report},
        )
