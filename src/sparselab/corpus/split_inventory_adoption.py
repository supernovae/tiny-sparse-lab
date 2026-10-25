"""Explicit adoption of a completed local executor record, not cold production."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from sparselab.corpus.admission_inspection_cache import capture, unchanged
from sparselab.corpus.bounded_metadata import read_metadata
from sparselab.corpus.project import load_project, project_path
from sparselab.corpus.split_inventory_cache import InventoryCache
from sparselab.training.manifest import sha256_file
from sparselab.verification_proofs import _trusted

_SEAL = object()
_HEX = re.compile(r"[0-9a-f]{64}")
_ROLES = (
    "job_spec",
    "job_report",
    "session_spec",
    "session_report",
    "container_report",
    "stdout",
    "stderr",
    "events",
)


def _inside(root, value):
    if not isinstance(value, str) or not value or ".." in Path(value).parts:
        raise ValueError("invalid inventory completion path")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.absolute()
    if not path.is_relative_to(root) or any(
        p.is_symlink() for p in (path, *path.parents)
    ):
        raise ValueError("inventory completion path escapes its owned work root")
    return path


def _digest(value):
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ValueError("inventory completion requires exact SHA-256 identities")
    return value


def _reference(root, reference, *, control=True):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("invalid inventory completion reference")
    path = _inside(root, reference["path"])
    digest = _digest(reference["sha256"])
    if not path.is_file():
        raise ValueError("inventory completion reference is not a regular file")
    if control and (not _trusted(root, path) or path.stat().st_size > 4 * 1024**2):
        raise ValueError(
            "inventory completion control evidence is not owned and bounded"
        )
    observed = capture([path])
    if sha256_file(path) != digest:
        raise ValueError(f"inventory completion digest mismatch: {path.name}")
    unchanged(observed)
    return path, observed


def _successful(job, session, outer):
    if (
        job.get("launched") is not True
        or job.get("reason") != "exit"
        or type(job.get("returncode")) is not int
        or job["returncode"] != 0
        or job.get("process_success") is not True
        or any(
            job.get(k) is not True
            for k in ("cleanup_verified", "evidence_complete", "logs_complete")
        )
        or job.get("errors") != []
        or job.get("interruption_signal") is not None
        or session.get("reason") != "complete"
        or session.get("cleanup_verified") is not True
        or session.get("evidence_complete") is not True
        or session.get("pending_failure") is not None
        or session.get("incomplete_jobs") != []
        or session.get("slots_completed") != session.get("slots_total")
        or type(session.get("slots_total")) is not int
        or session["slots_total"] < 1
        or outer.get("reason") != "exit"
        or any(
            type(outer.get(k)) is not int or outer[k] != 0
            for k in ("engine_exit", "container_exit")
        )
        or outer.get("oom_killed") is not False
        or outer.get("cleanup_verified") is not True
        or outer.get("logs_complete") is not True
        or outer.get("errors") != []
    ):
        raise ValueError(
            "inventory adoption requires successful execution and verified cleanup"
        )


@dataclass(frozen=True)
class _AdoptedInventory:
    report: dict
    provenance: dict
    dependencies: dict
    request: dict
    issuer_pid: int
    seal: object

    def check(self, cache, project, path, expected):
        if (
            self.seal is not _SEAL
            or self.issuer_pid != os.getpid()
            or self.request != cache.request(project, path)
            or self.report.get("sha256") != expected
        ):
            raise ValueError(
                "adopted completion does not match this inventory invocation"
            )
        unchanged(self.dependencies)


def adopt_completed_inventory(project, root, inventory, reference):
    try:
        return _adopt_completed_inventory(project, root, inventory, reference)
    except (KeyError, TypeError, AttributeError, OSError) as error:
        raise ValueError("missing or malformed completed inventory evidence") from error


def _adopt_completed_inventory(project, root, inventory, reference):
    """Validate recorded execution and current input identities without replay.

    Executor records are integrity-bound local operator evidence, not signatures
    invented for the old run. Native acquisition/admission validation still runs
    before a frozen split can publish. No source-body verification is certified
    by this adoption operation.
    """
    root, inventory = Path(root).resolve(), Path(inventory).absolute()
    declaration, dependencies = _reference(root, reference)
    manifest = read_metadata(declaration)
    if (
        set(manifest) != {"kind", "producer", "project", "inventory", "evidence"}
        or manifest["kind"] != "completed_split_inventory_execution"
    ):
        raise ValueError("invalid completed inventory declaration")
    producer = manifest["producer"]
    if (
        set(producer) != {"revision", "runtime_sha256", "image"}
        or re.fullmatch(r"[0-9a-f]{40}", producer.get("revision", "")) is None
    ):
        raise ValueError("inventory adoption requires an exact producer revision")
    _digest(producer["runtime_sha256"])
    if (
        not isinstance(producer["image"], str)
        or re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", producer["image"]) is None
    ):
        raise ValueError("inventory adoption requires an exact image identity")
    refs = manifest["evidence"]
    if not isinstance(refs, dict) or set(refs) != set(_ROLES):
        raise ValueError("inventory adoption lacks closed execution evidence")
    paths = {}
    for role in _ROLES:
        paths[role], observed = _reference(root, refs[role])
        dependencies.update(observed)
    records = {
        role: read_metadata(paths[role])
        for role in _ROLES
        if role not in ("stderr", "events")
    }
    spec, job = records["job_spec"], records["job_report"]
    session_spec, session, outer = (
        records["session_spec"],
        records["session_report"],
        records["container_report"],
    )
    _successful(job, session, outer)
    if (
        job.get("spec_sha256") != refs["job_spec"]["sha256"]
        or outer.get("session_spec_sha256") != refs["session_spec"]["sha256"]
    ):
        raise ValueError("completed inventory specification identity mismatch")
    if (
        spec.get("source") != job.get("source")
        or spec.get("source", {}).get("commit") != producer["revision"]
        or spec.get("wrap") is not None
    ):
        raise ValueError("completed inventory producer identity mismatch")
    runtime = spec.get("runtime", {})
    identity = job.get("runtime_identity", {})
    if (
        job.get("runtime") != runtime
        or runtime.get("executable_sha256") != producer["runtime_sha256"]
        or identity.get("executable_sha256") != producer["runtime_sha256"]
    ):
        raise ValueError("completed inventory runtime identity mismatch")
    if (
        outer.get("image") != producer["image"]
        or session_spec.get("execution_policy", {}).get("image") != producer["image"]
        or session_spec.get("runtime_sha256") != producer["runtime_sha256"]
    ):
        raise ValueError("completed inventory container identity mismatch")
    project_file = _inside(root, manifest["project"]["path"])
    if _inside(root, manifest["inventory"]["path"]) != inventory:
        raise ValueError("completed inventory output path mismatch")
    expected = _digest(manifest["inventory"]["sha256"])
    argv = [
        identity.get("invocation"),
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
    if not argv[0] or spec.get("argv") != argv:
        raise ValueError("completed inventory command identity mismatch")
    progress = session.get("progress", {})
    jobs = progress.get("jobs", [])
    matching = [item for item in jobs if item.get("job_id") == job.get("job_id")]
    if len(matching) != 1 or spec.get("job_id") != job.get("job_id"):
        raise ValueError("completed inventory job is absent or ambiguous")
    item = matching[0]
    for field, role in (
        ("spec_sha256", "job_spec"),
        ("report_sha256", "job_report"),
        ("stdout_sha256", "stdout"),
        ("stderr_sha256", "stderr"),
    ):
        if item.get(field) != refs[role]["sha256"]:
            raise ValueError("completed inventory progress identity mismatch")
    slots = [
        slot
        for slot in session_spec.get("slots", [])
        if slot.get("id") == item.get("slot") and slot.get("root") == item.get("root")
    ]
    if (
        len(slots) != 1
        or slots[0].get("argv") != argv
        or session_spec.get("roots", {}).get(item.get("root")) != str(root)
    ):
        raise ValueError("completed inventory slot identity mismatch")
    if (
        session.get("session_id") != session_spec.get("session_id")
        or item.get("launched") is not True
        or type(item.get("returncode")) is not int
        or item["returncode"] != 0
        or progress.get("events_sha256") != refs["events"]["sha256"]
    ):
        raise ValueError("completed inventory session identity mismatch")
    if paths["stderr"].stat().st_size:
        raise ValueError("completed inventory contains unreviewed native errors")
    report = records["stdout"]
    counts = report.get("counts", {})
    if (
        report.get("schema_version") != 1
        or report.get("project_id") != project.config.id
        or report.get("path") != str(inventory)
        or report.get("sha256") != expected
        or not isinstance(counts, dict)
        or set(counts) - {s.id for s in project.sources}
        or any(type(v) is not int or v < 0 for v in counts.values())
        or type(report.get("documents")) is not int
        or sum(counts.values()) != report["documents"]
    ):
        raise ValueError("completed inventory native output identity mismatch")
    required = spec.get("required_files", {})
    scientific = [
        project_file,
        *(project_path(project.root, p) for p in project.config.sources),
        project_path(project.root, project.config.splits),
        project_path(project.root, project.config.release),
    ]
    for path in scientific:
        value = required.get(str(path))
        if value is None:
            raise ValueError(
                f"completed producer did not bind scientific input: {path.name}"
            )
        _, observed = _reference(
            root, {"path": str(path), "sha256": value}, control=False
        )
        dependencies.update(observed)
    if required.get(str(project_file)) != _digest(manifest["project"]["sha256"]):
        raise ValueError("completed inventory project identity mismatch")
    cache = InventoryCache(root)
    fresh = load_project(project_file, verify_inputs=False)
    if cache.request(fresh, inventory) != cache.request(project, inventory):
        raise ValueError(
            "current project differs from the completed producer configuration"
        )
    existing = cache.completed(project, inventory)
    if (
        existing is not None
        and existing[2].get("mode") == "adopted_execution"
        and existing[2].get("declaration_sha256") == reference["sha256"]
    ):
        previous_report, observed, provenance = existing
        if previous_report != report:
            raise ValueError("adopted cache disagrees with recorded native output")
        dependencies.update(observed)
        unchanged(dependencies)
        return _AdoptedInventory(
            report,
            provenance,
            dependencies,
            cache.request(project, inventory),
            os.getpid(),
            _SEAL,
        )
    lock = root / "corpora" / project.config.id / "acquisition.json"
    lock_sha = required.get(str(lock))
    if lock_sha is None:
        raise ValueError("completed producer did not bind its acquisition identity")
    _, observed = _reference(
        root, {"path": str(lock), "sha256": lock_sha}, control=False
    )
    dependencies.update(observed)
    _, observed = _reference(root, manifest["inventory"], control=False)
    dependencies.update(observed)
    unchanged(dependencies)
    provenance = {
        "mode": "adopted_execution",
        "declaration_sha256": reference["sha256"],
        "declaration": reference,
        "producer_revision": producer["revision"],
        "command": argv,
        "runtime_sha256": producer["runtime_sha256"],
        "image": producer["image"],
        "job_spec_sha256": refs["job_spec"]["sha256"],
        "evidence": refs,
        "input_identities": {str(p): required[str(p)] for p in [*scientific, lock]},
        "source_validation": "native acquisition and admission checks remain required before split publication",
    }
    # Configurations are checked against the recorded input hashes on every
    # adoption. Only owned, private evidence plus immutable large artifacts enter
    # the persistent cache; no existing permissions are changed for caching.
    persistent = {
        name: rows for name, rows in dependencies.items() if _trusted(root, Path(name))
    }
    if str(lock) in persistent and str(inventory) in persistent:
        cache.record_completed(
            project, inventory, report, persistent, completion=provenance
        )
    return _AdoptedInventory(
        report,
        provenance,
        dependencies,
        cache.request(project, inventory),
        os.getpid(),
        _SEAL,
    )
