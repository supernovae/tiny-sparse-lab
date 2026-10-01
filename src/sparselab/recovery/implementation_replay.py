"""Explicit, isolated replay of a committed historical Corpus Forge implementation.

The source tree is read from Git objects, never from the checkout. Operational
receipts are not inputs to corpus scientific identities.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sparselab.recovery.provenance import declaration_paths, repository_root

_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENTS = (
    "acquisition",
    "pipeline",
    "project",
    "provenance",
    "rights",
    "large_build",
    "release",
    "export",
)
_DIRECT = frozenset(
    ("acquisition", "pipeline", "project", "provenance", "rights", "large_build")
)
_FORMAT = "sparselab-implementation-replay-v1"
_SOURCE_FORMAT = "sparselab-materialized-source-v1"


def _reject_symlink_ancestors(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"UNSAFE_REPLAY_ROOT: symlinked destination {component}")


def _source_target(repo: Path, work_root: Path, commit: str) -> Path:
    work = Path(work_root).expanduser().absolute()
    _reject_symlink_ancestors(work)
    work = work.resolve()
    if work.is_relative_to(repo) or repo.is_relative_to(work):
        raise ValueError(
            "UNSAFE_REPLAY_ROOT: replay state must be external to the checkout"
        )
    target = work / "replay" / "source" / commit
    _reject_symlink_ancestors(target)
    return target


def _git(root: Path, *args: str, input: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        input=input,
        capture_output=True,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
        check=False,
    )
    if result.returncode:
        raise ValueError(
            f"MISSING_IMPLEMENTATION: Git object unavailable: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def _repository_identity(root: Path, commit: str, project: str) -> dict[str, Any]:
    remote = subprocess.run(
        ["git", "-C", str(root), "config", "--get", "remote.origin.url"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "root": str(root),
        "commit": commit,
        "project": project,
        "tree": _git(root, "rev-parse", f"{commit}^{{tree}}").decode().strip(),
        "object_format": _git(root, "rev-parse", "--show-object-format")
        .decode()
        .strip(),
        "origin_url": remote.stdout.strip() if remote.returncode == 0 else None,
    }


def _repository(source: Path, commit: str) -> tuple[Path, str]:
    root = repository_root(source)
    if root is None or not _COMMIT.fullmatch(commit):
        raise ValueError(
            "MISSING_IMPLEMENTATION: exact full commit and local Git repository required"
        )
    resolved = (
        _git(root, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    )
    if resolved != commit:
        raise ValueError(
            "MISSING_IMPLEMENTATION: commit must identify an exact commit object"
        )
    return root, resolved


def _relative(root: Path, path: Path) -> str:
    absolute = Path(path).absolute()
    if not absolute.is_relative_to(root):
        raise ValueError(f"SOURCE_COMMIT_MISMATCH: path outside repository: {path}")
    for ancestor in (absolute, *absolute.parents):
        if ancestor.is_symlink():
            raise ValueError(f"SOURCE_COMMIT_MISMATCH: symlinked source: {ancestor}")
        if ancestor == root:
            break
    return absolute.relative_to(root).as_posix()


def _tree(root: Path, commit: str) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for row in _git(root, "ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
        if not row:
            continue
        header, raw = row.split(b"\t", 1)
        mode, kind, oid = header.decode("ascii").split(" ")
        name = raw.decode("utf-8", errors="strict")
        if (
            mode not in {"100644", "100755"}
            or kind != "blob"
            or not name
            or name.startswith("/")
            or "\\" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or any(ord(c) < 32 for c in name)
        ):
            raise ValueError(
                f"UNSAFE_SOURCE_TREE: invalid Git entry {name!r} mode={mode}"
            )
        if name in entries:
            raise ValueError(f"UNSAFE_SOURCE_TREE: duplicate path {name}")
        entries[name] = mode, oid
    if not entries:
        raise ValueError("MISSING_IMPLEMENTATION: empty historical source tree")
    return entries


def _blob_digest(data: bytes, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    h.update(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()


def _source_inventory(
    root: Path, entries: dict[str, tuple[str, str]], algorithm: str
) -> dict[str, dict[str, str]]:
    inventory: dict[str, dict[str, str]] = {}
    found: set[str] = set()
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            directory = Path(base) / name
            if not stat.S_ISDIR(directory.lstat().st_mode):
                raise ValueError(f"SOURCE_TAMPERED: unsafe directory {directory}")
        for name in files:
            path = Path(base) / name
            relative = path.relative_to(root).as_posix()
            if relative not in entries or not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"SOURCE_TAMPERED: unexpected file {relative}")
            mode, oid = entries[relative]
            data = path.read_bytes()
            if _blob_digest(data, algorithm) != oid or bool(
                path.stat().st_mode & 0o111
            ) != (mode == "100755"):
                raise ValueError(
                    f"SOURCE_TAMPERED: historical bytes or mode differ: {relative}"
                )
            inventory[relative] = {
                "mode": mode,
                "object": oid,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            found.add(relative)
    if found != entries.keys():
        raise ValueError(
            f"SOURCE_TAMPERED: missing historical files: {sorted(entries.keys() - found)}"
        )
    return inventory


def implementation_preflight(
    source: Path, commit: str, project: Path
) -> dict[str, Any]:
    """Report historical/current component byte identity without creating files."""
    try:
        root, commit = _repository(source, commit)
        tree = _tree(root, commit)
    except ValueError as error:
        if str(error).startswith("MISSING_IMPLEMENTATION"):
            return {
                "status": "MISSING_IMPLEMENTATION",
                "components": {},
                "repository": None,
                "reason": str(error),
            }
        raise
    from sparselab.corpus.project import load_project

    recipe = load_project(project)
    authored = {Path(project).absolute()}
    authored.update(
        project.parent / relative
        for relative in (
            *recipe.config.sources,
            *recipe.config.transforms,
            recipe.config.splits,
            recipe.config.release,
        )
    )
    project_relative = _relative(root, project)
    for path in declaration_paths(project, "corpus"):
        relative = _relative(root, path)
        if not path.is_file() and path not in authored:
            continue  # Missing local payloads remain external prerequisites, not dirty declarations.
        if relative not in tree or not path.is_file():
            raise ValueError(
                f"SOURCE_COMMIT_MISMATCH: missing historical project input {relative}"
            )
        if path.read_bytes() != _git(root, "show", f"{commit}:{relative}"):
            raise ValueError(
                f"SOURCE_COMMIT_MISMATCH: dirty or changed project input {relative}"
            )
    execution_root = Path(__file__).resolve().parents[3]
    large_build = (
        recipe.release.schema_version == 3
        and len(recipe.transforms) == 1
        and recipe.transforms[0].kind == "lm_text"
        and recipe.release.lm.selected
        and not recipe.release.chat.selected
        and all(item.schema_version == 3 for item in recipe.sources)
    )
    direct = _DIRECT - {"rights", "large_build"}
    if recipe.release.schema_version in (2, 3):
        direct = direct | {"rights"}
    if large_build:
        direct = direct | {"large_build"}
    components: dict[str, dict[str, Any]] = {}
    for name in _COMPONENTS:
        relative = f"src/sparselab/corpus/{name}.py"
        if relative not in tree:
            return {
                "status": "MISSING_IMPLEMENTATION",
                "components": components,
                "repository": {"root": str(root), "commit": commit},
                "reason": f"historical source missing {relative}",
            }
        historical = hashlib.sha256(
            _git(root, "show", f"{commit}:{relative}")
        ).hexdigest()
        module = importlib.import_module(f"sparselab.corpus.{name}")
        current_path = Path(module.__file__).resolve(strict=True)
        current = hashlib.sha256(current_path.read_bytes()).hexdigest()
        stage = (
            "acquisition"
            if name == "acquisition"
            else "freeze"
            if name == "release"
            else "export"
            if name == "export"
            else "build"
        )
        components[name] = {
            "role": name,
            "affected_stage": stage,
            "source_commit": commit,
            "path": relative,
            "pinned_sha256": historical,
            "current_sha256": current,
            "direct_identity_hash": name in direct,
            "status": "MATCH"
            if current == historical
            else "PINNED_IMPLEMENTATION_REPLAY_REQUIRED",
        }
    for name, relative in (
        ("training_manifest", "src/sparselab/training/manifest.py"),
        ("project_config", "src/sparselab/config/models.py"),
        ("pyproject", "pyproject.toml"),
        ("uv_lock", "uv.lock"),
    ):
        if relative not in tree:
            return {
                "status": "MISSING_IMPLEMENTATION",
                "components": components,
                "repository": {"root": str(root), "commit": commit},
                "reason": f"historical source missing {relative}",
            }
        historical = hashlib.sha256(
            _git(root, "show", f"{commit}:{relative}")
        ).hexdigest()
        if name in {"pyproject", "uv_lock"}:
            current_path = execution_root / relative
        else:
            module_name = (
                "sparselab.training.manifest"
                if name == "training_manifest"
                else "sparselab.config.models"
            )
            current_path = Path(importlib.import_module(module_name).__file__).resolve(
                strict=True
            )
        current = (
            hashlib.sha256(current_path.read_bytes()).hexdigest()
            if current_path.is_file()
            else None
        )
        stage = "dependencies" if name in {"pyproject", "uv_lock"} else "build"
        components[name] = {
            "role": name,
            "affected_stage": stage,
            "source_commit": commit,
            "path": relative,
            "pinned_sha256": historical,
            "current_sha256": current,
            "direct_identity_hash": False,
            "status": "MATCH"
            if current == historical
            else "PINNED_IMPLEMENTATION_REPLAY_REQUIRED",
        }
    return {
        "status": "MATCH"
        if all(row["status"] == "MATCH" for row in components.values())
        else "PINNED_IMPLEMENTATION_REPLAY_REQUIRED",
        "components": components,
        "repository": _repository_identity(root, commit, project_relative),
    }


def verify_materialized_source(
    source_root: Path, commit: str | None = None
) -> dict[str, Any]:
    """Reauthenticate every committed source byte and mode against Git objects."""
    source_root = Path(source_root).absolute()
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("SOURCE_TAMPERED: symlinked or missing source tree")
    if any(parent.is_symlink() for parent in source_root.parents):
        raise ValueError("SOURCE_TAMPERED: symlinked source ancestor")
    receipt_path = (
        source_root.parent.parent / "source-manifests" / f"{source_root.name}.json"
    )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("SOURCE_TAMPERED: missing or symlinked source manifest")
    record = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        record.get("format") != _SOURCE_FORMAT
        or set(record)
        != {
            "format",
            "commit",
            "repository",
            "tree",
            "object_format",
            "files",
            "inventory_sha256",
        }
        or record["commit"] != source_root.name
        or (commit is not None and record["commit"] != commit)
    ):
        raise ValueError("SOURCE_TAMPERED: invalid source manifest")
    root, exact = _repository(Path(record["repository"]), record["commit"])
    entries = _tree(root, exact)
    if _git(root, "rev-parse", f"{exact}^{{tree}}").decode().strip() != record["tree"]:
        raise ValueError("SOURCE_TAMPERED: tree mismatch")
    algorithm = _git(root, "rev-parse", "--show-object-format").decode().strip()
    if algorithm != record["object_format"]:
        raise ValueError("SOURCE_TAMPERED: Git object format mismatch")
    files = _source_inventory(source_root, entries, algorithm)
    if (
        record["files"] != files
        or record["inventory_sha256"]
        != hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise ValueError("SOURCE_TAMPERED: source manifest differs from Git inventory")
    return record


def materialize_source(source: Path, commit: str, work_root: Path) -> dict[str, Any]:
    """Atomically publish a read-only full tree from the exact local commit."""
    root, commit = _repository(source, commit)
    entries = _tree(root, commit)
    target = _source_target(root, work_root, commit)
    receipt_path = target.parent.parent / "source-manifests" / f"{commit}.json"
    if target.exists() or receipt_path.exists():
        if not target.is_dir() or not receipt_path.is_file():
            raise ValueError("SOURCE_TAMPERED: incomplete existing source publication")
        return {
            "source_root": str(target),
            "manifest_path": str(receipt_path),
            "manifest": verify_materialized_source(target, commit),
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    algorithm = _git(root, "rev-parse", "--show-object-format").decode().strip()
    with tempfile.TemporaryDirectory(prefix=".source-", dir=target.parent) as temporary:
        staging = Path(temporary)
        archive = subprocess.Popen(
            [
                "git",
                "-c",
                "tar.umask=0022",
                "-C",
                str(root),
                "archive",
                "--format=tar",
                commit,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
        )
        assert archive.stdout is not None
        seen: set[str] = set()
        try:
            with tarfile.open(fileobj=archive.stdout, mode="r|") as stream:
                for member in stream:
                    name = member.name.rstrip("/")
                    if member.isdir():
                        if name and (
                            name.startswith("/")
                            or any(part in {"", ".", ".."} for part in name.split("/"))
                        ):
                            raise ValueError(f"UNSAFE_SOURCE_TREE: directory {name!r}")
                        continue
                    if (
                        name not in entries
                        or name in seen
                        or not member.isfile()
                        or member.mode & 0o7000
                    ):
                        raise ValueError(f"UNSAFE_SOURCE_TREE: archive entry {name!r}")
                    mode, oid = entries[name]
                    if bool(member.mode & 0o111) != (mode == "100755"):
                        raise ValueError(f"UNSAFE_SOURCE_TREE: archive mode {name!r}")
                    stream_file = stream.extractfile(member)
                    if stream_file is None:
                        raise ValueError(
                            f"UNSAFE_SOURCE_TREE: unreadable archive entry {name!r}"
                        )
                    data = stream_file.read()
                    if _blob_digest(data, algorithm) != oid:
                        raise ValueError(f"SOURCE_TAMPERED: Git blob mismatch {name}")
                    destination = staging / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    destination.chmod(0o555 if mode == "100755" else 0o444)
                    seen.add(name)
        finally:
            archive.stdout.close()
            archive.wait()
            if archive.returncode:
                raise ValueError(
                    "MISSING_IMPLEMENTATION: failed to archive pinned Git tree"
                )
        if seen != entries.keys():
            raise ValueError("SOURCE_TAMPERED: incomplete historical archive")
        files = _source_inventory(staging, entries, algorithm)
        record = {
            "format": _SOURCE_FORMAT,
            "commit": commit,
            "repository": str(root),
            "tree": _git(root, "rev-parse", f"{commit}^{{tree}}").decode().strip(),
            "object_format": algorithm,
            "files": files,
            "inventory_sha256": hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        for directory, _, _ in os.walk(staging):
            Path(directory).chmod(0o555)
        staging.rename(target)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True)
    receipt_path.chmod(0o444)
    return {
        "source_root": str(target),
        "manifest_path": str(receipt_path),
        "manifest": verify_materialized_source(target, commit),
    }


def _log_identity(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(max(0, stream.tell() - 4096))
        excerpt = stream.read(4096).decode(errors="replace")
    return digest.hexdigest(), excerpt


def _run_logged(
    command: list[str],
    phase: str,
    logs: Path,
    env: dict[str, str],
    cwd: Path | None = None,
) -> dict[str, Any]:
    logs.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    stdout_path = logs / f"{token}.{phase}.stdout"
    stderr_path = logs / f"{token}.{phase}.stderr"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        result = subprocess.run(
            command, stdout=stdout, stderr=stderr, env=env, cwd=cwd, check=False
        )
    stdout_hash, stdout_excerpt = _log_identity(stdout_path)
    stderr_hash, stderr_excerpt = _log_identity(stderr_path)
    return {
        "phase": phase,
        "argv": command,
        "returncode": result.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_sha256": stdout_hash,
        "stderr_sha256": stderr_hash,
        "stdout": stdout_excerpt,
        "stderr": stderr_excerpt,
    }


def _verify_producer_provenance(
    source: dict[str, Any],
    lock: dict[str, Any],
    build: dict[str, Any],
    release: dict[str, Any],
) -> list[dict[str, str]]:
    """Bind already byte-verified artifacts to the actual materialized producer."""
    files = source["files"]
    acquisition_sha = files["src/sparselab/corpus/acquisition.py"]["sha256"]
    snapshots = []
    for source_id, entry in lock["sources"].items():
        if entry["snapshot_sha256"] is None:
            continue
        manifest = json.loads(
            (Path(entry["snapshot_path"]) / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest["adapter"]["module_sha256"] != acquisition_sha:
            raise ValueError(
                f"IMPLEMENTATION_PROVENANCE_MISMATCH: snapshot {source_id}"
            )
        snapshots.append({"source_id": source_id, "sha256": entry["snapshot_sha256"]})
    identity = build["identity"]
    fields = {
        "implementation_sha256": "pipeline",
        "schema_implementation_sha256": "project",
        "provenance_implementation_sha256": "provenance",
    }
    if identity["release"]["schema_version"] in (2, 3):
        fields["rights_implementation_sha256"] = "rights"
    transforms = identity["transforms"]
    if (
        identity["release"]["schema_version"] == 3
        and len(transforms) == 1
        and transforms[0]["kind"] == "lm_text"
        and identity["release"]["lm"]["selected"]
        and not identity["release"]["chat"]["selected"]
    ):
        fields["large_builder_implementation_sha256"] = "large_build"
    for field, name in fields.items():
        if identity.get(field) != files[f"src/sparselab/corpus/{name}.py"]["sha256"]:
            raise ValueError(f"IMPLEMENTATION_PROVENANCE_MISMATCH: build {name}")
    if (
        release["build_id"] != build["build_id"]
        or release["build_identity"] != identity
    ):
        raise ValueError("IMPLEMENTATION_PROVENANCE_MISMATCH: release/build binding")
    return snapshots


def replay_corpus(
    source: Path,
    commit: str,
    project: Path,
    work_root: Path,
    *,
    allow_network: bool,
    expected_release_sha256: str | None,
    expected_build_sha256: str | None = None,
) -> dict[str, Any]:
    """Execute historical producers in a dedicated uv environment and verify outputs."""
    root = Path(work_root).expanduser().absolute()
    _reject_symlink_ancestors(root)
    root = root.resolve()
    repository, _ = _repository(source, commit)
    _source_target(repository, root, commit)
    receipts = root / "replay" / "receipts"
    _reject_symlink_ancestors(receipts)
    receipts.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts / f"{uuid.uuid4().hex}.json"
    receipt: dict[str, Any] = {
        "format": _FORMAT,
        "status": "FAILED",
        "stage": "preflight",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": commit,
        "requested_source": str(Path(source).absolute()),
        "project": str(Path(project).absolute()),
        "expected_build_sha256": expected_build_sha256,
        "expected_release_sha256": expected_release_sha256,
        "allow_network": allow_network,
        "operations": [],
        "scientific_identity": {},
        "operational_identity": {
            "work_root": str(root),
            "orchestrator": str(Path(__file__).resolve()),
            "orchestrator_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
    }
    try:
        for value in (expected_build_sha256, expected_release_sha256):
            if value is not None and not _HEX.fullmatch(value):
                raise ValueError(
                    "INVALID_EXPECTED_DIGEST: expected SHA-256 must be lowercase full hex"
                )
        preflight = implementation_preflight(source, commit, project)
        receipt["preflight"] = preflight
        receipt["repository"] = preflight["repository"]
        if preflight["status"] == "MISSING_IMPLEMENTATION":
            raise ValueError(f"MISSING_IMPLEMENTATION: {preflight['reason']}")
        receipt["stage"] = "materialization"
        source_record = materialize_source(source, commit, root)
        historical = Path(source_record["source_root"])
        verify_materialized_source(historical, commit)
        receipt["source"] = {
            "tree": source_record["manifest"]["tree"],
            "inventory_sha256": source_record["manifest"]["inventory_sha256"],
            "manifest_path": source_record["manifest_path"],
        }
        relative = preflight["repository"]["project"]
        destination = historical / relative
        if not destination.is_file():
            raise ValueError(
                "SOURCE_COMMIT_MISMATCH: missing historical corpus declaration"
            )
        receipt["stage"] = "dependencies"
        environment = root / "replay" / "env" / commit
        _reject_symlink_ancestors(environment)
        environment.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        for key in (
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "UV_ACTIVE",
            "PYTHONHOME",
            "PYTHONPATH",
            "UV_PROJECT_ENVIRONMENT",
        ):
            env.pop(key, None)
        env.update(
            {
                "UV_PROJECT_ENVIRONMENT": str(environment),
                "UV_NO_ENV_FILE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "UV_NO_PROGRESS": "1",
            }
        )
        uv = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=False
        )
        if uv.returncode:
            raise ValueError(f"ENVIRONMENT_UNAVAILABLE: uv unavailable: {uv.stderr}")
        receipt["operational_identity"].update(
            {
                "uv": uv.stdout.strip(),
                "python": sys.version,
                "environment": str(environment),
                "lock_sha256": hashlib.sha256(
                    (historical / "uv.lock").read_bytes()
                ).hexdigest(),
            }
        )
        command = [
            "uv",
            "sync",
            "--project",
            str(historical),
            "--locked",
            "--no-install-project",
            "--no-dev",
            "--no-default-groups",
            "--python",
            sys.executable,
        ]
        historical_project = tomllib.loads(
            (historical / "pyproject.toml").read_text(encoding="utf-8")
        )
        if "cpu" in historical_project.get("project", {}).get(
            "optional-dependencies", {}
        ):
            command.extend(["--extra", "cpu"])
        if not allow_network:
            command.append("--offline")
        install = _run_logged(command, "dependencies", root / "replay" / "logs", env)
        receipt["operations"].append(install)
        if install["returncode"]:
            raise ValueError(
                "ENVIRONMENT_UNAVAILABLE: historical locked dependencies could not be installed"
            )
        verify_materialized_source(historical, commit)
        python = environment / "bin" / "python"
        if not python.is_file():
            raise ValueError("ENVIRONMENT_UNAVAILABLE: dedicated uv python missing")
        dependencies = subprocess.run(
            ["uv", "pip", "freeze", "--python", str(python)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if dependencies.returncode:
            raise ValueError(
                f"ENVIRONMENT_UNAVAILABLE: unable to inventory dedicated dependencies: {dependencies.stderr}"
            )
        interpreter = subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--no-sync",
                "--python",
                str(python),
                "python",
                "-I",
                "-c",
                "import sys; print(sys.version)",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if interpreter.returncode:
            raise ValueError(
                f"ENVIRONMENT_UNAVAILABLE: dedicated interpreter failed: {interpreter.stderr}"
            )
        receipt["operational_identity"]["dependencies"] = (
            dependencies.stdout.splitlines()
        )
        receipt["operational_identity"]["environment_python"] = (
            interpreter.stdout.strip()
        )
        work = root / "replay" / "work" / commit
        work.mkdir(parents=True, exist_ok=True)
        receipt["stage"] = "historical_producers"
        request = {
            "version": 1,
            "source_root": str(historical),
            "project": str(destination),
            "work_root": str(work),
            "allow_network": allow_network,
            "expected_release_sha256": expected_release_sha256,
            "expected_build_sha256": expected_build_sha256,
            "components": preflight["components"],
        }
        with tempfile.TemporaryDirectory(
            prefix=".invocation-", dir=receipts
        ) as temporary:
            request_path = Path(temporary) / "request.json"
            output_path = Path(temporary) / "response.json"
            request_path.write_text(
                json.dumps(request, sort_keys=True), encoding="utf-8"
            )
            worker = Path(__file__).with_name("_implementation_worker.py")
            run = _run_logged(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "--no-sync",
                    "--python",
                    str(python),
                    "python",
                    "-I",
                    str(worker),
                    str(request_path),
                    str(output_path),
                ],
                "historical_producers",
                root / "replay" / "logs",
                {**env, "UV_OFFLINE": "1" if not allow_network else "0"},
                work,
            )
            receipt["operations"].append(run)
            result = (
                json.loads(output_path.read_text(encoding="utf-8"))
                if output_path.is_file()
                else None
            )
        if (
            type(result) is not dict
            or type(result.get("version")) is not int
            or result["version"] != 1
            or result.get("status") not in {"MATCH", "FAILED"}
        ):
            raise ValueError(
                "HISTORICAL_EXECUTION_FAILED: invalid historical worker result"
            )
        if result["status"] == "MATCH":
            required = {
                "version",
                "status",
                "path",
                "build_path",
                "build_sha256",
                "release_sha256",
                "snapshots",
                "modules",
            }
            if (
                set(result) != required
                or any(type(result[key]) is not str for key in ("path", "build_path"))
                or any(
                    type(result[key]) is not str or not _HEX.fullmatch(result[key])
                    for key in ("build_sha256", "release_sha256")
                )
                or type(result["snapshots"]) is not list
                or type(result["modules"]) is not dict
            ):
                raise ValueError(
                    "HISTORICAL_EXECUTION_FAILED: malformed successful worker result"
                )
        elif (
            set(result)
            != {"version", "status", "phase", "artifacts", "error", "error_type"}
            or any(
                type(result[key]) is not str for key in ("phase", "error", "error_type")
            )
            or type(result["artifacts"]) is not dict
        ):
            raise ValueError(
                "HISTORICAL_EXECUTION_FAILED: malformed failed worker result"
            )
        receipt["worker"] = result
        if run["returncode"] or result["status"] != "MATCH":
            raise ValueError(
                result.get("error", "HISTORICAL_EXECUTION_FAILED: worker failed")
            )
        receipt["stage"] = "independent_verification"
        from sparselab.corpus.acquisition import verify_acquisition
        from sparselab.corpus.project import load_project
        from sparselab.corpus.release import verify_build, verify_release

        corpus = load_project(Path(project))
        lock = verify_acquisition(corpus, work)
        built = Path(result["build_path"])
        frozen = Path(result["path"])
        if (
            any(
                path.is_symlink()
                for path in (built, *built.parents, frozen, *frozen.parents)
            )
            or not built.resolve(strict=True).is_relative_to(work.resolve(strict=True))
            or not frozen.resolve(strict=True).is_relative_to(work.resolve(strict=True))
        ):
            raise ValueError(
                "HISTORICAL_EXECUTION_FAILED: artifact escaped isolated root"
            )
        build = verify_build(built)
        release = verify_release(frozen)
        if (
            build["build_id"] != result["build_sha256"]
            or release["release_id"] != result["release_sha256"]
            or (expected_build_sha256 and build["build_id"] != expected_build_sha256)
            or (
                expected_release_sha256
                and release["release_id"] != expected_release_sha256
            )
        ):
            raise ValueError(
                "EXPECTED_DIGEST_MISMATCH: independent artifact identity mismatch"
            )
        verified_source = verify_materialized_source(historical, commit)
        snapshots = _verify_producer_provenance(verified_source, lock, build, release)
        if snapshots != result["snapshots"]:
            raise ValueError(
                "IMPLEMENTATION_PROVENANCE_MISMATCH: worker snapshot inventory"
            )
        receipt["scientific_identity"] = {
            "build_sha256": build["build_id"],
            "release_sha256": release["release_id"],
            "snapshots": snapshots,
        }
        receipt["stage"] = "complete"
        receipt["status"] = "MATCH"
        receipt["path"] = str(frozen)
    except Exception as error:
        if (
            isinstance(receipt.get("worker"), dict)
            and receipt["worker"].get("status") == "FAILED"
        ):
            receipt["stage"] = receipt["worker"].get("phase", "historical_producers")
        receipt["error"] = str(error)
        raise
    finally:
        receipt["finished_at_utc"] = datetime.now(UTC).isoformat()
        receipt["record_sha256"] = hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with receipt_path.open("x", encoding="utf-8") as output:
            json.dump(receipt, output, indent=2, sort_keys=True)
            output.write("\n")
    return {
        "path": receipt["path"],
        "receipt_path": str(receipt_path),
        "receipt": receipt,
    }


def verify_replay_receipt(path: Path) -> dict[str, Any]:
    """Validate operational evidence and independently authenticate published artifacts."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("INVALID_REPLAY_RECEIPT: missing or symlinked receipt")
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "format",
        "status",
        "stage",
        "started_at_utc",
        "finished_at_utc",
        "source_commit",
        "requested_source",
        "project",
        "allow_network",
        "expected_build_sha256",
        "expected_release_sha256",
        "operations",
        "scientific_identity",
        "operational_identity",
        "record_sha256",
    }
    if (
        not isinstance(record, dict)
        or not required.issubset(record)
        or record["format"] != _FORMAT
        or record["status"] not in {"MATCH", "FAILED"}
        or type(record["allow_network"]) is not bool
        or not isinstance(record["operations"], list)
        or not isinstance(record["stage"], str)
    ):
        raise ValueError("INVALID_REPLAY_RECEIPT: invalid versioned record")
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    if (
        record["record_sha256"]
        != hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise ValueError("INVALID_REPLAY_RECEIPT: record digest mismatch")
    if record.get("repository"):
        repo = record["repository"]
        root, commit = _repository(Path(repo["root"]), record["source_commit"])
        if (
            commit != repo["commit"]
            or _repository_identity(root, commit, repo["project"])["tree"]
            != repo["tree"]
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: repository tree mismatch")
    for operation in record["operations"]:
        for channel in ("stdout", "stderr"):
            stream = Path(operation[f"{channel}_path"])
            if (
                stream.is_symlink()
                or not stream.is_file()
                or _log_identity(stream)[0] != operation[f"{channel}_sha256"]
            ):
                raise ValueError(f"INVALID_REPLAY_RECEIPT: {channel} log mismatch")
    if record.get("source") and isinstance(record["source"], dict):
        manifest = Path(record["source"]["manifest_path"])
        source_root = manifest.parent.parent / "source" / record["source_commit"]
        verified = verify_materialized_source(source_root, record["source_commit"])
        if verified["inventory_sha256"] != record["source"]["inventory_sha256"]:
            raise ValueError("INVALID_REPLAY_RECEIPT: source digest mismatch")
    if record["status"] == "MATCH":
        if not record.get("source") or not record.get("repository"):
            raise ValueError(
                "INVALID_REPLAY_RECEIPT: missing historical producer identity"
            )
        from sparselab.corpus.acquisition import verify_acquisition
        from sparselab.corpus.project import load_project
        from sparselab.corpus.release import verify_build, verify_release

        worker = record["worker"]
        if worker["status"] != "MATCH" or record["stage"] != "complete":
            raise ValueError("INVALID_REPLAY_RECEIPT: contradictory success")
        work = (
            Path(record["operational_identity"]["work_root"])
            / "replay"
            / "work"
            / record["source_commit"]
        )
        lock = verify_acquisition(load_project(Path(record["project"])), work)
        for key in ("path", "build_path"):
            artifact = Path(worker[key])
            if not artifact.resolve(strict=True).is_relative_to(
                work.resolve(strict=True)
            ) or any(item.is_symlink() for item in (artifact, *artifact.parents)):
                raise ValueError(
                    "INVALID_REPLAY_RECEIPT: artifact outside isolated work root"
                )
        build = verify_build(Path(worker["build_path"]))
        release = verify_release(Path(worker["path"]))
        snapshots = _verify_producer_provenance(verified, lock, build, release)
        if (
            snapshots != record["scientific_identity"]["snapshots"]
            or snapshots != worker["snapshots"]
            or build["build_id"] != record["scientific_identity"]["build_sha256"]
            or release["release_id"] != record["scientific_identity"]["release_sha256"]
            or build["build_id"] != worker["build_sha256"]
            or release["release_id"] != worker["release_sha256"]
            or (
                record["expected_build_sha256"] is not None
                and build["build_id"] != record["expected_build_sha256"]
            )
            or (
                record["expected_release_sha256"] is not None
                and release["release_id"] != record["expected_release_sha256"]
            )
            or record["path"] != worker["path"]
        ):
            raise ValueError("INVALID_REPLAY_RECEIPT: scientific identity mismatch")
    elif not isinstance(record.get("error"), str):
        raise ValueError("INVALID_REPLAY_RECEIPT: failure lacks reason")
    return record
