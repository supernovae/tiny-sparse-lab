"""Pinned source acquisition and independently verifiable immutable byte snapshots."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse

from sparselab.corpus.project import (
    GitAcquisition,
    HttpAcquisition,
    HuggingFaceAcquisition,
    LocalAcquisition,
    Project,
    SourceDeclaration,
    project_path,
    safe_name,
)
from sparselab.engram.packs import _rename_noreplace
from sparselab.hf_auth import HUB_ACCESS_ERRORS, hub_auth_kwargs, raise_for_hub_auth
from sparselab.training.manifest import canonical_json, sha256_file

ADAPTER_VERSION = "corpus-acquisition-v1"


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _identity_declaration(source: SourceDeclaration) -> dict[str, Any]:
    declaration = source.model_dump(mode="json")
    if source.kind == "local":
        declaration["acquisition"]["files"] = [
            {"name": entry.name} for entry in source.acquisition.files
        ]
    return declaration


def declaration_sha256(source: SourceDeclaration) -> str:
    return _digest(_identity_declaration(source))


def _adapter(source: SourceDeclaration) -> dict[str, str]:
    return {
        "id": source.kind,
        "version": ADAPTER_VERSION,
        "module_sha256": sha256_file(Path(__file__)),
    }


def _write_json(path: Path, value: object) -> None:
    with path.open("wb") as handle:
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_stream(source: Any, target: Path, remaining: int) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with target.open("xb") as output:
        while chunk := source.read(min(1024 * 1024, remaining - size + 1)):
            size += len(chunk)
            if size > remaining:
                raise ValueError("source exceeds declared max_bytes")
            digest.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    return digest.hexdigest(), size


def _regular(path: Path) -> None:
    if path.is_symlink() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise ValueError(f"source is not a regular, nonsymlink file: {path}")


def _acquire_local(
    source: SourceDeclaration, root: Path, staging: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = source.acquisition
    assert isinstance(spec, LocalAcquisition)
    total = 0
    inventory = []
    for item in sorted(spec.files, key=lambda f: f.name):
        location = Path(item.path)
        location = location if location.is_absolute() else project_path(root, item.path)
        _regular(location)
        with location.open("rb") as stream:
            digest, size = _copy_stream(
                stream, staging / "files" / item.name, spec.max_bytes - total
            )
        total += size
        inventory.append({"path": item.name, "sha256": digest, "size": size})
    return inventory, {"acquisition_paths": [item.model_dump() for item in spec.files]}


def _git(args: list[str], *, cwd: Path | None = None) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True
    ).stdout


def _acquire_git(
    source: SourceDeclaration, staging: Path, cache_root: Path, offline: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if offline:
        raise ValueError("offline acquisition cannot fetch Git")
    spec = source.acquisition
    assert isinstance(spec, GitAcquisition)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / _digest(
        {"uri": source.canonical_uri, "revision": source.revision}
    )
    if not cache.exists():
        cache.mkdir()
        _git(["init", "--bare", str(cache)])
    _git(
        [
            "--git-dir",
            str(cache),
            "fetch",
            "--depth=1",
            "--no-tags",
            source.canonical_uri,
            source.revision,
        ]
    )
    commit = (
        _git(["--git-dir", str(cache), "rev-parse", "FETCH_HEAD^{commit}"])
        .decode()
        .strip()
    )
    if commit.lower() != source.revision.lower():
        raise ValueError("Git fetched commit differs from pinned revision")
    tree = _git(["--git-dir", str(cache), "ls-tree", "-rz", "--full-tree", commit])
    inventory = []
    total = 0
    selected = set()
    for record in tree.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8", errors="strict")
        if not any(
            fnmatch.fnmatchcase(path, pattern) for pattern in spec.include
        ) or any(fnmatch.fnmatchcase(path, pattern) for pattern in spec.exclude):
            continue
        safe_name(path)
        if mode not in ("100644", "100755") or kind != "blob":
            raise ValueError(f"Git selection contains unsafe symlink/submodule: {path}")
        selected.add(path)
        size = int(_git(["--git-dir", str(cache), "cat-file", "-s", blob]))
        if size > spec.max_bytes - total:
            raise ValueError("Git selection exceeds max_bytes")
        target = staging / "files" / path
        with subprocess.Popen(
            ["git", "--git-dir", str(cache), "cat-file", "blob", blob],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            assert process.stdout is not None
            digest, copied = _copy_stream(
                process.stdout, target, spec.max_bytes - total
            )
            if process.wait() != 0 or copied != size:
                raise ValueError(f"Git blob read failed: {path}")
        total += copied
        inventory.append(
            {"path": path, "sha256": digest, "size": copied, "git_blob_id": blob}
        )
    if not selected:
        raise ValueError("Git include patterns selected no files")
    return sorted(inventory, key=lambda item: item["path"]), {"commit": commit}


class _PrivateHubRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward a Hub bearer credential to another origin."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
        if redirected is not None:
            before = urlparse(request.full_url)
            after = urlparse(newurl)
            if (before.scheme, before.netloc) != (after.scheme, after.netloc):
                redirected.remove_header("Authorization")
        return redirected


def _acquire_http(
    source: SourceDeclaration, staging: Path, offline: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if offline:
        raise ValueError("offline acquisition cannot fetch HTTP")
    spec = source.acquisition
    assert isinstance(spec, HttpAcquisition)
    if urlparse(source.canonical_uri).scheme not in ("http", "https"):
        raise ValueError("HTTP source requires an HTTP(S) canonical URI")
    suffix = Path(urlparse(source.canonical_uri).path).suffix.lower()
    if suffix not in (".txt", ".md", ".markdown"):
        raise ValueError("HTTP source must explicitly name a text/Markdown document")
    name = "document" + suffix
    headers = {"User-Agent": "SparseLab-Corpus-Forge/1"}
    origin = urlparse(source.canonical_uri)
    if origin.scheme == "https" and origin.hostname == "huggingface.co":
        token = hub_auth_kwargs().get("token")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(source.canonical_uri, headers=headers)
    try:
        with urllib.request.build_opener(_PrivateHubRedirect()).open(
            request, timeout=30
        ) as response:
            if urlparse(response.url).scheme not in ("http", "https"):
                raise ValueError("HTTP redirect has unsafe scheme")
            digest, size = _copy_stream(
                response, staging / "files" / name, spec.max_bytes
            )
            retrieval = {
                "final_url": response.url,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
            }
    except HTTPError as error:
        if origin.scheme == "https" and origin.hostname == "huggingface.co":
            raise_for_hub_auth(error, credential_supplied="Authorization" in headers)
        raise
    if digest != spec.expected_sha256:
        raise ValueError("HTTP expected SHA-256 mismatch")
    return [{"path": name, "sha256": digest, "size": size}], retrieval


def _validate_hf_rows(path: Path, spec: HuggingFaceAcquisition, available: int) -> int:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        if spec.text_field not in parquet.schema_arrow.names:
            raise ValueError(f"HF text field missing: {spec.text_field}")
        count = parquet.metadata.num_rows
        if count > available:
            raise ValueError("HF selection exceeds max_rows")
        return count
    if path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list):
            raise ValueError("HF JSON requires object or array of objects")
        if len(records) > available:
            raise ValueError("HF selection exceeds max_rows")
    else:
        with path.open(encoding="utf-8") as handle:
            records = []
            for line in handle:
                if len(records) >= available:
                    raise ValueError("HF selection exceeds max_rows")
                records.append(json.loads(line))
    if any(
        not isinstance(record, dict) or not isinstance(record.get(spec.text_field), str)
        for record in records
    ):
        raise ValueError(f"HF text field missing/non-string: {spec.text_field}")
    return len(records)


def _acquire_hf(
    source: SourceDeclaration, staging: Path, cache_root: Path, offline: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if offline:
        raise ValueError("offline acquisition cannot fetch Hugging Face")
    from huggingface_hub import snapshot_download

    spec = source.acquisition
    assert isinstance(spec, HuggingFaceAcquisition)
    auth = hub_auth_kwargs()
    try:
        location = Path(
            snapshot_download(
                repo_id=source.canonical_uri.removeprefix(
                    "https://huggingface.co/datasets/"
                ),
                repo_type="dataset",
                revision=source.revision,
                allow_patterns=list(spec.include),
                cache_dir=str(cache_root),
                **auth,
            )
        )
    except HUB_ACCESS_ERRORS as error:
        raise_for_hub_auth(error, credential_supplied=bool(auth))
    if location.name.lower() != source.revision.lower():
        raise ValueError("HF snapshot revision does not match pinned commit")
    inventory = []
    total = rows = 0
    for path in sorted(location.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(location).as_posix()
        if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in spec.include):
            continue
        safe_name(relative)
        if path.is_symlink():
            # hub cache often provides symlinks to immutable blobs; resolve within the cache.
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(cache_root.resolve()):
                raise ValueError("HF snapshot symlink escapes cache")
            path = resolved
        _regular(path)
        parts = relative.split("/")
        if spec.config not in parts and not Path(relative).name.startswith(
            spec.config + "-"
        ):
            raise ValueError(
                f"HF file is not unambiguously in config {spec.config}: {relative}"
            )
        if spec.split not in parts and not any(
            part.startswith((spec.split + "-", spec.split + ".")) for part in parts
        ):
            raise ValueError(
                f"HF file is not unambiguously in split {spec.split}: {relative}"
            )
        if path.suffix.lower() not in (".jsonl", ".json", ".parquet"):
            raise ValueError(f"unsupported HF file format: {relative}")
        if path.stat().st_size > spec.max_bytes - total:
            raise ValueError("HF selection exceeds max_bytes")
        rows += _validate_hf_rows(path, spec, spec.max_rows - rows)
        with path.open("rb") as stream:
            digest, size = _copy_stream(
                stream, staging / "files" / relative, spec.max_bytes - total
            )
        total += size
        inventory.append({"path": relative, "sha256": digest, "size": size})
    if not inventory:
        raise ValueError("HF include patterns selected no files")
    return inventory, {
        "config": spec.config,
        "split": spec.split,
        "text_field": spec.text_field,
        "max_rows": spec.max_rows,
        "rows": rows,
    }


def verify_snapshot(path: Path | str, *, _staged: bool = False) -> dict[str, Any]:
    path = Path(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["schema_version"] != 1
            or manifest["source_id"] != manifest["declaration"]["id"]
        ):
            raise ValueError("snapshot metadata mismatch")
        source = SourceDeclaration.model_validate(manifest["declaration"])
        if (
            declaration_sha256(source) != manifest["declaration_sha256"]
            or manifest["adapter"]["id"] != source.kind
        ):
            raise ValueError("snapshot declaration/adapter identity mismatch")
        expected = _digest(
            {
                "declaration_sha256": manifest["declaration_sha256"],
                "adapter": manifest["adapter"],
                "files": manifest["files"],
            }
        )
        if expected != manifest["snapshot_sha256"] or (
            not _staged and path.name != expected
        ):
            raise ValueError("snapshot identity mismatch")
        names = set()
        for entry in manifest["files"]:
            name = safe_name(entry["path"])
            if name in names:
                raise ValueError("duplicate snapshot path")
            names.add(name)
            file = path / "files" / name
            _regular(file)
            if (
                not file.resolve().is_relative_to((path / "files").resolve())
                or file.stat().st_size != entry["size"]
                or sha256_file(file) != entry["sha256"]
            ):
                raise ValueError(f"snapshot byte mismatch: {name}")
        actual = {
            p.relative_to(path / "files").as_posix()
            for p in (path / "files").rglob("*")
            if p.is_file() or p.is_symlink()
        }
        if actual != names:
            raise ValueError("snapshot file inventory mismatch")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid/incomplete snapshot: {path}") from error


def _project_sha(project: Project) -> str:
    """Only acquisition declarations bind the lock; release variants share snapshots."""
    return _digest(
        {
            "project_id": project.config.id,
            "sources": [
                s.model_dump(mode="json")
                for s in sorted(project.sources, key=lambda s: s.id)
            ],
        }
    )


def verify_acquisition(project: Project, work_root: Path | str) -> dict[str, Any]:
    base = Path(work_root) / "corpora" / project.config.id
    try:
        lock = json.loads((base / "acquisition.json").read_text(encoding="utf-8"))
        if (
            lock["schema_version"] != 1
            or lock["project_id"] != project.config.id
            or lock["project_sha256"] != _project_sha(project)
            or set(lock["sources"]) != {s.id for s in project.sources}
        ):
            raise ValueError("acquisition lock does not match project")
        for source in project.sources:
            entry = lock["sources"][source.id]
            if entry["declaration_sha256"] != declaration_sha256(source):
                raise ValueError(f"declaration changed: {source.id}")
            if source.redistribution == "rejected":
                if (
                    entry["receipt"]["status"] != "rejected"
                    or entry["receipt"]["reason"] != source.rejection_reason
                    or entry["snapshot_path"] is not None
                    or entry["snapshot_sha256"] is not None
                ):
                    raise ValueError("invalid rejected receipt")
                continue
            snapshot = base / "snapshots" / source.id / entry["snapshot_sha256"]
            if entry["snapshot_path"] != str(snapshot.resolve()):
                raise ValueError("snapshot path mismatch")
            manifest = verify_snapshot(snapshot)
            if (
                manifest["snapshot_sha256"] != entry["snapshot_sha256"]
                or manifest["declaration_sha256"] != entry["declaration_sha256"]
                or manifest["declaration"] != source.model_dump(mode="json")
            ):
                raise ValueError("snapshot provenance mismatch")
            status = "generator" if source.kind.endswith("generator") else "acquired"
            if entry["receipt"] != {
                "status": status,
                "retrieval": manifest["retrieval"],
            }:
                raise ValueError("snapshot receipt mismatch")
        return lock
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("missing or invalid acquisition lock") from error


def acquire(
    project: Project, work_root: Path | str, offline: bool = False
) -> dict[str, Any]:
    """Publish verified snapshots independently; replace the active lock only on success."""
    base = Path(work_root) / "corpora" / project.config.id
    if offline:
        return verify_acquisition(project, work_root)
    snapshot_root = base / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    entries = {}
    for source in project.sources:
        declared_digest = declaration_sha256(source)
        if source.redistribution == "rejected":
            entries[source.id] = {
                "declaration_sha256": declared_digest,
                "snapshot_sha256": None,
                "snapshot_path": None,
                "receipt": {"status": "rejected", "reason": source.rejection_reason},
            }
            continue
        # A pre-existing lock is advisory for reuse, but never trusted without verification.
        try:
            old = json.loads((base / "acquisition.json").read_text(encoding="utf-8"))[
                "sources"
            ][source.id]
            old_path = snapshot_root / source.id / old["snapshot_sha256"]
            if old["declaration_sha256"] == declared_digest:
                manifest = verify_snapshot(old_path)
                if source.kind not in ("local", "http_document") and manifest[
                    "adapter"
                ] == _adapter(source):
                    entries[source.id] = old
                    continue
        except OSError, ValueError, TypeError, KeyError, json.JSONDecodeError:
            pass
        parent = snapshot_root / source.id
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".acquire-", dir=parent))
        try:
            (staging / "files").mkdir()
            retrieval: dict[str, Any] = {}
            if source.kind == "local":
                files, retrieval = _acquire_local(source, project.root, staging)
            elif source.kind == "git":
                files, retrieval = _acquire_git(
                    source, staging, base / "git-cache", False
                )
            elif source.kind == "http_document":
                files, retrieval = _acquire_http(source, staging, False)
            elif source.kind == "huggingface_dataset":
                files, retrieval = _acquire_hf(
                    source, staging, base / "hf-cache", False
                )
            else:
                files = []
                retrieval = {
                    "generator_config_sha256": _digest(
                        source.acquisition.model_dump(mode="json")
                    )
                }
            adapter = _adapter(source)
            identity = _digest(
                {
                    "declaration_sha256": declared_digest,
                    "adapter": adapter,
                    "files": files,
                }
            )
            manifest = {
                "schema_version": 1,
                "source_id": source.id,
                "declaration": source.model_dump(mode="json"),
                "declaration_sha256": declared_digest,
                "adapter": adapter,
                "files": files,
                "retrieval": retrieval,
                "snapshot_sha256": identity,
            }
            _write_json(staging / "manifest.json", manifest)
            if verify_snapshot(staging, _staged=True)["snapshot_sha256"] != identity:
                raise ValueError("staged snapshot identity mismatch")
            destination = parent / identity
            if destination.exists():
                verify_snapshot(destination)
            else:
                for directory in sorted((staging / "files").rglob("*"), reverse=True):
                    if directory.is_dir():
                        _sync_dir(directory)
                _sync_dir(staging / "files")
                _sync_dir(staging)
                _rename_noreplace(staging, destination)
                _sync_dir(parent)
                verify_snapshot(destination)
            entries[source.id] = {
                "declaration_sha256": declared_digest,
                "snapshot_sha256": identity,
                "snapshot_path": str(destination.resolve()),
                "receipt": {
                    "status": "generator"
                    if not files and source.kind.endswith("generator")
                    else "acquired",
                    "retrieval": retrieval,
                },
            }
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    lock = {
        "schema_version": 1,
        "project_id": project.config.id,
        "project_sha256": _project_sha(project),
        "sources": entries,
    }
    handle, name = tempfile.mkstemp(prefix=".acquisition-", dir=base)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(canonical_json(lock) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, base / "acquisition.json")
        _sync_dir(base)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return verify_acquisition(project, work_root)
