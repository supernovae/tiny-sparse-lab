"""Immutable, engine-neutral dispatch bundles for independent workers."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sparselab.config.models import RunConfig
from sparselab.data.packing import load_prepared_data
from sparselab.data.tokenizer import load_tokenizer
from sparselab.model.portable_engram import load_portable_engram
from sparselab.staging import (
    materialize_prepared_inputs,
    verify_prepared_inputs,
    verify_stage_bundle,
)
from sparselab.training.checkpoints import (
    CheckpointManager,
    _fsync_directory,
    _safe_member,
    strict_json,
)
from sparselab.training.manifest import (
    ArtifactIdentity,
    canonical_json,
    config_sha256,
    read_manifest,
    sha256_file,
    source_identity,
)

_CHUNK = 1024 * 1024
_CACHE = ".dispatch-cache"


def _models() -> Any:
    from sparselab.workers import models

    return models


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        os.chmod(destination, 0o600)
        while block := incoming.read(_CHUNK):
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def _copy_tree(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise ValueError(f"symlinked dispatch source: {source}")
    if source.is_file():
        _atomic_copy(source, destination)
        return
    if not source.is_dir():
        raise ValueError(f"missing dispatch source: {source}")
    destination.mkdir(parents=True, mode=0o700)
    for child in sorted(source.rglob("*")):
        target = destination / child.relative_to(source)
        if child.is_symlink():
            raise ValueError(f"symlinked dispatch source: {child}")
        if child.is_dir():
            target.mkdir(mode=0o700)
        elif child.is_file():
            _atomic_copy(child, target)
        else:
            raise ValueError(f"unsupported dispatch source member: {child}")


def _inventory(root: Path) -> list[ArtifactIdentity]:
    result: list[ArtifactIdentity] = []
    for member in sorted(root.rglob("*")):
        if member.is_symlink():
            raise ValueError(f"symlinked bundle member: {member}")
        if member.is_file():
            result.append(
                ArtifactIdentity(
                    member.relative_to(root).as_posix(),
                    sha256_file(member),
                    member.stat().st_size,
                )
            )
    return result


def _require_safe(root: Path, identity: ArtifactIdentity) -> Path:
    path = _safe_member(root, identity.relative_path)
    if path is None or not path.is_file() or path.stat().st_size != identity.size_bytes:
        raise ValueError(f"missing or unsafe bundle asset: {identity.relative_path}")
    if sha256_file(path) != identity.sha256:
        raise ValueError(f"bundle asset digest mismatch: {identity.relative_path}")
    return path


def _destination(root: Path, name: str) -> Path:
    relative = Path(name)
    if (
        not name
        or any(ord(character) < 32 for character in name)
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError("unsafe dispatch member path")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError("unsafe dispatch member destination")
    return path


def _manifest_payload(manifest: Any) -> bytes:
    return canonical_json(manifest.model_dump(mode="json"))


def _write_bundle_manifest(path: Path, manifest: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        os.chmod(path, 0o600)
        handle.write(_manifest_payload(manifest))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_bundle_manifest(path: Path) -> Any:
    import json

    from sparselab.training.manifest import _reject_duplicate_keys, _reject_nonfinite

    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid dispatch bundle manifest: {error}") from error
    return _models().BundleManifest.model_validate(raw)


def _selected_run(selected: Path) -> tuple[Path, Path]:
    selected = selected.resolve(strict=True)
    if selected.name in {"latest.json", "best.json"}:
        pointer = strict_json(selected)
        relative = pointer.get("relative_path")
        if not isinstance(relative, str):
            raise ValueError("invalid checkpoint pointer")
        selected = _safe_member(selected.parent, relative)
        if selected is None:
            raise ValueError("unsafe checkpoint pointer")
    if not selected.is_dir() or selected.parent.name != "checkpoints":
        raise ValueError("continuation must select a finalized checkpoint generation")
    return selected.parent.parent, selected


def _versions() -> dict[str, object]:
    return _models().current_required_versions()


def _drift_authorized(continuation: Any) -> bool:
    return continuation.kind == "RESUMED" and continuation.allow_runtime_drift


def _verify_parent_closure(
    config: RunConfig, selected: Path, *, resume: bool
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    """Verify a selected generation and every parent-declared local input."""
    run, generation = _selected_run(selected)
    parent = read_manifest(run / "manifest.json")
    parent_digest = hashlib.sha256(canonical_json(parent)).hexdigest()
    # Reconstitute the already-verified sealed representation for the bundle.
    parent["sha256"] = parent_digest
    manager = CheckpointManager(run)
    report = manager.verify(
        selected,
        expected_manifest=parent_digest,
        require_training_state=resume,
        expected_config=config,
    )
    if not report.valid:
        raise ValueError(f"selected continuation is invalid: {report.errors}")
    generation_manifest = strict_json(generation / "manifest.json")
    generation_digest = generation_manifest.get("sha256")
    if not isinstance(generation_digest, str):
        raise TypeError("selected generation has no embedded digest")
    artifacts = parent.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("parent manifest artifact inventory is invalid")
    identities = [ArtifactIdentity(**item) for item in artifacts]
    by_name = {item.relative_path: item for item in identities}
    if len(by_name) != len(identities):
        raise ValueError("parent manifest has duplicate artifact entries")
    for item in identities:
        _require_safe(run, item)
    required = {"tokenizer.json"}
    if resume:
        required.update({"data/manifest.json", "data/train.npy", "data/validation.npy"})
    if not required <= by_name.keys():
        raise ValueError("selected continuation lacks bound run-owned artifacts")
    parent_config = RunConfig.model_validate(parent["effective_config"])
    if parent_config.model.vocab_size != config.model.vocab_size:
        raise ValueError("continuation tokenizer/model vocabulary differs")
    return run, generation, parent, generation_manifest


def _continuation_assets(destination: Path, selected: Path, *, resume: bool) -> None:
    run, generation = _selected_run(selected)
    manifest = read_manifest(run / "manifest.json")
    _copy_tree(
        run / "manifest.json", destination / "continuation" / "parent_manifest.json"
    )
    for name in ("resolved_config.yaml", "tokenizer.json", "tokenizer_manifest.json"):
        candidate = run / name
        if candidate.is_file():
            _copy_tree(candidate, destination / "continuation" / name)
    _copy_tree(
        generation, destination / "continuation" / "checkpoints" / generation.name
    )
    artifacts = {item["relative_path"] for item in manifest.get("artifacts", [])}
    required = {"tokenizer.json"}
    if resume:
        required.update({"data/manifest.json", "data/train.npy", "data/validation.npy"})
    if not required <= artifacts:
        raise ValueError("selected continuation lacks bound run-owned artifacts")
    for name in sorted(artifacts):
        if (
            resume
            or name in {"tokenizer.json", "portable_package"}
            or name.startswith("portable_package/")
        ):
            source = _safe_member(run, name)
            if source is None:
                raise ValueError(f"unsafe continuation artifact: {name}")
            _copy_tree(source, destination / "continuation" / "run_assets" / name)


def prepare_dispatch_bundle(
    config: RunConfig,
    output: Path,
    *,
    stage_bundle: Path | None = None,
    promote: Path | None = None,
    resume: Path | None = None,
    allow_runtime_drift: bool = False,
) -> Any:
    """Freeze offline inputs and verified parent state without probing a device."""
    if promote is not None and resume is not None:
        raise ValueError("promotion and full resume are mutually exclusive")
    if allow_runtime_drift and resume is None:
        raise ValueError("runtime drift requires an explicit full resume")
    selected = resume or promote
    parent_details = (
        _verify_parent_closure(config, selected, resume=resume is not None)
        if selected is not None
        else None
    )
    evidence = None
    if stage_bundle is not None:
        verified_stage = verify_stage_bundle(
            stage_bundle, config, allow_runtime_drift=allow_runtime_drift
        )
        evidence = {"stage_bundle_sha256": verified_stage["sha256"]}
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"dispatch bundle exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temp:
        work = Path(temp)
        if parent_details is not None and resume is not None and stage_bundle is None:
            run, _, parent, _ = parent_details
            for identity in parent["artifacts"]:
                _copy_tree(
                    run / identity["relative_path"],
                    work / "assets" / identity["relative_path"],
                )
        else:
            prepared_config = config
            if parent_details is not None:
                run = parent_details[0]
                parent_tokenizer = run / "tokenizer.json"
                if (
                    not config.tokenizer.path.is_file()
                    or config.tokenizer.path.is_symlink()
                    or sha256_file(config.tokenizer.path)
                    != sha256_file(parent_tokenizer)
                ):
                    raise ValueError("promotion requires matching tokenizer identity")
                prepared_config = config.model_copy(
                    update={
                        "tokenizer": config.tokenizer.model_copy(
                            update={"path": parent_tokenizer}
                        )
                    }
                )
                if config.model.memory_package_path is not None:
                    from sparselab.training.continuation import _package_identity

                    if _package_identity(
                        config.model.memory_package_path
                    ) != _package_identity(run / "portable_package"):
                        raise ValueError(
                            "promotion requires matching portable package identity"
                        )
                    prepared_config = prepared_config.model_copy(
                        update={
                            "model": config.model.model_copy(
                                update={"memory_package_path": run / "portable_package"}
                            )
                        }
                    )
            prepared = work / "prepared"
            materialize_prepared_inputs(
                prepared_config, prepared, prepared_inputs=stage_bundle
            )
            os.rename(prepared / "assets", work / "assets")
            shutil.rmtree(prepared)
        if parent_details is not None:
            run, _, parent, generation_manifest = parent_details
            parent_artifacts = {
                item["relative_path"]: item for item in parent["artifacts"]
            }
            for name in (
                "tokenizer.json",
                *(
                    ("data/manifest.json", "data/train.npy", "data/validation.npy")
                    if resume is not None
                    else ()
                ),
            ):
                staged = work / "assets" / name
                identity = parent_artifacts[name]
                if (
                    not staged.is_file()
                    or staged.stat().st_size != identity["size_bytes"]
                    or sha256_file(staged) != identity["sha256"]
                ):
                    raise ValueError(
                        f"continuation {name} differs from prepared inputs"
                    )
            continuation = _models().ContinuationSpec.model_validate(
                {
                    "kind": "RESUMED" if resume is not None else "PROMOTED",
                    "parent_run_id": parent["run_id"],
                    "checkpoint_sha256": generation_manifest["sha256"],
                    "artifact_identity": {
                        "relative_path": "continuation/parent_manifest.json",
                        "sha256": parent["sha256"],
                        "size_bytes": (run / "manifest.json").stat().st_size,
                    },
                    "allow_runtime_drift": allow_runtime_drift,
                }
            )
            assert selected is not None
            _continuation_assets(work, selected, resume=resume is not None)
        else:
            continuation = _models().ContinuationSpec(kind="FRESH")
        manifest = _models().BundleManifest.model_validate(
            {
                "schema_version": 1,
                "config": config.model_dump(mode="json"),
                "config_sha256": config_sha256(config.model_dump(mode="json")),
                "required_versions": _versions(),
                "source_identity_sha256": source_identity()["sha256"],
                "continuation": continuation.model_dump(mode="json"),
                "files": [item.__dict__ for item in _inventory(work)],
                "stage_evidence": evidence,
            }
        )
        _write_bundle_manifest(work / "bundle.json", manifest)
        verify_dispatch_bundle(work)
        os.rename(work, output)
        _fsync_directory(output.parent)
    return manifest


def _verify_embedded_continuation(root: Path, manifest: Any, config: RunConfig) -> None:
    continuation = manifest.continuation
    if continuation.kind == "FRESH":
        if any(
            item.relative_path.startswith("continuation/") for item in manifest.files
        ):
            raise ValueError("fresh bundle contains continuation assets")
        return
    identity = continuation.artifact_identity
    if (
        identity is None
        or identity.relative_path != "continuation/parent_manifest.json"
    ):
        raise ValueError("continuation parent manifest identity is invalid")
    parent_path = root / identity.relative_path
    parent = read_manifest(parent_path)
    parent_digest = hashlib.sha256(canonical_json(parent)).hexdigest()
    if (
        parent_digest != identity.sha256
        or parent_path.stat().st_size != identity.size_bytes
    ):
        raise ValueError("embedded parent manifest identity mismatch")
    if parent.get("run_id") != continuation.parent_run_id:
        raise ValueError("embedded parent run ID mismatch")
    generations = [
        path
        for path in (root / "continuation" / "checkpoints").glob("step_*_gen_*")
        if path.is_dir() and not path.is_symlink()
    ]
    if len(generations) != 1:
        raise ValueError("embedded continuation must contain exactly one generation")
    generation = generations[0]
    report = CheckpointManager(root).verify(
        generation,
        expected_manifest=parent_digest,
        require_training_state=continuation.kind == "RESUMED",
        expected_config=config,
    )
    if not report.valid:
        raise ValueError(f"embedded continuation verification failed: {report.errors}")
    raw = strict_json(generation / "manifest.json")
    if raw.get("sha256") != continuation.checkpoint_sha256:
        raise ValueError("embedded continuation digest mismatch")
    parent_assets = {
        item["relative_path"]: item
        for item in parent.get("artifacts", [])
        if isinstance(item, dict)
    }
    tokenizer = root / "assets" / "tokenizer.json"
    tokenizer_identity = parent_assets.get("tokenizer.json")
    if (
        not isinstance(tokenizer_identity, dict)
        or not tokenizer.is_file()
        or sha256_file(tokenizer) != tokenizer_identity.get("sha256")
    ):
        raise ValueError("embedded continuation tokenizer differs from bundle")
    if continuation.kind == "RESUMED":
        for name in ("data/manifest.json", "data/train.npy", "data/validation.npy"):
            if not (root / "assets" / name).is_file():
                raise ValueError("full continuation lacks offline prepared data")


def verify_dispatch_bundle(root: Path) -> Any:
    root = root.resolve(strict=True)
    manifest = _read_bundle_manifest(root / "bundle.json")
    _models().validate_required_versions(manifest.required_versions)
    config = RunConfig.model_validate(manifest.config)
    if manifest.config_sha256 != config_sha256(config.model_dump(mode="json")):
        raise ValueError("dispatch bundle config hash mismatch")
    if manifest.source_identity_sha256 != source_identity()[
        "sha256"
    ] and not _drift_authorized(manifest.continuation):
        raise ValueError("dispatch bundle source identity differs")
    cache_assets = (
        root.parent.parent / "assets"
        if not (root / "assets").is_dir() and root.parent.name == "bundles"
        else None
    )
    names: set[str] = set()
    for value in manifest.files:
        identity = ArtifactIdentity(
            **(value.model_dump() if hasattr(value, "model_dump") else value)
        )
        if identity.relative_path in names:
            raise ValueError("duplicate dispatch bundle asset")
        names.add(identity.relative_path)
        if cache_assets is None:
            _require_safe(root, identity)
        else:
            _require_safe(
                cache_assets,
                ArtifactIdentity(identity.sha256, identity.sha256, identity.size_bytes),
            )
    if not {"assets/tokenizer.json", "assets/data/manifest.json"} <= names:
        raise ValueError("dispatch bundle lacks required offline assets")
    if cache_assets is None:
        _verify_bundle_inputs(root, config, manifest)
        _verify_embedded_continuation(root, manifest, config)
    return manifest


def _verify_bundle_inputs(bundle_root: Path, config: RunConfig, manifest: Any) -> None:
    """Verify offline inputs in place; cache verification never creates a bundle copy."""
    assets = bundle_root / "assets"
    data = load_prepared_data(
        assets / "data", byte_enabled=config.model.memory in {"byte", "portable"}
    )
    tokenizer = load_tokenizer(assets / "tokenizer.json")
    if tokenizer.get_vocab_size() != config.model.vocab_size:
        raise ValueError("bundle tokenizer vocabulary does not match model")
    if not len(data.train) or not len(data.validation):
        raise ValueError("bundle prepared data is empty")
    if config.model.memory_package_path is not None:
        load_portable_engram(
            assets / "portable_package",
            expected_shape=(config.model.memory_table_size, config.model.memory_dim),
            expected_ngram_size=config.model.memory_ngram_size,
        )


def _cache_paths(worker_root: Path, digest: str) -> tuple[Path, Path]:
    cache = worker_root / _CACHE
    for directory in (
        cache,
        cache / "assets",
        cache / "bundles",
        cache / "bundles" / digest,
    ):
        if directory.is_symlink():
            raise ValueError("symlinked dispatch cache directory")
    return cache / "assets", cache / "bundles" / digest


def install_dispatch_bundle(
    worker_root: Path,
    manifest: Any,
    attachments: Mapping[str, Path],
    *,
    check_only: bool,
) -> dict[str, object]:
    """Install only missing immutable content-addressed files, then publish manifest."""
    worker_root = worker_root.resolve()
    config = RunConfig.model_validate(manifest.config)
    _models().validate_required_versions(manifest.required_versions)
    if manifest.config_sha256 != config_sha256(config.model_dump(mode="json")):
        raise ValueError("dispatch bundle schema or config identity mismatch")
    if manifest.source_identity_sha256 != source_identity()[
        "sha256"
    ] and not _drift_authorized(manifest.continuation):
        raise ValueError("dispatch bundle source identity mismatch")
    assets, bundle_dir = _cache_paths(worker_root, manifest.digest())
    assets.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(assets.parent, 0o700)
    os.chmod(assets, 0o700)
    bundle_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(bundle_dir.parent, 0o700)
    missing: list[str] = []
    identities = [
        ArtifactIdentity(**(item.model_dump() if hasattr(item, "model_dump") else item))
        for item in manifest.files
    ]
    for item in identities:
        cached = assets / item.sha256
        if cached.is_symlink():
            raise ValueError("symlinked cached dispatch asset")
        if (
            not cached.is_file()
            or cached.stat().st_size != item.size_bytes
            or sha256_file(cached) != item.sha256
        ):
            missing.append(item.sha256)
        else:
            os.chmod(cached, 0o600)
    if check_only:
        return {
            "bundle_digest": manifest.digest(),
            "missing_asset_digests": sorted(set(missing)),
            "installed": False,
        }
    required = {f"assets/{digest}" for digest in set(missing)}
    allowed = {f"assets/{item.sha256}" for item in identities}
    asset_attachments = {
        name: path for name, path in attachments.items() if name != "bundle.json"
    }
    if not required <= set(asset_attachments) <= allowed:
        raise ValueError(
            "install attachments must cover missing assets and belong to the bundle"
        )
    need = sum(
        next(item.size_bytes for item in identities if item.sha256 == digest)
        for digest in set(missing)
    )
    if shutil.disk_usage(worker_root).free < need:
        raise OSError("insufficient free disk for dispatch bundle")
    for item in identities:
        incoming = asset_attachments.get(f"assets/{item.sha256}")
        if incoming is None:
            continue
        if (
            incoming.is_symlink()
            or incoming.stat().st_size != item.size_bytes
            or sha256_file(incoming) != item.sha256
        ):
            raise ValueError("dispatch attachment integrity failure")
        target = assets / item.sha256
        if item.sha256 not in missing:
            _require_safe(
                assets, ArtifactIdentity(item.sha256, item.sha256, item.size_bytes)
            )
            continue
        with tempfile.NamedTemporaryFile(
            prefix=".asset.", dir=assets, delete=False
        ) as handle:
            temporary = Path(handle.name)
        temporary.unlink()
        try:
            _atomic_copy(incoming, temporary)
            if sha256_file(temporary) != item.sha256:
                raise ValueError("dispatch attachment changed during install")
            if target.exists():
                _require_safe(
                    assets, ArtifactIdentity(item.sha256, item.sha256, item.size_bytes)
                )
                if sha256_file(target) != item.sha256:
                    raise ValueError("conflicting cached dispatch asset")
            else:
                os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    with (worker_root / _CACHE / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if bundle_dir.exists():
            existing = _read_bundle_manifest(bundle_dir / "bundle.json")
            if existing.digest() != manifest.digest():
                raise ValueError("conflicting bundle digest publication")
        else:
            with tempfile.TemporaryDirectory(
                prefix=f".{bundle_dir.name}.", dir=bundle_dir.parent
            ) as directory:
                temporary = Path(directory)
                _write_bundle_manifest(temporary / "bundle.json", manifest)
                verify_dispatch_bundle(temporary)
                os.rename(temporary, bundle_dir)
                _fsync_directory(bundle_dir.parent)
    verify_dispatch_bundle(bundle_dir)
    return {
        "bundle_digest": manifest.digest(),
        "missing_asset_digests": [],
        "installed": True,
    }


def materialize_dispatch_bundle(
    worker_root: Path, bundle_digest: str, destination: Path
) -> Any:
    """Make an isolated worker-private layout from an already verified cache."""
    assets, bundle_dir = _cache_paths(worker_root.resolve(), bundle_digest)
    manifest = _read_bundle_manifest(bundle_dir / "bundle.json")
    _models().validate_required_versions(manifest.required_versions)
    if manifest.source_identity_sha256 != source_identity()[
        "sha256"
    ] and not _drift_authorized(manifest.continuation):
        raise ValueError("cached dispatch bundle source identity differs")
    if manifest.digest() != bundle_digest:
        raise ValueError("bundle digest mismatch")
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError("dispatch materialization destination exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temp:
        work = Path(temp)
        for value in manifest.files:
            item = ArtifactIdentity(
                **(value.model_dump() if hasattr(value, "model_dump") else value)
            )
            cached = assets / item.sha256
            if (
                not cached.is_file()
                or cached.stat().st_size != item.size_bytes
                or sha256_file(cached) != item.sha256
            ):
                raise ValueError("cached dispatch asset is missing or corrupt")
            target = _destination(work, item.relative_path)
            _atomic_copy(cached, target)
        _verify_bundle_inputs(work, RunConfig.model_validate(manifest.config), manifest)
        _verify_embedded_continuation(
            work, manifest, RunConfig.model_validate(manifest.config)
        )
        _write_bundle_manifest(work / "bundle.json", manifest)
        # staging consumes this standardized input root without source paths.
        input_root = work
        payload = {
            "format_version": 1,
            "requested_config": manifest.config.model_dump(mode="json"),
            "source_identity_sha256": manifest.source_identity_sha256,
            "artifacts": [item.__dict__ for item in _inventory(input_root / "assets")],
        }
        (input_root / "inputs.json").write_bytes(
            canonical_json(
                {
                    **payload,
                    "sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
                }
            )
            + b"\n"
        )
        os.chmod(input_root / "inputs.json", 0o600)
        verify_prepared_inputs(
            input_root,
            RunConfig.model_validate(manifest.config),
            allow_runtime_drift=_drift_authorized(manifest.continuation),
        )
        os.rename(work, destination)
        _fsync_directory(destination.parent)
    return manifest
