"""Verify existing typed experiment inputs without acquiring or preparing data."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import yaml

from sparselab.training.manifest import sha256_file

if TYPE_CHECKING:
    from sparselab.experiments.plan import Artifact, ExperimentPlan


def _safe_path(reference: str, source: Path) -> Path:
    declared = Path(reference)
    if ".." in declared.parts:
        raise ValueError("artifact path cannot contain parent traversal")
    target = declared if declared.is_absolute() else source.absolute().parent / declared
    target = target.absolute()
    if not target.exists():
        raise ValueError(f"missing artifact: {target}")
    for component in (target, *target.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked artifact path: {component}")
    if target.is_dir():
        for member in target.rglob("*"):
            if member.is_symlink():
                raise ValueError(f"symlinked artifact member: {member}")
    elif not target.is_file():
        raise ValueError(f"artifact is not a regular file or directory: {target}")
    return target


def _json_file(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"artifact manifest must be an object: {path}")
    return raw


def _require_directory(path: Path, manifest: str) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"expected a directory with {manifest}: {path}")
    return _json_file(path / manifest)


def _identity(artifact: Artifact, actual: object, digest: object) -> None:
    if actual != artifact.identifier:
        raise ValueError(
            f"logical identifier mismatch: expected {artifact.identifier}, got {actual}"
        )
    if digest != artifact.sha256:
        raise ValueError(
            f"domain digest mismatch: expected {artifact.sha256}, got {digest}"
        )


def _check_snapshot_paths(path: Path, manifest: dict[str, Any]) -> None:
    snapshots = manifest.get("snapshots")
    if not isinstance(snapshots, list):
        raise TypeError("missing build/release snapshot inventory")
    for item in snapshots:
        if not isinstance(item, dict):
            raise TypeError("invalid snapshot inventory entry")
        source_id, digest = item.get("source_id"), item.get("sha256")
        if (
            not isinstance(source_id, str)
            or not source_id
            or Path(source_id).name != source_id
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("invalid snapshot identity")
        _safe_path(str(path.parent.parent / "snapshots" / source_id / digest), path)


def _verify_domain(artifact: Artifact, path: Path) -> None:
    kind = artifact.kind
    if kind == "source_snapshot":
        from sparselab.corpus.acquisition import verify_snapshot

        manifest = _require_directory(path, "manifest.json")
        verified = verify_snapshot(path)
        if manifest != verified or manifest.get("schema_version") != artifact.version:
            raise ValueError("snapshot manifest/version mismatch")
        _identity(artifact, verified["source_id"], verified["snapshot_sha256"])
    elif kind == "corpus_build":
        from sparselab.corpus.release import verify_build

        manifest = _require_directory(path, "build.json")
        if manifest.get("schema_version") != artifact.version:
            raise ValueError("build format version mismatch")
        _check_snapshot_paths(path, manifest)
        verified = verify_build(path)
        _identity(artifact, verified["build_id"], verified["build_id"])
    elif kind == "corpus_release":
        from sparselab.corpus.release import verify_release

        manifest = _require_directory(path, "manifest.json")
        if manifest.get("schema_version") != artifact.version:
            raise ValueError("release format version mismatch")
        _check_snapshot_paths(path, manifest)
        verified = verify_release(path)
        _identity(artifact, verified["release_id"], verified["release_id"])
    elif kind == "corpus_export":
        from sparselab.config.models import RunConfig
        from sparselab.corpus.export import verify_release_export

        manifest = _require_directory(path, "export.json")
        if manifest.get("schema_version") != artifact.version:
            raise ValueError("export format version mismatch")
        run = RunConfig.model_validate(
            yaml.safe_load((path / "run.yaml").read_text(encoding="utf-8"))
        )
        if run.dataset.corpus_export_path != path:
            raise ValueError(
                "export run configuration references another export directory"
            )
        if run.dataset.corpus_release_path is None:
            raise ValueError("export lacks a pinned release")
        _safe_path(str(run.dataset.corpus_release_path), path)
        verified = verify_release_export(run.dataset)
        _identity(artifact, path.name, path.name)
        if verified["export_sha256"] != sha256_file(path / "export.json"):
            raise ValueError("export sidecar changed")
    elif kind == "tokenizer":
        from sparselab.data.tokenizer import load_tokenizer, verify_tokenizer_artifact

        if not path.is_file() or path.name != "tokenizer.json":
            raise ValueError(
                "tokenizer requires tokenizer.json and its provenance manifest"
            )
        manifest = _json_file(
            _safe_path(str(path.with_name("tokenizer_manifest.json")), path)
        )
        if artifact.version != 1:
            raise ValueError("unsupported tokenizer artifact version")
        source, revision, vocab = (
            manifest.get("source"),
            manifest.get("revision"),
            manifest.get("vocab_size"),
        )
        if not isinstance(source, str) or type(vocab) is not int:
            raise ValueError("invalid tokenizer provenance")
        dataset = None
        bakeoff = manifest.get("corpus_forge_bakeoff")
        if bakeoff is not None:
            if (
                source != "local_text"
                or not isinstance(bakeoff, dict)
                or not isinstance(bakeoff.get("release_id"), str)
            ):
                raise ValueError("invalid bakeoff tokenizer provenance")
            _safe_path(str(path.parent.parent.parent), path)
            report = _json_file(path.parent.parent.parent / "report.json")
            _safe_path(report["identity"]["release_path"], path)
        elif manifest.get("corpus_export") is not None:
            from sparselab.config.models import RunConfig

            export = path.parent.parent
            _safe_path(str(export), path)
            run = RunConfig.model_validate(
                yaml.safe_load((export / "run.yaml").read_text(encoding="utf-8"))
            )
            if run.tokenizer.path != path or run.dataset.corpus_export_path != export:
                raise ValueError("tokenizer is not bound to its corpus export")
            if run.dataset.corpus_release_path is None:
                raise ValueError("corpus tokenizer lacks a pinned release")
            _safe_path(str(run.dataset.corpus_release_path), path)
            dataset = run.dataset
        elif source == "local_stories":
            raise ValueError("local_stories tokenizer requires a pinned source dataset")
        verify_tokenizer_artifact(
            path, source=source, revision=revision, vocab_size=vocab, dataset=dataset
        )
        tokenizer = load_tokenizer(path)
        if tokenizer.get_vocab_size() != vocab:
            raise ValueError("tokenizer vocabulary differs from manifest")
        _identity(
            artifact,
            path.parent.name,
            sha256_file(path),
        )
    elif kind == "prepared_data":
        from sparselab.data.packing import load_prepared_data

        manifest = _require_directory(path, "manifest.json")
        if artifact.version != 1:
            raise ValueError("unsupported prepared-data artifact version")
        data = load_prepared_data(
            path, byte_enabled=manifest.get("byte_addressing") is not None
        )
        del data
        _identity(
            artifact, manifest.get("settings_sha256"), manifest.get("manifest_sha256")
        )
    elif kind == "stage_bundle":
        from sparselab.config.models import RunConfig
        from sparselab.staging import verify_stage_bundle

        inputs = _require_directory(path, "inputs.json")
        if artifact.version != 1:
            raise ValueError("unsupported stage-bundle version")
        config = RunConfig.model_validate(inputs["requested_config"])
        verified = verify_stage_bundle(path, config)
        _identity(artifact, verified.get("sha256"), verified.get("sha256"))
    elif kind == "checkpoint":
        from sparselab.training.checkpoints import FORMAT_VERSION, CheckpointManager

        raw = _require_directory(path, "manifest.json")
        if (
            path.name in {"latest.json", "best.json"}
            or artifact.version != FORMAT_VERSION
        ):
            raise ValueError("checkpoint must pin a supported immutable generation")
        if path.parent.name != "checkpoints":
            raise ValueError(
                "checkpoint must be a generation under a run's checkpoints directory"
            )
        if artifact.state not in {"full", "weights"}:
            raise ValueError("checkpoint state must be full or weights")
        expected_level = "full" if artifact.state == "full" else "weights_only"
        if raw.get("resume_level") != expected_level:
            raise ValueError(f"checkpoint does not have {expected_level} state")
        report = CheckpointManager(path.parent.parent).verify(
            path, require_training_state=artifact.state == "full"
        )
        if not report.valid:
            raise ValueError(f"checkpoint generation invalid: {report.errors}")
        if raw.get("format_version") != artifact.version:
            raise ValueError("checkpoint format version mismatch")
        _identity(artifact, path.name, raw.get("sha256"))
    elif kind == "capability_card":
        from sparselab.evaluation.capabilities import load_capability_card

        if not path.is_file():
            raise ValueError("capability card must be a JSON file")
        card = load_capability_card(path)
        if card.version != artifact.version:
            raise ValueError("capability card version mismatch")
        _identity(artifact, card.name, sha256_file(path))
    elif kind == "prompt_set":
        if artifact.version != 1 or not path.is_file() or path.suffix != ".json":
            raise ValueError("prompt_set v1 requires a JSON file")
        panel = _json_file(path)
        prompts = panel.get("prompts")
        if (
            panel.get("format") != "dense_lm_decoding_prompts_v1"
            or panel.get("split") not in {"development", "test"}
            or not isinstance(prompts, list)
            or not prompts
        ):
            raise ValueError("unsupported prompt-set format or empty panel")
        ids: set[str] = set()
        for row in prompts:
            if (
                not isinstance(row, dict)
                or any(
                    not isinstance(row.get(key), str) or not row[key]
                    for key in ("id", "text", "category")
                )
                or row["id"] in ids
            ):
                raise ValueError("invalid or duplicate prompt in frozen panel")
            ids.add(row["id"])
        _identity(artifact, path.stem, sha256_file(path))
    else:
        raise ValueError(f"no immutable external verifier for {kind}")


_MEMO_SEAL = object()


class _VerifiedArtifact:
    __slots__ = ("_seal", "identity")

    def __init__(self, identity: dict[str, object], *, _seal: object = None) -> None:
        if _seal is not _MEMO_SEAL:
            raise TypeError("artifact evidence cannot be constructed from metadata")
        self.identity = MappingProxyType(dict(identity))
        self._seal = _MEMO_SEAL


def _fingerprint(path: Path) -> tuple[tuple[str, int, int, int, int], ...]:
    members = [path] if path.is_file() else [path, *sorted(path.rglob("*"))]
    return tuple(
        (
            member.relative_to(path).as_posix() if member != path else ".",
            member.stat().st_dev,
            member.stat().st_ino,
            member.stat().st_size,
            member.stat().st_mtime_ns,
        )
        for member in members
    )


def verify_artifact(
    artifact: Artifact,
    source: Path,
    *,
    memo: dict[tuple[object, ...], _VerifiedArtifact] | None = None,
) -> dict[str, object]:
    """Verify one pinned external identity and return JSON-native identity/availability."""
    if artifact.from_phase is not None:
        raise ValueError(
            f"{artifact.kind} planned output {artifact.identifier} is unresolved until execution"
        )
    if artifact.path is None or artifact.sha256 is None:
        raise ValueError(f"{artifact.kind} external artifact lacks path or digest")
    try:
        path = _safe_path(artifact.path, source)
        key = (
            path.resolve(strict=True),
            artifact.kind,
            artifact.version,
            artifact.identifier,
            artifact.sha256,
            _fingerprint(path),
        )
        cached = memo.get(key) if memo is not None else None
        if isinstance(cached, _VerifiedArtifact) and cached._seal is _MEMO_SEAL:
            return dict(cached.identity)
        _verify_domain(artifact, path)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as error:
        raise ValueError(
            f"invalid {artifact.kind} artifact at {artifact.path}: {error}"
        ) from error
    verified = {
        "kind": artifact.kind,
        "version": artifact.version,
        "identifier": artifact.identifier,
        "sha256": artifact.sha256,
        "path": str(path),
    }
    if memo is not None:
        memo[key] = _VerifiedArtifact(verified, _seal=_MEMO_SEAL)
    return dict(verified)


def verify_inputs(
    plan: ExperimentPlan,
    source: Path,
    *,
    memo: dict[tuple[object, ...], _VerifiedArtifact] | None = None,
) -> dict[str, dict[str, object]]:
    """Resolve named plan inputs only; planned phase outputs remain unresolved."""
    return {
        name: verify_artifact(plan.artifacts[reference], source, memo=memo)
        for name, reference in plan.inputs.items()
    }
