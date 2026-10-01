"""Task-scoped, gated replay of the four pinned DevMind corpus generations.

Only operational receipts and immutable snapshots are copied; no historical lock,
manifest, declaration or adapter provenance is rewritten.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import stat
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sparselab.corpus.acquisition import (
    _project_sha,
    declaration_sha256,
    verify_acquisition,
    verify_snapshot,
)
from sparselab.corpus.project import (
    Project,
    ProjectConfig,
    load_project,
    source_declaration_payload,
)
from sparselab.recovery.implementation_replay import (
    materialize_source,
    replay_corpus,
    verify_materialized_source,
    verify_replay_receipt,
)
from sparselab.training.manifest import sha256_file

LINEAGE = Path(__file__).with_name("ancestry-replay-lineage.json")
STAGES = ("v2", "v3", "v4-intermediate", "v4-final")
# Reservations are operational admission limits, not estimates of corpus identity.
_RESERVATION = {
    "v2": (128 << 30, 300_000),
    "v3": (160 << 30, 500_000),
    "v4-intermediate": (240 << 30, 1_000_000),
    "v4-final": (64 << 30, 300_000),
}
_HEX = frozenset("0123456789abcdef")


def _sha(path: Path) -> str:
    return sha256_file(path)


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked operational record: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"invalid operational record: {path}")
    return value


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    with path.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def _no_links(path: Path) -> None:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"symlinked path refused: {path}")


def _publish_directory(source: Path, target: Path) -> None:
    """Linux exclusive atomic directory publish (never replace even an empty target)."""
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def _git_bytes(repo: Path, commit: str, path: str) -> bytes:
    if (
        not path
        or path.startswith("/")
        or any(p in ("", ".", "..") for p in path.split("/"))
    ):
        raise ValueError(f"unsafe Git declaration path: {path}")
    env = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1", "GIT_NO_LAZY_FETCH": "1"}
    tree = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", commit, "--", path],
        env=env,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    fields = tree.split(None, 3)
    if (
        len(fields) != 4
        or fields[0] != "100644"
        and fields[0] != "100755"
        or fields[1] != "blob"
        or fields[3] != path
    ):
        raise ValueError(f"missing regular Git declaration: {commit}:{path}")
    return subprocess.run(
        ["git", "-C", str(repo), "cat-file", "blob", fields[2]],
        env=env,
        capture_output=True,
        check=True,
    ).stdout


def _project(state: Path, repo: Path, stage: dict[str, Any]) -> tuple[Project, Path]:
    commit = stage["historical_producing_commit"]
    relative = stage["project_path"]
    # Git-object checks precede materialization, uv sync and network operations.
    _git_bytes(repo, commit, relative)
    source = Path(materialize_source(repo / relative, commit, state)["source_root"])
    verify_materialized_source(source, commit)
    path = source / relative
    project = load_project(path)
    declarations = [
        relative,
        *(
            str(Path(relative).parent / name)
            for name in (
                *project.config.sources,
                *project.config.transforms,
                project.config.splits,
                project.config.release,
            )
        ),
    ]
    for name in declarations:
        if (source / name).read_bytes() != _git_bytes(repo, commit, name):
            raise ValueError(f"historical declaration differs from Git blob: {name}")
    if project.config.id != stage["source_corpus_generation"]:
        raise ValueError("historical corpus generation mismatch")
    expected = stage["expected_project_sha256"]
    if expected is not None and _project_sha(project) != expected:
        raise ValueError(
            f"project identity mismatch: actual={_project_sha(project)} expected={expected}"
        )
    return project, path


def _lock(receipt: dict[str, Any]) -> dict[str, Any]:
    closure = receipt["acquisition_closure"]
    work = Path(receipt["corpus_work_root"])
    project = load_project(Path(receipt["historical_project"]))
    return verify_acquisition(project, work, lock_path=Path(closure["path"]))


def _safe_tree(path: Path) -> None:
    _no_links(path)
    if not path.is_dir():
        raise ValueError(f"missing snapshot directory: {path}")
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in (*dirs, *files):
            item = Path(root) / name
            mode = item.lstat().st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError(f"non-regular snapshot entry: {item}")


def _inventory(path: Path) -> dict[str, tuple[int, str]]:
    _safe_tree(path)
    return {
        file.relative_to(path).as_posix(): (file.stat().st_size, _sha(file))
        for file in path.rglob("*")
        if file.is_file()
    }


def _source_bytes(project: Project) -> dict[str, bytes]:
    return {
        source.id: (project.root / reference).read_bytes()
        for reference, source in zip(
            project.config.sources, project.sources, strict=True
        )
    }


def import_snapshots(
    parent_receipt: Path,
    target: Project,
    target_work: Path,
    inherited: set[str],
) -> dict[str, str]:
    """Import only IDs bound to an independently authenticated parent receipt."""
    parent = verify_replay_receipt(parent_receipt)
    if parent["status"] != "MATCH":
        raise ValueError("parent replay did not match")
    return _import_verified_snapshots(parent, target, target_work, inherited)


def _import_verified_snapshots(
    parent: dict[str, Any],
    target: Project,
    target_work: Path,
    inherited: set[str],
    *,
    previous: Project | None = None,
) -> dict[str, str]:
    """Publish each verified snapshot exclusively, without copying its lock."""
    old = previous or load_project(Path(parent["historical_project"]))
    old_sources = {source.id: source for source in old.sources}
    new_sources = {source.id: source for source in target.sources}
    if inherited != set(old_sources) & set(new_sources):
        raise ValueError(
            "imported source IDs are not the exact unchanged parent subset"
        )
    lock = _lock(parent)
    if set(lock["sources"]) != set(old_sources):
        raise ValueError("parent lock source set mismatch")
    previous_bytes = _source_bytes(old)
    target_bytes = _source_bytes(target)
    result: dict[str, str] = {}
    for source_id in sorted(inherited):
        old_source, new_source = old_sources[source_id], new_sources[source_id]
        if (
            source_declaration_payload(old_source)
            != source_declaration_payload(new_source)
            or previous_bytes[source_id] != target_bytes[source_id]
        ):
            raise ValueError(f"inherited source declaration changed: {source_id}")
        entry = lock["sources"][source_id]
        snapshot_sha = entry["snapshot_sha256"]
        if snapshot_sha is None:
            raise ValueError(f"inherited source lacks snapshot: {source_id}")
        original = (
            Path(parent["corpus_work_root"])
            / "corpora"
            / old.config.id
            / "snapshots"
            / source_id
            / snapshot_sha
        )
        destination = (
            target_work
            / "corpora"
            / target.config.id
            / "snapshots"
            / source_id
            / snapshot_sha
        )
        _safe_tree(original)
        manifest = verify_snapshot(original)
        if (
            manifest["source_id"] != source_id
            or manifest["snapshot_sha256"] != snapshot_sha
            or manifest["declaration_sha256"] != declaration_sha256(new_source)
            or manifest["declaration"] != source_declaration_payload(new_source)
        ):
            raise ValueError(
                f"parent snapshot declaration or identity mismatch: {source_id}"
            )
        _no_links(destination)
        if destination.exists():
            if (
                _inventory(destination) != _inventory(original)
                or verify_snapshot(destination) != manifest
            ):
                raise ValueError(f"conflicting imported snapshot: {source_id}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.parent / f".import-{uuid.uuid4().hex}"
            try:
                shutil.copytree(original, temporary, symlinks=True)
                if _inventory(temporary) != _inventory(original):
                    raise ValueError(f"snapshot copy byte mismatch: {source_id}")
                # Verify at canonical SHA path before publishing: verify_snapshot's
                # staged mode accepts a temporary basename but verifies every byte.
                if verify_snapshot(temporary, _staged=True) != manifest:
                    raise ValueError(f"snapshot copy manifest mismatch: {source_id}")
                try:
                    _publish_directory(temporary, destination)
                except FileExistsError as error:
                    raise ValueError(
                        f"conflicting snapshot publication: {source_id}"
                    ) from error
                if verify_snapshot(destination) != manifest:
                    raise ValueError(f"published snapshot mismatch: {source_id}")
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        result[source_id] = snapshot_sha
    return result


def _lineage(path: Path, state: Path) -> dict[str, Any]:
    record = _json(path)
    if (
        record.get("lineage_version") != 1
        or [s["id"] for s in record["stages"]] != list(STAGES)
        or Path(record["state_root"]) != state
    ):
        raise ValueError("lineage stage order or task state root mismatch")
    for stage in record["stages"]:
        if (
            len(stage["historical_producing_commit"]) != 40
            or set(stage["historical_producing_commit"]) - _HEX
        ):
            raise ValueError("invalid pinned producing commit")
    return record


def _preflight(state: Path, stage: str) -> dict[str, int]:
    _no_links(state)
    state.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(state)
    vfs = os.statvfs(state)
    reserved_bytes, reserved_inodes = _RESERVATION[stage]
    total_inodes = vfs.f_files
    free_inodes = vfs.f_favail
    if usage.free - reserved_bytes < usage.total // 4 or (
        total_inodes and free_inodes - reserved_inodes < total_inodes // 4
    ):
        raise ValueError(
            "insufficient phase byte/inode reservation plus 25% safety floor"
        )
    return {
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "free_inodes": free_inodes,
        "total_inodes": total_inodes,
        "reserved_bytes": reserved_bytes,
        "reserved_inodes": reserved_inodes,
    }


def _stage_path(state: Path, stage: str) -> Path:
    return state / "replay" / "ancestry" / "stages" / f"{stage}.json"


def _task_state_owned(state: Path) -> bool:
    if not state.exists():
        return True
    _no_links(state)
    if not state.is_dir():
        return False
    names = {entry.name for entry in state.iterdir()}
    if names - {"initial-preflight.json", "replay", "v4"}:
        return False
    initial = state / "initial-preflight.json"
    if initial.exists() and (initial.is_symlink() or not initial.is_file()):
        return False
    if names - {"initial-preflight.json"}:
        ancestry = state / "replay" / "ancestry"
        stages = ancestry / "stages"
        launches = ancestry / "launches"
        if not (
            (stages.is_dir() and any(stages.glob("*.json")))
            or (launches.is_dir() and any(launches.glob("*.json")))
        ):
            return False
    return True


def _advance(
    state_root: Path,
    through: str,
    *,
    allow_network: bool,
    lineage_path: Path = LINEAGE,
    repository: Path | None = None,
    validate_only: bool = False,
) -> dict[str, Any]:
    """Run at most one stage, only after all earlier stage records authenticate."""
    if through not in STAGES:
        raise ValueError(f"unknown ancestry stage: {through}")
    if not Path(state_root).expanduser().is_absolute():
        raise ValueError("state root must be absolute")
    state = Path(state_root).expanduser().absolute()
    _no_links(state)
    if not _task_state_owned(state):
        raise ValueError("conflicting pre-existing task root; refusing unrelated state")
    repo = (repository or Path(__file__).resolve().parents[3]).resolve()
    lineage = _lineage(lineage_path, state)
    lineage_sha = _sha(lineage_path)
    stages = lineage["stages"]
    launches = state / "replay" / "ancestry" / "launches"
    if launches.exists():
        for launch in launches.glob("*.json"):
            binding = _json(launch)
            if launch.stem not in STAGES or binding != {
                "stage": launch.stem,
                "lineage_sha256": lineage_sha,
                "source_commit": stages[STAGES.index(launch.stem)][
                    "historical_producing_commit"
                ],
            }:
                raise ValueError("task restart stage/source binding mismatch")
        pending = launches / f"{through}.json"
        if pending.exists() and not _stage_path(state, through).exists():
            raise ValueError(f"unreconciled prior phase attempt; no retry: {through}")
    parents: list[dict[str, Any]] = []
    for index in range(STAGES.index(through) + 1):
        stage = stages[index]
        name = STAGES[index]
        path = _stage_path(state, name)
        if not path.exists():
            if index < STAGES.index(through):
                raise ValueError(f"missing verified parent stage: {name}")
            break
        recorded = _json(path)
        if (
            recorded.get("lineage_sha256") != lineage_sha
            or recorded.get("stage") != name
        ):
            raise ValueError(f"task restart lineage binding mismatch: {name}")
        if recorded.get("status") != "MATCH":
            raise ValueError(f"previous stage attempt failed; no retry: {name}")
        receipt_path = Path(recorded["receipt_path"])
        if _sha(receipt_path) != recorded["receipt_sha256"]:
            raise ValueError(f"stage receipt byte mismatch: {name}")
        receipt = verify_replay_receipt(receipt_path)
        if (
            receipt["status"] != "MATCH"
            or receipt["source_commit"] != stage["historical_producing_commit"]
            or receipt["phase"] != stage["phase"]
            or receipt["build_id"] != stage["expected_build_sha256"]
            or receipt.get("release_id") != stage["expected_release_sha256"]
            or receipt["scientific_identity"]["project_sha256"]
            != recorded["project_sha256"]
            or receipt["corpus_work_root"] != str(_work(state, stage))
        ):
            raise ValueError(f"authenticated parent stage identity mismatch: {name}")
        if index and receipt["parent_receipt"] != {
            "path": parents[-1]["receipt_path"],
            "sha256": parents[-1]["receipt_sha256"],
        }:
            raise ValueError(f"parent receipt chain mismatch: {name}")
        if not index and receipt.get("parent_receipt") is not None:
            raise ValueError("v2 has an unexpected parent")
        lock = _lock(receipt)
        if (
            recorded.get("snapshots")
            != {key: entry["snapshot_sha256"] for key, entry in lock["sources"].items()}
            or recorded.get("acquisition_closure") != receipt["acquisition_closure"]
        ):
            raise ValueError(f"restart snapshot/closure binding mismatch: {name}")
        parents.append(recorded)
    if len(parents) == STAGES.index(through) + 1:
        return {"status": "ALREADY_VERIFIED", "stage": through, **parents[-1]}
    if validate_only:
        # Read-only: no source export, environment or acquisition is created.
        for stage in stages:
            import yaml

            project_bytes = _git_bytes(
                repo, stage["historical_producing_commit"], stage["project_path"]
            )
            config = ProjectConfig.model_validate(yaml.safe_load(project_bytes))
            directory = str(Path(stage["project_path"]).parent)
            for name in (
                *config.sources,
                *config.transforms,
                config.splits,
                config.release,
            ):
                _git_bytes(
                    repo, stage["historical_producing_commit"], f"{directory}/{name}"
                )
        return {
            "status": "LINEAGE_VALID",
            "next_stage": through,
            "lineage_sha256": lineage_sha,
        }
    if not allow_network:
        raise ValueError(
            "network permission required before running an uncompleted stage"
        )
    name = through
    stage = stages[STAGES.index(name)]
    state.mkdir(parents=True, exist_ok=True)
    launches.mkdir(parents=True, exist_ok=True)
    _exclusive_json(
        launches / f"{name}.json",
        {
            "stage": name,
            "lineage_sha256": lineage_sha,
            "source_commit": stage["historical_producing_commit"],
        },
    )
    measurements = _preflight(state, name)
    target_work = _work(state, stage)
    # Authenticate declarations and parent before copying or contacting any source.
    project, _historical_path = _project(state, repo, stage)
    expected_count = (
        stage["inherited_snapshot_count"]
        + stage["new_source_count"]
        + stage.get("changed_snapshot_count", 0)
    )
    if len(project.sources) != expected_count:
        raise ValueError(
            f"source count mismatch: actual={len(project.sources)} expected={expected_count}"
        )
    if any(s.redistribution == "rejected" for s in project.sources):
        raise ValueError("rejected source in required snapshot lineage")
    parent_receipt: Path | None = None
    unchanged: dict[str, str] = {}
    changed: dict[str, str] = {}
    if parents:
        parent_receipt = Path(parents[-1]["receipt_path"])
        parent = verify_replay_receipt(parent_receipt)
        old_project = load_project(Path(parent["historical_project"]))
        old = {s.id: s for s in old_project.sources}
        current = {s.id: s for s in project.sources}
        old_bytes = _source_bytes(old_project)
        current_bytes = _source_bytes(project)
        previous = _lock(parent)["sources"]
        if name == "v4-final":
            changed_ids = set(stage["changed_source_ids"])
            if (
                set(old) != set(current)
                or {
                    key
                    for key in old
                    if source_declaration_payload(old[key])
                    != source_declaration_payload(current[key])
                }
                != changed_ids
                or {key for key in old if old_bytes[key] != current_bytes[key]}
                != changed_ids
                or len(previous) != stage["previous_snapshot_count"]
                or len(changed_ids) != stage["changed_snapshot_count"]
            ):
                raise ValueError(
                    "final-v4 changed declaration set is not exactly the two pinned IDs"
                )
            changed = {
                key: previous[key]["snapshot_sha256"] for key in sorted(changed_ids)
            }
            inherited = set(old) - changed_ids
        else:
            if not set(old).issubset(current):
                raise ValueError("parent source declaration missing in child project")
            inherited = set(old)
        if len(inherited) != stage["inherited_snapshot_count"]:
            raise ValueError("inherited source count mismatch")
        if name != "v4-final":
            unchanged = _import_verified_snapshots(
                parent, project, target_work, inherited, previous=old_project
            )
        else:
            for source_id in sorted(inherited):
                if (
                    source_declaration_payload(old[source_id])
                    != source_declaration_payload(current[source_id])
                    or old_bytes[source_id] != current_bytes[source_id]
                ):
                    raise ValueError(
                        f"unchanged source declaration differs: {source_id}"
                    )
                entry = previous[source_id]
                original = (
                    Path(parent["corpus_work_root"])
                    / "corpora"
                    / project.config.id
                    / "snapshots"
                    / source_id
                    / entry["snapshot_sha256"]
                )
                if (
                    verify_snapshot(original)["snapshot_sha256"]
                    != entry["snapshot_sha256"]
                ):
                    raise ValueError(f"final inherited snapshot differs: {source_id}")
                unchanged[source_id] = entry["snapshot_sha256"]
    attempts = state / "replay" / "ancestry" / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt_path = attempts / f"{name}-{uuid.uuid4().hex}.json"
    result: dict[str, Any] = {
        "stage": name,
        "status": "FAILED",
        "lineage_sha256": lineage_sha,
        "source_commit": stage["historical_producing_commit"],
        "parent_receipt_path": str(parent_receipt) if parent_receipt else None,
        "expected_project_sha256": stage["expected_project_sha256"],
        "expected_build_sha256": stage["expected_build_sha256"],
        "expected_release_sha256": stage["expected_release_sha256"],
        "inherited_snapshots": unchanged,
        "changed_snapshots": changed,
        "observed_project_sha256": _project_sha(project),
        "admission": measurements,
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    prior_receipts = set((state / "replay" / "receipts").glob("*.json"))
    try:
        request = replay_corpus(
            repo / stage["project_path"],
            stage["historical_producing_commit"],
            repo / stage["project_path"],
            state,
            allow_network=allow_network,
            expected_project_sha256=stage["expected_project_sha256"],
            expected_build_sha256=stage["expected_build_sha256"],
            expected_release_sha256=stage["expected_release_sha256"],
            phase=stage["phase"],
            parent_receipt=parent_receipt,
            inherited_source_ids=tuple(sorted(unchanged)),
            expected_unchanged_snapshots=unchanged or None,
            expected_changed_snapshots=changed or None,
            corpus_work_root=target_work if name.startswith("v4-") else None,
            use_historical_project=True,
        )
        receipt_path = Path(request["receipt_path"])
        result["receipt_path"] = str(receipt_path)
        result["receipt_sha256"] = _sha(receipt_path)
        receipt = verify_replay_receipt(receipt_path)
        if receipt["status"] != "MATCH":
            raise ValueError(f"replay gate failed: {receipt.get('error', 'unknown')}")
        lock = _lock(receipt)
        observed = {
            key: entry["snapshot_sha256"] for key, entry in lock["sources"].items()
        }
        if len(observed) != expected_count:
            raise ValueError(
                f"snapshot count mismatch: actual={len(observed)} expected={expected_count}"
            )
        for source_id, expected in sorted(unchanged.items()):
            if observed.get(source_id) != expected:
                raise ValueError(
                    f"unchanged snapshot differs: {source_id}: actual={observed.get(source_id)} expected={expected}"
                )
        for source_id, previous_sha in sorted(changed.items()):
            if observed.get(source_id) == previous_sha:
                raise ValueError(
                    f"changed snapshot did not change: {source_id}: {previous_sha}"
                )
        for field, actual, expected in (
            (
                "project_sha256",
                receipt["scientific_identity"]["project_sha256"],
                stage["expected_project_sha256"],
            ),
            ("build_id", receipt["build_id"], stage["expected_build_sha256"]),
            ("release_id", receipt.get("release_id"), stage["expected_release_sha256"]),
        ):
            if (
                expected is not None
                and actual != expected
                or field == "release_id"
                and actual != expected
            ):
                raise ValueError(
                    f"{field} mismatch: actual={actual} expected={expected}"
                )
        if name == "v4-intermediate" and receipt.get("release") is not None:
            raise ValueError("intermediate v4 unexpectedly froze a release")
        result.update(
            status="MATCH",
            project_sha256=receipt["scientific_identity"]["project_sha256"],
            snapshots=observed,
            build_id=receipt["build_id"],
            release_id=receipt.get("release_id"),
            acquisition_closure=receipt["acquisition_closure"],
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        new_receipts = (
            set((state / "replay" / "receipts").glob("*.json")) - prior_receipts
        )
        if len(new_receipts) == 1:
            failed_receipt = new_receipts.pop()
            result["receipt_path"] = str(failed_receipt)
            result["receipt_sha256"] = _sha(failed_receipt)
            try:
                failure = verify_replay_receipt(failed_receipt)
            except (ValueError, KeyError, OSError) as verification_error:
                result["receipt_verification_error"] = str(verification_error)
            else:
                result["failure_gate"] = failure.get("stage")
                result["observed_scientific_identity"] = failure.get(
                    "scientific_identity"
                )
                worker = failure.get("worker", {})
                if isinstance(worker, dict):
                    result["observed_worker_identities"] = {
                        key: worker[key]
                        for key in (
                            "project_sha256",
                            "build_sha256",
                            "release_sha256",
                            "pre_snapshots",
                            "post_snapshots",
                            "snapshots",
                        )
                        if key in worker
                    }
                result["acquisition_closure"] = failure.get("acquisition_closure")
        raise
    finally:
        result["finished_at_utc"] = datetime.now(UTC).isoformat()
        _exclusive_json(attempt_path, result)
        _exclusive_json(
            _stage_path(state, name), {**result, "attempt_path": str(attempt_path)}
        )
    return {**result, "attempt_path": str(attempt_path)}


def advance(
    state_root: Path,
    through: str,
    *,
    allow_network: bool,
    lineage_path: Path = LINEAGE,
    repository: Path | None = None,
    validate_only: bool = False,
) -> dict[str, Any]:
    """Reverify the lineage and persist a first failure without retrying it."""
    try:
        return _advance(
            state_root,
            through,
            allow_network=allow_network,
            lineage_path=lineage_path,
            repository=repository,
            validate_only=validate_only,
        )
    except Exception as error:
        state = Path(state_root).expanduser().absolute()
        if (
            allow_network
            and not validate_only
            and through in STAGES
            and state.is_dir()
            and _task_state_owned(state)
            and not _stage_path(state, through).exists()
        ):
            failure = {
                "stage": through,
                "status": "FAILED",
                "lineage_sha256": _sha(lineage_path),
                "error": f"{type(error).__name__}: {error}",
                "finished_at_utc": datetime.now(UTC).isoformat(),
            }
            receipts = state / "replay" / "receipts"
            # When execution itself failed, preserve its independently readable
            # failure receipt; never claim which payload leaf diverged without it.
            matching = list(receipts.glob("*.json"))
            if len(matching) == 1 and through == "v2":
                failure["receipt_path"] = str(matching[0])
                failure["receipt_sha256"] = _sha(matching[0])
            attempt = (
                state
                / "replay"
                / "ancestry"
                / "attempts"
                / f"{through}-{uuid.uuid4().hex}.json"
            )
            _exclusive_json(attempt, failure)
            _exclusive_json(
                _stage_path(state, through), {**failure, "attempt_path": str(attempt)}
            )
        raise


def _work(state: Path, stage: dict[str, Any]) -> Path:
    return (
        state / "v4"
        if stage["id"].startswith("v4-")
        else state / "replay" / "work" / stage["historical_producing_commit"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--through", choices=STAGES, required=True)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        outcome = advance(
            args.state_root,
            args.through,
            allow_network=args.allow_network,
            validate_only=args.validate_only,
        )
        print(json.dumps(outcome, sort_keys=True))
        return 0
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        failure: dict[str, Any] = {
            "status": "FAILED",
            "error": str(error),
            "stage": args.through,
        }
        stage_path = _stage_path(args.state_root.expanduser().absolute(), args.through)
        if stage_path.is_file() and not stage_path.is_symlink():
            result = _json(stage_path)
            for key in (
                "attempt_path",
                "failure_gate",
                "receipt_path",
                "receipt_sha256",
                "acquisition_closure",
                "observed_project_sha256",
                "project_sha256",
                "build_id",
                "release_id",
                "observed_scientific_identity",
                "expected_project_sha256",
                "observed_worker_identities",
                "expected_build_sha256",
                "expected_release_sha256",
                "inherited_snapshots",
                "changed_snapshots",
                "snapshots",
            ):
                if key in result:
                    failure[key] = result[key]
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
