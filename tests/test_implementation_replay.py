"""Real Git-object materialization and isolated historical Corpus Forge execution."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from sparselab.corpus.acquisition import verify_acquisition, verify_snapshot
from sparselab.corpus.project import load_project
from sparselab.corpus.release import verify_build, verify_release
from sparselab.recovery.implementation_replay import (
    implementation_preflight,
    materialize_source,
    replay_corpus,
    verify_materialized_source,
    verify_replay_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def pinned_project(tmp_path: Path) -> Iterator[tuple[Path, Path, str]]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "Historical Fixture")
    shutil.copytree(
        ROOT / "src",
        repository / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        ROOT / "corpora" / "devmind-sample-v0",
        repository / "corpora" / "devmind-sample-v0",
    )
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copy2(ROOT / name, repository / name)
    project = repository / "corpora" / "devmind-sample-v0" / "corpus.yaml"
    pipeline = repository / "src" / "sparselab" / "corpus" / "pipeline.py"
    pipeline.write_bytes(
        pipeline.read_bytes().replace(b"import hashlib\n", b"import  hashlib\n", 1)
        + b"\n# historical corpus implementation fixture\n"
    )
    acquisition = repository / "src/sparselab/corpus/acquisition.py"
    acquisition.write_bytes(
        acquisition.read_bytes().replace(b"import hashlib\n", b"import  hashlib\n", 1)
    )
    _git(repository, "add", "src", "corpora", "pyproject.toml", "uv.lock", "README.md")
    _git(repository, "commit", "-m", "pin genuine historical producer")
    historical = _git(repository, "rev-parse", "HEAD")
    pipeline.write_bytes(
        (ROOT / "src" / "sparselab" / "corpus" / "pipeline.py").read_bytes()
    )
    yield repository, project, historical
    # Keep artifact/receipt evidence, but do not retain multi-gigabyte test environments.
    for environment_root in tmp_path.glob("*/replay/env"):
        shutil.rmtree(environment_root)


def test_preflight_byte_difference_and_missing_commit(
    pinned_project: tuple[Path, Path, str],
) -> None:
    _, project, commit = pinned_project
    report = implementation_preflight(project, commit, project)
    assert report["status"] == "PINNED_IMPLEMENTATION_REPLAY_REQUIRED"
    assert (
        report["components"]["pipeline"]["status"]
        == "PINNED_IMPLEMENTATION_REPLAY_REQUIRED"
    )
    assert report["components"]["release"]["direct_identity_hash"] is False
    assert report["repository"]["commit"] == commit
    assert (
        implementation_preflight(project, "0" * len(commit), project)["status"]
        == "MISSING_IMPLEMENTATION"
    )


def test_materialization_reuses_verified_git_bytes_and_rejects_tamper(
    pinned_project: tuple[Path, Path, str], tmp_path: Path
) -> None:
    _, project, commit = pinned_project
    first = materialize_source(project, commit, tmp_path / "work")
    second = materialize_source(project, commit, tmp_path / "work")
    assert first["source_root"] == second["source_root"]
    source = Path(first["source_root"])
    historical = source / "src" / "sparselab" / "corpus" / "pipeline.py"
    assert historical.read_bytes().endswith(
        b"# historical corpus implementation fixture\n"
    )
    assert (
        verify_materialized_source(source, commit)["inventory_sha256"]
        == first["manifest"]["inventory_sha256"]
    )
    historical.chmod(0o644)
    historical.write_bytes(
        historical.read_bytes().replace(b"def build(", b"def substituted_build(", 1)
    )
    with pytest.raises(ValueError, match="SOURCE_TAMPERED"):
        verify_materialized_source(source, commit)
    with pytest.raises(ValueError, match="SOURCE_TAMPERED"):
        materialize_source(project, commit, tmp_path / "work")


def test_git_replace_refs_cannot_redirect_committed_source(
    pinned_project: tuple[Path, Path, str], tmp_path: Path
) -> None:
    repository, project, commit = pinned_project
    pipeline = repository / "src" / "sparselab" / "corpus" / "pipeline.py"
    pipeline.write_bytes(pipeline.read_bytes() + b"\n# replacement implementation\n")
    _git(repository, "add", "src")
    _git(repository, "commit", "-m", "different implementation")
    replacement = _git(repository, "rev-parse", "HEAD")
    _git(repository, "replace", commit, replacement)
    work = tmp_path / "work"
    published = materialize_source(project, commit, work)
    assert not Path(published["source_root"]).is_relative_to(repository)
    assert (
        (
            Path(published["source_root"])
            / "src"
            / "sparselab"
            / "corpus"
            / "pipeline.py"
        )
        .read_bytes()
        .endswith(b"# historical corpus implementation fixture\n")
    )
    assert published["manifest"]["commit"] == commit
    assert (
        verify_materialized_source(Path(published["source_root"]), commit)["tree"]
        == published["manifest"]["tree"]
    )


def test_in_checkout_replay_root_is_rejected_without_relocation(pinned_project) -> None:
    repository, project, commit = pinned_project
    work = repository / "ignored-work"
    with pytest.raises(ValueError, match="UNSAFE_REPLAY_ROOT"):
        materialize_source(project, commit, work)
    with pytest.raises(ValueError, match="UNSAFE_REPLAY_ROOT"):
        replay_corpus(
            project,
            commit,
            project,
            work,
            allow_network=False,
            expected_release_sha256=None,
        )
    assert not work.exists()
    assert not (repository.parent / ".sparselab-replay-sources").exists()


def test_source_manifest_symlink_rejected_before_execution(
    pinned_project: tuple[Path, Path, str], tmp_path: Path
) -> None:
    _, project, commit = pinned_project
    root = tmp_path / "output"
    external = tmp_path / "unrelated"
    external.mkdir()
    (root / "replay").mkdir(parents=True)
    (root / "replay" / "source").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="UNSAFE_REPLAY_ROOT"):
        replay_corpus(
            project,
            commit,
            project,
            root,
            allow_network=False,
            expected_release_sha256=None,
        )
    assert not list(external.iterdir())


def test_dirty_project_input_rejected_before_environment_creation(
    pinned_project: tuple[Path, Path, str], tmp_path: Path
) -> None:
    _, project, commit = pinned_project
    project.write_bytes(
        project.read_bytes().replace(
            b"id: devmind-sample-v0", b"id: devmind-altered-v0"
        )
    )
    work = tmp_path / "output"
    with pytest.raises(ValueError, match="SOURCE_COMMIT_MISMATCH"):
        replay_corpus(
            project,
            commit,
            project,
            work,
            allow_network=False,
            expected_release_sha256=None,
        )
    assert not (work / "replay" / "env").exists()
    receipts = list((work / "replay" / "receipts").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "FAILED"


def test_historical_worker_publishes_authentic_artifacts_offline(
    pinned_project: tuple[Path, Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, project, commit = pinned_project
    real_run = subprocess.run
    uv_operations: set[str] = set()

    def managed_execution(command, **kwargs):
        executable = Path(command[0]).name
        assert executable in {"git", "uv"}, "Replay bypassed uv-managed execution"
        if executable == "uv":
            uv_operations.add(command[1])
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", managed_execution)
    root = tmp_path / "output"
    result = replay_corpus(
        project,
        commit,
        project,
        root,
        allow_network=False,
        expected_release_sha256=None,
    )
    receipt = result["receipt"]
    assert receipt["status"] == "MATCH"
    assert (
        receipt["scientific_identity"]["release_sha256"]
        == verify_release(Path(result["path"]))["release_id"]
    )
    assert (
        receipt["scientific_identity"]["build_sha256"]
        == verify_build(Path(receipt["worker"]["build_path"]))["build_id"]
    )
    assert receipt["worker"]["modules"]["pipeline"] == str(
        Path(receipt["source"]["manifest_path"]).parent.parent
        / "source"
        / commit
        / "src"
        / "sparselab"
        / "corpus"
        / "pipeline.py"
    )
    assert receipt["operational_identity"]["environment"] == str(
        root / "replay" / "env" / commit
    )
    historical_lock = subprocess.check_output(
        ["git", "-C", str(project.parents[2]), "show", f"{commit}:uv.lock"]
    )
    assert (
        receipt["operational_identity"]["lock_sha256"]
        == hashlib.sha256(historical_lock).hexdigest()
    )
    lock = verify_acquisition(load_project(project), root / "replay" / "work" / commit)
    for entry in lock["sources"].values():
        if entry["snapshot_sha256"]:
            assert (
                verify_snapshot(Path(entry["snapshot_path"]))["snapshot_sha256"]
                == entry["snapshot_sha256"]
            )
    assert verify_replay_receipt(Path(result["receipt_path"]))["status"] == "MATCH"
    assert Path(receipt["operational_identity"]["environment"]) != Path(sys.prefix)
    assert {"sync", "pip", "run"} <= uv_operations


def test_receipt_rejects_tampered_record_and_operational_log(
    pinned_project: tuple[Path, Path, str], tmp_path: Path
) -> None:
    _, project, commit = pinned_project
    result = replay_corpus(
        project,
        commit,
        project,
        tmp_path / "output",
        allow_network=False,
        expected_release_sha256=None,
    )
    path = Path(result["receipt_path"])
    record = verify_replay_receipt(path)
    log = Path(record["operations"][-1]["stderr_path"])
    log.write_bytes(log.read_bytes() + b"tampered log")
    with pytest.raises(ValueError, match="INVALID_REPLAY_RECEIPT: stderr log mismatch"):
        verify_replay_receipt(path)
    log.write_bytes(log.read_bytes().removesuffix(b"tampered log"))
    raw = json.loads(path.read_text())
    raw["scientific_identity"]["release_sha256"] = "0" * 64
    path.write_text(json.dumps(raw))
    with pytest.raises(
        ValueError, match="INVALID_REPLAY_RECEIPT: record digest mismatch"
    ):
        verify_replay_receipt(path)
    raw["scientific_identity"]["release_sha256"] = result["receipt"][
        "scientific_identity"
    ]["release_sha256"]
    raw["expected_release_sha256"] = "0" * 64
    raw.pop("record_sha256")
    raw["record_sha256"] = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(raw))
    with pytest.raises(
        ValueError, match="INVALID_REPLAY_RECEIPT: scientific identity mismatch"
    ):
        verify_replay_receipt(path)


def test_release_mismatch_preserves_verified_partial_outputs(
    pinned_project: tuple[Path, Path, str], tmp_path: Path
) -> None:
    _, project, commit = pinned_project
    root = tmp_path / "output"
    with pytest.raises(ValueError, match="EXPECTED_DIGEST_MISMATCH"):
        replay_corpus(
            project,
            commit,
            project,
            root,
            allow_network=False,
            expected_release_sha256="0" * 64,
        )
    receipts = list((root / "replay" / "receipts").glob("*.json"))
    assert len(receipts) == 1
    record = json.loads(receipts[0].read_text())
    assert record["status"] == "FAILED"
    assert "EXPECTED_DIGEST_MISMATCH" in record["error"]
    assert record["stage"] == "release_verification"
    assert verify_replay_receipt(receipts[0])["status"] == "FAILED"
    releases = list(
        (
            root
            / "replay"
            / "work"
            / commit
            / "corpora"
            / "devmind-sample-v0"
            / "releases"
        ).glob("*")
    )
    assert len(releases) == 1
    assert record["worker"]["artifacts"]["release_sha256"] == releases[0].name
    assert verify_release(releases[0])["release_id"] == releases[0].name


def test_replay_in_independent_roots_preserves_all_scientific_identities(
    pinned_project: tuple[Path, Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, project, commit = pinned_project
    unrelated_environment = tmp_path / "registered-model-runtime"
    unrelated_environment.mkdir()
    marker = unrelated_environment / "sentinel"
    marker.write_bytes(b"registered environment is not a replay target")
    checkout_environment = repository / ".venv"
    checkout_environment.mkdir()
    checkout_marker = checkout_environment / "sentinel"
    checkout_marker.write_bytes(b"checkout environment is not a replay target")
    monkeypatch.setenv("VIRTUAL_ENV", str(checkout_environment))
    monkeypatch.setenv("CONDA_PREFIX", str(unrelated_environment))
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(unrelated_environment))
    monkeypatch.setenv("SPARSELAB_RUNTIME_DIR", str(unrelated_environment))
    first = replay_corpus(
        project,
        commit,
        project,
        tmp_path / "first",
        allow_network=False,
        expected_release_sha256=None,
    )
    second = replay_corpus(
        project,
        commit,
        project,
        tmp_path / "second",
        allow_network=False,
        expected_release_sha256=first["receipt"]["scientific_identity"][
            "release_sha256"
        ],
        expected_build_sha256=first["receipt"]["scientific_identity"]["build_sha256"],
    )
    assert (
        first["receipt"]["scientific_identity"]
        == second["receipt"]["scientific_identity"]
    )
    assert first["path"] != second["path"]
    assert marker.read_bytes() == b"registered environment is not a replay target"
    assert set(unrelated_environment.iterdir()) == {marker}
    assert (
        checkout_marker.read_bytes() == b"checkout environment is not a replay target"
    )
    assert set(checkout_environment.iterdir()) == {checkout_marker}
    for item in (first, second):
        assert verify_replay_receipt(Path(item["receipt_path"]))["status"] == "MATCH"


def test_expected_build_mismatch_stops_before_release_and_retains_failure_receipt(
    pinned_project: tuple[Path, Path, str], tmp_path: Path
) -> None:
    _, project, commit = pinned_project
    work = tmp_path / "output"
    with pytest.raises(ValueError, match="EXPECTED_BUILD_MISMATCH"):
        replay_corpus(
            project,
            commit,
            project,
            work,
            allow_network=False,
            expected_release_sha256=None,
            expected_build_sha256="0" * 64,
        )
    receipts = list((work / "replay" / "receipts").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["status"] == "FAILED"
    assert "EXPECTED_BUILD_MISMATCH" in receipt["error"]
    assert receipt["stage"] == "build"
    assert receipt["worker"]["artifacts"]["build_sha256"] != "0" * 64
    assert verify_replay_receipt(receipts[0])["status"] == "FAILED"
    assert "EXPECTED_BUILD_MISMATCH" in receipt["operations"][-1]["stderr"]
    assert not list(
        (
            work
            / "replay"
            / "work"
            / commit
            / "corpora"
            / "devmind-sample-v0"
            / "releases"
        ).glob("*")
    )


def test_current_artifacts_cannot_masquerade_as_historical_replay(
    pinned_project, tmp_path: Path
) -> None:
    from sparselab.corpus.acquisition import acquire
    from sparselab.corpus.pipeline import build
    from sparselab.corpus.release import freeze

    _, project, commit = pinned_project
    root = tmp_path / "output"
    historical = replay_corpus(
        project,
        commit,
        project,
        root,
        allow_network=False,
        expected_release_sha256=None,
    )
    work = root / "replay" / "work" / commit
    recipe = load_project(project)
    lock = acquire(recipe, work)
    built = build(recipe, work, offline=True)
    frozen = freeze(built, work)
    assert built.name != historical["receipt"]["scientific_identity"]["build_sha256"]
    record = historical["receipt"]
    snapshots = [
        {"source_id": key, "sha256": entry["snapshot_sha256"]}
        for key, entry in lock["sources"].items()
        if entry["snapshot_sha256"]
    ]
    record["path"] = str(frozen)
    record["worker"].update(
        path=str(frozen),
        build_path=str(built),
        build_sha256=built.name,
        release_sha256=frozen.name,
        snapshots=snapshots,
    )
    record["scientific_identity"] = {
        "build_sha256": built.name,
        "release_sha256": frozen.name,
        "snapshots": snapshots,
    }
    record.pop("record_sha256")
    record["record_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    forged = tmp_path / "forged-replay.json"
    forged.write_text(json.dumps(record))
    with pytest.raises(
        ValueError, match="IMPLEMENTATION_PROVENANCE_MISMATCH: snapshot"
    ):
        verify_replay_receipt(forged)
