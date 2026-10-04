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
    pinned_project: tuple[Path, Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, project, commit = pinned_project
    rename = Path.rename

    def rename_writable_directory(source: Path, target: Path) -> Path:
        # Enforce macOS's directory-rename permission rule on every test host.
        assert source.stat().st_mode & 0o200
        return rename(source, target)

    monkeypatch.setattr(Path, "rename", rename_writable_directory)
    first = materialize_source(project, commit, tmp_path / "work")
    second = materialize_source(project, commit, tmp_path / "work")
    assert first["source_root"] == second["source_root"]
    source = Path(first["source_root"])
    assert source.stat().st_mode & 0o777 == 0o555
    assert all(path.stat().st_mode & 0o222 == 0 for path in source.rglob("*"))
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


@pytest.fixture
def ancestry_chain(tmp_path: Path):
    """Four real historical recipes, with mixed original immutable adapters."""
    import yaml

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init")
    _git(upstream, "config", "user.email", "fixture@example.invalid")
    _git(upstream, "config", "user.name", "Historical Fixture")
    ids = [
        "foundation_a",
        "foundation_b",
        "foundation_c",
        "v4_iac_cmake_build",
        "v4_runtime_metro_js",
    ]
    for source_id in ids:
        folder = upstream / source_id
        folder.mkdir()
        (folder / "guide.md").write_text(
            f"# {source_id}\nA distinct authored guide about {source_id} invariants.\n"
        )
        (folder / "extra.md").write_text(
            f"# Additional {source_id}\nAdditional contracts for {source_id}.\n"
        )
    (upstream / "LICENSE").write_text("SPDX-License-Identifier: MIT\n")
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-m", "immutable local source")
    revision = _git(upstream, "rev-parse", "HEAD")
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
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copy2(ROOT / name, repository / name)
    adapter = repository / "src/sparselab/corpus/acquisition.py"
    current_adapter = adapter.read_bytes()
    stages = []

    def declaration(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False))

    for index, (generation, count) in enumerate(
        (("chain-v2", 3), ("chain-v3", 4), ("chain-v4", 5), ("chain-v4", 5))
    ):
        recipe = repository / "corpora" / generation / "corpus.yaml"
        selected = ids[:count]
        declaration(
            recipe,
            {
                "schema_version": 1,
                "id": generation,
                "sources": [f"sources/{item}.yaml" for item in selected],
                "transforms": ["transforms/lm.yaml"],
                "splits": "splits.yaml",
                "release": "release.yaml",
            },
        )
        declaration(
            recipe.parent / "splits.yaml",
            {
                "schema_version": 1,
                "unit": "source_repository",
                "assignments": {
                    item: (
                        "validation"
                        if item == "foundation_b"
                        else "test"
                        if item == "foundation_c"
                        else "train"
                    )
                    for item in selected
                },
            },
        )
        declaration(
            recipe.parent / "transforms/lm.yaml",
            {
                "id": generation.replace("-", "_") + "_lm",
                "version": "1",
                "kind": "lm_text",
                "parameters": {},
                "inputs": selected,
            },
        )
        declaration(
            recipe.parent / "release.yaml",
            {
                "schema_version": 3,
                "mixture": {"developer_systems": 1.0},
                "publication_mode": "metadata_reconstruction_only",
                "training_use_policy": "allowed_unless_explicitly_prohibited",
                "lm": {"selected": True, "training_splits": ["train", "validation"]},
                "chat": {"selected": False, "training_splits": []},
            },
        )
        for item in selected:
            declaration(
                recipe.parent / f"sources/{item}.yaml",
                {
                    "schema_version": 3,
                    "id": item,
                    "kind": "git",
                    "canonical_uri": str(upstream),
                    "revision": revision,
                    "license": "MIT",
                    "license_url": (upstream / "LICENSE").as_uri(),
                    "explicit_training_restriction": "none_found",
                    "rights": {
                        "training_eligibility": "eligible_with_obligations",
                        "redistribution_mode": "metadata_reconstruction_only",
                        "spdx_expression": "MIT",
                        "license_references": [(upstream / "LICENSE").as_uri()],
                        "notices": ["Retain source notice."],
                    },
                    "domains": ["developer_systems"],
                    "document_kinds": ["markdown"],
                    "source_family": item,
                    "acquisition": {
                        "include": [
                            f"{item}/*.md"
                            if index == 3 and item in ids[3:]
                            else f"{item}/guide.md"
                        ],
                        "max_bytes": 4096,
                    },
                },
            )
        adapter.write_bytes(
            current_adapter.replace(b"import hashlib\n", b"import  hashlib\n", 1)
            if index == 0
            else current_adapter
        )
        _git(
            repository,
            "add",
            "src",
            "corpora",
            "pyproject.toml",
            "uv.lock",
            "README.md",
        )
        _git(repository, "commit", "-m", f"historical ancestry stage {index}")
        stages.append((recipe, _git(repository, "rev-parse", "HEAD")))
    yield repository, stages
    for environment_root in tmp_path.glob("*/replay/env"):
        shutil.rmtree(environment_root)


def _historical_oracle(source: Path, recipe: Path, work: Path) -> dict:
    """Compute gates with genuine exported producers, never injected identities."""
    code = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + '/src')
from sparselab.corpus.project import load_project
from sparselab.corpus.acquisition import acquire
from sparselab.corpus.pipeline import build
from sparselab.corpus.release import freeze
project = load_project(Path(sys.argv[2]))
work = Path(sys.argv[3])
acquire(project, work)
built = build(project, work, offline=True)
frozen = freeze(built, work)
print(json.dumps({'build': built.name, 'release': frozen.name}))
"""
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--no-sync",
            "--python",
            sys.executable,
            "python",
            "-I",
            "-c",
            code,
            str(source),
            str(recipe),
            str(work),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_authentic_ancestry_chain_and_archived_intermediate_lock(
    ancestry_chain, tmp_path: Path
) -> None:
    """Mixed adapters survive imports; final active lock cannot erase a parent."""
    repository, stages = ancestry_chain
    root = tmp_path / "ancestry"
    parent = None
    parent_lock = None
    successes = []
    for index, (recipe, commit) in enumerate(stages):
        exported = materialize_source(recipe, commit, root)
        source = Path(exported["source_root"])
        historical_recipe = source / recipe.relative_to(repository)
        project = load_project(historical_recipe)
        work = root / "v4" if index >= 2 else root / "replay" / "work" / commit
        changed = (
            {
                item: parent_lock["sources"][item]["snapshot_sha256"]
                for item in ("v4_iac_cmake_build", "v4_runtime_metro_js")
            }
            if index == 3
            else {}
        )
        inherited = {
            key: entry["snapshot_sha256"]
            for key, entry in (parent_lock or {"sources": {}})["sources"].items()
            if key not in changed
        }
        oracle_work = tmp_path / f"oracle-{index}"
        for key in inherited:
            original = Path(parent_lock["sources"][key]["snapshot_path"])
            for target_work in (work, oracle_work):
                target = (
                    target_work
                    / "corpora"
                    / project.config.id
                    / "snapshots"
                    / key
                    / original.name
                )
                if not target.exists():
                    shutil.copytree(original, target)
        oracle = _historical_oracle(source, historical_recipe, oracle_work)
        result = replay_corpus(
            recipe,
            commit,
            recipe,
            root,
            allow_network=True,  # Only the fixture's local Git repository.
            use_historical_project=True,
            corpus_work_root=work,
            phase="build" if index == 2 else "release",
            expected_build_sha256=oracle["build"],
            expected_release_sha256=None if index == 2 else oracle["release"],
            parent_receipt=Path(parent["receipt_path"]) if parent else None,
            inherited_source_ids=tuple(inherited),
            expected_unchanged_snapshots=inherited,
            expected_changed_snapshots=changed,
        )
        verified = verify_replay_receipt(Path(result["receipt_path"]))
        assert verified["status"] == "MATCH"
        assert (
            verify_build(Path(verified["worker"]["build_path"]))["build_id"]
            == oracle["build"]
        )
        observed = verify_acquisition(project, work)
        actual = {
            key: entry["snapshot_sha256"] for key, entry in observed["sources"].items()
        }
        assert {key: actual[key] for key in inherited} == inherited
        assert all(actual[key] != old for key, old in changed.items())
        if index == 2:
            assert verified["phase"] == "build"
            assert verified["release"] is None
            assert "release_sha256" not in verified["worker"]
            assert not (work / "corpora" / project.config.id / "releases").exists()
        else:
            assert (
                verify_release(Path(result["path"]))["release_id"] == oracle["release"]
            )
        if index == 1:
            inherited_sha = verify_snapshot(
                Path(observed["sources"]["foundation_a"]["snapshot_path"])
            )["adapter"]["module_sha256"]
            new_sha = verify_snapshot(
                Path(observed["sources"]["v4_iac_cmake_build"]["snapshot_path"])
            )["adapter"]["module_sha256"]
            assert inherited_sha != new_sha
        parent, parent_lock = result, observed
        successes.append(result)
    intermediate = Path(successes[2]["receipt_path"])
    assert verify_replay_receipt(intermediate)["phase"] == "build"
    assert (
        verify_replay_receipt(Path(successes[3]["receipt_path"]))["status"] == "MATCH"
    )
    record = json.loads(intermediate.read_text())
    archived = Path(record["acquisition_closure"]["path"])
    original_bytes = archived.read_bytes()
    archived.chmod(0o644)
    archived.write_bytes(original_bytes + b" ")
    with pytest.raises(ValueError):
        verify_replay_receipt(Path(successes[3]["receipt_path"]))
    archived.write_bytes(original_bytes)
    assert verify_replay_receipt(intermediate)["phase"] == "build"
    last_record = verify_replay_receipt(Path(successes[3]["receipt_path"]))
    log = Path(last_record["operations"][-1]["stderr_path"])
    log_bytes = log.read_bytes()
    log.write_bytes(log_bytes + b"tampered ancestry log\n")
    with pytest.raises(ValueError):
        verify_replay_receipt(Path(successes[3]["receipt_path"]))
    log.write_bytes(log_bytes)
    snapshot = Path(parent_lock["sources"]["foundation_a"]["snapshot_path"])
    file = next((snapshot / "files").rglob("guide.md"))
    original_text = file.read_bytes()
    file.chmod(0o644)
    file.write_bytes(original_text + b"tampered inherited source\n")
    with pytest.raises(ValueError):
        verify_replay_receipt(Path(successes[3]["receipt_path"]))
    file.write_bytes(original_text)


@pytest.mark.parametrize(
    "options",
    [
        {"expected_project_sha256": "0" * 64},
        {"inherited_source_ids": ("unknown",)},
        {"expected_unchanged_snapshots": {"unknown": "0" * 64}},
        {"expected_changed_snapshots": {"unknown": "0" * 64}},
        {"phase": "release", "expected_release_sha256": None},
    ],
)
def test_ancestry_refuses_invalid_request_before_dependencies(
    pinned_project, tmp_path: Path, options
) -> None:
    _, project, commit = pinned_project
    root = tmp_path / "refused"
    request = {
        "use_historical_project": True,
        "phase": "build",
        "expected_release_sha256": None,
        "expected_build_sha256": "0" * 64,
        **options,
    }
    with pytest.raises(ValueError):
        replay_corpus(project, commit, project, root, allow_network=False, **request)
    assert not (root / "replay" / "env").exists()
    assert not (root / "replay" / "work").exists()


@pytest.mark.parametrize(
    "unsafe", ["relative", "checkout", "export", "environment", "symlink"]
)
def test_ancestry_refuses_colliding_corpus_roots(
    pinned_project, tmp_path: Path, unsafe: str
) -> None:
    repository, project, commit = pinned_project
    root = tmp_path / "refused"
    if unsafe == "relative":
        work = Path("relative-corpus-root")
    elif unsafe == "checkout":
        work = repository / "corpus-work"
    elif unsafe == "export":
        work = root / "replay" / "source" / commit
    elif unsafe == "environment":
        work = root / "replay" / "env" / commit
    else:
        work = tmp_path / "linked"
        work.symlink_to(repository, target_is_directory=True)
    with pytest.raises(ValueError):
        replay_corpus(
            project,
            commit,
            project,
            root,
            allow_network=False,
            use_historical_project=True,
            corpus_work_root=work,
            phase="build",
            expected_release_sha256=None,
            expected_build_sha256="0" * 64,
        )
    assert not (root / "replay" / "env").exists()


def test_ancestry_build_mismatch_retains_archived_lock_without_freeze(
    pinned_project, tmp_path: Path
) -> None:
    _, project, commit = pinned_project
    root = tmp_path / "blocked"
    with pytest.raises(ValueError, match="EXPECTED_BUILD_MISMATCH"):
        replay_corpus(
            project,
            commit,
            project,
            root,
            allow_network=False,
            use_historical_project=True,
            phase="build",
            expected_build_sha256="0" * 64,
            expected_release_sha256=None,
        )
    receipt = next((root / "replay" / "receipts").glob("*.json"))
    record = verify_replay_receipt(receipt)
    assert record["status"] == "FAILED"
    archived = Path(record["acquisition_closure"]["path"])
    corpus = load_project(
        root / "replay" / "source" / commit / project.relative_to(project.parents[2])
    )
    work = root / "replay" / "work" / commit
    assert (
        verify_acquisition(corpus, work, lock_path=archived)["project_id"]
        == corpus.config.id
    )
    assert not (work / "corpora" / corpus.config.id / "releases").exists()
