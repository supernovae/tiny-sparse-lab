"""Task-scoped ancestry importer and read-only coordinator refusal boundaries."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sparselab.corpus.acquisition import acquire, verify_snapshot
from sparselab.corpus.project import load_project

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/research/devmind-pretrain-v4/replay_ancestry.py"
spec = importlib.util.spec_from_file_location("replay_ancestry_test", SCRIPT)
assert spec and spec.loader
ancestry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ancestry)


def test_publish_directory_is_atomic(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    source.mkdir()
    (source / "evidence").write_bytes(b"verified bytes")
    target = tmp_path / "published"
    ancestry._publish_directory(source, target)
    assert not source.exists()
    assert (target / "evidence").read_bytes() == b"verified bytes"


@pytest.mark.parametrize("existing", ["empty", "populated", "file", "symlink"])
def test_publish_directory_never_replaces_existing_target(
    tmp_path: Path, existing: str
) -> None:
    source = tmp_path / "staging"
    source.mkdir()
    (source / "evidence").write_bytes(b"verified bytes")
    target = tmp_path / "published"
    if existing in {"empty", "populated"}:
        target.mkdir()
        if existing == "populated":
            (target / "original").write_bytes(b"existing evidence")
    elif existing == "file":
        target.write_bytes(b"existing evidence")
    else:
        target.symlink_to(tmp_path / "missing", target_is_directory=True)
    before = target.lstat()
    with pytest.raises(FileExistsError):
        ancestry._publish_directory(source, target)
    assert target.lstat() == before
    assert (source / "evidence").read_bytes() == b"verified bytes"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True, capture_output=True
    ).stdout.strip()


def _projects(tmp_path: Path) -> tuple[Path, Path]:
    sample = ROOT / "corpora/devmind-sample-v0"
    old = tmp_path / "old" / "devmind-sample-v0"
    new = tmp_path / "new" / "devmind-sample-v0"
    shutil.copytree(sample, old)
    shutil.copytree(sample, new)
    added = (
        (new / "sources/docs.yaml")
        .read_text(encoding="utf-8")
        .replace("id: sample_docs", "id: sample_extra", 1)
    )
    (new / "sources/extra.yaml").write_text(added, encoding="utf-8")
    declaration = new / "corpus.yaml"
    declaration.write_text(
        declaration.read_text(encoding="utf-8").replace(
            "  - sources/docs.yaml\n",
            "  - sources/docs.yaml\n  - sources/extra.yaml\n",
            1,
        ),
        encoding="utf-8",
    )
    return old / "corpus.yaml", new / "corpus.yaml"


@pytest.fixture
def imported(tmp_path: Path):
    original, target = _projects(tmp_path)
    project = load_project(original)
    parent_work = tmp_path / "parent-work"
    lock = acquire(project, parent_work)
    closure = tmp_path / "archived-lock.json"
    active = parent_work / "corpora" / project.config.id / "acquisition.json"
    shutil.copyfile(active, closure)
    parent = {
        "historical_project": str(original),
        "corpus_work_root": str(parent_work),
        "acquisition_closure": {"path": str(closure)},
    }
    child = load_project(target)
    destination = tmp_path / "child-work"
    inherited = {source.id for source in project.sources}
    return parent, child, destination, inherited, lock, active, closure


def test_import_subset_is_byte_identical_and_does_not_transfer_lock(imported) -> None:
    parent, child, work, inherited, lock, active, closure = imported
    before = active.read_bytes()
    result = ancestry._import_verified_snapshots(parent, child, work, inherited)
    assert result == {key: lock["sources"][key]["snapshot_sha256"] for key in inherited}
    for source_id, digest in result.items():
        old = (
            Path(parent["corpus_work_root"])
            / "corpora"
            / child.config.id
            / "snapshots"
            / source_id
            / digest
        )
        new = work / "corpora" / child.config.id / "snapshots" / source_id / digest
        assert ancestry._inventory(old) == ancestry._inventory(new)
        assert verify_snapshot(old) == verify_snapshot(new)
    assert not (work / "corpora" / child.config.id / "acquisition.json").exists()
    assert active.read_bytes() == before == closure.read_bytes()
    assert ancestry._import_verified_snapshots(parent, child, work, inherited) == result


def test_import_rejects_mismatched_set_declaration_and_conflict(imported) -> None:
    parent, child, work, inherited, lock, _, _ = imported
    with pytest.raises(ValueError, match="exact unchanged parent subset"):
        ancestry._import_verified_snapshots(
            parent, child, work, inherited - {min(inherited)}
        )
    changed = child.model_copy(
        update={
            "sources": tuple(
                source.model_copy(update={"revision": "changed"})
                if source.id == min(inherited)
                else source
                for source in child.sources
            )
        }
    )
    with pytest.raises(ValueError, match="inherited source declaration changed"):
        ancestry._import_verified_snapshots(parent, changed, work, inherited)
    declaration = child.root / "sources/code.yaml"
    original = declaration.read_bytes()
    declaration.write_bytes(original + b"\n# no semantic change\n")
    try:
        with pytest.raises(ValueError, match="inherited source declaration changed"):
            ancestry._import_verified_snapshots(parent, child, work, inherited)
    finally:
        declaration.write_bytes(original)
    identity = lock["sources"][min(inherited)]["snapshot_sha256"]
    destination = (
        work / "corpora" / child.config.id / "snapshots" / min(inherited) / identity
    )
    destination.mkdir(parents=True)
    (destination / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting imported snapshot"):
        ancestry._import_verified_snapshots(parent, child, work, inherited)


def test_import_refuses_symlink_and_unverified_parent(imported, tmp_path: Path) -> None:
    parent, child, work, inherited, lock, _, _ = imported
    identity = lock["sources"][min(inherited)]["snapshot_sha256"]
    dest = work / "corpora" / child.config.id / "snapshots" / min(inherited)
    dest.mkdir(parents=True)
    (dest / identity).symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked path refused"):
        ancestry._import_verified_snapshots(parent, child, work, inherited)
    with pytest.raises(ValueError, match="missing or symlinked receipt"):
        ancestry.import_snapshots(tmp_path / "unverified.json", child, work, inherited)


def test_cli_preflight_refuses_network_and_incorrect_parent_without_side_effects(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Ancestry Fixture")
    recipe = repo / "corpora" / "devmind-sample-v0"
    shutil.copytree(ROOT / "corpora/devmind-sample-v0", recipe)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "pin local fixture declarations")
    commit = _git(repo, "rev-parse", "HEAD")
    state = tmp_path / "state"
    lineage = tmp_path / "lineage.json"
    stages = [
        {
            "id": name,
            "historical_producing_commit": commit,
            "project_path": "corpora/devmind-sample-v0/corpus.yaml",
            "source_corpus_generation": "devmind-sample-v0",
            "phase": "build" if name == "v4-intermediate" else "release",
            "expected_project_sha256": None,
            "expected_build_sha256": "0" * 64,
            "expected_release_sha256": None if name == "v4-intermediate" else "1" * 64,
            "inherited_snapshot_count": 0,
            "new_source_count": 4,
        }
        for name in ancestry.STAGES
    ]
    lineage.write_text(
        json.dumps({"lineage_version": 1, "state_root": str(state), "stages": stages}),
        encoding="utf-8",
    )
    result = ancestry.advance(
        state,
        "v2",
        allow_network=False,
        lineage_path=lineage,
        repository=repo,
        validate_only=True,
    )
    assert result["status"] == "LINEAGE_VALID"
    assert not state.exists()
    with pytest.raises(ValueError, match="network permission required"):
        ancestry.advance(
            state, "v2", allow_network=False, lineage_path=lineage, repository=repo
        )
    with pytest.raises(ValueError, match="missing verified parent stage"):
        ancestry.advance(
            state, "v3", allow_network=True, lineage_path=lineage, repository=repo
        )
    assert not state.exists()
    state.mkdir()
    (state / "unrelated-source.txt").write_text("not this task", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting pre-existing task root"):
        ancestry.advance(
            state, "v2", allow_network=True, lineage_path=lineage, repository=repo
        )
    assert {path.name for path in state.iterdir()} == {"unrelated-source.txt"}
