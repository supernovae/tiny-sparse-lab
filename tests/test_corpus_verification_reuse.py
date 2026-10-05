"""Trusted corpus receipts and operation-local closure seals preserve cold integrity."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.corpus import acquisition as acquisition_module
from sparselab.corpus import export as export_module
from sparselab.corpus import release as release_module
from sparselab.corpus.acquisition import acquire, verify_snapshot
from sparselab.corpus.export import export_release, verify_release_export
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import Project, load_project
from sparselab.corpus.release import _verification_operation, freeze, verify_release
from sparselab.data.tokenizer import train_tokenizer, verify_tokenizer_artifact
from sparselab.training.manifest import sha256_file
from sparselab.verification_proofs import ProofStore

PROJECT = Path(__file__).resolve().parents[1] / "corpora/devmind-sample-v0/corpus.yaml"
CONFIG = Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path, Path]:
    root = tmp_path_factory.mktemp("corpus-proof")
    project = load_project(PROJECT)
    acquire(project, root)
    release = freeze(build(project, root, offline=True), root)
    export = export_release(release, "lm", CONFIG, 300, root)
    tokenizer = train_tokenizer(load_tokenizer_config(export / "tokenizer.yaml"))
    return root, release, export, tokenizer


def _snapshot_paths(release: Path) -> list[Path]:
    manifest = json.loads((release / "manifest.json").read_text())
    return [
        release.parent.parent / "snapshots" / item["source_id"] / item["sha256"]
        for item in manifest["snapshots"]
    ]


def _consumer(
    root: Path, release: Path, export: Path, tokenizer: Path, mode: str
) -> dict:
    """Count SHA reads of source and release payloads through public consumers."""
    script = r"""
import json
import sys
from pathlib import Path
from sparselab.config.loading import load_config
from sparselab.corpus import acquisition, export as exports, release as releases
from sparselab.corpus.acquisition import verify_snapshot
from sparselab.corpus.export import verify_release_export
from sparselab.corpus.release import verify_release
from sparselab.data.tokenizer import verify_tokenizer_artifact
from sparselab.experiments import artifacts
from sparselab.training.manifest import sha256_file
from sparselab.verification_proofs import ProofStore
root, release, export, tokenizer = map(Path, sys.argv[1:5])
mode = sys.argv[5]
reads = []
for module in (acquisition, releases, exports, artifacts):
    original = module.sha256_file
    def counted(path, *args, _original=original, **kwargs):
        path = Path(path)
        if '/files/' in str(path) or path.is_relative_to(release / 'lm'):
            reads.append(str(path))
        return _original(path, *args, **kwargs)
    module.sha256_file = counted
store = ProofStore(root)
options = {'proof_store': store, 'verification_mode': mode}
manifest = verify_release(release, **options)
config = load_config(export / 'run.yaml')
dataset = config.dataset
binding = verify_release_export(dataset, **options)
provenance = verify_tokenizer_artifact(tokenizer, source=dataset.source,
    revision=dataset.revision, vocab_size=config.model.vocab_size, dataset=dataset, **options)
snapshots = [verify_snapshot(release.parent.parent / 'snapshots' / item['source_id'] /
    item['sha256'], **options)['snapshot_sha256'] for item in manifest['snapshots']]
print(json.dumps({'reads': reads, 'hits': store.hits, 'recorded': store.recorded,
    'release': manifest['release_id'], 'export': binding['release_id'],
    'tokenizer': provenance['sha256'], 'snapshots': snapshots}))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            *(str(p) for p in (root, release, export, tokenizer)),
            mode,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _snapshot_probe(root: Path, snapshots: list[Path]) -> dict:
    script = r"""
import json
import sys
from pathlib import Path
from sparselab.corpus import acquisition
from sparselab.corpus.acquisition import verify_snapshot
from sparselab.verification_proofs import ProofStore
count = 0
original = acquisition.sha256_file
def counted(path):
    global count
    if '/files/' in str(path):
        count += 1
    return original(path)
acquisition.sha256_file = counted
root = Path(sys.argv[1])
store = ProofStore(root)
identities = [verify_snapshot(Path(value), proof_store=store,
    verification_mode='verified_reuse')['snapshot_sha256'] for value in sys.argv[2:]]
print(json.dumps({'payload_reads': count, 'hits': store.hits, 'identities': identities}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(root), *(str(p) for p in snapshots)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _export_probe(root: Path, export: Path) -> dict:
    script = r"""
import json
import sys
from pathlib import Path
from sparselab.config.loading import load_config
from sparselab.corpus import acquisition, release
from sparselab.corpus.export import verify_release_export
from sparselab.verification_proofs import ProofStore
reads = []
for module in (acquisition, release):
    original = module.sha256_file
    def counted(path, _original=original):
        if '/files/' in str(path) or '/lm/' in str(path):
            reads.append(str(path))
        return _original(path)
    module.sha256_file = counted
store = ProofStore(Path(sys.argv[1]))
binding = verify_release_export(load_config(Path(sys.argv[2]) / 'run.yaml').dataset,
    proof_store=store, verification_mode='verified_reuse')
print(json.dumps({'reads': reads, 'hits': store.hits, 'release': binding['release_id']}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(root), str(export)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_distinct_process_receipts_and_independent_cold(
    corpus: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, release, export, tokenizer = corpus
    release_bytes = (release / "manifest.json").read_bytes()
    export_bytes = (export / "export.json").read_bytes()
    release_sha = sha256_file(release / "manifest.json")
    snapshot_shas = [sha256_file(p / "manifest.json") for p in _snapshot_paths(release)]
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    first = _consumer(root, release, export, tokenizer, "verified_reuse")
    assert first["reads"]
    assert first["recorded"] >= len(_snapshot_paths(release)) + 3
    warm = _consumer(root, release, export, tokenizer, "verified_reuse")
    assert warm["reads"] == []
    assert warm["hits"] >= len(_snapshot_paths(release)) + 3
    cold = _consumer(root, release, export, tokenizer, "cold")
    assert cold["reads"]
    identities = ("release", "export", "tokenizer", "snapshots")
    assert tuple(first[k] for k in identities) == tuple(warm[k] for k in identities)
    assert tuple(first[k] for k in identities) == tuple(cold[k] for k in identities)
    assert first["release"] == release.name
    assert (release / "manifest.json").read_bytes() == release_bytes
    assert (export / "export.json").read_bytes() == export_bytes
    assert sha256_file(release / "manifest.json") == release_sha
    assert [
        sha256_file(p / "manifest.json") for p in _snapshot_paths(release)
    ] == snapshot_shas


def test_operation_memo_rejects_payload_and_snapshot_change(
    corpus: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    _, original, _, _ = corpus
    release = (
        tmp_path / "corpora" / original.parent.parent.name / "releases" / original.name
    )
    release.parent.mkdir(parents=True)
    shutil.copytree(original, release)
    snapshots = release.parent.parent / "snapshots"
    shutil.copytree(original.parent.parent / "snapshots", snapshots)
    source = next(p for p in _snapshot_paths(release) if p.parent.name == "sample_code")
    payload = next(path for path in (source / "files").rglob("*") if path.is_file())
    for target in (release / "lm" / "train.jsonl", payload):
        before = target.stat()
        content = target.read_bytes()
        with _verification_operation():
            assert verify_release(release)["release_id"] == original.name
            target.write_bytes(bytes([content[0] ^ 1]) + content[1:])
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
            with pytest.raises(
                ValueError, match="changed within verification operation"
            ):
                verify_release(release)
        with pytest.raises(ValueError):
            verify_release(release)
        target.write_bytes(content)
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_added_source_preserves_snapshot_receipt(
    corpus: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, original, _, _ = corpus
    root = tmp_path / "work"
    release = (
        root / "corpora" / original.parent.parent.name / "releases" / original.name
    )
    release.parent.mkdir(parents=True)
    shutil.copytree(original, release)
    shutil.copytree(
        original.parent.parent / "snapshots", release.parent.parent / "snapshots"
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    store = ProofStore(root)
    old_snapshots = _snapshot_paths(release)
    for snapshot in old_snapshots:
        verify_snapshot(snapshot, proof_store=store, verification_mode="verified_reuse")
    project = load_project(PROJECT)
    specification = project.model_dump(mode="json")
    additional = next(
        source for source in specification["sources"] if source["id"] == "sample_docs"
    ).copy()
    additional["id"] = "additional_docs"
    additional["canonical_uri"] = "sparselab://corpora/devmind-sample-v0/extra-guide.md"
    additional["source_family"] = "additional_docs_train"
    specification["sources"].append(additional)
    specification["splits"]["assignments"]["additional_docs_train"] = "train"
    changed = Project.model_validate(specification)
    options = {"proof_store": store, "verification_mode": "verified_reuse"}
    rehashed_ancestors = []
    with monkeypatch.context() as counting:
        original_sha = acquisition_module.sha256_file

        def counted(path: Path) -> str:
            if any(path.is_relative_to(snapshot) for snapshot in old_snapshots):
                rehashed_ancestors.append(path)
            return original_sha(path)

        counting.setattr(acquisition_module, "sha256_file", counted)
        acquire(changed, root, **options)
        new_release = freeze(
            build(changed, root, offline=True, **options), root, **options
        )
    assert rehashed_ancestors == []
    assert new_release != release
    assert set(old_snapshots).issubset(_snapshot_paths(new_release))
    assert all(old.is_dir() for old in old_snapshots)
    probe = _snapshot_probe(root, old_snapshots)
    assert probe["payload_reads"] == 0
    assert probe["hits"] == len(old_snapshots)
    assert probe["identities"] == [old.name for old in old_snapshots]
    assert verify_release(new_release)["release_id"] == new_release.name


def test_signed_snapshot_receipt_falls_back_to_cold_on_changed_bytes(
    corpus: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, release, _, _ = corpus
    root = tmp_path / "work"
    snapshots = root / "corpora" / release.parent.parent.name / "snapshots"
    snapshots.parent.mkdir(parents=True)
    shutil.copytree(release.parent.parent / "snapshots", snapshots)
    snapshot = next(snapshots.glob("sample_code/*"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    store = ProofStore(root)
    assert (
        verify_snapshot(
            snapshot, proof_store=store, verification_mode="verified_reuse"
        )["snapshot_sha256"]
        == snapshot.name
    )
    assert _snapshot_probe(root, [snapshot])["payload_reads"] == 0
    payload = next(path for path in (snapshot / "files").rglob("*") if path.is_file())
    original = payload.read_bytes()
    before = payload.stat()
    payload.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    os.utime(payload, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError):
        verify_snapshot(snapshot, proof_store=store, verification_mode="verified_reuse")
    with pytest.raises(ValueError):
        verify_snapshot(snapshot)


def test_config_and_tokenizer_changes_leave_ancestor_receipts(
    corpus: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, release, export, tokenizer = corpus
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    store = ProofStore(root)
    options = {"proof_store": store, "verification_mode": "verified_reuse"}
    dataset = load_config(export / "run.yaml").dataset
    verify_tokenizer_artifact(
        tokenizer,
        source=dataset.source,
        revision=dataset.revision,
        vocab_size=300,
        dataset=dataset,
        **options,
    )
    old = store.hits
    alternate = tmp_path / "base.yaml"
    alternate.write_text(CONFIG.read_text().replace("seed: 7", "seed: 11"))
    hashed_payloads = []
    with monkeypatch.context() as counting:
        for module in (acquisition_module, release_module, export_module):
            original = module.sha256_file

            def counted(path: Path, *, _original=original) -> str:
                if "/files/" in str(path) or path.is_relative_to(release / "lm"):
                    hashed_payloads.append(path)
                return _original(path)

            counting.setattr(module, "sha256_file", counted)
        changed_export = export_release(release, "lm", alternate, 300, root, **options)
    assert hashed_payloads == []
    assert changed_export != export
    assert (
        verify_release_export(
            load_config(changed_export / "run.yaml").dataset, **options
        )["release_id"]
        == release.name
    )
    assert store.hits > old
    config_probe = _export_probe(root, changed_export)
    assert config_probe["reads"] == []
    assert config_probe["hits"] >= 1
    assert config_probe["release"] == release.name
    assert (export / "export.json").read_bytes() != (
        changed_export / "export.json"
    ).read_bytes()
    hashed_payloads.clear()
    with monkeypatch.context() as counting:
        for module in (acquisition_module, release_module, export_module):
            original = module.sha256_file

            def counted(path: Path, *, _original=original) -> str:
                if "/files/" in str(path) or path.is_relative_to(release / "lm"):
                    hashed_payloads.append(path)
                return _original(path)

            counting.setattr(module, "sha256_file", counted)
        tokenizer_export = export_release(release, "lm", CONFIG, 301, root, **options)
        changed_tokenizer = train_tokenizer(
            load_tokenizer_config(tokenizer_export / "tokenizer.yaml"), **options
        )
    assert hashed_payloads == []
    assert changed_tokenizer != tokenizer
    dataset = load_config(tokenizer_export / "run.yaml").dataset
    old = store.hits
    verify_tokenizer_artifact(
        changed_tokenizer,
        source=dataset.source,
        revision=dataset.revision,
        vocab_size=301,
        dataset=dataset,
        **options,
    )
    assert store.hits > old
    variant = _consumer(
        root, release, tokenizer_export, changed_tokenizer, "verified_reuse"
    )
    assert variant["reads"] == []
    assert variant["hits"] >= len(_snapshot_paths(release)) + 3
