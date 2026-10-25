"""Native producer completion, warm recovery and authenticated invalidation."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_corpus_admission_inspection import _fixture

from sparselab.corpus import split_inventory
from sparselab.corpus.project import load_project
from sparselab.corpus.split_inventory_cache import InventoryCache, verified_inventory


@pytest.fixture
def completed_inventory(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    previous = os.umask(0o077)
    try:
        root = tmp_path / "work"
        root.mkdir()
        recipe, *_ = _fixture(root)
        project = load_project(recipe / "acquire.yaml")
        output = root / "inventory.jsonl"
        report = split_inventory.write_split_inventory(project, root, output)
    finally:
        os.umask(previous)
    return root, project, output, report


def _no_replay(*args):
    raise AssertionError("completed native inventory was regenerated")


def test_native_producer_completion_reuses_without_source_pass(completed_inventory):
    root, project, path, report = completed_inventory
    before = path.stat().st_mtime_ns
    actual, reused = verified_inventory(
        project, root, path, report["sha256"], _no_replay
    )
    assert reused and actual == report
    assert path.stat().st_mtime_ns == before
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from sparselab.corpus.project import load_project
from sparselab.corpus.split_inventory_cache import verified_inventory
def no_replay(*args):raise AssertionError('source replay in fresh process')
project=load_project(Path(sys.argv[1]))
report,reused=verified_inventory(project,Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],no_replay)
assert reused and report['sha256']==sys.argv[4]
print('fresh process reused native completion')
""",
            str(project.root / "acquire.yaml"),
            str(root),
            str(path),
            report["sha256"],
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "fresh process reused" in child.stdout


def test_cold_verification_records_original_path_for_recovery(completed_inventory):
    root, project, path, report = completed_inventory
    old_scratch = root / "corpora" / project.config.id / "split-inventory-check"
    old_scratch.mkdir(mode=0o775)
    before_mode = old_scratch.stat().st_mode
    cache = InventoryCache(root)
    cache._path("inventory", cache.request(project, path)).unlink()
    actual, reused = verified_inventory(
        project, root, path, report["sha256"], split_inventory.write_split_inventory
    )
    assert not reused and actual == report
    assert verified_inventory(project, root, path, report["sha256"], _no_replay) == (
        report,
        True,
    )
    assert old_scratch.stat().st_mode == before_mode


def test_changed_scientific_configuration_invalidates_completion(completed_inventory):
    root, project, path, report = completed_inventory
    changed = SimpleNamespace(
        config=project.config,
        sources=project.sources,
        splits=project.splits,
        release=SimpleNamespace(normalizer="a-different-normalizer"),
    )
    assert InventoryCache(root).completed(changed, path) is None
    with pytest.raises(AssertionError, match="regenerated"):
        verified_inventory(changed, root, path, report["sha256"], _no_replay)


def test_same_size_output_mutation_refuses_completion(completed_inventory):
    root, project, path, report = completed_inventory
    original = path.stat()
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(ValueError, match="reviewed inventory digest"):
        verified_inventory(project, root, path, report["sha256"], _no_replay)


def test_same_size_source_mutation_invalidates_completion(completed_inventory):
    root, project, path, report = completed_inventory
    completed = InventoryCache(root).completed(project, path)
    assert completed is not None
    _, dependencies, _ = completed
    snapshot = next(Path(p) for p in dependencies if Path(p).is_dir())
    source = next(p for p in (snapshot / "files").rglob("*") if p.is_file())
    old = source.stat()
    raw = source.read_bytes()
    assert raw
    source.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    os.utime(source, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert InventoryCache(root).completed(project, path) is None
    with pytest.raises(AssertionError, match="regenerated"):
        verified_inventory(project, root, path, report["sha256"], _no_replay)


def test_changed_producer_authority_invalidates_completion(completed_inventory):
    root, project, path, _ = completed_inventory
    cache = InventoryCache(root)
    cache.authority["inventory"]["split_inventory.py"] = "0" * 64
    assert cache.completed(project, path) is None


@pytest.mark.parametrize("raw", ['{"dependencies": [], "payload": {}}', '{"payload":'])
def test_corrupt_or_unsigned_cache_never_becomes_completion(completed_inventory, raw):
    root, project, path, _ = completed_inventory
    cache = InventoryCache(root)
    metadata = cache._path("inventory", cache.request(project, path))
    metadata.write_text(raw)
    assert InventoryCache(root).completed(project, path) is None


def test_unavailable_authority_preserves_valid_cold_processing(
    completed_inventory, monkeypatch
):
    from sparselab.corpus import split_inventory_cache

    root, project, path, report = completed_inventory

    def unavailable(*args, **kwargs):
        raise OSError("fixture cannot read optional producer authority")

    monkeypatch.setattr(split_inventory_cache, "sha256_file", unavailable)
    assert InventoryCache(root).proofs is None
    # Only the optional authority reader is unavailable; the native file hash
    # remains required and independent of cache availability.
    monkeypatch.setattr(
        split_inventory_cache, "sha256_file", split_inventory.sha256_file
    )
    monkeypatch.setattr(InventoryCache, "completed", lambda *args, **kwargs: None)
    actual, reused = verified_inventory(
        project, root, path, report["sha256"], split_inventory.write_split_inventory
    )
    assert not reused and actual == report


def test_interrupted_producer_has_no_completed_output(completed_inventory, monkeypatch):
    root, project, _, _ = completed_inventory
    output = root / "interrupted.jsonl"

    def interrupt(*args, **kwargs):
        raise RuntimeError("fixture producer interrupted")

    monkeypatch.setattr(split_inventory, "_file_documents", interrupt)
    with pytest.raises(RuntimeError, match="producer interrupted"):
        split_inventory.write_split_inventory(project, root, output)
    assert not output.exists()
    assert InventoryCache(root).completed(project, output) is None


def test_unavailable_dependency_metadata_keeps_native_processing_without_proof(
    completed_inventory, monkeypatch
):
    root, project, _, report = completed_inventory
    output = root / "uncached-inventory.jsonl"

    def unavailable(*args):
        raise PermissionError("optional metadata capture unavailable")

    monkeypatch.setattr(
        "sparselab.corpus.split_inventory_cache.inventory_inputs", unavailable
    )
    result = split_inventory.write_split_inventory(project, root, output)
    assert result["sha256"] == report["sha256"]
    assert InventoryCache(root).completed(project, output) is None
