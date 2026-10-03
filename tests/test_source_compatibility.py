"""Audited source mappings preserve provenance without trusting unsigned metadata."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sparselab.experiments import source_compatibility as compatibility
from sparselab.training.manifest import canonical_json, source_identity


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _commit(repo: Path) -> str:
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = tmp_path / "evidence"
    package = repo / "src" / "sparselab"
    shutil.copytree(
        Path(compatibility.__file__).parents[1],
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    _git(repo, "init", "-q")
    path = package / "cli" / "main.py"
    execution_bytes = path.read_bytes()
    path.write_bytes(b"# Historical operational fixture.\n" + execution_bytes)
    baseline_commit = _commit(repo)
    path.write_bytes(execution_bytes)
    execution_commit = _commit(repo)
    baseline = compatibility._committed_identity(repo, baseline_commit)
    execution = compatibility._committed_identity(repo, execution_commit)
    record = {
        "compatibility_version": 1,
        "baseline_commit": baseline_commit,
        "execution_commit": execution_commit,
        "baseline_source_identity": baseline,
        "execution_source_identity": execution,
        "operational_changes": compatibility._changes(baseline, execution),
        "authorization": "Explicit reviewed operational compatibility fixture",
    }
    evidence = repo / "compatibility.json"
    evidence.write_bytes(canonical_json(record) + b"\n")
    evidence_commit = _commit(repo)
    monkeypatch.setenv("SPARSELAB_SOURCE_COMPATIBILITY", str(evidence))
    monkeypatch.setenv(
        "SPARSELAB_SOURCE_COMPATIBILITY_SHA256",
        hashlib.sha256(evidence.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("SPARSELAB_SOURCE_COMPATIBILITY_COMMIT", evidence_commit)
    return repo, evidence, record, evidence_commit


def test_reviewed_mapping_keeps_actual_execution_source(binding):
    _, _, record, _ = binding
    actual = source_identity()
    historical = compatibility.locked_source_identity()
    assert historical["sha256"] == record["baseline_source_identity"]["sha256"]
    assert actual["sha256"] == record["execution_source_identity"]["sha256"]
    assert actual["sha256"] != historical["sha256"]
    assert compatibility.source_identities_compatible(
        historical["sha256"], actual["sha256"]
    )
    assert not compatibility.source_identities_compatible(
        actual["sha256"], historical["sha256"]
    )
    assert not compatibility.source_identities_compatible("0" * 64, actual["sha256"])
    historical["sha256"] = "0" * 64
    assert (
        compatibility.locked_source_identity()["sha256"]
        == record["baseline_source_identity"]["sha256"]
    )


def test_changed_evidence_rejected_even_after_verified_cache_hit(binding):
    _, evidence, _, _ = binding
    compatibility.locked_source_identity()
    evidence.write_text(evidence.read_text() + " ")
    with pytest.raises(ValueError, match="digest changed"):
        compatibility.locked_source_identity()


def test_reconstructed_uncommitted_mapping_is_not_authority(binding, monkeypatch):
    _, evidence, record, _ = binding
    record["authorization"] = "uncommitted replacement"
    evidence.write_bytes(canonical_json(record) + b"\n")
    monkeypatch.setenv(
        "SPARSELAB_SOURCE_COMPATIBILITY_SHA256",
        hashlib.sha256(evidence.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="trusted committed evidence"):
        compatibility.locked_source_identity()


def test_wrong_source_inventory_fails_closed(binding, monkeypatch):
    repo, evidence, record, _ = binding
    record["execution_source_identity"]["sha256"] = "0" * 64
    evidence.write_bytes(canonical_json(record) + b"\n")
    commit = _commit(repo)
    monkeypatch.setenv(
        "SPARSELAB_SOURCE_COMPATIBILITY_SHA256",
        hashlib.sha256(evidence.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("SPARSELAB_SOURCE_COMPATIBILITY_COMMIT", commit)
    with pytest.raises(ValueError, match="inventory changed"):
        compatibility.locked_source_identity()


def test_changed_declared_delta_cannot_hide_source_change(binding, monkeypatch):
    repo, evidence, record, _ = binding
    record["operational_changes"] = []
    evidence.write_bytes(canonical_json(record) + b"\n")
    commit = _commit(repo)
    monkeypatch.setenv(
        "SPARSELAB_SOURCE_COMPATIBILITY_SHA256",
        hashlib.sha256(evidence.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("SPARSELAB_SOURCE_COMPATIBILITY_COMMIT", commit)
    with pytest.raises(ValueError, match="change inventory differs"):
        compatibility.locked_source_identity()


def test_incomplete_pin_fails_closed(monkeypatch):
    monkeypatch.setenv("SPARSELAB_SOURCE_COMPATIBILITY", "/not-authority")
    monkeypatch.delenv("SPARSELAB_SOURCE_COMPATIBILITY_SHA256", raising=False)
    monkeypatch.delenv("SPARSELAB_SOURCE_COMPATIBILITY_COMMIT", raising=False)
    with pytest.raises(ValueError, match="path, SHA and trusted commit"):
        compatibility.locked_source_identity()


def test_symlink_replacement_cannot_reuse_verified_mapping(binding):
    _, evidence, _, _ = binding
    compatibility.locked_source_identity()
    copy = evidence.with_name("same-bytes.json")
    copy.write_bytes(evidence.read_bytes())
    evidence.unlink()
    evidence.symlink_to(copy)
    with pytest.raises(ValueError, match="symlinked"):
        compatibility.locked_source_identity()


def test_normal_source_without_compatibility_stays_current(monkeypatch):
    for name in compatibility._ENV:
        monkeypatch.delenv(name, raising=False)
    assert compatibility.locked_source_identity() == source_identity()
    assert not compatibility.source_identities_compatible(
        "0" * 64, str(source_identity()["sha256"])
    )


def test_scientific_code_change_is_not_operational_authority(binding, monkeypatch):
    repo, evidence, record, _ = binding
    path = repo / "src" / "sparselab" / "training" / "manifest.py"
    execution_bytes = path.read_bytes()
    path.write_bytes(
        b"# Not an allowed operational integration point.\n" + execution_bytes
    )
    baseline_commit = _commit(repo)
    path.write_bytes(execution_bytes)
    execution_commit = _commit(repo)
    baseline = compatibility._committed_identity(repo, baseline_commit)
    execution = compatibility._committed_identity(repo, execution_commit)
    record.update(
        {
            "baseline_commit": baseline_commit,
            "execution_commit": execution_commit,
            "baseline_source_identity": baseline,
            "execution_source_identity": execution,
            "operational_changes": compatibility._changes(baseline, execution),
        }
    )
    evidence.write_bytes(canonical_json(record) + b"\n")
    commit = _commit(repo)
    monkeypatch.setenv(
        "SPARSELAB_SOURCE_COMPATIBILITY_SHA256",
        hashlib.sha256(evidence.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("SPARSELAB_SOURCE_COMPATIBILITY_COMMIT", commit)
    with pytest.raises(ValueError, match="unsupported scientific code changes"):
        compatibility.locked_source_identity()


def test_authenticated_historical_cache_is_reused_without_repreparation(
    binding, tmp_path
):
    from sparselab.config.loading import load_config
    from sparselab.data.packing import prepare_data
    from sparselab.data.tokenizer import load_tokenizer

    repo, _, record, _ = binding
    historical_source = tmp_path / "historical-source"
    shutil.copytree(repo / "src", historical_source)
    cli = historical_source / "sparselab" / "cli" / "main.py"
    cli.write_bytes(b"# Historical operational fixture.\n" + cli.read_bytes())
    config_path = tmp_path / "run.json"
    env = dict(os.environ)
    for key in compatibility._ENV:
        env.pop(key, None)
    tests_root = Path(__file__).parent
    env["PYTHONPATH"] = os.pathsep.join([str(historical_source), str(tests_root)])
    script = (
        "from pathlib import Path; from test_training import config; "
        "from sparselab.data.packing import prepare_data; "
        "from sparselab.data.tokenizer import load_tokenizer; "
        "from sparselab.training.manifest import source_identity; "
        f"assert source_identity()['sha256'] == {record['baseline_source_identity']['sha256']!r}; "
        f"cfg=config(Path({str(tmp_path / 'data')!r})); "
        "data=prepare_data(cfg,load_tokenizer(cfg.tokenizer.path)); "
        f"Path({str(config_path)!r}).write_text(cfg.model_dump_json()); "
        "print(data.root)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    original_root = Path(result.stdout.strip())
    original_files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in original_root.iterdir()
        if path.is_file()
    }
    config = load_config(config_path)
    reused = prepare_data(config, load_tokenizer(config.tokenizer.path))
    assert reused.root == original_root
    assert (
        reused.manifest["source_identity_sha256"]
        == record["baseline_source_identity"]["sha256"]
    )
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in original_root.iterdir()
        if path.is_file()
    } == original_files
    array = original_root / "train.npy"
    data = bytearray(array.read_bytes())
    data[-1] ^= 1
    array.write_bytes(data)
    with pytest.raises(ValueError):
        prepare_data(config, load_tokenizer(config.tokenizer.path))
