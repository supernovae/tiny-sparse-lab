"""Hand-authored tiny retained bytes: no acquisition, fitting or model execution."""

import hashlib
import json

import pytest
from test_corpus_acquisition import _fixture

from sparselab import verifier_authority as authority
from sparselab.corpus.acquisition import (
    ADAPTER_VERSION,
    _digest,
    declaration_sha256,
    verify_snapshot,
)
from sparselab.corpus.project import load_project, source_declaration_payload
from sparselab.verification_proofs import ProofStore


def retained_snapshot(tmp_path):
    project = load_project(
        _fixture(
            tmp_path,
            kind="git",
            revision="a" * 40,
            acquisition={"include": ["*.md"], "max_bytes": 1024},
        )
    )
    source = project.sources[0]
    payload = b"Offline retained fixture.\n"
    identity = {
        "declaration_sha256": declaration_sha256(source),
        "adapter": {"id": "git", "version": ADAPTER_VERSION, "module_sha256": "1" * 64},
        "files": [
            {
                "path": "data.md",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    sha = _digest(identity)
    root = tmp_path / "work"
    path = root / "corpora" / project.config.id / "snapshots" / source.id / sha
    (path / "files").mkdir(parents=True)
    (path / "files/data.md").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "source_id": source.id,
        "declaration": source_declaration_payload(source),
        **identity,
        "retrieval": {},
        "snapshot_sha256": sha,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return project, root, path, manifest


def test_snapshot_authenticated_warm_hit_and_cold_invalidation(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    _, root, path, manifest = retained_snapshot(tmp_path)
    store = ProofStore(root)

    def verify():
        return verify_snapshot(
            path, proof_store=store, verification_mode="verified_reuse"
        )

    assert verify() == manifest
    assert store.recorded == 1
    assert verify() == manifest
    assert store.hits == 1
    original = authority.Path.read_bytes

    def changed(file):
        raw = original(file)
        return (
            raw + b"\n# changed verifier authority\n"
            if file.name == "acquisition.py"
            else raw
        )

    monkeypatch.setattr(authority.Path, "read_bytes", changed)
    assert verify() == manifest
    assert store.hits == 1
    assert store.diagnostics()["reasons"]["verifier_authority_changed"] == 1
    (path / "files/data.md").write_bytes(b"Changed retained fixture.\n")
    with pytest.raises(ValueError, match="mismatch"):
        verify()
    assert store.hits == 1


def test_snapshot_transport_exclusion_requires_reaudit(monkeypatch):
    before = authority.verifier_authority("source_snapshot", 1)
    assert "workers.models" in before["modules"]
    assert "workers.transport" not in before["modules"]
    assert "recovery.implementation_replay" not in before["modules"]
    original = authority.Path.read_bytes

    def changed(path):
        raw = original(path)
        if path.name == "__init__.py" and path.parent.name == "workers":
            raw += b"\ndef verify_new_snapshot(value):\n    return call_worker(value)\n"
        return raw

    monkeypatch.setattr(authority.Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="exclusion needs review"):
        authority.verifier_authority("source_snapshot", 1)


@pytest.mark.parametrize("mutation", ["symlink", "foreign_proof"])
def test_snapshot_unsafe_proof_falls_back(tmp_path, monkeypatch, mutation):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    _, root, path, _ = retained_snapshot(tmp_path)
    store = ProofStore(root)
    verify_snapshot(path, proof_store=store, verification_mode="verified_reuse")
    if mutation == "symlink":
        body = path / "files/data.md"
        target = tmp_path / "body"
        body.rename(target)
        body.symlink_to(target)
        with pytest.raises(ValueError):
            verify_snapshot(path, proof_store=store, verification_mode="verified_reuse")
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "foreign"))
        foreign = ProofStore(root)
        verify_snapshot(path, proof_store=foreign, verification_mode="verified_reuse")
        assert foreign.hits == 0
