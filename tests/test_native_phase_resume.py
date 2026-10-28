"""Tiny native checkpoint continuation; no corpus-scale replay or qualification."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from sparselab.corpus.admission_inspection_cache import InspectionCache, capture
from sparselab.corpus.split_runs import SplitStages, write_rows
from sparselab.verification_proofs import ProofStore


@pytest.fixture
def owned(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    mask = os.umask(0o077)
    root = tmp_path / "work"
    root.mkdir()
    recipe = root / "recipe"
    recipe.mkdir()
    recipe.chmod(0o775)
    declaration = recipe / "source.yaml"
    declaration.write_text("source: retained\n")
    declaration.chmod(0o664)
    try:
        yield root, declaration
    finally:
        os.umask(mask)


def test_group_writable_declaration_is_bound_without_changing_permissions(owned):
    root, declaration = owned
    cache = InspectionCache(root)
    dependencies = capture([declaration])
    cache.publish("fixture", {"id": "fixed"}, {"result": 3}, dependencies, limit=65536)
    assert cache.load("fixture", {"id": "fixed"}, limit=65536)[0] == {"result": 3}
    assert declaration.stat().st_mode & 0o777 == 0o664
    before = declaration.stat()
    declaration.write_text("source: changed!\n")  # Same length, restored mtime.
    os.utime(declaration, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert cache.load("fixture", {"id": "fixed"}, limit=65536) is None


@pytest.mark.parametrize("unsafe", ["world_write", "shared_root"])
def test_mutable_declaration_requires_private_root_and_safe_permissions(owned, unsafe):
    root, declaration = owned
    if unsafe == "world_write":
        declaration.chmod(0o666)
    else:
        root.chmod(0o755)
    cache = InspectionCache(root)
    cache.publish(
        "fixture", {"id": "fixed"}, {"result": 3}, capture([declaration]), limit=65536
    )
    assert cache.load("fixture", {"id": "fixed"}, limit=65536) is None


def legacy_phase(root, declaration, monkeypatch):
    request = {"producer": "tiny-native-producer", "inputs": capture([declaration])}
    stages = SplitStages(root / "stages", request, work_root=root)
    # Prior code wrote the atomic native commit and cache, but declined to seal
    # the optional proof when a declaration was group writable.
    with monkeypatch.context() as old:
        old.setattr(ProofStore, "record", lambda *_: None)
        result = stages.phase(
            "render",
            lambda directory: {"files": [write_rows(directory / "rows", [[1], [2]])]},
        )
    return request, stages, result


def test_resume_uses_native_commit_when_optional_proof_was_never_published(
    owned, monkeypatch
):
    root, declaration = owned
    request, first, result = legacy_phase(root, declaration, monkeypatch)
    before = {p: p.read_bytes() for p in first.root.rglob("*") if p.is_file()}
    resumed = SplitStages(root / "stages", request, work_root=root)
    assert (
        resumed.phase("render", lambda _: pytest.fail("replayed completed native work"))
        == result
    )
    assert resumed.reused == ["render"]
    assert resumed.cache.proofs.diagnostics()["reasons"]["owned_native_completion"] == 1
    assert before == {p: p.read_bytes() for p in first.root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "damage", ["output", "completion", "missing", "untrusted", "proof"]
)
def test_native_commit_fallback_refuses_changed_incomplete_or_untrusted_state(
    owned, monkeypatch, damage
):
    root, declaration = owned
    request, first, result = legacy_phase(root, declaration, monkeypatch)
    completion = next(first.root.glob("completion-*.json"))
    if damage == "output":
        Path(result["files"][0]["path"]).write_bytes(b"different\n")
    elif damage == "completion":
        value = json.loads(completion.read_bytes())
        value["record"]["result"]["changed"] = True
        completion.write_text(json.dumps(value))
    elif damage == "missing":
        completion.unlink()
    elif damage == "untrusted":
        completion.chmod(0o666)
    else:
        cache_path = next(first.cache.directory.glob("split-phase-*.json"))
        value = json.loads(cache_path.read_bytes())
        binding = first.cache._binding(
            cache_path,
            cache_path.read_bytes(),
            {
                "request": first.identity,
                "completion": json.loads(completion.read_bytes())["sha256"],
            },
            capture(value["dependencies"]),
        )
        proof = first.cache.proofs._receipt(binding)
        proof.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        proof.write_text(
            json.dumps(
                {
                    "proof": {"untrusted": True},
                    "hmac": hashlib.sha256(b"bad").hexdigest(),
                }
            )
        )
        proof.chmod(0o600)
    if damage == "completion":
        with pytest.raises(ValueError, match="corrupt split completion chain"):
            SplitStages(root / "stages", request, work_root=root)
        return
    resumed = SplitStages(root / "stages", request, work_root=root)
    assert resumed.completed_phase("render") is None


def test_changed_input_gets_a_new_native_request(owned, monkeypatch):
    root, declaration = owned
    request, first, _ = legacy_phase(root, declaration, monkeypatch)
    declaration.write_text("source: changed!\n")
    changed = {**request, "inputs": capture([declaration])}
    resumed = SplitStages(root / "stages", changed, work_root=root)
    assert resumed.identity != first.identity
    assert resumed.completed_phase("render") is None
