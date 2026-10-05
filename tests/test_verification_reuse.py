from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys

import pytest
from test_training import config as training_config

from sparselab.data.packing import load_prepared_data, prepare_data
from sparselab.data.tokenizer import load_tokenizer
from sparselab.data.verification import verify_file
from sparselab.experiments.artifacts import verify_artifact
from sparselab.experiments.plan import Artifact
from sparselab.training import manifest as manifests
from sparselab.verification_proofs import ProofStore, file_binding
from sparselab.verifier_authority import verifier_authority


@pytest.fixture
def trusted_prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))
    root = tmp_path / "work"
    root.mkdir(mode=0o700)
    config = training_config(root)
    prepared = prepare_data(config, load_tokenizer(config.tokenizer.path))
    return root, config, prepared


def _load(prepared, store, mode="verified_reuse"):
    return load_prepared_data(
        prepared.root, byte_enabled=False, proof_store=store, verification_mode=mode
    )


def test_separate_process_reuses_arrays_without_payload_sha(trusted_prepared):
    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    _load(prepared, store)
    assert store.recorded == 2
    code = """
import json, sys
from pathlib import Path
from sparselab.training import manifest
from sparselab.data.packing import load_prepared_data
from sparselab.verification_proofs import ProofStore
reads=[]
original=manifest.sha256_file
def counted(path, *a, **kw):
    if path.suffix == '.npy': reads.append(path.name)
    return original(path,*a,**kw)
manifest.sha256_file=counted
store=ProofStore(Path(sys.argv[1]))
data=load_prepared_data(Path(sys.argv[2]),byte_enabled=False,proof_store=store,verification_mode=sys.argv[3])
print(json.dumps({'reads':reads,'hits':store.hits,'sha':data.manifest['manifest_sha256']}))
"""
    result = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                code,
                str(root),
                str(prepared.root),
                "verified_reuse",
            ],
            text=True,
        )
    )
    assert result == {
        "reads": [],
        "hits": 2,
        "sha": prepared.manifest["manifest_sha256"],
    }
    cold = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code, str(root), str(prepared.root), "cold"],
            text=True,
        )
    )
    assert sorted(cold["reads"]) == ["train.npy", "validation.npy"]
    assert cold["hits"] == 0


@pytest.mark.parametrize("mutation", ["bytes", "inode", "truncate", "manifest"])
def test_changed_array_or_manifest_fails(trusted_prepared, mutation):
    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    _load(prepared, store)
    path = prepared.root / "train.npy"
    before = path.stat()
    raw = path.read_bytes()
    if mutation == "manifest":
        manifest_path = prepared.root / "manifest.json"
        document = json.loads(manifest_path.read_text())
        document["train"]["tokens"] += 1
        manifest_path.write_text(json.dumps(document))
    elif mutation == "inode":
        new = path.with_name("replacement")
        new.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        new.replace(path)
    elif mutation == "truncate":
        path.write_bytes(raw[:-1])
    else:
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError):
        _load(prepared, store)
    with pytest.raises(ValueError):
        _load(prepared, store, "cold")


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "forged",
        "stale",
        "authentic_stale_implementation",
        "missing_key",
        "foreign_key",
        "permissions",
    ],
)
def test_invalid_receipt_falls_back_to_cold(trusted_prepared, invalid, monkeypatch):
    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    _load(prepared, store)
    if invalid == "authentic_stale_implementation":
        # A correctly signed schema-1 receipt cannot be promoted by this verifier.
        for receipt in store.directory.glob("*.json"):
            raw = json.loads(receipt.read_text())
            raw["proof"].pop("verifier_authority")
            raw["proof"]["verification_schema"] = 1
            raw["proof"]["implementation_sha256"] = "0" * 64
            raw["hmac"] = hmac.new(
                store._key(create=False),
                manifests.canonical_json(raw["proof"]),
                hashlib.sha256,
            ).hexdigest()
            receipt.write_text(json.dumps(raw))
    receipts = list(store.directory.glob("*.json"))
    if invalid == "missing":
        for path in receipts:
            path.unlink()
    elif invalid == "authentic_stale_implementation":
        store = ProofStore(root)
    elif invalid in {"forged", "stale"}:
        for path in receipts:
            payload = json.loads(path.read_text())
            if invalid == "forged":
                payload.pop("hmac")
            else:
                payload["proof"]["verification_schema"] += 1
            path.write_text(json.dumps(payload))
    elif invalid == "missing_key":
        store.key_path.unlink()
    elif invalid == "foreign_key":
        store.key_path.write_bytes(b"x" * 32)
    else:
        os.chmod(root, 0o777)
    reads = []
    original = manifests.sha256_file

    def counted(path, *args, **kwargs):
        if path.suffix == ".npy":
            reads.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(manifests, "sha256_file", counted)
    _load(prepared, store)
    assert sorted(reads) == ["train.npy", "validation.npy"]


def test_relocation_and_links_are_not_reuse(trusted_prepared):
    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    _load(prepared, store)
    moved = root / "relocated"
    shutil.copytree(prepared.root, moved)
    data = load_prepared_data(
        moved, byte_enabled=False, proof_store=store, verification_mode="verified_reuse"
    )
    assert all(p.cold_verified for p in data.receipt.proofs.values())
    linked = root / "linked"
    linked.symlink_to(prepared.root, target_is_directory=True)
    assert not store.trusted(linked)
    hard = root / "hard.npy"
    os.link(prepared.root / "train.npy", hard)
    assert not store.trusted(hard)


def test_only_exact_cold_process_owned_evidence_can_record(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    path = tmp_path / "asset"
    path.write_bytes(b"original")
    store = ProofStore(tmp_path)
    digest = manifests.sha256_file(path)
    binding = file_binding(path, digest)
    with pytest.raises(TypeError, match="process-owned"):
        store.record(binding, {"sha256": digest})
    proof = verify_file(path, expected_sha256=digest)
    store.record(binding, proof)
    assert store.lookup(binding)
    warm = verify_file(
        path,
        expected_sha256=digest,
        proof_store=store,
        verification_mode="verified_reuse",
    )
    with pytest.raises(TypeError, match="process-owned"):
        store.record(binding, warm)
    assert oct(store.directory.stat().st_mode & 0o777) == "0o700"
    assert oct(store.key_path.stat().st_mode & 0o777) == "0o600"
    with pytest.raises(ValueError, match="binding differs"):
        verify_file(
            path,
            expected_sha256="a" * 64,
            proof_store=store,
            verification_mode="verified_reuse",
            binding=binding,
        )


def test_typed_dependency_changes_miss(trusted_prepared):
    root, config, _ = trusted_prepared
    path = config.tokenizer.path
    artifact = Artifact(
        kind="tokenizer",
        version=1,
        producer="fixture",
        identifier=path.parent.name,
        sha256=manifests.sha256_file(path),
        path=str(path),
    )
    store = ProofStore(root)
    before = verify_artifact(
        artifact,
        root / "plan.yaml",
        proof_store=store,
        verification_mode="verified_reuse",
    )
    assert (
        verify_artifact(
            artifact,
            root / "plan.yaml",
            proof_store=store,
            verification_mode="verified_reuse",
        )
        == before
    )
    manifest = path.with_name("tokenizer_manifest.json")
    payload = json.loads(manifest.read_text())
    payload["vocab_size"] += 1
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        verify_artifact(
            artifact,
            root / "plan.yaml",
            proof_store=store,
            verification_mode="verified_reuse",
        )


def test_prepared_worker_choices_preserve_exact_manifest_and_sha(
    trusted_prepared, monkeypatch
):
    from dataclasses import replace

    from sparselab.data import packing
    from sparselab.host_capacity import sha_work_plan

    _, _, prepared = trusted_prepared
    manifest_bytes = (prepared.root / "manifest.json").read_bytes()
    expected = {name: proof.sha256 for name, proof in prepared.receipt.proofs.items()}
    plan = sha_work_plan(operator_cap=4)
    for workers in (1, 2, plan.workers):
        monkeypatch.setattr(
            packing,
            "sha_work_plan",
            lambda workers=workers: replace(plan, workers=workers),
        )
        loaded = load_prepared_data(prepared.root, byte_enabled=False)
        assert {
            name: proof.sha256 for name, proof in loaded.receipt.proofs.items()
        } == expected
        assert (prepared.root / "manifest.json").read_bytes() == manifest_bytes


def test_unchanged_array_node_remains_reusable(trusted_prepared, monkeypatch):
    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    _load(prepared, store)
    # Identical bytes at a new inode still require cold verification for that node.
    path = prepared.root / "train.npy"
    replacement = path.with_name("replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    original = manifests.sha256_file
    reads = []

    def counted(path, *args, **kwargs):
        if path.suffix == ".npy":
            reads.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(manifests, "sha256_file", counted)
    _load(prepared, store)
    assert reads == ["train.npy"]


def test_large_inventory_rejects_mutation_after_a_successful_check(tmp_path):
    from sparselab.experiments.artifacts import _fingerprint as artifact_fingerprint

    root = tmp_path / "work"
    root.mkdir(mode=0o700)
    tree = root / "tree"
    nested = tree / "nested"
    nested.mkdir(parents=True)
    for index in range(64):
        (nested / f"{index:04}.bin").write_bytes(bytes([index]))
    dependency = root / "upstream.json"
    dependency.write_text("{}")
    binding = {
        "path": str(tree),
        "members": [list(row) for row in artifact_fingerprint(tree)],
        "dependencies": [
            [str(dependency), [list(row) for row in artifact_fingerprint(dependency)]]
        ],
    }
    store = ProofStore(root)
    assert store._safe_binding(binding)
    changed = nested / "0000.bin"
    previous = changed.stat()
    changed.write_bytes(b"x")
    os.utime(changed, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert not store._safe_binding(binding)


def test_proof_diagnostics_reasons_and_measured_bytes(trusted_prepared):
    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    _load(prepared, store)
    first = store.diagnostics()
    assert first["reasons"] == {"no_receipt": 2}
    assert sorted(event["bytes_hashed"] for event in first["events"]) == sorted(
        (prepared.root / name).stat().st_size
        for name in ("train.npy", "validation.npy")
    )
    _load(prepared, store)
    warm = store.diagnostics()
    assert warm["reasons"] == {"no_receipt": 2, "hit": 2}
    assert sorted(event["bytes_avoided"] for event in warm["events"][2:]) == sorted(
        (prepared.root / name).stat().st_size
        for name in ("train.npy", "validation.npy")
    )
    assert all(
        event["verifier_authority"]["kind"] == "prepared_array"
        for event in warm["events"]
    )


def test_manifest_binding_miss_is_not_missing_receipt(trusted_prepared):
    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    path = prepared.root / "train.npy"
    digest = manifests.sha256_file(path)
    proof = verify_file(path, expected_sha256=digest)
    old = file_binding(
        path, digest, kind="prepared_array", closure={"manifest": "before"}
    )
    store.record(old, proof)
    changed = file_binding(
        path, digest, kind="prepared_array", closure={"manifest": "after"}
    )
    assert not store.lookup(changed)
    assert store.diagnostics()["events"][-1]["reason"] == "changed_manifest_binding"


def test_dependency_fingerprint_miss_is_distinct(trusted_prepared):
    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    path = prepared.root / "train.npy"
    dependency = prepared.root / "manifest.json"
    digest = manifests.sha256_file(path)
    binding = file_binding(path, digest)
    info = dependency.stat()
    binding["dependencies"] = [
        [
            str(dependency),
            [
                [
                    ".",
                    info.st_dev,
                    info.st_ino,
                    info.st_mode,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                ]
            ],
        ]
    ]
    proof = verify_file(path, expected_sha256=digest)
    store.record(binding, proof)
    dependency.write_bytes(dependency.read_bytes())
    changed = file_binding(path, digest)
    current = dependency.stat()
    changed["dependencies"] = [
        [
            str(dependency),
            [
                [
                    ".",
                    current.st_dev,
                    current.st_ino,
                    current.st_mode,
                    current.st_size,
                    current.st_mtime_ns,
                    current.st_ctime_ns,
                ]
            ],
        ]
    ]
    assert not store.lookup(changed)
    assert store.diagnostics()["events"][-1]["reason"] == "changed_dependency"


def test_restored_mtime_same_size_requires_cold_hash(trusted_prepared):
    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    _load(prepared, store)
    path = prepared.root / "train.npy"
    old = path.stat()
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(ValueError):
        _load(prepared, store)
    assert store.diagnostics()["events"][-1]["reason"] == "changed_fingerprint"


@pytest.mark.parametrize(
    "dependency_import",
    [
        b"    from sparselab import proof_helper\n",
        (
            b"    from importlib import import_module as load_helper\n"
            b"    return load_helper('sparselab.proof_helper')\n"
        ),
    ],
)
def test_lazy_helper_changes_invalidate_signed_array_proofs(
    trusted_prepared, monkeypatch, dependency_import
):
    root, _, prepared = trusted_prepared
    from sparselab import verifier_authority as authority_module

    original_read = authority_module.Path.read_bytes
    original_is_file = authority_module.Path.is_file
    helper = b"SEMANTICS = 1\n"
    present = True

    def source(self):
        if self.name == "proof_helper.py":
            return helper
        raw = original_read(self)
        if self.name == "packing.py":
            return raw + b"\ndef _new_verifier():\n" + dependency_import
        return raw

    def exists(self):
        return (present and self.name == "proof_helper.py") or original_is_file(self)

    monkeypatch.setattr(authority_module.Path, "read_bytes", source)
    monkeypatch.setattr(authority_module.Path, "is_file", exists)
    store = ProofStore(root)
    _load(prepared, store)
    _load(prepared, store)
    assert store.hits == 2
    helper = b"SEMANTICS = 2\n"
    _load(prepared, store)
    assert store.hits == 2
    assert store.diagnostics()["reasons"]["verifier_authority_changed"] == 2
    assert store.recorded == 4
    present = False
    _load(prepared, store)
    assert store.hits == 2
    assert store.recorded == 4
    assert store.diagnostics()["events"][-1]["reason"] == "unknown"
    assert "missing verifier" in store.diagnostics()["events"][-1]["authority_error"]


@pytest.mark.parametrize(
    "indirect_access",
    [
        (
            "    from importlib import import_module as load_helper\n"
            "    loader = load_helper\n"
            "    return loader('sparselab.proof_helper')\n"
        ),
        "    return globals()['table_address'](b'train.npy', 1024)\n",
    ],
)
def test_indirect_verifier_dependencies_cannot_reuse_or_publish_proofs(
    trusted_prepared, monkeypatch, indirect_access
):
    root, _, prepared = trusted_prepared
    from sparselab import verifier_authority as authority_module

    store = ProofStore(root)
    _load(prepared, store)
    _load(prepared, store)
    assert store.hits == 2
    assert store.recorded == 2
    original = authority_module.Path.read_bytes

    def changed(self):
        raw = original(self)
        if self.name == "packing.py":
            return raw + b"\ndef _new_verifier():\n" + indirect_access.encode()
        return raw

    monkeypatch.setattr(authority_module.Path, "read_bytes", changed)
    _load(prepared, store)
    _load(prepared, store)
    assert store.hits == 2
    assert store.recorded == 2
    assert store.diagnostics()["reasons"]["unknown"] == 4
    assert "dynamic verifier" in store.diagnostics()["events"][-1]["authority_error"]


def test_package_initializer_and_nonliteral_import_are_fail_closed(monkeypatch):
    from sparselab import verifier_authority as authority_module

    before = verifier_authority("prepared_array", 1)
    assert "data" in before["modules"]
    original = authority_module.Path.read_bytes

    def changed_initializer(self):
        raw = original(self)
        if self.name == "__init__.py" and self.parent.name == "data":
            return raw + b"\n# changed package initialization\n"
        return raw

    monkeypatch.setattr(authority_module.Path, "read_bytes", changed_initializer)
    assert verifier_authority("prepared_array", 1)["sha256"] != before["sha256"]

    def dynamic(self):
        raw = original(self)
        if self.name == "packing.py":
            return (
                raw + b"\ndef _dynamic_verifier(name):\n    return __import__(name)\n"
            )
        return raw

    monkeypatch.setattr(authority_module.Path, "read_bytes", dynamic)
    with pytest.raises(ValueError, match="dynamic verifier import"):
        verifier_authority("prepared_array", 1)


@pytest.mark.parametrize(
    "declaration",
    [
        "class UsesExcludedBase(AllocationManifest):\n    pass\n",
        "class UsesExcludedMetaclass(metaclass=AllocationManifest):\n    pass\n",
        "@AllocationManifest\nclass UsesExcludedDecorator:\n    pass\n",
        "class UsesExcludedBody:\n    field = AllocationManifest\n",
    ],
)
def test_new_excluded_helper_class_use_requires_audit(monkeypatch, declaration):
    from sparselab import verifier_authority as authority_module

    original = authority_module.Path.read_bytes

    def changed(self):
        raw = original(self)
        return raw + b"\n" + declaration.encode() if self.name == "packing.py" else raw

    monkeypatch.setattr(authority_module.Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="verifier exclusion needs review"):
        verifier_authority("prepared_array", 1)


def test_unknown_verifier_kind_is_cold_without_receipt(trusted_prepared):
    root, _, prepared = trusted_prepared
    path = prepared.root / "train.npy"
    digest = manifests.sha256_file(path)
    store = ProofStore(root)
    binding = file_binding(path, digest, kind="unsupported_future_kind")
    proof = verify_file(
        path,
        expected_sha256=digest,
        proof_store=store,
        verification_mode="verified_reuse",
        binding=binding,
    )
    assert proof.cold_verified
    assert store.diagnostics()["reasons"] == {"unknown": 1}
    assert store.recorded == 0


@pytest.mark.parametrize(
    "kind, module",
    [
        ("stage_inventory_file", "staging"),
        ("checkpoint_member", "training.checkpoints"),
        ("dispatch_cache_asset", "workers.bundles"),
        ("run_artifact", "evaluation.evidence"),
        ("ingested_run_artifact", "experiments.evidence"),
    ],
)
def test_file_receipt_kind_rechecks_changed_verifier(
    kind, module, trusted_prepared, monkeypatch
):
    from sparselab import verifier_authority as authority_module

    root, _, prepared = trusted_prepared
    path = prepared.root / "train.npy"
    digest = manifests.sha256_file(path)
    binding = file_binding(path, digest, kind=kind, closure={"verified": "fixture"})
    store = ProofStore(root)
    proof = verify_file(path, expected_sha256=digest)
    store.record(binding, proof)
    assert store.lookup(binding)

    original = authority_module.Path.read_bytes
    target = authority_module.Path(module.replace(".", "/") + ".py").name

    def changed(self):
        raw = original(self)
        return raw + b"\n# changed domain verifier\n" if self.name == target else raw

    monkeypatch.setattr(authority_module.Path, "read_bytes", changed)
    assert not store.lookup(binding)
    assert store.diagnostics()["events"][-1]["reason"] == "verifier_authority_changed"
    new_proof = verify_file(
        path,
        expected_sha256=digest,
        proof_store=store,
        verification_mode="verified_reuse",
        binding=binding,
    )
    assert new_proof.cold_verified
    assert store.lookup(binding)


def test_unrelated_module_change_keeps_prepared_receipt_warm(
    trusted_prepared, monkeypatch
):
    from sparselab import verifier_authority as authority_module

    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    _load(prepared, store)
    original = authority_module.Path.read_bytes

    def changed(self):
        raw = original(self)
        return (
            raw + b"unrelated training/UI"
            if self.name in {"trainer.py", "cli.py"}
            else raw
        )

    monkeypatch.setattr(authority_module.Path, "read_bytes", changed)
    _load(prepared, store)
    assert store.hits == 2
    assert store.recorded == 2


def test_full_prepared_artifact_remains_warm_across_training_change(
    trusted_prepared, monkeypatch
):
    from sparselab import verifier_authority as authority_module

    root, _, prepared = trusted_prepared
    artifact = Artifact(
        kind="prepared_data",
        version=1,
        producer="fixture",
        identifier=prepared.manifest["settings_sha256"],
        sha256=prepared.manifest["manifest_sha256"],
        path=str(prepared.root),
    )
    store = ProofStore(root)
    options = {"proof_store": store, "verification_mode": "verified_reuse"}
    first = verify_artifact(artifact, root / "plan.yaml", **options)
    assert verify_artifact(artifact, root / "plan.yaml", **options) == first
    recorded = store.recorded
    original = authority_module.Path.read_bytes

    def unrelated(self):
        raw = original(self)
        if self.name in {"trainer.py", "cli.py"}:
            return raw + b"\n# unrelated execution or UI\n"
        return raw

    monkeypatch.setattr(authority_module.Path, "read_bytes", unrelated)
    assert verify_artifact(artifact, root / "plan.yaml", **options) == first
    assert store.diagnostics()["events"][-1]["kind"] == "prepared_data"
    assert store.diagnostics()["events"][-1]["reason"] == "hit"
    assert store.recorded == recorded


def test_relevant_verifier_change_cold_reseals_not_reblesses(
    trusted_prepared, monkeypatch
):
    from sparselab import verifier_authority as authority_module

    root, _, prepared = trusted_prepared
    store = ProofStore(root)
    _load(prepared, store)
    original = authority_module.Path.read_bytes

    def changed(self):
        raw = original(self)
        return raw + b"\n# changed verifier\n" if self.name == "packing.py" else raw

    monkeypatch.setattr(authority_module.Path, "read_bytes", changed)
    _load(prepared, store)
    assert store.diagnostics()["reasons"]["verifier_authority_changed"] == 2
    assert store.recorded == 4


def test_read_only_store_never_publishes_on_cold_fallback(trusted_prepared):
    root, _, prepared = trusted_prepared
    store = ProofStore(root, read_only=True)
    _load(prepared, store)
    assert store.recorded == 0
    assert not store.directory.exists()


def test_explicit_cold_file_verifier_diagnostic(trusted_prepared):
    root, _, prepared = trusted_prepared
    store = ProofStore(root, read_only=True)
    path = prepared.root / "train.npy"
    digest = manifests.sha256_file(path)
    verify_file(
        path, expected_sha256=digest, proof_store=store, verification_mode="cold"
    )
    assert store.diagnostics()["events"][-1]["reason"] == "explicit_cold"
    assert store.diagnostics()["events"][-1]["bytes_hashed"] == path.stat().st_size
    assert store.misses == store.recorded == 0


def test_unsafe_config_ancestry_never_creates_key_outside_selected_path(
    trusted_prepared, tmp_path, monkeypatch
):
    root, _, prepared = trusted_prepared
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "config-alias"
    alias.symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(alias / "new-config"))
    store = ProofStore(root)
    verified = _load(prepared, store)
    assert all(proof.cold_verified for proof in verified.receipt.proofs.values())
    assert store.recorded == 0
    assert not (outside / "new-config").exists()


def test_execution_config_and_evaluation_protocol_reuse_prepared_ancestors(
    trusted_prepared, monkeypatch
):
    from sparselab.experiments.lock import resolve_plan
    from sparselab.experiments.plan import ExperimentPlan

    root, config, prepared = trusted_prepared
    store = ProofStore(root)
    panel = root / "panel.json"
    panel.write_text(
        json.dumps(
            {
                "evaluation_suite_version": 1,
                "id": "first-panel",
                "evaluations": [
                    {"id": "heldout", "role": "gate", "kind": "heldout_lm"}
                ],
            }
        )
    )
    plan = ExperimentPlan.model_validate(
        {
            "plan_version": 1,
            "id": "ancestor-reuse",
            "base_run": config,
            "evaluation_suite": panel.name,
            "artifacts": {
                "tokenizer": {
                    "kind": "tokenizer",
                    "version": 1,
                    "producer": "sparselab",
                    "identifier": config.tokenizer.path.parent.name,
                    "sha256": manifests.sha256_file(config.tokenizer.path),
                    "path": str(config.tokenizer.path),
                },
                "packed": {
                    "kind": "prepared_data",
                    "version": 1,
                    "producer": "sparselab",
                    "identifier": prepared.manifest["settings_sha256"],
                    "sha256": prepared.manifest["manifest_sha256"],
                    "path": str(prepared.root),
                },
            },
            "inputs": {"tokenizer": "tokenizer", "training": "packed"},
        }
    )
    source = root / "plan.json"
    source.write_text(plan.model_dump_json())
    options = {"proof_store": store, "verification_mode": "verified_reuse"}
    original = resolve_plan(plan, source, **options)
    reads = []
    hasher = manifests.sha256_file

    def counted(path, *args, **kwargs):
        if path.suffix == ".npy":
            reads.append(path.name)
        return hasher(path, *args, **kwargs)

    monkeypatch.setattr(manifests, "sha256_file", counted)
    changed_config = config.model_copy(
        update={
            "optimizer": config.optimizer.model_copy(
                update={
                    "peak": config.optimizer.peak / 2,
                    "floor": config.optimizer.floor / 2,
                }
            ),
        }
    )
    config_plan = plan.model_copy(update={"base_run": changed_config})
    source.write_text(config_plan.model_dump_json())
    config_lock = resolve_plan(config_plan, source, **options)
    assert config_lock.scientific_sha256 != original.scientific_sha256
    assert reads == []
    panel.write_text(
        json.dumps(
            {
                "evaluation_suite_version": 1,
                "id": "revised-panel",
                "evaluations": [
                    {"id": "revised-heldout", "role": "gate", "kind": "heldout_lm"}
                ],
            }
        )
    )
    revised = resolve_plan(config_plan, source, **options)
    assert revised.scientific_sha256 != config_lock.scientific_sha256
    assert reads == []
    assert (
        prepared.manifest["manifest_sha256"]
        == _load(prepared, store).manifest["manifest_sha256"]
    )
