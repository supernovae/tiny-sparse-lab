"""Tiny native artifact/relay integrity checks; no tokenizer or model execution."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from sparselab.training.manifest import ArtifactIdentity, canonical_json
from sparselab.training.metrics import ExperimentStore, _envelope_digest
from sparselab.workers.artifacts import _restore_crash_projection, _verify_complete_run
from sparselab.workers.relay import RelayStore
from sparselab.workers.relay_collection import import_commit_records
from sparselab.workers.relay_models import RelayArtifactIdentity, RelayLocation


def _manifest(path, payload):
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    path.write_bytes(canonical_json({**payload, "sha256": digest}))
    return digest


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("verify_only", [False, True])
@pytest.mark.parametrize("damage", [None, "missing", "size", "digest", "shape"])
def test_relay_reference_requires_its_matching_native_manifest(
    tmp_path, history, verify_only, damage
):
    source = ExperimentStore(tmp_path / "source")
    source.create_run("run", {"name": "run"}, {})
    payload = {"manifest_version": 1, "run_id": "run"}
    path = tmp_path / "manifest.json"
    digest = _manifest(path, payload)
    source.register_manifest("run", digest, payload)
    records = source.export_records()["records"]
    reference = records[-1]["payload"]
    if damage == "size":
        reference["size_bytes"] += 1
    elif damage == "digest":
        reference["sha256"] = "0" * 64
    elif damage == "shape":
        del reference["sha256"]
    records[-1]["sha256"] = _envelope_digest(records[-1])
    relay = RelayStore(
        RelayLocation(kind="file", root=str(tmp_path / "relay")),
        "fixture",
        scratch_root=tmp_path / "scratch",
    )

    def publish(file, relative):
        item = RelayArtifactIdentity(
            relative_path=relative,
            sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            size_bytes=file.stat().st_size,
        )
        relay.put_verified(file, item)
        return item

    prefix = "history/run/run/" if history else "run/"
    manifest = publish(path, prefix + "manifest.json")
    page = tmp_path / "page.json"
    origin = records[0]["origin_id"]
    page.write_bytes(canonical_json({"origin_id": origin, "records": records}))
    page_item = publish(page, "outbox/page.json")
    descriptor = SimpleNamespace(
        run_id="current" if history else "run",
        inventory=[] if damage == "missing" else [manifest],
        inventory_pages=[],
        outbox_pages=[
            SimpleNamespace(
                artifact=page_item,
                origin_id=origin,
                first_sequence=1,
                last_sequence=len(records),
            )
        ],
    )
    root = tmp_path / "controller"
    target = ExperimentStore(root)
    controller = SimpleNamespace(
        root=root,
        store=SimpleNamespace(metrics=target, import_records=target.import_records),
    )
    if damage:
        message = (
            "lacks its immutable closure"
            if damage == "missing"
            else "malformed manifest"
            if damage == "shape"
            else "differs from its artifact"
        )
        with pytest.raises(ValueError, match=message):
            import_commit_records(
                controller, relay, descriptor, verify_only=verify_only
            )
        with target._connect() as con:
            assert con.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    else:
        import_commit_records(controller, relay, descriptor, verify_only=verify_only)
        with target._connect() as con:
            assert con.execute("SELECT count(*) FROM manifests").fetchone()[0] == (
                0 if verify_only else 1
            )


@pytest.mark.parametrize("projection", ["reference", "full", "wrong-reference"])
def test_complete_run_verifies_reference_before_hydrating(tmp_path, projection):
    from sparselab.config.models import RunConfig
    from sparselab.training.manifest import config_sha256

    config = RunConfig.model_validate(
        {
            "schema_version": 2,
            "name": "fixture",
            "seed": 7,
            "model": {
                "vocab_size": 260,
                "hidden_dim": 16,
                "ffn_dim": 32,
                "num_layers": 1,
                "num_heads": 2,
                "max_seq_len": 32,
            },
            "tokenizer": {"path": str(tmp_path / "tokenizer.json")},
            "dataset": {
                "source": "synthetic",
                "synthetic_seed": 7,
                "cache_dir": str(tmp_path / "cache"),
                "train_max_documents": 8,
                "validation_max_documents": 4,
                "train_max_tokens": 256,
                "validation_max_tokens": 128,
            },
            "training": {
                "micro_batch_size": 1,
                "seq_len": 16,
                "max_steps": 1,
                "max_tokens": 16,
            },
            "optimizer": {"name": "adamw", "warmup_steps": 0},
            "logging": {"root_dir": str(tmp_path / "runs")},
        }
    )
    requested = config.model_dump(mode="json")
    effective = json.loads(json.dumps(requested))
    effective["runtime"].update(backend="cpu", precision="fp32")
    worker = SimpleNamespace(
        worker_id="worker", engine="pytorch", backend="cpu", device_index=0
    )
    spec = SimpleNamespace(
        config=config,
        digest=lambda: "a" * 64,
        dispatch_bundle_digest="b" * 64,
        source_identity_sha256="c" * 64,
        continuation=SimpleNamespace(
            kind="FRESH",
            parent_run_id=None,
            checkpoint_sha256=None,
            allow_runtime_drift=False,
        ),
        matrix=None,
        plan=None,
    )
    bundle = SimpleNamespace(config=config, digest=lambda: "b" * 64, files=[])
    receipt = SimpleNamespace(
        run_id="run",
        attempt_id="attempt",
        experiment_id="experiment",
        worker_id="worker",
        spec_digest="a" * 64,
        bundle_digest="b" * 64,
        state="FAILED",
    )
    payload = {
        "manifest_version": 1,
        "run_id": "run",
        "attempt_id": "attempt",
        "experiment_id": "experiment",
        "worker_id": "worker",
        "requested_config": requested,
        "requested_config_sha256": config_sha256(requested),
        "effective_config": effective,
        "effective_config_sha256": config_sha256(effective),
        "source_identity": {"sha256": "c" * 64},
        "continuation_kind": "FRESH",
        "parent_run_id": None,
        "checkpoint_sha256": None,
        "resource_decisions": [
            {
                "kind": "worker_dispatch",
                "spec_digest": "a" * 64,
                "bundle_digest": "b" * 64,
                "matrix": None,
                "plan": None,
            }
        ],
        "artifacts": [],
    }
    root = tmp_path / "run"
    root.mkdir()
    digest = _manifest(root / "manifest.json", payload)
    (root / "resolved_config.yaml").write_bytes(canonical_json(effective))
    items = [
        ArtifactIdentity(
            "run/" + f.name,
            hashlib.sha256(f.read_bytes()).hexdigest(),
            f.stat().st_size,
        )
        for f in root.iterdir()
    ]
    source = ExperimentStore(tmp_path / "source")
    source.create_run("run", requested, {})
    source.register_manifest("run", digest, payload)
    target = source if projection == "full" else ExperimentStore(tmp_path / "target")
    if target is not source:
        target.import_records(source.export_records()["records"])
    if projection == "wrong-reference":
        with target._connect() as con:
            marker = json.loads(con.execute("SELECT json FROM manifests").fetchone()[0])
            marker["manifest_reference"]["size_bytes"] += 1
            con.execute("UPDATE manifests SET json=?", (json.dumps(marker),))
        with pytest.raises(ValueError, match="replicated manifest differs"):
            _verify_complete_run(root, receipt, items, worker, spec, bundle, target)
    else:
        verified = _verify_complete_run(
            root, receipt, items, worker, spec, bundle, target
        )
        assert verified == (payload, [])
        _restore_crash_projection(target, receipt, verified)
        with target._connect() as con:
            assert (
                json.loads(con.execute("SELECT json FROM manifests").fetchone()[0])
                == payload
            )
