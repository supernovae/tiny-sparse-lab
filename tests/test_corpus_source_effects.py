"""Offline public-CLI qualification of exact cross-project source effects."""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest
import yaml

from sparselab.cli.main import main
from sparselab.corpus import acquisition
from sparselab.corpus.acquisition import acquire, declaration_sha256, verify_acquisition
from sparselab.corpus.project import SourceDeclaration, load_project
from sparselab.corpus.transport_budget import TransportBudget
from sparselab.verification_proofs import ProofStore


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    recipe = tmp_path / "recipe"
    source_ids = ("gutenberg", "pagerduty", "scoutflo", "wikimedia")
    for source_id in source_ids:
        _write(
            recipe / "sources" / f"{source_id}.yaml",
            {
                "schema_version": 1,
                "id": source_id,
                "kind": "huggingface_dataset",
                "canonical_uri": f"example/{source_id}",
                "revision": "a" * 40,
                "license": "MIT",
                "redistribution": "redistributable",
                "domains": ["technical_docs"],
                "document_kinds": ["prose"],
                "source_family": source_id,
                "acquisition": {
                    "config": "default",
                    "split": "train",
                    "text_field": "text",
                    "max_rows": 2,
                    "max_bytes": 1024,
                    "bounded_shards": [
                        {
                            "path": f"{source_id}.jsonl",
                            "expected_sha256": "b" * 64,
                            "max_shard_bytes": 1024,
                            "max_scanned_rows": 2,
                            "hash_modulus": 1,
                            "hash_remainders": [0],
                            "declared_config": "default",
                            "declared_split": "train",
                        }
                    ],
                },
            },
        )
    common = {
        "schema_version": 1,
        "sources": [f"sources/{source_id}.yaml" for source_id in source_ids],
        "transforms": [],
        "splits": "splits.yaml",
        "release": "release.yaml",
    }
    _write(
        recipe / "splits.yaml",
        {"schema_version": 1, "unit": "document", "assignments": {"document": "train"}},
    )
    _write(
        recipe / "release.yaml",
        {
            "schema_version": 1,
            "mixture": {"technical_docs": 1.0},
            "lm": {"selected": False, "training_splits": []},
            "chat": {"selected": False, "training_splits": []},
        },
    )
    origin_recipe = recipe / "origin.yaml"
    _write(
        origin_recipe,
        {**common, "id": "approved-origin", "sources": common["sources"][:-1]},
    )
    root = tmp_path / "work"
    root.mkdir(mode=0o700)
    config = tmp_path / "config"
    config.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))

    def origin_transport(source, staging, _cache, _offline, _budget):
        name = f"{source.id}.jsonl"
        body = (json.dumps({"text": source.id}) + "\n").encode()
        target = staging / "files" / name
        target.write_bytes(body)
        return [
            {
                "path": name,
                "size": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        ], {"fixture": source.id}

    with monkeypatch.context() as setup:
        setup.setattr(acquisition, "_acquire_hf", origin_transport)
        lock = acquire(load_project(origin_recipe), root)
    effects = []
    for source_id in source_ids:
        source = SourceDeclaration.model_validate(
            yaml.safe_load((recipe / "sources" / f"{source_id}.yaml").read_text())
        )
        # The acquired-only source has the same declaration format but no origin.
        effects.append(
            {
                "source_id": source_id,
                "declaration_sha256": declaration_sha256(source),
                "effect": "acquire" if source_id == "wikimedia" else "reuse_only",
                **(
                    {
                        "origin_project_id": "approved-origin",
                        "snapshot_sha256": lock["sources"][source_id][
                            "snapshot_sha256"
                        ],
                    }
                    if source_id != "wikimedia"
                    else {}
                ),
            }
        )
    target_recipe = recipe / "target.yaml"
    _write(
        target_recipe,
        {
            **common,
            "id": "candidate",
            "source_effects": effects,
            "transport_budget": {
                "attempt_id": "fixture",
                "max_source_body_bytes": 4096,
                "max_metadata_body_bytes": 4096,
                "max_transfers": 2,
                "max_retries_per_shard": 0,
                "max_wall_seconds": 120,
                "max_disk_bytes": 1_000_000,
            },
        },
    )
    return root, origin_recipe, target_recipe


def _cli(
    monkeypatch: pytest.MonkeyPatch, root: Path, recipe: Path, *extra: str
) -> dict:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "--work-dir",
            str(root),
            "corpus",
            "acquire",
            str(recipe),
            *extra,
        ],
    )
    from contextlib import redirect_stdout
    from io import StringIO

    output = StringIO()
    with redirect_stdout(output):
        main()
    return json.loads(output.getvalue())


def _tripwire(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    attempted: list[str] = []
    original_popen = subprocess.Popen

    def blocked(*args, **kwargs):
        attempted.append(str(args[:1]))
        raise AssertionError("undeclared network or subprocess transport")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(urllib.request, "build_opener", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)

    def checked_popen(*args, **kwargs):
        command = args[0] if args else kwargs.get("args")
        if (
            isinstance(command, list)
            and len(command) == 5
            and command[0] == "git"
            and command[1] == "-C"
            and command[3:] == ["rev-parse", "--show-toplevel"]
        ):
            return original_popen(*args, **kwargs)
        return blocked(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", checked_popen)
    return attempted


def test_public_cli_verified_reuse_empty_warm_and_cold_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, recipe = _project(tmp_path, monkeypatch)
    project = load_project(recipe)
    budget = TransportBudget.initialize(
        root / "corpora/candidate/transport-budget.sqlite", project
    )
    observed = []

    def wiki_only(source, staging, _cache, _offline, transport):
        observed.append(source.id)
        assert source.id == "wikimedia" and transport is not None
        body = b'{"text":"Wikimedia fixture"}\n'
        target = staging / "files/wikimedia.jsonl"
        target.write_bytes(body)
        transfer = transport.reserve_transfer(source.id, "wikimedia.jsonl", len(body))
        transport.charge_transfer_actual(transfer, len(body))
        transport.finish_transfer(transfer, success=True)
        return [
            {
                "path": target.name,
                "size": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        ], {"fixture": "wikimedia-only"}

    monkeypatch.setattr(acquisition, "_acquire_hf", wiki_only)
    observed_stores = []
    real_verify = acquisition.verify_snapshot

    def observed_verify(*args, **kwargs):
        if kwargs.get("proof_store") is not None:
            observed_stores.append(kwargs["proof_store"])
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(acquisition, "verify_snapshot", observed_verify)
    attempted = _tripwire(monkeypatch)
    first = _cli(monkeypatch, root, recipe)
    assert observed_stores and any(store.misses for store in observed_stores)
    first_recorded = any(store.recorded for store in observed_stores)
    assert observed == ["wikimedia"] and not attempted
    assert {
        key: value["reuse_origin"]["project_id"]
        for key, value in first["sources"].items()
        if "reuse_origin" in value
    } == {key: "approved-origin" for key in ("gutenberg", "pagerduty", "scoutflo")}
    assert {row["key"].split(":", 1)[0] for row in budget.receipt()["transfers"]} == {
        "wikimedia"
    }
    assert budget.receipt()["metadata_actual"] == 0
    assert _cli(monkeypatch, root, recipe, "--offline") == first
    observed_stores.clear()
    warm = _cli(monkeypatch, root, recipe)
    assert warm == first and observed == ["wikimedia"] and not attempted
    assert observed_stores and all(
        store.root == root and store.misses + store.hits > 0
        for store in observed_stores
    )
    if first_recorded:
        assert any(store.hits for store in observed_stores)
    else:
        # The current source-snapshot verifier authority declines signing due
        # an unrelated dynamic import; verified_reuse must safely recheck cold.
        assert any(
            event["reason"] == "unknown"
            for store in observed_stores
            for event in store.diagnostics()["events"]
        )
    assert (
        verify_acquisition(
            project,
            root,
            proof_store=ProofStore(root),
            verification_mode="verified_reuse",
        )
        == first
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "corrupt",
        "retarget",
        "unsafe",
        "member_symlink",
        "declaration",
        "incomplete",
    ],
)
def test_preflight_fails_before_any_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    root, _, recipe = _project(tmp_path, monkeypatch)
    project = load_project(recipe)
    TransportBudget.initialize(
        root / "corpora/candidate/transport-budget.sqlite", project
    )
    retained = project.config.source_effects[0]
    origin = (
        root
        / "corpora"
        / retained.origin_project_id
        / "snapshots"
        / retained.source_id
        / retained.snapshot_sha256
    )
    if damage == "missing":
        shutil.rmtree(origin)
    elif damage == "corrupt":
        next((origin / "files").iterdir()).write_bytes(b"corrupt")
    elif damage in {"retarget", "unsafe"}:
        moved = origin.with_name("safe-original")
        origin.rename(moved)
        origin.symlink_to(moved if damage == "retarget" else tmp_path)
    elif damage == "member_symlink":
        (origin / "unexpected-link").symlink_to(origin / "manifest.json")
    else:
        raw = yaml.safe_load(recipe.read_text())
        if damage == "declaration":
            raw["source_effects"][0]["declaration_sha256"] = "0" * 64
        else:
            raw["source_effects"].pop()
        _write(recipe, raw)
    attempted = _tripwire(monkeypatch)
    with pytest.raises(ValueError):
        _cli(monkeypatch, root, recipe)
    assert not attempted
    assert not (root / "corpora/candidate/acquisition.json").exists()
    state = (
        TransportBudget(
            root / "corpora/candidate/transport-budget.sqlite", load_project(recipe)
        ).receipt()
        if damage not in {"declaration", "incomplete"}
        else None
    )
    if state is not None:
        assert state["transfers"] == [] and state["metadata_actual"] == 0


def test_kml_declaration_permits_only_pinned_wikimedia_acquisition() -> None:
    recipe = Path(
        "experiments/research/kernel-memory-lab/card05-base-50m/project-acquire-reuse-v2.yaml"
    )
    project = load_project(recipe)
    effects = {item.source_id: item.effect for item in project.config.source_effects}
    assert effects == {
        "kml_scale_pagerduty": "reuse_only",
        "kml_scale_project_gutenberg": "reuse_only",
        "kml_scale_scoutflo": "reuse_only",
        "kml_scale_wikimedia": "acquire",
    }


def test_stale_proof_reverified_and_bad_copy_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, recipe = _project(tmp_path, monkeypatch)
    project = load_project(recipe)
    TransportBudget.initialize(
        root / "corpora/candidate/transport-budget.sqlite", project
    )
    store = ProofStore(root)
    effect = project.config.source_effects[0]
    origin = (
        root
        / "corpora"
        / effect.origin_project_id
        / "snapshots"
        / effect.source_id
        / effect.snapshot_sha256
    )
    acquisition.verify_snapshot(
        origin, proof_store=store, verification_mode="verified_reuse"
    )
    content = next((origin / "files").iterdir())
    original = content.read_bytes()
    content.write_bytes(b"tampered")
    observed_stores = []
    real_verify = acquisition.verify_snapshot

    def observed_verify(*args, **kwargs):
        if kwargs.get("proof_store") is not None:
            observed_stores.append(
                (kwargs["proof_store"], kwargs.get("verification_mode"))
            )
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(acquisition, "verify_snapshot", observed_verify)
    _tripwire(monkeypatch)
    with pytest.raises(ValueError):
        _cli(monkeypatch, root, recipe)
    assert observed_stores and all(
        mode == "verified_reuse" for _, mode in observed_stores
    )
    assert any(session.misses for session, _ in observed_stores)
    content.write_bytes(original)
    # A separate bad local copy cannot be treated as a valid retained origin.
    copy = (
        root / "corpora/candidate/snapshots" / effect.source_id / effect.snapshot_sha256
    )
    copy.parent.mkdir(parents=True)
    shutil.copytree(origin, copy)
    next((copy / "files").iterdir()).write_bytes(b"bad copy")
    with pytest.raises(ValueError):
        _cli(monkeypatch, root, recipe)
    assert not (root / "corpora/candidate/acquisition.json").exists()


def test_acquisition_lock_rejects_changed_origin_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, recipe = _project(tmp_path, monkeypatch)
    project = load_project(recipe)
    TransportBudget.initialize(
        root / "corpora/candidate/transport-budget.sqlite", project
    )

    def wiki_only(source, staging, _cache, _offline, _budget):
        assert source.id == "wikimedia"
        body = b'{"text":"fixture"}\n'
        file = staging / "files/wikimedia.jsonl"
        file.write_bytes(body)
        return [
            {
                "path": file.name,
                "size": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        ], {"fixture": "wikimedia-only"}

    monkeypatch.setattr(acquisition, "_acquire_hf", wiki_only)
    _tripwire(monkeypatch)
    lock = _cli(monkeypatch, root, recipe)
    selected = root / "corpora/candidate/acquisition.json"
    lock["sources"]["gutenberg"]["reuse_origin"]["project_id"] = "substituted"
    selected.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="retained origin binding mismatch"):
        _cli(monkeypatch, root, recipe, "--offline")
    with pytest.raises(ValueError, match="retained origin binding mismatch"):
        verify_acquisition(project, root, proof_store=ProofStore(root))


def test_origin_mutation_during_only_permitted_transport_prevents_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, recipe = _project(tmp_path, monkeypatch)
    project = load_project(recipe)
    TransportBudget.initialize(
        root / "corpora/candidate/transport-budget.sqlite", project
    )
    effect = project.config.source_effects[0]
    origin = (
        root
        / "corpora"
        / effect.origin_project_id
        / "snapshots"
        / effect.source_id
        / effect.snapshot_sha256
    )

    def wiki_only(source, staging, _cache, _offline, _budget):
        assert source.id == "wikimedia"
        next((origin / "files").iterdir()).write_bytes(b"changed during transfer")
        body = b'{"text":"fixture"}\n'
        file = staging / "files/wikimedia.jsonl"
        file.write_bytes(body)
        return [
            {
                "path": file.name,
                "size": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        ], {"fixture": "wikimedia-only"}

    monkeypatch.setattr(acquisition, "_acquire_hf", wiki_only)
    _tripwire(monkeypatch)
    with pytest.raises(ValueError):
        _cli(monkeypatch, root, recipe)
    assert not (root / "corpora/candidate/acquisition.json").exists()
