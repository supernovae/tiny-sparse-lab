"""Generic offline ancestry: static retained manifests, no producer execution."""

import json

import pytest
from test_snapshot_warm_reuse import retained_snapshot

from sparselab.corpus.acquisition import _project_sha
from sparselab.recovery import engine, implementation_replay
from sparselab.recovery.manifest import CorpusRelease, SnapshotInheritance
from sparselab.training.manifest import sha256_file


def ancestry(tmp_path, monkeypatch):
    project, root, snapshot, manifest = retained_snapshot(tmp_path)
    lock = root / "corpora" / project.config.id / "acquisition.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_id": project.config.id,
                "project_sha256": _project_sha(project),
                "sources": {
                    "one": {
                        "declaration_sha256": manifest["declaration_sha256"],
                        "snapshot_sha256": snapshot.name,
                        "snapshot_path": str(snapshot),
                        "receipt": {"status": "acquired", "retrieval": {}},
                    }
                },
            }
        )
    )
    parent = root / "replay/receipts/parent.json"
    parent.parent.mkdir(parents=True)
    parent.write_text("{}")
    record = {
        "status": "MATCH",
        "format": implementation_replay._ANCESTRY_FORMAT,
        "project_id": project.config.id,
        "historical_project": str(project.root / "corpus.yaml"),
        "corpus_work_root": str(root),
        "acquisition_closure": {"path": str(lock)},
        "scientific_identity": {"snapshots": {"one": snapshot.name}},
    }
    monkeypatch.setattr(
        implementation_replay, "verify_replay_receipt", lambda path: record
    )
    step = CorpusRelease(
        kind="corpus_release",
        id="corpus",
        project="recipe/corpus.yaml",
        expected_build_sha256="a" * 64,
        expected_release_sha256="b" * 64,
        snapshot_inheritance=SnapshotInheritance(
            parent_receipt="replay/receipts/parent.json",
            parent_receipt_sha256=sha256_file(parent),
            snapshots={"one": snapshot.name},
        ),
    )
    return step, tmp_path / "recovery.json", root, parent, record


def test_native_generic_parent_identity_and_declaration_checks(tmp_path, monkeypatch):
    step, declaration, root, parent, record = ancestry(tmp_path, monkeypatch)
    options = engine._snapshot_inheritance(step, declaration, root)
    assert options["parent_receipt"] == parent
    assert options["inherited_source_ids"] == ("one",)
    assert (
        options["expected_unchanged_snapshots"]
        == record["scientific_identity"]["snapshots"]
    )
    assert options["use_historical_project"] is True
    parent.write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="declared parent digest mismatch"):
        engine._snapshot_inheritance(step, declaration, root)


@pytest.mark.parametrize(
    "mutation", ["source", "snapshot", "parent_map", "parent_failed", "symlink"]
)
def test_generic_ancestry_rejects_changed_inputs(tmp_path, monkeypatch, mutation):
    step, declaration, root, parent, record = ancestry(tmp_path, monkeypatch)
    if mutation == "source":
        import shutil

        shutil.copytree(tmp_path / "recipe", tmp_path / "child")
        step = step.model_copy(update={"project": "child/corpus.yaml"})
        source = tmp_path / "child/sources/one.yaml"
        source.write_text(source.read_text() + "\n# changed declaration bytes\n")
    elif mutation == "snapshot":
        step.snapshot_inheritance.snapshots["one"] = "0" * 64
    elif mutation == "parent_map":
        record["scientific_identity"]["snapshots"]["missing"] = "0" * 64
    elif mutation == "parent_failed":
        record["status"] = "FAILED"
    else:
        target = tmp_path / "other.json"
        parent.rename(target)
        parent.symlink_to(target)
    with pytest.raises(ValueError, match="INVALID_PARENT_RECEIPT"):
        engine._snapshot_inheritance(step, declaration, root)


def test_declared_parent_is_cold_authenticated(tmp_path, monkeypatch):
    original = implementation_replay.verify_replay_receipt
    step, declaration, root, _, _ = ancestry(tmp_path, monkeypatch)
    monkeypatch.setattr(implementation_replay, "verify_replay_receipt", original)
    with pytest.raises(ValueError):
        engine._snapshot_inheritance(step, declaration, root)


def test_recovery_cli_routes_declared_ancestry_without_acquisition(
    tmp_path, monkeypatch, capsys
):
    from sparselab.cli.main import build_parser
    from sparselab.recovery import provenance

    step, declaration, root, _, _ = ancestry(tmp_path, monkeypatch)
    declaration.write_text(
        json.dumps(
            {
                "recovery_version": 1,
                "id": "inheritance",
                "source_commit": "a" * 40,
                "steps": [
                    step.model_dump(mode="json"),
                    {
                        "id": "model",
                        "kind": "external_required",
                        "role": "model",
                        "reason": "Offline fixture",
                    },
                ],
            }
        )
    )
    monkeypatch.setattr(provenance, "declaration_preflight", lambda *a, **kw: {})
    monkeypatch.setattr(engine, "_pinned_inputs", lambda *a: [])
    rows = [
        {
            "id": "corpus",
            "kind": "corpus_release",
            "classification": "PINNED_IMPLEMENTATION_REPLAY_REQUIRED",
            "reason": "declared inheritance",
            "path": None,
        }
    ]
    monkeypatch.setattr(engine, "inspect_manifest", lambda *a: {"steps": rows})
    calls = []

    def replay(*a, **kw):
        calls.append(kw)
        raise ValueError("FIXTURE_STOP: replay producer intentionally not executed")

    monkeypatch.setattr(implementation_replay, "replay_corpus", replay)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--work-dir",
            str(root),
            "recovery",
            "reconstruct",
            str(declaration),
            "--replay-pinned-implementation",
            "--allow-uncommitted-declaration",
            "--json",
        ]
    )
    with pytest.raises(SystemExit):
        args.handler(args)
    assert calls[0]["inherited_source_ids"] == ("one",)
    assert calls[0]["allow_network"] is False
    assert "FIXTURE_STOP" in capsys.readouterr().out
    calls.clear()
    args.replay_pinned_implementation = False
    with pytest.raises(SystemExit):
        args.handler(args)
    assert not calls
    assert "PINNED_IMPLEMENTATION_REPLAY_REQUIRED" in capsys.readouterr().out


def test_legacy_step_serialization_preserves_identity():
    step = CorpusRelease(kind="corpus_release", id="corpus", project="corpus.yaml")
    assert "snapshot_inheritance" not in step.model_dump(mode="json")


@pytest.mark.parametrize(
    "changes",
    [
        {"parent_receipt": "../parent.json"},
        {"parent_receipt_sha256": "bad"},
        {"snapshots": {}},
        {"snapshots": {"../one": "a" * 64}},
        {"changed_snapshots": {"one": "b" * 64}},
    ],
)
def test_inheritance_rejects_invalid_declarations(changes):
    with pytest.raises(ValueError):
        SnapshotInheritance.model_validate(
            {
                "parent_receipt": "replay/receipts/parent.json",
                "parent_receipt_sha256": "a" * 64,
                "snapshots": {"one": "a" * 64},
                **changes,
            }
        )


def test_fresh_native_import_and_worker_reuse_preflight(tmp_path, monkeypatch):
    from sparselab.corpus import acquisition
    from sparselab.corpus.project import load_project
    from sparselab.recovery._implementation_worker import _prepare_inheritance

    step, _declaration, root, _, record = ancestry(tmp_path, monkeypatch)
    corpus = load_project(tmp_path / "recipe/corpus.yaml")
    _, inherited, _ = implementation_replay._ancestry_parent_verified(
        record, ("one",), corpus
    )
    fresh = root / "replay/work/child"
    implementation_replay._stage_ancestry_imports(record, inherited, fresh, corpus)
    implementation_replay._ancestry_target_imports(
        record, inherited, fresh, corpus.config.id
    )
    request = {
        "inherited_snapshots": inherited,
        "expected_unchanged_snapshots": step.snapshot_inheritance.snapshots,
        "expected_changed_snapshots": {},
    }

    # Patch producers before worker installs its refusal guards. Nothing can acquire.
    def forbidden(*a, **kw):
        raise AssertionError("fixture must never acquire")

    for name in (
        "_acquire_git",
        "_acquire_hf",
        "_acquire_http",
        "_acquire_local",
        "_acquire_wikimedia",
    ):
        monkeypatch.setattr(acquisition, name, forbidden)
    before = _prepare_inheritance(acquisition, corpus, fresh, request)
    assert before == step.snapshot_inheritance.snapshots
    lock = acquisition.verify_acquisition(corpus, fresh)
    assert lock["sources"]["one"]["snapshot_sha256"] == before["one"]
    copied = fresh / "corpora" / corpus.config.id / "snapshots/one" / before["one"]
    original = root / "corpora" / corpus.config.id / "snapshots/one" / before["one"]
    assert (copied / "manifest.json").read_bytes() == (
        original / "manifest.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="INHERITED_REACQUISITION_FORBIDDEN"):
        acquisition._acquire_git(corpus.sources[0])
    (copied / "files/data.md").write_text("tamper")
    with pytest.raises(ValueError):
        implementation_replay._ancestry_target_imports(
            record, inherited, fresh, corpus.config.id
        )


def test_fresh_changed_and_new_sources_keep_prior_inventory(tmp_path, monkeypatch):
    import yaml

    from sparselab.corpus import acquisition
    from sparselab.corpus.project import load_project
    from sparselab.recovery._implementation_worker import _prepare_inheritance

    step, _, root, _, record = ancestry(tmp_path, monkeypatch)
    old = load_project(tmp_path / "recipe/corpus.yaml")
    _, imports, _ = implementation_replay._ancestry_parent_verified(
        record, ("one",), old
    )
    import shutil

    shutil.copytree(old.root, tmp_path / "child")
    source_path = tmp_path / "child/sources/one.yaml"
    source = yaml.safe_load(source_path.read_text())
    source["revision"] = "b" * 40
    source_path.write_text(yaml.safe_dump(source))
    source["id"] = "new_source"
    (source_path.parent / "new.yaml").write_text(yaml.safe_dump(source))
    project_path = tmp_path / "child/corpus.yaml"
    project = yaml.safe_load(project_path.read_text())
    project["sources"].append("sources/new.yaml")
    project["sources"].sort()
    project_path.write_text(yaml.safe_dump(project))
    corpus = load_project(project_path)
    implementation_replay._ancestry_changed_declarations(
        record, corpus, step.snapshot_inheritance.snapshots
    )
    fresh = root / "replay/work/changed-child"
    implementation_replay._stage_ancestry_imports(record, imports, fresh, corpus)
    implementation_replay._ancestry_target_imports(
        record, imports, fresh, corpus.config.id
    )
    request = {
        "inherited_snapshots": {},
        "expected_unchanged_snapshots": {},
        "expected_changed_snapshots": step.snapshot_inheritance.snapshots,
    }

    def forbidden(*a, **kw):
        raise AssertionError("producer must not execute")

    for name in (
        "_acquire_git",
        "_acquire_hf",
        "_acquire_http",
        "_acquire_local",
        "_acquire_wikimedia",
    ):
        monkeypatch.setattr(acquisition, name, forbidden)
    assert (
        _prepare_inheritance(acquisition, corpus, fresh, request)
        == step.snapshot_inheritance.snapshots
    )
    # The old inventory establishes ancestry; it cannot certify changed/new inputs.
    with pytest.raises(ValueError, match="acquisition lock does not match project"):
        acquisition.verify_acquisition(corpus, fresh)


@pytest.mark.parametrize("source_count", [3, 123])
def test_worker_generic_changed_source_sets(source_count, tmp_path):
    from types import SimpleNamespace

    from sparselab.recovery._implementation_worker import _prepare_inheritance

    changed = {f"generic_{number}": f"{number:064x}" for number in range(source_count)}
    corpus = SimpleNamespace(
        config=SimpleNamespace(id="generic"),
        sources=[
            SimpleNamespace(id=key, redistribution="redistributable") for key in changed
        ],
    )
    base = tmp_path / "corpora/generic"
    base.mkdir(parents=True)
    lock = base / "acquisition.json"
    lock.write_text(
        json.dumps(
            {"sources": {key: {"snapshot_sha256": sha} for key, sha in changed.items()}}
        )
    )
    acquisition = SimpleNamespace(
        verify_snapshot=lambda path: {
            "snapshot_sha256": path.name,
            "declaration": {"revision": "old"},
        },
        source_declaration_payload=lambda item: {"revision": "new"},
    )
    request = {
        "inherited_snapshots": {},
        "expected_unchanged_snapshots": {},
        "expected_changed_snapshots": changed,
    }
    assert _prepare_inheritance(acquisition, corpus, tmp_path, request) == changed
    lock.write_text(
        json.dumps({"sources": {key: {"snapshot_sha256": "f" * 64} for key in changed}})
    )
    with pytest.raises(ValueError, match="CHANGED_IDENTITY_MISMATCH"):
        _prepare_inheritance(acquisition, corpus, tmp_path, request)


def test_import_rejects_unsafe_members_before_publication(tmp_path, monkeypatch):
    from sparselab.corpus.project import load_project

    step, _, root, _, record = ancestry(tmp_path, monkeypatch)
    corpus = load_project(tmp_path / "recipe/corpus.yaml")
    _, imports, _ = implementation_replay._ancestry_parent_verified(
        record, ("one",), corpus
    )
    source = (
        root
        / "corpora"
        / corpus.config.id
        / "snapshots/one"
        / step.snapshot_inheritance.snapshots["one"]
    )
    (source / "foreign").symlink_to(tmp_path / "outside")
    destination = root / "replay/work/unsafe"
    with pytest.raises(ValueError, match="unsafe inherited member"):
        implementation_replay._stage_ancestry_imports(
            record, imports, destination, corpus
        )
    assert not destination.exists()
