"""Offline native HTTP fixtures for bounded steps on a single full project."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import yaml
from test_corpus_acquisition import _budgeted_hf_fixture, _mock_budgeted_hf_http, _yaml
from test_corpus_source_effects import _cli, _project

from sparselab.corpus import acquisition, acquisition_step
from sparselab.corpus.acquisition import acquire, verify_acquisition
from sparselab.corpus.project import load_project
from sparselab.corpus.transport_budget import TransportBudget

ALLOWANCE = 2 * 1024**3


def fixture(tmp_path, monkeypatch, *, second=True):
    content = gzip.compress(b'{"text":"one"}\n', mtime=0)
    recipe = _budgeted_hf_fixture(tmp_path, content)
    config = yaml.safe_load(recipe.read_text())
    config["transport_budget"].update(max_disk_bytes=ALLOWANCE, max_wall_seconds=600)
    if second:
        source = yaml.safe_load((recipe.parent / "sources/one.yaml").read_text())
        source["id"] = "two"
        _yaml(recipe.parent / "sources/two.yaml", source)
        config["sources"].append("sources/two.yaml")
    _yaml(recipe, config)
    project = load_project(recipe)
    root = tmp_path / "work"
    ledger = TransportBudget.initialize(
        root / "corpora/example/transport-budget.sqlite", project
    )
    calls, _ = _mock_budgeted_hf_http(monkeypatch, content)
    return recipe, project, root, ledger, calls, content


def run(project, root, source_id="one", **kwargs):
    return acquire(
        project,
        root,
        source_id=source_id,
        max_new_bytes=kwargs.pop("max_new_bytes", ALLOWANCE),
        min_free_bytes=kwargs.pop("min_free_bytes", 0),
        **kwargs,
    )


def test_public_steps_keep_full_binding_and_publish_lock_only_on_completion(
    tmp_path, monkeypatch
):
    recipe, project, root, ledger, calls, content = fixture(tmp_path, monkeypatch)
    initial = ledger.receipt()
    before = ledger.path.read_bytes()
    plan = _cli(
        monkeypatch,
        root,
        recipe,
        "--source-id",
        "one",
        "--max-new-bytes",
        str(ALLOWANCE),
        "--min-free-bytes",
        "0",
        "--plan",
    )
    assert plan["status"] == "planned" and plan["complete"] is False
    assert not calls and ledger.path.read_bytes() == before
    result = _cli(
        monkeypatch,
        root,
        recipe,
        "--source-id",
        "one",
        "--max-new-bytes",
        str(ALLOWANCE),
        "--min-free-bytes",
        "0",
    )
    assert result["status"] == "source_verified" and result["complete"] is False
    assert result["admission"]["transport_budget_binding"] == initial["binding"]
    assert result["admission"]["project_sha256"] == acquisition._project_sha(project)
    base = root / "corpora/example"
    assert (
        not (base / "acquisition.json").exists()
        and not (base / "snapshots/two").exists()
    )
    source = Path(result["source"]["snapshot_path"])
    first_manifest = (source / "manifest.json").read_bytes()
    first_calls = list(calls)
    again = run(project, root)
    assert again["source"] == result["source"] and calls == first_calls
    assert (source / "manifest.json").read_bytes() == first_manifest
    second = run(project, root, "two")
    assert second["admission"]["transport_budget_binding"] == initial["binding"]
    assert not (base / "acquisition.json").exists()
    saved = ledger.receipt()
    assert saved["source_actual"] == saved["source_charged"] == 2 * len(content)
    assert (
        saved["created_at"] == initial["created_at"]
        and saved["deadline"] == initial["deadline"]
    )
    calls_before = list(calls)
    complete = acquire(project, root)
    assert set(complete["sources"]) == {"one", "two"} and calls == calls_before
    assert verify_acquisition(project, root) == complete
    summary = _cli(monkeypatch, root, recipe, "--require-existing", "--summary")
    assert summary["source_count"] == 2
    assert summary["complete"] is True
    assert summary["lock_path"] == str(root / "corpora/example/acquisition.json")
    assert calls == calls_before
    assert (source / "manifest.json").read_bytes() == first_manifest


def test_finalize_existing_snapshots_never_fetches_and_publishes_lock(
    tmp_path, monkeypatch
):
    recipe, project, root, _ledger, calls, _content = fixture(tmp_path, monkeypatch)
    one = run(project, root)
    two = run(project, root, "two")
    before = list(calls)

    def no_fetch(*_args, **_kwargs):
        pytest.fail("existing-snapshot finalization attempted a fetch")

    monkeypatch.setattr(acquisition, "_acquire_hf", no_fetch)
    summary = _cli(monkeypatch, root, recipe, "--require-existing")
    assert summary["source_count"] == 2
    complete = verify_acquisition(project, root)
    assert set(complete["sources"]) == {"one", "two"}
    assert (
        complete["sources"]["one"]["snapshot_sha256"]
        == one["source"]["snapshot_sha256"]
    )
    assert (
        complete["sources"]["two"]["snapshot_sha256"]
        == two["source"]["snapshot_sha256"]
    )
    assert calls == before
    assert verify_acquisition(project, root) == complete


def test_finalize_existing_refuses_missing_or_changed_snapshot_without_fetch(
    tmp_path, monkeypatch
):
    _recipe, project, root, _ledger, calls, _content = fixture(tmp_path, monkeypatch)
    first = run(project, root)
    before = list(calls)

    def no_fetch(*_args, **_kwargs):
        pytest.fail("existing-snapshot finalization attempted a fetch")

    monkeypatch.setattr(acquisition, "_acquire_hf", no_fetch)
    with pytest.raises(ValueError, match="existing verified snapshot unavailable: two"):
        acquire(project, root, require_existing=True)
    assert calls == before and not (root / "corpora/example/acquisition.json").exists()
    source = Path(first["source"]["snapshot_path"])
    manifest = source / "manifest.json"
    changed = json.loads(manifest.read_bytes())
    changed["snapshot_sha256"] = "0" * 64
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="existing verified snapshot unavailable: one"):
        acquire(project, root, require_existing=True)
    assert calls == before and not (root / "corpora/example/acquisition.json").exists()


@pytest.mark.parametrize(
    "kwargs", [{"offline": True}, {"source_id": "one"}, {"plan_only": True}]
)
def test_finalize_existing_refuses_incompatible_modes(tmp_path, monkeypatch, kwargs):
    _recipe, project, root, _ledger, calls, _content = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="existing-snapshot finalization"):
        acquire(project, root, require_existing=True, **kwargs)
    assert not calls and not (root / "corpora/example/acquisition.json").exists()


@pytest.mark.parametrize(
    "override,match",
    [
        ({"source_id": "missing"}, "exact declared"),
        ({"max_new_bytes": 0}, "positive remaining"),
        ({"max_new_bytes": 1}, "cumulative"),
        ({"max_new_bytes": None}, "positive remaining"),
        ({"min_free_bytes": None}, "explicit free"),
        ({"min_free_bytes": -1}, "explicit free"),
        ({"min_free_bytes": 10**20}, "disk headroom"),
        ({"offline": True}, "cannot use"),
        ({"verification_mode": "verified_reuse"}, "cold verification"),
        ({"proof_store": object()}, "cold verification"),
    ],
)
def test_refusals_before_output_or_http(tmp_path, monkeypatch, override, match):
    _, project, root, ledger, calls, _ = fixture(tmp_path, monkeypatch)
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match=match):
        run(project, root, **override)
    assert not calls and ledger.path.read_bytes() == before
    assert not (root / "corpora/example/snapshots").exists()


def test_exact_cumulative_and_free_boundaries(tmp_path, monkeypatch):
    _, project, root, _, calls, content = fixture(tmp_path, monkeypatch)
    plan = run(project, root, plan_only=True)["admission"]
    expected = (
        2 * (project.sources[0].acquisition.max_bytes + acquisition_step.MANIFEST_BYTES)
        + len(content)
        + acquisition_step.LEDGER_BYTES
        + acquisition_step.JOURNAL_BYTES
    )
    assert plan["native_new_path_bytes"] == expected
    with pytest.raises(ValueError, match="cumulative"):
        run(project, root, max_new_bytes=expected - 1)
    assert (
        run(project, root, max_new_bytes=expected, plan_only=True)["admission"][
            "native_new_path_bytes"
        ]
        == expected
    )
    from types import SimpleNamespace

    peak = plan["native_peak_added_bytes"]
    monkeypatch.setattr(
        acquisition_step.os,
        "statvfs",
        lambda _: SimpleNamespace(f_bavail=peak + 123, f_frsize=1),
    )
    run(project, root, min_free_bytes=123, plan_only=True)
    with pytest.raises(ValueError, match="disk headroom"):
        run(project, root, min_free_bytes=124)
    assert not calls


def test_selected_projection_does_not_reserve_unselected_expansion(
    tmp_path, monkeypatch
):
    recipe, _, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    # Change fixture configuration before initializing a fresh full-project ledger.
    source_file = recipe.parent / "sources/two.yaml"
    source = yaml.safe_load(source_file.read_text())
    source["acquisition"]["max_bytes"] = 100 * ALLOWANCE
    _yaml(source_file, source)
    project = load_project(recipe)
    other = tmp_path / "other"
    ledger = TransportBudget.initialize(
        other / "corpora/example/transport-budget.sqlite", project
    )
    assert ledger.projected_disk_bytes() > ledger.spec.max_disk_bytes
    assert run(project, other)["status"] == "source_verified"
    assert calls


def test_manifest_cap_precedes_write_and_publication(tmp_path, monkeypatch):
    _, project, root, ledger, calls, content = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(acquisition_step, "MANIFEST_BYTES", 8)
    with pytest.raises(ValueError, match="manifest byte cap"):
        run(project, root)
    assert calls and ledger.receipt()["source_actual"] == len(content)
    assert not list((root / "corpora/example").rglob("manifest.json"))
    assert not (root / "corpora/example/acquisition.json").exists()


def test_failure_preserves_prior_snapshot_and_charges(tmp_path, monkeypatch):
    _, project, root, ledger, _, content = fixture(tmp_path, monkeypatch)
    first = run(project, root)
    manifest = Path(first["source"]["snapshot_path"]) / "manifest.json"
    saved = manifest.read_bytes()
    _mock_budgeted_hf_http(monkeypatch, content, fail_http=True)
    with pytest.raises(Exception, match="unavailable|503"):
        run(project, root, "two")
    state = ledger.receipt()
    assert state["source_charged"] == 2 * len(content)
    assert state["source_actual"] == len(content)
    assert manifest.read_bytes() == saved
    assert not (root / "corpora/example/acquisition.json").exists()
    with pytest.raises(ValueError, match="terminal"):
        run(project, root, "two")
    assert ledger.receipt() == state


def test_corrupt_or_unsafe_existing_paths_refuse_before_http(tmp_path, monkeypatch):
    _, project, root, _, calls, _ = fixture(tmp_path, monkeypatch)
    result = run(project, root)
    source = Path(result["source"]["snapshot_path"])
    data = next((source / "files").rglob("*.jsonl"))
    data.write_bytes(b"tampered")
    calls.clear()
    with pytest.raises(ValueError):
        run(project, root)
    assert not calls
    data.unlink()
    data.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError):
        run(project, root)
    assert not calls


def test_step_journal_is_stable_and_ledger_counters_durable(tmp_path, monkeypatch):
    _, project, _, ledger, _, _ = fixture(tmp_path, monkeypatch)
    ledger.bounded_storage = True
    ledger.reserve_metadata(10)
    journal = Path(str(ledger.path) + "-journal")
    inode = journal.stat().st_ino
    ledger.reserve_metadata(10)
    assert journal.stat().st_ino == inode
    assert TransportBudget(ledger.path, project).receipt()["metadata_charged"] == 20
    ledger.reserve_metadata(10)
    assert journal.stat().st_ino == inode
    state = ledger.receipt()
    with pytest.raises(ValueError, match="metadata cap"):
        ledger._mutate(lambda state: state.update(extra="x" * (16 * 1024**2)))
    assert ledger.receipt() == state
    assert ledger.path.stat().st_size <= acquisition_step.LEDGER_BYTES
    assert journal.stat().st_size <= acquisition_step.JOURNAL_BYTES


def test_reuse_only_step_copies_once_without_network(tmp_path, monkeypatch):
    root, _, recipe = _project(tmp_path, monkeypatch)
    config = yaml.safe_load(recipe.read_text())
    config["transport_budget"]["max_disk_bytes"] = ALLOWANCE
    _yaml(recipe, config)
    project = load_project(recipe)
    ledger = TransportBudget.initialize(
        root / "corpora/candidate/transport-budget.sqlite", project
    )
    monkeypatch.setattr(
        acquisition, "_acquire_hf", lambda *a: pytest.fail("reuse opened transport")
    )
    result = run(project, root, "cached_alpha")
    assert result["status"] == "source_verified"
    assert ledger.receipt()["source_actual"] == 0
    manifest = Path(result["source"]["snapshot_path"]) / "manifest.json"
    inode = manifest.stat().st_ino
    assert run(project, root, "cached_alpha")["source"] == result["source"]
    assert manifest.stat().st_ino == inode
    assert not (root / "corpora/candidate/acquisition.json").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "journal_symlink",
        "hot_journal",
        "ledger_oversize",
        "wal",
        "inventory_symlink",
        "wal_symlink",
    ],
)
def test_unsafe_metadata_refuses_before_sqlite_or_network(tmp_path, monkeypatch, kind):
    _, project, root, ledger, calls, _ = fixture(tmp_path, monkeypatch)
    if kind == "journal_symlink":
        Path(str(ledger.path) + "-journal").symlink_to(tmp_path / "missing")
    elif kind == "hot_journal":
        Path(str(ledger.path) + "-journal").write_bytes(b"hotdata!")
    elif kind == "ledger_oversize":
        monkeypatch.setattr(
            acquisition_step, "LEDGER_BYTES", ledger.path.stat().st_size - 1
        )
    elif kind == "wal_symlink":
        Path(str(ledger.path) + "-wal").symlink_to(tmp_path / "missing")
    elif kind == "wal":
        Path(str(ledger.path) + "-wal").touch()
    else:
        (ledger.path.parent / "alias").symlink_to(tmp_path / "missing")
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError):
        run(project, root)
    assert not calls and ledger.path.read_bytes() == before
    assert not (ledger.path.parent / "snapshots").exists()


def test_full_project_disk_cap_includes_existing_bytes(tmp_path, monkeypatch):
    recipe, project, root, ledger, calls, _ = fixture(tmp_path, monkeypatch)
    plan = run(project, root, plan_only=True)["admission"]
    config = yaml.safe_load(recipe.read_text())
    config["transport_budget"]["max_disk_bytes"] = (
        plan["native_peak_added_bytes"] + ledger.path.stat().st_size - 1
    )
    _yaml(recipe, config)
    project = load_project(recipe)
    other = tmp_path / "other"
    TransportBudget.initialize(
        other / "corpora/example/transport-budget.sqlite", project
    )
    with pytest.raises(ValueError, match="disk allowance"):
        run(project, other)
    assert not calls and not (other / "corpora/example/snapshots").exists()


def test_all_selected_shards_reserve_transport_together(tmp_path, monkeypatch):
    recipe, _, _, _, calls, _ = fixture(tmp_path, monkeypatch, second=False)
    source_path = recipe.parent / "sources/one.yaml"
    source = yaml.safe_load(source_path.read_text())
    shard = dict(source["acquisition"]["bounded_shards"][0])
    shard["path"] = "another.json.gz"
    source["acquisition"]["bounded_shards"].append(shard)
    _yaml(source_path, source)
    config = yaml.safe_load(recipe.read_text())
    config["transport_budget"]["max_transfers"] = 1
    _yaml(recipe, config)
    project = load_project(recipe)
    other = tmp_path / "other"
    ledger = TransportBudget.initialize(
        other / "corpora/example/transport-budget.sqlite", project
    )
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="transfer count"):
        run(project, other)
    assert not calls and ledger.path.read_bytes() == before


def test_binding_change_and_missing_ledger_refuse(tmp_path, monkeypatch):
    recipe, project, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="initialize"):
        run(project, tmp_path / "absent")
    source_path = recipe.parent / "sources/two.yaml"
    source = yaml.safe_load(source_path.read_text())
    source["revision"] = "b" * 40
    _yaml(source_path, source)
    with pytest.raises(ValueError, match="binding changed"):
        run(load_project(recipe), tmp_path / "work")
    assert not calls


def test_no_selector_options_and_active_scratch_refuse(tmp_path, monkeypatch):
    _, project, root, _, calls, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="require --source-id"):
        acquire(project, root, max_new_bytes=ALLOWANCE)
    from sparselab.corpus import bounded_metadata

    token = bounded_metadata._ACTIVE_STORE.set(object())
    try:
        with pytest.raises(ValueError, match="without metadata scratch"):
            run(project, root)
    finally:
        bounded_metadata._ACTIVE_STORE.reset(token)
    assert not calls


def test_unsupported_platform_refuses_before_work(tmp_path, monkeypatch):
    _, project, root, ledger, calls, _ = fixture(tmp_path, monkeypatch)
    before = ledger.path.read_bytes()
    monkeypatch.delattr(acquisition_step.os, "statvfs")
    with pytest.raises(ValueError, match="POSIX filesystem"):
        run(project, root)
    assert not calls and ledger.path.read_bytes() == before


def test_reuse_admission_only_credits_the_exact_pinned_origin(tmp_path, monkeypatch):
    import hashlib
    import json
    import shutil

    root, _, recipe = _project(tmp_path, monkeypatch)
    config = yaml.safe_load(recipe.read_text())
    config["transport_budget"]["max_disk_bytes"] = ALLOWANCE
    _yaml(recipe, config)
    project = load_project(recipe)
    ledger = TransportBudget.initialize(
        root / "corpora/candidate/transport-budget.sqlite", project
    )
    effect = project.config.source_effects[0]
    origin = acquisition._retained_origin(root, effect)
    manifest = json.loads((origin / "manifest.json").read_text())
    # A different, internally valid historical snapshot of the same declaration
    # must not count as the exact source_effects origin already copied locally.
    candidate = tmp_path / "candidate"
    shutil.copytree(origin, candidate)
    row = manifest["files"][0]
    body = b'{"text":"different historical content"}\n'
    (candidate / "files" / row["path"]).write_bytes(body)
    row.update(size=len(body), sha256=hashlib.sha256(body).hexdigest())
    manifest["snapshot_sha256"] = acquisition._digest(
        {key: manifest[key] for key in ("declaration_sha256", "adapter", "files")}
    )
    acquisition._write_json(candidate / "manifest.json", manifest)
    destination = (
        ledger.path.parent
        / "snapshots"
        / effect.source_id
        / manifest["snapshot_sha256"]
    )
    destination.parent.mkdir(parents=True)
    candidate.rename(destination)
    acquisition.verify_snapshot(destination)
    plan = run(project, root, effect.source_id, plan_only=True)["admission"]
    original = json.loads((origin / "manifest.json").read_text())
    expected = (
        2
        * (
            sum(row["size"] for row in original["files"])
            + (origin / "manifest.json").stat().st_size
        )
        + acquisition_step.LEDGER_BYTES
        + acquisition_step.JOURNAL_BYTES
    )
    assert (
        plan["existing_snapshot"] is None and plan["native_new_path_bytes"] == expected
    )
    result = run(project, root, effect.source_id)
    assert result["source"]["snapshot_sha256"] == effect.snapshot_sha256
    assert destination.exists()


def test_compact_finalization_rebuilds_stale_lock_from_saved_snapshots(
    tmp_path, monkeypatch
):
    recipe, project, root, _ledger, calls, _content = fixture(tmp_path, monkeypatch)
    run(project, root, "one")
    run(project, root, "two")
    complete = acquire(project, root, require_existing=True)
    lock = root / "corpora/example/acquisition.json"
    stale = dict(complete, sources={"one": complete["sources"]["one"]})
    lock.write_text(json.dumps(stale))
    before = list(calls)
    summary = _cli(monkeypatch, root, recipe, "--require-existing")
    assert summary["source_count"] == 2
    assert verify_acquisition(project, root) == complete
    assert calls == before
