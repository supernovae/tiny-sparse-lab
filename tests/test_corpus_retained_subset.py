"""Offline retained selection preserves row ancestry and refuses altered derivatives."""

import hashlib
import json
from pathlib import Path

import pytest
from test_corpus_acquisition import _bounded_hf_fixture, _mock_hf_stream

from sparselab.corpus.acquisition import _digest, acquire, verify_snapshot
from sparselab.corpus.project import Project, load_project
from sparselab.training.manifest import canonical_json, sha256_file


def fixture(tmp_path, monkeypatch):
    raw = b"".join(
        canonical_json({"text": f"Retained tiny fixture row {i}."}) + b"\n"
        for i in range(1000)
    )
    recipe = _bounded_hf_fixture(tmp_path, raw, "default/train.jsonl", rows=1000)
    spec = load_project(recipe).model_dump(mode="json")
    spec["sources"][0]["acquisition"].update(max_rows=1000, max_bytes=1024**2)
    spec["sources"][0]["acquisition"]["bounded_shards"][0].update(
        hash_modulus=1, hash_remainders=[0]
    )
    project = Project.model_validate(spec)
    _mock_hf_stream(monkeypatch, raw)
    parent = Path(
        acquire(project, tmp_path / "origin")["sources"]["one"]["snapshot_path"]
    )
    pin = verify_snapshot(parent)
    spec["config"]["id"] = "retained-derivative"
    spec["sources"][0]["acquisition"].update(
        max_rows=100,
        max_bytes=100000,
        retained_subset={
            "snapshot_path": str(parent),
            "snapshot_sha256": pin["snapshot_sha256"],
            "manifest_sha256": sha256_file(parent / "manifest.json"),
            "profile_id": "first32-fixture",
            "seed": 86,
        },
    )
    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_: pytest.fail("subset attempted network"),
    )
    return Project.model_validate(spec), parent


def test_public_subset_cold_replay_and_unchanged_parent(tmp_path, monkeypatch):
    project, parent = fixture(tmp_path, monkeypatch)
    before = sha256_file(parent / "manifest.json")
    lock = acquire(project, tmp_path / "work")
    child = Path(lock["sources"]["one"]["snapshot_path"])
    result = verify_snapshot(child)
    assert acquire(project, tmp_path / "work", offline=True) == lock
    spec = project.sources[0].acquisition.retained_subset
    original = verify_snapshot(parent)
    path = original["files"][0]["path"]
    expected = []
    for line in (parent / "files" / path).read_bytes().splitlines(keepends=True):
        row = json.loads(line)["_sparselab_source"]
        rank = hashlib.sha256(
            canonical_json(
                [
                    spec.profile_id,
                    spec.seed,
                    "one",
                    spec.snapshot_sha256,
                    row["source_row_index"],
                    row["source_row_sha256"],
                ]
            )
        ).hexdigest()
        if int(rank, 16) % 100 == 0:
            expected.append(line)
    assert 0 < len(expected) < 100
    assert (child / "files" / path).read_bytes() == b"".join(expected)
    assert result["retrieval"]["retained_subset"]["scanned_rows"] == 1000
    assert sha256_file(parent / "manifest.json") == before
    assert child.name != parent.name


@pytest.mark.parametrize(
    "fault", ["manifest", "policy", "symlink", "row_cap", "byte_cap"]
)
def test_subset_refusals_leave_no_lock_or_staging(tmp_path, monkeypatch, fault):
    project, parent = fixture(tmp_path, monkeypatch)
    spec = project.model_dump(mode="json")
    acquisition = spec["sources"][0]["acquisition"]
    if fault == "manifest":
        acquisition["retained_subset"]["manifest_sha256"] = "0" * 64
        match = "parent manifest digest"
    elif fault == "policy":
        acquisition["text_field"] = "code"
        match = "parent source policy"
    elif fault == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(parent, target_is_directory=True)
        acquisition["retained_subset"]["snapshot_path"] = str(alias)
        match = "unsafe snapshot origin"
    else:
        acquisition["max_rows" if fault == "row_cap" else "max_bytes"] = 1
        match = "row/byte cap"
    project = Project.model_validate(spec)
    work = tmp_path / "failed"
    with pytest.raises(ValueError, match=match):
        acquire(project, work)
    assert not (work / "corpora" / project.config.id / "acquisition.json").exists()
    assert not list(work.rglob(".acquire-*"))


def test_resealed_child_cannot_omit_selected_rows(tmp_path, monkeypatch):
    project, _ = fixture(tmp_path, monkeypatch)
    lock = acquire(project, tmp_path / "work")
    child = Path(lock["sources"]["one"]["snapshot_path"])
    manifest = json.loads((child / "manifest.json").read_text())
    file = child / "files" / manifest["files"][0]["path"]
    lines = file.read_bytes().splitlines(keepends=True)
    file.write_bytes(b"".join(lines[1:]))
    manifest["files"][0].update(size=file.stat().st_size, sha256=sha256_file(file))
    manifest["retrieval"]["shards"][0]["selected_rows"].pop(0)
    manifest["snapshot_sha256"] = _digest(
        {
            k: manifest[k]
            for k in ("declaration_sha256", "adapter", "files", "retrieval")
        }
    )
    (child / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    with pytest.raises(ValueError, match="cold parent replay"):
        verify_snapshot(child, _staged=True)


def test_parent_changes_invalidate_cold_child(tmp_path, monkeypatch):
    project, parent = fixture(tmp_path, monkeypatch)
    lock = acquire(project, tmp_path / "work")
    child = Path(lock["sources"]["one"]["snapshot_path"])
    file = next((parent / "files").rglob("*.jsonl"))
    with file.open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(ValueError, match="snapshot byte mismatch"):
        verify_snapshot(child)


def test_subset_interruption_retains_parent_and_cleans_owned_staging(
    tmp_path, monkeypatch
):
    project, parent = fixture(tmp_path, monkeypatch)
    from sparselab.corpus import retained_subset

    original = retained_subset.records_from_path

    def interrupted(*args, **kwargs):
        for index, record in enumerate(original(*args, **kwargs)):
            if index == 500:
                raise KeyboardInterrupt
            yield record

    monkeypatch.setattr(retained_subset, "records_from_path", interrupted)
    with pytest.raises(KeyboardInterrupt):
        acquire(project, tmp_path / "work")
    assert (parent / "manifest.json").exists()
    assert not list((tmp_path / "work").rglob(".acquire-*"))
    assert not list((tmp_path / "work").rglob("acquisition.json"))


def test_long_selected_rows_are_reported_not_truncated(tmp_path, monkeypatch):
    project, _ = fixture(tmp_path, monkeypatch)
    spec = project.model_dump(mode="json")
    spec["sources"][0]["acquisition"]["retained_subset"]["max_record_bytes"] = 1
    with pytest.raises(ValueError, match="selected no rows"):
        acquire(Project.model_validate(spec), tmp_path / "work")


def test_nested_subset_refuses_before_recursive_verification(tmp_path, monkeypatch):
    project, _ = fixture(tmp_path, monkeypatch)
    lock = acquire(project, tmp_path / "first")
    child = Path(lock["sources"]["one"]["snapshot_path"])
    spec = project.model_dump(mode="json")
    spec["sources"][0]["acquisition"]["retained_subset"].update(
        snapshot_path=str(child),
        snapshot_sha256=child.name,
        manifest_sha256=sha256_file(child / "manifest.json"),
    )
    with pytest.raises(ValueError, match="nested retained subsets"):
        acquire(Project.model_validate(spec), tmp_path / "second")


def test_parent_scratch_rows_released_in_shared_metadata_store(tmp_path, monkeypatch):
    from sparselab.corpus.bounded_metadata import metadata_operation
    from sparselab.corpus.retained_subset import derive_subset

    project, _ = fixture(tmp_path, monkeypatch)
    with metadata_operation(tmp_path) as store:
        retained = store.new_rows()
        store.append(retained, {"keep": True})
        before = store.connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
        first = derive_subset(project.sources[0])
        assert derive_subset(project.sources[0]) == first
        assert (
            store.connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
            == before
        )
        assert list(retained) == [{"keep": True}]


def test_aggregate_caps_refuse_before_acquisition(tmp_path, monkeypatch):
    project, _ = fixture(tmp_path, monkeypatch)
    spec = project.model_dump(mode="json")
    spec["sources"][0]["acquisition"]["max_rows"] = 20000
    import copy

    second = copy.deepcopy(spec["sources"][0])
    second["id"] = "second"
    spec["sources"].append(second)
    spec["config"]["sources"].append("sources/second.yaml")
    with pytest.raises(ValueError, match="aggregate selection caps"):
        acquire(Project.model_validate(spec), tmp_path / "work")
    assert not (tmp_path / "work").exists()


def test_existing_derivative_lock_is_not_replaced(tmp_path, monkeypatch):
    project, _ = fixture(tmp_path, monkeypatch)
    work = tmp_path / "work"
    acquire(project, work)
    lock = work / "corpora" / project.config.id / "acquisition.json"
    before = lock.read_bytes()
    spec = project.model_dump(mode="json")
    spec["sources"][0]["acquisition"]["retained_subset"]["seed"] += 1
    with pytest.raises(ValueError):
        acquire(Project.model_validate(spec), work)
    assert lock.read_bytes() == before


def test_warm_proof_invalidates_when_parent_changes(tmp_path, monkeypatch):
    from sparselab.verification_proofs import ProofStore

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    project, parent = fixture(tmp_path, monkeypatch)
    work = tmp_path / "work"
    child = Path(acquire(project, work)["sources"]["one"]["snapshot_path"])
    store = ProofStore(tmp_path)
    expected = verify_snapshot(
        child, proof_store=store, verification_mode="verified_reuse"
    )
    assert (
        verify_snapshot(child, proof_store=store, verification_mode="verified_reuse")
        == expected
    )
    hits = store.hits
    assert hits > 0, store.diagnostics()
    file = next((parent / "files").rglob("*.jsonl"))
    with file.open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(ValueError, match="snapshot byte mismatch"):
        verify_snapshot(child, proof_store=store, verification_mode="verified_reuse")
    assert store.hits == hits


@pytest.mark.parametrize(
    "change", [None, "live", "mixed", "aggregate", "unbound", "wrong_pin"]
)
def test_supervised_subset_is_offline_bound_and_zero_model(
    tmp_path, monkeypatch, change
):
    import yaml
    from test_attempt_packet_dispatch import _fixture
    from test_preparation_tokenizer_dispatch import run_words

    from sparselab.corpus.acquisition import verify_acquisition
    from sparselab.training.attempt_budget import AttemptBudget, AttemptBudgetError

    paths = _fixture(tmp_path, initialize=False)
    root = paths["root"]
    project, _ = fixture(root, monkeypatch)
    recipe = root / "subset"
    (recipe / "sources").mkdir(parents=True)
    payload = project.model_dump(mode="json")
    source = payload["sources"][0]
    if change == "live":
        source["acquisition"].pop("retained_subset")
    (recipe / "sources/source.yaml").write_text(yaml.safe_dump(source))
    for name in ("splits", "release"):
        (recipe / (name + ".yaml")).write_text(yaml.safe_dump(payload[name]))
    config = payload["config"]
    config.update(
        sources=["sources/source.yaml"], splits="splits.yaml", release="release.yaml"
    )
    if change in {"mixed", "aggregate"}:
        import copy

        second = copy.deepcopy(source)
        second["id"] = "second"
        if change == "mixed":
            second["acquisition"].pop("retained_subset")
        else:
            source["acquisition"]["max_rows"] = 20000
            second["acquisition"]["max_rows"] = 20000
            (recipe / "sources/source.yaml").write_text(yaml.safe_dump(source))
        (recipe / "sources/second.yaml").write_text(yaml.safe_dump(second))
        config["sources"].append("sources/second.yaml")
        config["sources"].sort()
    target = recipe / "project.yaml"
    target.write_text(yaml.safe_dump(config))
    contract = json.loads(paths["contract"].read_text())
    contract.update(
        preparation_only=True,
        offline_retained_sources_only=True,
        acquisition_project_sha256=None
        if change == "unbound"
        else "0" * 64
        if change == "wrong_pin"
        else sha256_file(target),
    )
    paths["contract"].write_text(json.dumps(contract))
    AttemptBudget.create_contract(
        paths["ledger"],
        contract_path=paths["contract"],
        expected_sha256=sha256_file(paths["contract"]),
    )
    if change is not None:
        with pytest.raises(AttemptBudgetError):
            run_words(paths, "corpus", "acquire", str(target))
        assert AttemptBudget(paths["ledger"]).status()["reservations"] == []
    else:
        assert run_words(paths, "corpus", "acquire", str(target)) == 0
        verified = verify_acquisition(load_project(target), root)
        assert verified["project_id"] == project.config.id
        status = AttemptBudget(paths["ledger"]).status()
        assert len(status["reservations"]) == 1
        assert status["reservations"][0]["updates"] == 0
        assert not list(root.rglob("transport-budget.sqlite"))
