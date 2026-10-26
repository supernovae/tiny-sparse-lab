"""Public local preparation uses native outputs without replaying admission.

Only the historical executor-adoption boundary is substituted: this tiny fixture
does not pretend to be a previously executed production job. Rendering, frozen
family joins, deduplication, publication and native completion storage are real.
"""

from types import SimpleNamespace

import pytest
import yaml
from test_corpus_row_qualification import declared_fixture

from sparselab.corpus import completed_preparation as completed
from sparselab.corpus import pipeline
from sparselab.corpus.admission_inspection_cache import capture
from sparselab.corpus.project import SourceDeclaration, load_project
from sparselab.corpus.release import freeze, verify_release
from sparselab.corpus.rights import verify_record_admission
from sparselab.corpus.split_freeze import finalize_family_inventory
from sparselab.training.manifest import canonical_json, sha256_file


def fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    root = tmp_path / "work"
    snapshots = root / "corpora/completed-fixture/snapshots"
    source_rows = [
        {
            "text": f"Generated explanation number {i} with enough clear words for native parsing. "
            * 3,
            **({"seed": {"id": i}} if i != 1 else {}),
        }
        for i in range(12)
    ]
    admission, source, snapshot, raw = declared_fixture(snapshots, rows=source_rows)
    admission["schema_version"] = 2
    source["acquisition"]["max_rows"] = 12
    source["acquisition"]["bounded_shards"][0]["max_scanned_rows"] = 12
    model = SourceDeclaration.model_validate(source)
    snapshot["declaration"] = source
    folder = snapshots / model.id / snapshot["snapshot_sha256"]
    (folder / "manifest.json").write_bytes(canonical_json(snapshot) + b"\n")
    admitted = verify_record_admission(
        admission, {model.id: source}, {model.id: snapshot}, snapshots
    )[model.id, snapshot["files"][0]["path"]]
    parsed = pipeline._records_for_file(
        raw,
        snapshot["files"][0]["path"],
        model,
        snapshot["snapshot_sha256"],
        file_rights=admitted["rights"],
        record_admission=admitted,
        normalizer_version="normalizer-structure-v3",
    )
    docs = sorted((row[0] for row in parsed), key=lambda doc: doc["document_id"])
    assert len(docs) == 11
    recipe = root / "recipe"
    recipe.mkdir()

    def write_yaml(name, value):
        (recipe / name).parent.mkdir(parents=True, exist_ok=True)
        (recipe / name).write_text(yaml.safe_dump(value), encoding="utf-8")

    write_yaml("sources/source.yaml", source)
    write_yaml(
        "transforms/lm.yaml",
        {
            "id": "fixture_lm",
            "kind": "lm_text",
            "version": "1",
            "parameters": {},
            "inputs": [model.id],
        },
    )
    (recipe / "admission.json").write_bytes(canonical_json(admission) + b"\n")
    write_yaml(
        "release.yaml",
        {
            "schema_version": 2,
            "publication_mode": "metadata_reconstruction_only",
            "mixture": {"general_prose": 1.0},
            "lm": {"selected": True, "training_splits": ["train", "validation"]},
            "chat": {"selected": False, "training_splits": []},
            "record_admission": {
                "path": "admission.json",
                "sha256": sha256_file(recipe / "admission.json"),
            },
            "normalizer": "normalizer-structure-v3",
        },
    )
    assignments = {
        doc["document_id"]: ("validation" if i == 0 else "test" if i == 1 else "train")
        for i, doc in enumerate(docs)
    }
    write_yaml(
        "splits.yaml",
        {"schema_version": 1, "unit": "document", "assignments": assignments},
    )
    write_yaml(
        "project.yaml",
        {
            "schema_version": 1,
            "id": "completed-fixture",
            "sources": ["sources/source.yaml"],
            "transforms": ["transforms/lm.yaml"],
            "splits": "splits.yaml",
            "release": "release.yaml",
        },
    )
    families = [
        {
            "document_id": doc["document_id"],
            "family_id": "reviewed-" + doc["document_id"],
            "split": assignments[doc["document_id"]],
            "stratum": "general_prose",
            "content_sha256": doc["content_sha256"],
        }
        for doc in docs
    ]
    candidate = recipe / "splits.family-candidates.jsonl"
    candidate.write_bytes(b"".join(canonical_json(row) + b"\n" for row in families))
    inventory = recipe / "inventory.jsonl"
    inventory.write_bytes(b"fixture historical inventory\n")
    declaration = recipe / "completion.json"
    declaration.write_bytes(
        canonical_json(
            {
                "project": {"path": str(recipe / "project.yaml")},
                "inventory": {"path": str(inventory)},
            }
        )
        + b"\n"
    )
    (recipe / "splits.receipt.json").write_bytes(
        canonical_json(
            {
                "schema_version": 1,
                "inventory_sha256": sha256_file(inventory),
                "cluster_sha256": "e" * 64,
                "family_counts": {},
                "split_sha256": sha256_file(recipe / "splits.yaml"),
                "family_candidate_sha256": sha256_file(candidate),
                "admission_sha256": sha256_file(recipe / "admission.json"),
                "preparation_reuse": {
                    "declaration": {
                        "path": str(declaration),
                        "sha256": sha256_file(declaration),
                    }
                },
            }
        )
        + b"\n"
    )
    from sparselab.corpus import acquisition, split_inventory_adoption

    monkeypatch.setattr(
        split_inventory_adoption,
        "adopt_completed_inventory",
        lambda *_: SimpleNamespace(
            provenance={"fixture": "historical executor boundary substituted"},
            dependencies=capture([declaration, inventory]),
        ),
    )
    monkeypatch.setattr(
        acquisition,
        "verify_acquisition",
        lambda *_args, **_kwargs: {
            "sources": {
                model.id: {
                    "declaration_sha256": acquisition.declaration_sha256(model),
                    "snapshot_sha256": snapshot["snapshot_sha256"],
                    "snapshot_path": str(folder),
                }
            }
        },
    )
    monkeypatch.setattr(
        acquisition, "verify_snapshot", lambda *_args, **_kwargs: snapshot
    )
    return root, recipe, load_project(recipe / "project.yaml")


def test_public_completed_build_release_and_family_export_match_native(
    tmp_path, monkeypatch
):
    root, recipe, project = fixture(tmp_path, monkeypatch)
    ordinary = pipeline.build(project, root, offline=True)
    from sparselab.corpus import acquisition, measurement

    def replay(*_args, **_kwargs):
        pytest.fail("completed preparation replayed source-body verification or census")

    monkeypatch.setattr(acquisition, "verify_acquisition", replay)
    monkeypatch.setattr(acquisition, "verify_snapshot", replay)
    monkeypatch.setattr(pipeline, "verify_record_admission", replay)
    monkeypatch.setattr(measurement, "summarize_release", replay)
    local = pipeline.build(
        project, root, offline=True, reuse_completed_preparation=True
    )
    for name in (
        "documents.jsonl",
        "spans.jsonl",
        "lineage.jsonl",
        "sources.json",
        "license-report.json",
        "rejected.jsonl",
        "audit.json",
        "splits.json",
        "lm/train.jsonl",
        "lm/validation.jsonl",
        "lm/test.jsonl",
        "stages/fixture_lm.jsonl",
    ):
        assert (local / name).read_bytes() == (ordinary / name).read_bytes(), name
    release = freeze(local, root)
    assert verify_release(release)["build_id"] == local.name
    monkeypatch.setattr(completed, "_admissions", replay)
    monkeypatch.setattr(completed, "_prepared_source", replay)
    assert pipeline.build(project, root, reuse_completed_preparation=True) == local
    from sparselab.corpus import split_freeze

    monkeypatch.setattr(split_freeze, "RowIndex", replay)
    exported = recipe / "final-families.jsonl"
    report = finalize_family_inventory(release, recipe / "splits.yaml", exported)
    assert exported.read_bytes() == (release / "family-inventory.jsonl").read_bytes()
    assert report["kept_documents"] == 11
    (release / "lm/train.jsonl").write_bytes(b"corrupted\n")
    with pytest.raises(ValueError, match="tampered artifact"):
        verify_release(release)


def test_interrupted_public_build_reuses_committed_shard_rendering(
    tmp_path, monkeypatch
):
    root, _recipe, project = fixture(tmp_path, monkeypatch)
    original = completed.StreamBuildRows.create
    monkeypatch.setattr(
        completed.StreamBuildRows,
        "create",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("fixture interruption")),
    )
    with pytest.raises(RuntimeError, match="fixture interruption"):
        pipeline.build(project, root, reuse_completed_preparation=True)
    monkeypatch.setattr(completed.StreamBuildRows, "create", original)
    monkeypatch.setattr(
        completed,
        "_prepared_source",
        lambda *_: pytest.fail("committed source rendering replayed"),
    )
    assert pipeline.build(project, root, reuse_completed_preparation=True).is_dir()


def test_unique_content_floor_excludes_eos(tmp_path, monkeypatch):
    from test_corpus_mixture import _fixture

    from sparselab.corpus import mixture

    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    spec = yaml.safe_load(declaration.read_text())
    spec["unique_train_content_floors"] = {
        "general": 9,
        "explanatory": 6,
        "incident": 3,
    }
    declaration.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(ValueError, match=r"8 available, 9 required \(EOS excluded\)"):
        mixture.materialize_mixture(declaration, tmp_path / "mixture")


def test_completed_mixture_counts_content_once_and_reuses_execution(
    tmp_path, monkeypatch
):
    from test_kml_50m_preparation_sequence import _authenticated_fixture_tokenizer

    from sparselab.corpus import mixture

    root, recipe, project = fixture(tmp_path, monkeypatch)
    built = pipeline.build(project, root, reuse_completed_preparation=True)
    release = freeze(built, root)
    tokenizer_config, _ = _authenticated_fixture_tokenizer(root, release)
    declaration = recipe / "mixture.yaml"
    spec = {
        "schema_version": 1,
        "release_path": str(release),
        "tokenizer_config": str(tokenizer_config),
        "family_inventory": str(release / "family-inventory.jsonl"),
        "source_strata": {project.sources[0].id: "general_prose"},
        "target_quotas": {"general_prose": 20},
        "unique_train_content_floors": {"general_prose": 20},
        "seed": 42,
        "max_exposures": 1,
    }
    declaration.write_text(yaml.safe_dump(spec), encoding="utf-8")
    output = root / "mixture"
    receipt = mixture.materialize_mixture(declaration, output)
    assert receipt["available_unique_content_tokens"]["general_prose"] > 20
    assert receipt["unique_train_content_floors"] == {"general_prose": 20}
    monkeypatch.setattr(
        mixture, "_schedule", lambda *_: pytest.fail("completed schedule was replayed")
    )
    assert mixture.verify_mixture(declaration, output) == receipt
    (output / "train.tokens.jsonl").write_bytes(b"changed\n")
    with pytest.raises(
        ValueError, match="completed native mixture evidence unavailable"
    ):
        mixture.verify_mixture(declaration, output)
