"""Exercise all Corpus Forge BPE candidates against a real frozen local release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from sparselab.config.loading import load_config
from sparselab.corpus.acquisition import acquire
from sparselab.corpus.export import export_release
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import load_project
from sparselab.corpus.release import freeze
from sparselab.corpus.tokenizer_bakeoff import GROUPS, bakeoff
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer, verify_tokenizer_artifact
from sparselab.experiments.artifacts import verify_artifact
from sparselab.experiments.plan import Artifact


def _project(
    tmp_path: Path, *, release_schema: int = 1, declaration_schema: int = 1
) -> tuple[Path, Path]:
    project_dir = tmp_path / "project"
    sources = project_dir / "sources"
    sources.mkdir(parents=True)
    references = []
    assignments = {}
    for split in ("train", "validation"):
        for group in GROUPS:
            identifier = f"{group}_{split}"
            source_path = sources / f"{identifier}.yaml"
            text_path = sources / f"{identifier}.txt"
            words = (
                f"{group}_{index:05x}_{(index * 7919 + GROUPS.index(group) * 104729) % 1048576:05x}"
                for index in range(24000 if split == "train" else 24)
            )
            passage = " ".join(words)
            text_path.write_text(
                f"{group} {split}: {passage} {passage}\n"
                if split == "train"
                else f"{group} {split}: {passage}\n",
                encoding="utf-8",
            )
            source = {
                "schema_version": release_schema,
                "id": identifier,
                "kind": "local",
                "canonical_uri": f"fixture:{identifier}",
                "revision": "v1",
                "license": "MIT",
                "domains": ["fixture"],
                "document_kinds": [group],
                "source_family": identifier,
                "acquisition": {
                    "files": [
                        {
                            "path": f"sources/{identifier}.txt",
                            "name": f"{identifier}.txt",
                        }
                    ],
                    "max_bytes": 2000000,
                },
            }
            if release_schema == 1:
                source["redistribution"] = "redistributable"
            else:
                source["rights"] = {
                    "training_eligibility": "eligible",
                    "redistribution_mode": "metadata_reconstruction_only",
                    "spdx_expression": "MIT",
                    "license_references": ["https://example.org/notice"],
                    "notices": ["Keep upstream copyright and license notices"],
                }
                if release_schema == 3:
                    source["explicit_training_restriction"] = "none_found"
            source_path.write_text(yaml.safe_dump(source))
            references.append(f"sources/{identifier}.yaml")
            assignments[identifier] = split
    (project_dir / "transforms").mkdir()
    (project_dir / "transforms/lm.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "lm",
                "version": "1",
                "kind": "lm_text",
                "parameters": {},
                "inputs": [],
            }
        )
    )
    (project_dir / "splits.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "unit": "source_document_family",
                "family_key": "source_family",
                "assignments": assignments,
            }
        )
    )
    release_spec = {
        "schema_version": release_schema,
        "mixture": {"fixture": 1.0},
        "lm": {"selected": True, "training_splits": ["train", "validation"]},
        "chat": {"selected": False, "training_splits": []},
    }
    if release_schema in (2, 3):
        release_spec["publication_mode"] = "metadata_reconstruction_only"
    if release_schema == 3:
        release_spec["training_use_policy"] = "allowed_unless_explicitly_prohibited"
    (project_dir / "release.yaml").write_text(yaml.safe_dump(release_spec))
    (project_dir / "corpus.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "tokenizer-bakeoff-fixture",
                "sources": sorted(references),
                "transforms": ["transforms/lm.yaml"],
                "splits": "splits.yaml",
                "release": "release.yaml",
            }
        )
    )
    project = load_project(project_dir / "corpus.yaml")
    workspace = tmp_path / "workspace"
    acquire(project, workspace)
    release = freeze(build(project, workspace, offline=True), workspace)
    declaration = project_dir / "tokenizer-bakeoff.yaml"
    declaration.write_text(
        yaml.safe_dump(
            {
                "schema_version": declaration_schema,
                "release_path": str(release),
                "vocab_sizes": [16384, 24576, 32768],
                "max_fit_bytes": 268435456,
                "eval_split": "validation",
                "eval_max_docs_per_group": 200,
                "groups": list(GROUPS),
                "near_best_ratio": 0.98,
            }
        )
    )
    return declaration, workspace


def test_selected_tokenizer_survives_export_and_artifact_verification(
    tmp_path: Path,
) -> None:
    declaration, workspace = _project(tmp_path, release_schema=3, declaration_schema=2)
    report_path = bakeoff(
        declaration, workspace / "tokenizer-bakeoff", work_root=workspace
    )
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    selected = Path(report["selected_tokenizer"])
    manifest_path = selected.with_name("tokenizer_manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    sample_path = report_path.with_name("fit.jsonl")
    sample_bytes = sample_path.read_bytes()
    validation_path = report_path.with_name("validation.jsonl")
    validation_bytes = validation_path.read_bytes()
    chosen = report["selected_vocab_size"]
    release_id = report["identity"]["release_id"]
    export = export_release(
        Path(report["identity"]["release_path"]),
        "lm",
        Path("configs/runtime_smoke_cpu.yaml"),
        chosen,
        workspace,
    )
    dataset = load_config(export / "run.yaml").dataset
    assert (
        verify_tokenizer_artifact(
            selected,
            source="local_text",
            revision=dataset.revision,
            vocab_size=chosen,
            dataset=dataset,
        )["revision"]
        == hashlib.sha256(sample_bytes).hexdigest()
    )
    artifact = Artifact(
        kind="tokenizer",
        version=1,
        producer="sparselab",
        identifier=str(chosen),
        sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
        path=str(selected),
    )
    assert verify_artifact(artifact, declaration)["identifier"] == artifact.identifier
    run = load_config(export / "run.yaml")
    run = run.model_copy(
        update={
            "tokenizer": run.tokenizer.model_copy(update={"path": selected}),
            "training": run.training.model_copy(
                update={"max_steps": 2, "max_tokens": 64}
            ),
            "optimizer": run.optimizer.model_copy(update={"warmup_steps": 1}),
            "logging": run.logging.model_copy(update={"root_dir": tmp_path / "runs"}),
        }
    )
    prepared = prepare_data(run, load_tokenizer(selected))
    assert prepared.manifest["corpus_export"]["release_id"] == release_id
    for other in report["candidates"]:
        if other["vocab_size"] != chosen:
            with pytest.raises(ValueError):
                verify_tokenizer_artifact(
                    Path(other["tokenizer_path"]),
                    source="local_text",
                    revision=other["manifest"]["revision"],
                    vocab_size=other["vocab_size"],
                )
            break
    with pytest.raises(ValueError):
        verify_artifact(
            artifact.model_copy(update={"identifier": "wrong"}), declaration
        )
    bad_dataset = dataset.model_copy(update={"revision": "f" * 64})
    with pytest.raises(ValueError):
        verify_tokenizer_artifact(
            selected,
            source="local_text",
            revision=bad_dataset.revision,
            vocab_size=chosen,
            dataset=bad_dataset,
        )
    for target, original, corrupted in (
        (
            report_path,
            report_bytes,
            report_bytes.replace(b'"fit_sample_sha256":"', b'"fit_sample_sha256":"f'),
        ),
        (sample_path, sample_bytes, sample_bytes + b'{"text":"injected"}\n'),
        (
            validation_path,
            validation_bytes,
            validation_bytes + b'{"text":"injected"}\n',
        ),
        (
            manifest_path,
            manifest_bytes,
            manifest_bytes.replace(b'"release_id":"', b'"release_id":"wrong-'),
        ),
    ):
        target.write_bytes(corrupted)
        try:
            with pytest.raises(ValueError):
                verify_artifact(artifact, declaration)
        finally:
            target.write_bytes(original)
    selected_bytes = selected.read_bytes()
    selected.write_bytes(selected_bytes + b"\n")
    with pytest.raises(ValueError):
        verify_artifact(artifact, declaration)
    selected.write_bytes(selected_bytes)
    from sparselab.staging import verify_prepared_inputs
    from sparselab.workers.bundles import (
        install_dispatch_bundle,
        materialize_dispatch_bundle,
        prepare_dispatch_bundle,
        verify_portable_corpus_binding,
    )

    dispatch = tmp_path / "dispatch"
    bundle = prepare_dispatch_bundle(run, dispatch)
    workspace.rename(tmp_path / "unavailable-original-inputs")
    worker = tmp_path / "worker"
    attachments = {
        f"assets/{item.sha256}": dispatch / item.relative_path for item in bundle.files
    }
    install_dispatch_bundle(worker, bundle, attachments, check_only=False)
    materialized = tmp_path / "materialized"
    materialize_dispatch_bundle(worker, bundle.digest(), materialized)
    verify_prepared_inputs(materialized, run)
    portable = verify_portable_corpus_binding(run, materialized / "assets")
    assert portable["tokenizer_bakeoff"]["release_id"] == release_id
    assert (
        portable["tokenizer_bakeoff"]["fit_sample_sha256"]
        == hashlib.sha256(sample_bytes).hexdigest()
        != run.dataset.revision
    )
    from sparselab.staging import stage
    from sparselab.training.checkpoints import CheckpointManager
    from sparselab.training.trainer import train

    staged = stage(
        run, tmp_path / "worker-stage", through="warmup", prepared_inputs=materialized
    )

    run_id = train(run, run_id="selected-offline", stage_bundle=staged)
    native_run = run.logging.root_dir / run_id
    checkpoint = CheckpointManager(native_run).load(
        native_run / "checkpoints/latest.json"
    )
    assert checkpoint.tokens_seen == 64
    assert verify_portable_corpus_binding(run, native_run) == portable
    portable_report = materialized / "assets/corpus/tokenizer-selection.json"
    portable_report.write_bytes(portable_report.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        verify_portable_corpus_binding(run, materialized / "assets")


def test_real_frozen_release_bakeoff_candidates_and_tamper(tmp_path: Path) -> None:
    declaration, workspace = _project(tmp_path)
    output = workspace / "tokenizer-bakeoff"
    report_path = bakeoff(declaration, output, work_root=workspace)
    report = json.loads(report_path.read_text())
    assert [row["vocab_size"] for row in report["candidates"]] == [16384, 24576, 32768]
    assert all(
        report["sample_receipt"]["fit"][group]["document_ids"] for group in GROUPS
    )
    assert all(
        report["sample_receipt"]["heldout"][group]["document_ids"] for group in GROUPS
    )
    assert {
        group: row["documents"]
        for group, row in report["candidates"][0]["per_kind"].items()
    } == dict.fromkeys(GROUPS, 1)
    assert report["selected_tokenizer"] in {
        candidate["tokenizer_path"] for candidate in report["candidates"]
    }
    assert bakeoff(declaration, output, work_root=workspace) == report_path
    manifest = output / "candidates/16384/tokenizer_manifest.json"
    manifest.write_text(
        manifest.read_text().replace('"release_id":"', '"release_id":"tampered-')
    )
    with pytest.raises(ValueError, match="manifest|mismatch|identity"):
        bakeoff(declaration, output, work_root=workspace)


@pytest.mark.parametrize("release_schema", [2, 3])
def test_pilot_bakeoff_on_verified_prospective_release(
    tmp_path: Path, release_schema: int
) -> None:
    declaration, workspace = _project(
        tmp_path, release_schema=release_schema, declaration_schema=2
    )
    output = workspace / "pilot-bakeoff"
    report_path = bakeoff(declaration, output, work_root=workspace)
    report = json.loads(report_path.read_text())
    assert [row["vocab_size"] for row in report["candidates"]] == [
        16384,
        24576,
        32768,
    ]
    assert report["selected_vocab_size"] == min(
        candidate["vocab_size"]
        for candidate in report["candidates"]
        if candidate["weighted_bytes_per_token"]
        >= 0.98 * max(row["weighted_bytes_per_token"] for row in report["candidates"])
    )
    assert all(
        report["sample_receipt"]["fit"][group]["document_ids"]
        and report["sample_receipt"]["heldout"][group]["document_ids"]
        for group in GROUPS
    )
    assert bakeoff(declaration, output, work_root=workspace) == report_path
    release = Path(yaml.safe_load(declaration.read_text())["release_path"])
    rights_report = release / "license-report.json"
    rights = json.loads(rights_report.read_text())
    rights["files"][0]["rights"]["training_eligibility"] = "ineligible"
    rights_report.write_text(json.dumps(rights))
    with pytest.raises(ValueError, match="tampered|mismatch|policy|identity"):
        bakeoff(declaration, output, work_root=workspace)


def test_pilot_rejects_legacy_and_unsupported_release(tmp_path: Path) -> None:
    declaration, workspace = _project(tmp_path, declaration_schema=2)
    output = workspace / "pilot-bakeoff"
    with pytest.raises(ValueError, match="prospective rights-tracked release"):
        bakeoff(declaration, output, work_root=workspace)
    assert not output.exists()
    release = Path(yaml.safe_load(declaration.read_text())["release_path"])
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["build_identity"]["release"]["schema_version"] = 4
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="release manifest identity mismatch"):
        bakeoff(declaration, output, work_root=workspace)
    assert not output.exists()
