"""Prospective file rights survive acquisition, training selection and publication."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from sparselab.config.loading import load_config
from sparselab.corpus.acquisition import acquire
from sparselab.corpus.export import export_release, verify_release_export
from sparselab.corpus.measurement import measure_source_rights
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import Project, load_project, source_declaration_payload
from sparselab.corpus.publication import publication_manifest
from sparselab.corpus.release import describe, freeze, verify_release

PROJECT = Path(__file__).resolve().parents[1] / "corpora/devmind-sample-v0/corpus.yaml"


def _prospective(tmp_path: Path) -> Project:
    original = load_project(PROJECT)
    text = tmp_path / "outside-library-secret.txt"
    text.write_text(
        "# SPDX-License-Identifier: GPL-3.0-only\nA differently licensed passage.\n"
    )
    nested = tmp_path / "external-passage.txt"
    nested.write_text("# SPDX-License-Identifier: MIT\nA third party review passage.\n")
    spec = original.model_dump(mode="json")
    for source in spec["sources"]:
        source["schema_version"] = 2
        source.pop("redistribution")
        source["rights"] = {
            "training_eligibility": "eligible",
            "redistribution_mode": "metadata_reconstruction_only",
            "spdx_expression": source["license"],
            "license_references": ["https://example.org/notice"],
            "notices": ["Keep the upstream copyright and license notices"],
        }
        if source["id"] == "sample_docs":
            source["acquisition"]["files"].extend(
                [
                    {"path": str(text), "name": "different-license.txt"},
                    {"path": str(nested), "name": "vendor/external.txt"},
                ]
            )
    spec["release"]["schema_version"] = 2
    spec["release"]["publication_mode"] = "metadata_reconstruction_only"
    return Project.model_validate(spec)


def test_prospective_release_preserves_file_decisions_and_no_source_bytes(
    tmp_path: Path,
) -> None:
    project = _prospective(tmp_path)
    acquire(project, tmp_path)
    release = freeze(build(project, tmp_path, offline=True), tmp_path)
    verify_release(release)
    report = json.loads((release / "license-report.json").read_text())
    files = {item["path"]: item for item in report["files"]}
    assert (
        files["different-license.txt"]["rights"]["training_eligibility"]
        == "eligible_with_obligations"
    )
    assert (
        files["different-license.txt"]["rights"]["detected_spdx_expression"]
        == "GPL-3.0-only"
    )
    assert (
        files["vendor/external.txt"]["rights"]["training_eligibility"]
        == "review_required"
    )
    assert report["training_eligibility"]["review_required"]["files"] == 1
    documents = [
        json.loads(line)
        for line in (release / "documents.jsonl").read_text().splitlines()
    ]
    assert any(item["license"] == "GPL-3.0-only" for item in documents)
    spans = [
        json.loads(line) for line in (release / "spans.jsonl").read_text().splitlines()
    ]
    assert not any(item["raw_path"] == "vendor/external.txt" for item in spans)
    assert not any("third party review passage" in item["text"] for item in documents)
    pub = publication_manifest(release)
    assert pub["publication_mode"] == "metadata_reconstruction_only"
    assert pub["sources"]
    serialized = json.dumps(pub)
    assert "differently licensed passage" not in serialized
    assert "outside-library-secret.txt" not in serialized
    assert "GPL-3.0-only" in serialized
    observed_cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "sparselab",
            "--work-dir",
            str(tmp_path),
            "corpus",
            "publication",
            str(release),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(observed_cli.stdout) == pub
    exported = export_release(
        release, "lm", Path("configs/runtime_smoke_cpu.yaml"), 300, tmp_path
    )
    sidecar = json.loads((exported / "export.json").read_text())
    assert sidecar["publication_mode"] == "metadata_reconstruction_only"
    assert sidecar["weight_license_status"] == "separate_analysis_required"
    assert "GPL-3.0-only" in load_config(exported / "run.yaml").dataset.license
    assert (
        verify_release_export(load_config(exported / "run.yaml").dataset)["release_id"]
        == release.name
    )
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "A": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    totals = measure_source_rights(release, tokenizer_path)
    assert totals["eligible_with_obligations"]["source_tokens"] > 0
    observed = describe(release, tokenizer=tokenizer_path)
    assert (
        observed["report"]["rights"]["training_eligibility"][
            "eligible_with_obligations"
        ]["source_tokens"]
        == totals["eligible_with_obligations"]["source_tokens"]
    )


def test_v3_private_research_freeze_and_public_manifest_remain_separate(
    tmp_path: Path,
) -> None:
    spec = _prospective(tmp_path).model_dump(mode="json")
    for source in spec["sources"]:
        source["schema_version"] = 3
        source["explicit_training_restriction"] = "none_found"
    spec["release"]["schema_version"] = 3
    spec["release"]["training_use_policy"] = "allowed_unless_explicitly_prohibited"
    project = Project.model_validate(spec)
    acquire(project, tmp_path)
    release = freeze(build(project, tmp_path, offline=True), tmp_path)
    report = json.loads((release / "license-report.json").read_text())
    assert report["schema_version"] == 3
    assert report["training_use_policy"] == "allowed_unless_explicitly_prohibited"
    assert all(
        item["schema_version"] == 3
        for item in [
            json.loads(line)
            for line in (release / "documents.jsonl").read_text().splitlines()
        ]
    )
    public = publication_manifest(release)
    assert public["training_use_policy"] == report["training_use_policy"]
    assert all(
        source["explicit_training_restriction"] == "none_found"
        for source in public["sources"]
    )
    assert "A differently licensed passage." not in json.dumps(public)


def test_conflicting_duplicate_rights_fail_closed(tmp_path: Path) -> None:
    spec = _prospective(tmp_path).model_dump(mode="json")
    original = next(
        source for source in spec["sources"] if source["id"] == "sample_docs"
    )
    conflicting = json.loads(json.dumps(original))
    conflicting["id"] = "second_license_for_same_text"
    conflicting["canonical_uri"] = "sparselab://alternate-owner/guide.md"
    conflicting["license"] = "GPL-3.0-only"
    conflicting["rights"]["spdx_expression"] = "GPL-3.0-only"
    conflicting["acquisition"]["files"] = [original["acquisition"]["files"][0]]
    spec["sources"].append(conflicting)
    project = Project.model_validate(spec)
    acquire(project, tmp_path)
    with pytest.raises(
        ValueError, match="duplicate source text has incompatible rights"
    ):
        build(project, tmp_path, offline=True)


def test_historical_receipt_serialization_is_unchanged() -> None:
    source = load_project(PROJECT).sources[0]
    declaration = source_declaration_payload(source)
    assert "rights" not in declaration
    assert declaration["redistribution"] == source.redistribution
