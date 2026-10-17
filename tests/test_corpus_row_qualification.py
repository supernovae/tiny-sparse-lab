"""Offline source-declared qualification and original-row provenance fixtures."""

import hashlib
import json
from copy import deepcopy

import pytest
from test_corpus_record_admission import _fixture, _verify

from sparselab.corpus.pipeline import _records_for_file
from sparselab.corpus.project import SourceDeclaration, source_declaration_payload
from sparselab.corpus.row_qualification import (
    RowQualification,
    exceptions,
    family_hint,
    provenance,
)
from sparselab.corpus.split_inventory import _family_hint, _verified_hf_metadata
from sparselab.training.manifest import canonical_json


def code_policy():
    return RowQualification(
        profile="code",
        family_fields=("repo_name",),
        language_field="language",
        allowed_languages=("Python", "Rust"),
        license_field="license",
        license_labels={"mit": "MIT", "apache-2.0": "Apache-2.0"},
        repository_field="repo_name",
        path_field="path",
    )


def code_row():
    return {
        "code": "def value():\n    return 42\n",
        "repo_name": "fixture/example",
        "path": "src/example.py",
        "language": "Python",
        "license": "mit",
    }


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"license": None}, "license"),
        ({"license": "gpl-3.0"}, "license"),
        ({"language": "C++"}, "language"),
        ({"repo_name": ""}, "provenance"),
        ({"path": "vendor/example.py"}, "vendor"),
        ({"path": "../example.py"}, "unsafe"),
        ({"code": "# Automatically generated, do not edit\nprint(1)"}, "generated"),
        ({"code": "# SPDX-License-Identifier: GPL-3.0-only\nprint(1)"}, "conflicting"),
    ],
)
def test_code_qualification_narrows_permissive_repository_labels(change, reason):
    row = {**code_row(), **change}
    conflict, _ = exceptions(row, code_policy(), "code")
    assert conflict is not None and reason in conflict


def test_code_provenance_and_repository_grouping_preserve_attribution():
    row = code_row()
    policy = code_policy()
    assert exceptions(row, policy, "code") == (None, ())
    assert family_hint(row, policy) == family_hint(
        {**row, "path": "other.py", "code": "different"}, policy
    )
    assert provenance(row, policy)["attribution"] == {
        "repository": row["repo_name"],
        "path": row["path"],
        "license_label": "mit",
        "language": "Python",
    }


def test_synthetic_seed_provenance_is_required_and_hashed_without_copying_prompt():
    policy = RowQualification(
        profile="synthetic_text",
        family_fields=("seed",),
        synthetic_provenance_fields=("seed",),
    )
    row = {"text": "Generated response", "seed": {"dataset": "fixture", "id": 7}}
    assert exceptions(row, policy, "text") == (None, ())
    assert family_hint(row, policy) == family_hint(
        {**row, "text": "Another response"}, policy
    )
    evidence = provenance(row, policy)
    assert (
        evidence["profile"] == "synthetic_text"
        and len(evidence["synthetic_provenance_sha256"]) == 64
    )
    assert "Generated response" not in json.dumps(evidence)
    assert exceptions({"text": "Generated response"}, policy, "text")[0]


def test_absent_qualification_preserves_existing_source_serialization(tmp_path):
    _, _, _, _, source = _fixture(tmp_path)
    assert "qualification" not in source_declaration_payload(source)["acquisition"]


def declared_fixture(tmp_path):
    manifest, source, snapshot, raw, _ = _fixture(tmp_path)
    source["canonical_uri"] = "https://huggingface.co/datasets/fixture/synthetic"
    source["acquisition"]["qualification"] = {
        "profile": "synthetic_text",
        "family_fields": ["seed"],
        "synthetic_provenance_fields": ["seed"],
    }
    rows = [
        {
            "text": "Generated explanatory fixture text with enough words for native parsing. "
            * 3,
            "seed": {"id": 7},
        },
        {"text": "Missing provenance row remains quarantined. " * 3},
    ]
    original = json.loads(raw.splitlines()[0])
    entry = manifest["sources"][0]
    rendered = []
    selected = []
    for index, row in enumerate(rows):
        digest = hashlib.sha256(canonical_json(row)).hexdigest()
        envelope = {
            **original["_sparselab_source"],
            "source_row_index": index,
            "source_row_sha256": digest,
        }
        envelope.pop("id", None)
        rendered.append({**row, "_sparselab_source": envelope})
        selected.append({"source_row_index": index, "source_row_sha256": digest})
        entry["records"][index]["source_row_sha256"] = digest
    payload = b"".join(canonical_json(row) + b"\n" for row in rendered)
    digest = hashlib.sha256(payload).hexdigest()
    entry["sample_sha256"] = digest
    snapshot["declaration"] = source
    snapshot["files"][0].update(sha256=digest, size=len(payload))
    snapshot["retrieval"]["shards"][0]["selected_rows"] = selected
    path = (
        tmp_path
        / source["id"]
        / snapshot["snapshot_sha256"]
        / "files"
        / entry["sample_path"]
    )
    path.write_bytes(payload)
    return manifest, source, snapshot, payload


def test_declared_rows_cold_verify_parse_and_group_with_original_envelope(tmp_path):
    manifest, source, snapshot, raw = declared_fixture(tmp_path)
    admitted = _verify(manifest, source, snapshot, tmp_path)
    model = SourceDeclaration.model_validate(source)
    file = snapshot["files"][0]
    rights = admitted[model.id, file["path"]]
    docs = _records_for_file(
        raw,
        file["path"],
        model,
        snapshot["snapshot_sha256"],
        file_rights=rights["rights"],
        record_admission=rights,
    )
    assert len(docs) == 1
    doc = docs[0][0]
    assert doc["metadata"]["qualification_profile"] == "synthetic_text"
    metadata = _verified_hf_metadata(
        raw, file, snapshot["retrieval"], source["revision"], original_rows=True
    )
    assert _family_hint(model, doc, metadata[0]) == family_hint(
        metadata[0], model.acquisition.qualification
    )
    bad = deepcopy(manifest)
    bad["sources"][0]["records"][1]["decision"] = "qualify"
    with pytest.raises(ValueError, match="material exception"):
        _verify(bad, source, snapshot, tmp_path)


def test_declared_rows_do_not_bypass_acquired_digest_binding(tmp_path):
    manifest, source, snapshot, _raw = declared_fixture(tmp_path)
    manifest["sources"][0]["records"][0]["source_row_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance|receipt"):
        _verify(manifest, source, snapshot, tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"family_fields": ["repo_name", "path"]},
        {"license_labels": {"gpl": "GPL-3.0-only"}},
        {"allowed_languages": []},
        {"family_fields": ["_sparselab_source"]},
    ],
)
def test_code_policy_cannot_split_repository_or_omit_required_screen(change):
    payload = code_policy().model_dump(mode="json")
    with pytest.raises(ValueError):
        RowQualification.model_validate({**payload, **change})


def test_declared_draft_cold_verifies_qualified_and_quarantined_rows(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from sparselab.corpus.admission_draft import draft_admission_manifest
    from sparselab.corpus.rights import verify_record_admission

    root = tmp_path / "work"
    snapshot_root = root / "corpora" / "declared-fixture" / "snapshots"
    manifest, source, snapshot, _ = declared_fixture(snapshot_root)
    model = SourceDeclaration.model_validate(source)
    snapshot_dir = snapshot_root / model.id / snapshot["snapshot_sha256"]
    (snapshot_dir / "manifest.json").write_text(json.dumps(snapshot))
    monkeypatch.setattr(
        "sparselab.corpus.admission_draft.verify_acquisition",
        lambda *_: {"sources": {model.id: {"snapshot_path": str(snapshot_dir)}}},
    )
    project = SimpleNamespace(
        config=SimpleNamespace(id="declared-fixture"), sources=(model,)
    )
    policy = tmp_path / "policy.md"
    policy.write_text("Fixture reviewed policy only.\n")
    template = tmp_path / "policy.json"
    entry = manifest["sources"][0]
    template.write_text(
        json.dumps(
            {
                "policy_id": manifest["policy_id"],
                "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                "sources": [
                    {
                        key: entry[key]
                        for key in ("source_id", "license_label", "rights")
                    }
                ],
            }
        )
    )
    output = root / "draft.json"
    report = draft_admission_manifest(project, root, template, policy, output)
    assert report["counts"] == {"qualify": 1, "exclude": 0, "quarantine": 1}
    verify_record_admission(
        json.loads(output.read_text()),
        {model.id: source},
        {model.id: snapshot},
        snapshot_root,
    )
