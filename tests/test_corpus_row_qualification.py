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
        "license_expression": "MIT",
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


def declared_fixture(tmp_path, *, policy=None, rows=None):
    manifest, source, snapshot, raw, _ = _fixture(tmp_path)
    source["canonical_uri"] = "https://huggingface.co/datasets/fixture/synthetic"
    source["acquisition"]["qualification"] = policy or {
        "profile": "synthetic_text",
        "family_fields": ["seed"],
        "synthetic_provenance_fields": ["seed"],
    }
    rows = rows or [
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
    expected = provenance(
        json.loads(raw.splitlines()[0]), model.acquisition.qualification
    )
    assert doc["metadata"]["qualification_profile"] == "synthetic_text"
    assert doc["metadata"]["qualification_family_hint"] == expected["family_hint"]
    assert (
        doc["metadata"]["synthetic_provenance_sha256"]
        == expected["synthetic_provenance_sha256"]
    )
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


@pytest.mark.parametrize("profile", ["synthetic", "nested_web", "code"])
def test_declared_draft_cold_verifies_qualified_and_quarantined_rows(
    tmp_path, monkeypatch, profile
):
    from types import SimpleNamespace

    from sparselab.corpus.admission_draft import draft_admission_manifest
    from sparselab.corpus.rights import verify_record_admission

    root = tmp_path / "work"
    snapshot_root = root / "corpora" / "declared-fixture" / "snapshots"
    overrides = {}
    if profile == "nested_web":
        overrides = {
            "policy": {
                "profile": "web_text",
                "family_fields": ["metadata.url"],
                "language_field": "metadata.language",
                "allowed_languages": ["en"],
            },
            "rows": [
                {
                    "text": "Nested English fixture with enough content. " * 3,
                    "metadata": {
                        "url": "https://example.org/article",
                        "language": "en",
                    },
                },
                {
                    "text": "Missing nested provenance fixture. " * 3,
                    "metadata": {"language": "en"},
                },
            ],
        }
    if profile == "code":
        overrides = {
            "policy": code_policy().model_dump(mode="json"),
            "rows": [
                {**code_row(), "text": "Wrong undeclared field must not be inspected."},
                {**code_row(), "repo_name": ""},
            ],
        }
    manifest, source, snapshot, _ = declared_fixture(snapshot_root, **overrides)
    if profile == "code":
        source["acquisition"]["text_field"] = "code"
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
    assert json.loads(output.read_text())["sources"][0]["records"][0]["issues"] == []
    verify_record_admission(
        json.loads(output.read_text()),
        {model.id: source},
        {model.id: snapshot},
        snapshot_root,
    )
    # Follow the same drafted decisions into the native inspection candidate
    # reader, including a code row with a misleading undeclared text field.
    from sparselab.corpus.admission_inspection import _candidates

    candidates, consumed = _candidates(
        project,
        root,
        json.loads(output.read_text()),
        {"sources": {model.id: {"snapshot_path": str(snapshot_dir)}}},
        1024 * 1024,
    )
    assert consumed == snapshot["files"][0]["size"]
    assert len(candidates[model.id]) == 2
    if profile == "code":
        assert candidates[model.id][0]["raw"] == code_row()["code"]


@pytest.mark.parametrize("uri", ["project_gutenberg_filtered", "wikimedia_filtered"])
def test_legacy_sources_refuse_declared_policy_override(tmp_path, uri):
    from sparselab.corpus.rights import _admission_profile

    _, source, _, _, _ = _fixture(tmp_path)
    source["canonical_uri"] = "https://huggingface.co/datasets/common-pile/" + uri
    source["acquisition"]["qualification"] = {
        "profile": "web_text",
        "family_fields": ["id"],
    }
    with pytest.raises(ValueError, match="legacy HF row policies"):
        SourceDeclaration.model_validate(source)
    with pytest.raises(ValueError, match="legacy HF row policies"):
        _admission_profile(source)


def test_fineweb_nested_dictionary_fields_group_url_across_dumps():
    policy = RowQualification(
        profile="web_text",
        family_fields=("metadata.url",),
        language_field="metadata.language",
        allowed_languages=("en",),
    )
    row = {
        "text": "English source text",
        "metadata": {
            "url": "https://example.org/article",
            "language": "en",
            "dump": "first",
        },
    }
    assert exceptions(row, policy, "text") == (None, ())
    assert family_hint(row, policy) == family_hint(
        {**row, "metadata": {**row["metadata"], "dump": "second"}}, policy
    )
    assert (
        exceptions(
            {**row, "metadata": {**row["metadata"], "language": "fr"}}, policy, "text"
        )[0]
        == "record outside declared language policy"
    )
    for metadata in ([], "not a dictionary", {"metadata.url": row["metadata"]["url"]}):
        assert (
            exceptions({**row, "metadata": metadata}, policy, "text")[0]
            == "missing or invalid declared family/synthetic provenance"
        )


@pytest.mark.parametrize(
    "field",
    [
        "",
        "a" * 129,
        "_sparselab_source",
        "metadata._sparselab_source",
        "metadata..url",
        ".url",
        "metadata.",
        " url ",
    ],
)
def test_qualification_field_path_guards(field):
    with pytest.raises(ValueError, match="bounded original row field"):
        RowQualification(profile="web_text", family_fields=(field,))


@pytest.mark.parametrize(
    "payload,reason",
    [
        ({"profile": "web_text", "family_fields": ["id", "id"]}, "duplicate"),
        (
            {
                "profile": "web_text",
                "family_fields": ["id"],
                "language_field": "language",
                "allowed_languages": [" "],
            },
            "language field",
        ),
        (
            {"profile": "synthetic_text", "family_fields": ["id"]},
            "original provenance fields",
        ),
        (
            {
                "profile": "web_text",
                "family_fields": ["id"],
                "license_field": "license",
            },
            "code-only",
        ),
        (
            {
                "profile": "web_text",
                "family_fields": ["id"],
                "synthetic_provenance_fields": ["seed"],
            },
            "synthetic provenance fields",
        ),
    ],
)
def test_profile_policy_guards(payload, reason):
    with pytest.raises(ValueError, match=reason):
        RowQualification.model_validate(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"language_field": None, "allowed_languages": []},
        {"license_field": None},
        {"repository_field": None},
        {"path_field": None},
        {"license_labels": {}},
        {"license_labels": {" ": "MIT"}},
    ],
)
def test_code_policy_requires_complete_mapping(change):
    with pytest.raises(ValueError, match="code qualification requires"):
        RowQualification.model_validate(
            {**code_policy().model_dump(mode="json"), **change}
        )


def test_declared_policy_requires_pinned_shards(tmp_path):
    _, source, _, _ = declared_fixture(tmp_path)
    source["acquisition"].pop("bounded_shards")
    source["acquisition"]["include"] = ["*.jsonl"]
    with pytest.raises(ValueError, match="requires pinned bounded shards"):
        SourceDeclaration.model_validate(source)


@pytest.mark.parametrize("value", [None, "", "  ", [], {}])
def test_empty_family_values_are_not_provenance(value):
    policy = RowQualification(profile="web_text", family_fields=("id",))
    with pytest.raises(ValueError, match="missing declared family provenance"):
        family_hint({"id": value}, policy)


def test_synthetic_all_provenance_fields_are_required_and_bound():
    policy = RowQualification(
        profile="synthetic_text",
        family_fields=("seed_id",),
        synthetic_provenance_fields=("seed_id", "generator"),
    )
    row = {"text": "Generated fixture text", "seed_id": 7}
    assert (
        exceptions(row, policy, "text")[0]
        == "missing or invalid declared family/synthetic provenance"
    )
    first = provenance({**row, "generator": "one"}, policy)
    second = provenance({**row, "generator": "two"}, policy)
    assert first["synthetic_provenance_fields"] == ["seed_id", "generator"]
    assert first["synthetic_provenance_sha256"] != second["synthetic_provenance_sha256"]


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"code": ""}, "empty or nontext"),
        ({"code": "  "}, "empty or nontext"),
        ({"code": 7}, "empty or nontext"),
        (
            {"code": "-----BEGIN RSA PRIVATE KEY-----"},
            "private identifier or credential",
        ),
        ({"license": ["mit"]}, "missing or incompatible"),
        ({"repo_name": 42}, "missing code repository"),
        ({"repo_name": "  "}, "provenance"),
        ({"path": 7}, "missing code repository or path"),
        ({"path": ""}, "unsafe code path"),
        ({"path": "/etc/a.py"}, "unsafe code path"),
        ({"path": "src\\a.py"}, "unsafe code path"),
        ({"path": "src/./a.py"}, "unsafe code path"),
        ({"path": "Vendor/a.py"}, "vendor or generated"),
        ({"path": "node_modules/a.py"}, "vendor or generated"),
        ({"path": "build/a.py"}, "vendor or generated"),
    ],
)
def test_code_screen_guards(change, reason):
    conflict, _ = exceptions({**code_row(), **change}, code_policy(), "code")
    assert conflict is not None and reason in conflict


def test_code_license_mapping_review_flag_and_repository_case():
    policy = code_policy()
    row = {
        **code_row(),
        "license": "apache-2.0",
        "code": "# SPDX-License-Identifier: Apache-2.0\nprint(1)",
    }
    assert exceptions(row, policy, "code") == (None, ())
    assert (
        "conflicting"
        in exceptions(
            {**row, "code": "# SPDX-License-Identifier: MIT\nprint(1)"}, policy, "code"
        )[0]
    )
    assert exceptions(
        {**code_row(), "code": "# Copyright Example\nprint(1)"}, policy, "code"
    ) == (None, ("code_notice_review",))
    assert family_hint(code_row(), policy) == family_hint(
        {**code_row(), "repo_name": "Fixture/Example"}, policy
    )
    assert provenance(row, policy)["attribution"]["license_expression"] == "Apache-2.0"


def test_code_attribution_and_unresolved_metadata_are_retained(tmp_path):
    row = {
        **code_row(),
        "text": "# Code fixture with enough retained content\ndef value():\n    return 42\n",
    }
    policy = code_policy().model_dump(mode="json")
    manifest, source, snapshot, raw = declared_fixture(
        tmp_path,
        policy=policy,
        rows=[
            row,
            {"text": "Unresolved fixture row with enough retained content. " * 3},
        ],
    )
    model = SourceDeclaration.model_validate(source)
    admitted = _verify(manifest, source, snapshot, tmp_path)
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
    metadata = docs[0][0]["metadata"]
    assert metadata["qualification_family_hint"] == family_hint(
        row, model.acquisition.qualification
    )
    for key, value in provenance(row, model.acquisition.qualification)[
        "attribution"
    ].items():
        assert metadata["qualification_" + key] == value
    from sparselab.corpus.rights import resolve_file_rights

    inventory_docs = _records_for_file(
        raw,
        file["path"],
        model,
        snapshot["snapshot_sha256"],
        file_rights=resolve_file_rights(model.rights, file["path"], raw),
    )
    assert inventory_docs[1][0]["metadata"]["qualification_status"] == "unresolved"


def test_write_split_inventory_uses_verified_original_declared_rows(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from sparselab.corpus.split_inventory import write_split_inventory

    _, source, snapshot, raw = declared_fixture(tmp_path)
    model = SourceDeclaration.model_validate(source)
    folder = tmp_path / model.id / snapshot["snapshot_sha256"]
    (folder / "manifest.json").write_text(json.dumps(snapshot))
    monkeypatch.setattr(
        "sparselab.corpus.split_inventory.verify_acquisition",
        lambda *_: {
            "sources": {
                model.id: {
                    "snapshot_path": str(folder),
                    "snapshot_sha256": snapshot["snapshot_sha256"],
                }
            }
        },
    )
    project = SimpleNamespace(
        config=SimpleNamespace(id="declared-inventory-fixture"),
        sources=(model,),
        release=SimpleNamespace(normalizer="normalizer-nfc-markdown-v1"),
    )
    output = tmp_path / "inventory.jsonl"
    write_split_inventory(project, tmp_path, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    original = json.loads(raw.splitlines()[0])
    assert rows[0]["family_hint"] == family_hint(
        original, model.acquisition.qualification
    )
    assert rows[1]["family_hint"].startswith("unresolved-family:")


@pytest.mark.parametrize("change", ["id", "provenance"])
def test_legacy_row_identity_clauses_remain_required(tmp_path, change):
    manifest, source, snapshot, raw, _ = _fixture(tmp_path)
    rows = [json.loads(line) for line in raw.splitlines()]
    if change == "id":
        rows[0]["id"] = "different-from-envelope"
    else:
        rows[0]["metadata"]["provenance"] = "not-a-shard-row"
    original = {
        key: value for key, value in rows[0].items() if key != "_sparselab_source"
    }
    digest = hashlib.sha256(canonical_json(original)).hexdigest()
    rows[0]["_sparselab_source"]["source_row_sha256"] = digest
    manifest["sources"][0]["records"][0]["source_row_sha256"] = digest
    snapshot["retrieval"]["shards"][0]["selected_rows"][0]["source_row_sha256"] = digest
    payload = b"".join(canonical_json(row) + b"\n" for row in rows)
    digest = hashlib.sha256(payload).hexdigest()
    snapshot["files"][0].update(sha256=digest, size=len(payload))
    manifest["sources"][0]["sample_sha256"] = digest
    (
        tmp_path
        / source["id"]
        / snapshot["snapshot_sha256"]
        / "files"
        / snapshot["files"][0]["path"]
    ).write_bytes(payload)
    with pytest.raises(ValueError, match="row provenance mismatch"):
        _verify(manifest, source, snapshot, tmp_path)


def test_nested_path_bounds_and_code_paths_are_explicit():
    RowQualification(profile="web_text", family_fields=("a" * 128,))
    for path in (".".join(["a"] * 9), ".".join(["a" * 104] * 5)):
        with pytest.raises(ValueError, match="bounded original row field"):
            RowQualification(profile="web_text", family_fields=(path,))
    policy = code_policy().model_dump(mode="json")
    for key in ("language_field", "license_field", "repository_field", "path_field"):
        policy[key] = "metadata." + policy[key]
    policy["family_fields"] = ["metadata.repo_name"]
    policy = RowQualification.model_validate(policy)
    row = code_row()
    nested = {"code": row.pop("code"), "metadata": row}
    assert exceptions(nested, policy, "code") == (None, ())
    assert provenance(nested, policy)["attribution"]["license_expression"] == "MIT"


@pytest.mark.parametrize(
    "field,value",
    [
        ("license_field", "license"),
        ("license_labels", {"mit": "MIT"}),
        ("repository_field", "repo_name"),
        ("path_field", "path"),
    ],
)
def test_noncode_policy_refuses_each_code_only_field(field, value):
    with pytest.raises(ValueError, match="code-only"):
        RowQualification.model_validate(
            {"profile": "web_text", "family_fields": ["id"], field: value}
        )


def test_admission_verifies_each_row_before_reading_next(tmp_path, monkeypatch):
    from sparselab.corpus import jsonl_records, rights

    manifest, source, snapshot, _ = declared_fixture(tmp_path)
    original_reader = jsonl_records.records_from_path
    original_check = rights._admission_exceptions
    checked = 0
    closed = False

    def check(*args, **kwargs):
        nonlocal checked
        result = original_check(*args, **kwargs)
        checked += 1
        return result

    def guarded_reader(path):
        nonlocal closed
        try:
            for index, record in enumerate(original_reader(path)):
                assert checked == index, (
                    "source text was materialized before validation"
                )
                yield record
        finally:
            closed = True

    monkeypatch.setattr(rights, "_admission_exceptions", check)
    monkeypatch.setattr(jsonl_records, "records_from_path", guarded_reader)
    _verify(manifest, source, snapshot, tmp_path)
    assert checked == 2 and closed


@pytest.mark.parametrize("extra", [False, True])
def test_streamed_admission_refuses_row_count_mismatch(tmp_path, monkeypatch, extra):
    from sparselab.corpus import jsonl_records

    manifest, source, snapshot, _ = declared_fixture(tmp_path)
    original_reader = jsonl_records.records_from_path
    closed = False

    def changed_reader(path):
        nonlocal closed
        try:
            rows = original_reader(path)
            first = next(rows)
            yield first
            if extra:
                yield from rows
                yield first
            else:
                rows.close()
        finally:
            closed = True

    monkeypatch.setattr(jsonl_records, "records_from_path", changed_reader)
    with pytest.raises(ValueError, match="row count mismatch"):
        _verify(manifest, source, snapshot, tmp_path)
    assert closed


def test_streamed_admission_closes_source_on_provenance_failure(tmp_path, monkeypatch):
    from sparselab.corpus import jsonl_records

    manifest, source, snapshot, _ = declared_fixture(tmp_path)
    manifest["sources"][0]["records"][0]["source_row_sha256"] = "0" * 64
    original_reader = jsonl_records.records_from_path
    closed = False
    yielded = 0

    def tracked_reader(path):
        nonlocal closed, yielded
        try:
            for row in original_reader(path):
                yielded += 1
                yield row
        finally:
            closed = True

    monkeypatch.setattr(jsonl_records, "records_from_path", tracked_reader)
    with pytest.raises(ValueError, match="provenance|receipt"):
        _verify(manifest, source, snapshot, tmp_path)
    assert closed and yielded == 1
