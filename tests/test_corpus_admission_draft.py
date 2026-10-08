"""Admission drafts cover every verified row and file without clearing flags."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_corpus_record_admission import _mixed_fixture

from sparselab.corpus.admission_draft import (
    _additional_draft_flags,
    draft_admission_manifest,
)
from sparselab.corpus.project import SourceDeclaration
from sparselab.corpus.rights import verify_record_admission


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('The review said "copied passage".', ("quoted_text_context",)),
        (
            "The author discussed allegations.",
            ("sensitive_biography_or_allegation_context",),
        ),
        ("A plain description.", ()),
    ],
)
def test_scale_audit_patterns_only_narrow_wikimedia_draft(
    content: str, expected: tuple[str, ...]
) -> None:
    row = {"text": content}
    assert _additional_draft_flags(row, "wikimedia") == expected
    assert _additional_draft_flags(row, "gutenberg") == ()


def _setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, Path, Path, Path]:
    fixture_root = tmp_path / "fixture"
    manifest, sources, snapshots = _mixed_fixture(fixture_root)
    project_id = "draft-fixture"
    root = tmp_path / "work"
    snapshot_root = root / "corpora" / project_id / "snapshots"
    lock = {"sources": {}}
    for source_id, snapshot in snapshots.items():
        destination = snapshot_root / source_id / snapshot["snapshot_sha256"]
        shutil.copytree(
            fixture_root / source_id / snapshot["snapshot_sha256"] / "files",
            destination / "files",
        )
        (destination / "manifest.json").write_text(json.dumps(snapshot))
        lock["sources"][source_id] = {"snapshot_path": str(destination)}
    project = SimpleNamespace(
        config=SimpleNamespace(id=project_id),
        sources=tuple(
            SourceDeclaration.model_validate(source) for source in sources.values()
        ),
    )
    monkeypatch.setattr(
        "sparselab.corpus.admission_draft.verify_acquisition", lambda *_: lock
    )
    policy_document = tmp_path / "policy.md"
    policy_document.write_text("Reviewed source-policy fixture.\n")
    template = {
        "policy_id": manifest["policy_id"],
        "policy_sha256": hashlib.sha256(policy_document.read_bytes()).hexdigest(),
        "sources": [
            {key: item[key] for key in ("source_id", "license_label", "rights")}
            for item in manifest["sources"]
        ],
    }
    template_path = tmp_path / "policy-template.json"
    template_path.write_text(json.dumps(template))
    return project, root, template_path, policy_document


def test_draft_binds_all_rows_and_files_and_quarantines_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, root, template, policy_document = _setup(tmp_path, monkeypatch)
    output = root / "admission-draft.json"
    report = draft_admission_manifest(project, root, template, policy_document, output)
    assert report["counts"] == {"qualify": 2, "exclude": 0, "quarantine": 2}
    manifest = json.loads(output.read_text())
    entries = {item["source_id"]: item for item in manifest["sources"]}
    assert [item["decision"] for item in entries["gutenberg_fixture"]["records"]] == [
        "qualify",
        "quarantine",
    ]
    assert entries["gutenberg_fixture"]["records"][0]["issues"][0]["field"] == (
        "metadata.edition"
    )
    assert [item["decision"] for item in entries["incident_fixture"]["files"]] == [
        "qualify",
        "quarantine",
    ]
    snapshots = {
        source.id: json.loads(
            (
                root
                / "corpora"
                / project.config.id
                / "snapshots"
                / source.id
                / entries[source.id]["snapshot_sha256"]
                / "manifest.json"
            ).read_text()
        )
        for source in project.sources
    }
    verify_record_admission(
        manifest,
        {source.id: source.model_dump(mode="json") for source in project.sources},
        snapshots,
        root / "corpora" / project.config.id / "snapshots",
    )


def test_draft_rejects_output_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, root, template, policy_document = _setup(tmp_path, monkeypatch)
    output = root / "admission-draft.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("old evidence")
    with pytest.raises(ValueError, match="new output"):
        draft_admission_manifest(project, root, template, policy_document, output)
    assert output.read_text() == "old evidence"


def test_draft_rejects_changed_reviewed_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, root, template, policy_document = _setup(tmp_path, monkeypatch)
    policy_document.write_text("Policy changed after template review.\n")
    output = root / "admission-draft.json"
    with pytest.raises(ValueError, match="policy document SHA-256 mismatch"):
        draft_admission_manifest(project, root, template, policy_document, output)
    assert not output.exists()
