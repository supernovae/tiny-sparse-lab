"""Admission drafts cover every verified row and file without clearing flags."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_corpus_record_admission import _mixed_fixture

from sparselab.corpus.acquisition import declaration_sha256
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


@pytest.mark.parametrize(
    "changed",
    ("unchanged", "policy", "project", "lock", "declaration", "snapshot"),
)
def test_application_binding_fails_closed_before_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    project, root, template_path, policy_document = _setup(tmp_path, monkeypatch)
    lock_path = root / "corpora" / project.config.id / "acquisition.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = {
        "project_sha256": "a" * 64,
        "sources": {
            source.id: {
                "snapshot_path": str(
                    next(
                        (
                            root
                            / "corpora"
                            / project.config.id
                            / "snapshots"
                            / source.id
                        ).iterdir()
                    )
                ),
                "declaration_sha256": declaration_sha256(source),
                "snapshot_sha256": next(
                    (
                        root / "corpora" / project.config.id / "snapshots" / source.id
                    ).iterdir()
                ).name,
            }
            for source in project.sources
        },
    }
    lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(
        "sparselab.corpus.admission_draft.verify_acquisition", lambda *_: lock
    )
    template = json.loads(template_path.read_text())
    template["application_binding"] = {
        "project_sha256": lock["project_sha256"],
        "acquisition_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "sources": {
            source.id: {
                "declaration_sha256": lock["sources"][source.id]["declaration_sha256"],
                "snapshot_sha256": lock["sources"][source.id]["snapshot_sha256"],
            }
            for source in project.sources
        },
    }
    if changed == "policy":
        policy_document.write_text("Unreviewed replacement.\n")
    elif changed == "project":
        template["application_binding"]["project_sha256"] = "b" * 64
    elif changed == "lock":
        lock_path.write_text(json.dumps({**lock, "unexpected": True}))
    elif changed in {"declaration", "snapshot"}:
        source_id = project.sources[0].id
        key = f"{changed}_sha256"
        template["application_binding"]["sources"][source_id][key] = "b" * 64
    template_path.write_text(json.dumps(template))
    output = root / "application-draft.json"
    if changed == "unchanged":
        result = draft_admission_manifest(
            project, root, template_path, policy_document, output
        )
        assert result["counts"] == {"qualify": 2, "exclude": 0, "quarantine": 2}
        return
    if changed == "policy":
        error = "policy document SHA-256 mismatch"
    elif changed in {"project", "lock"}:
        error = "application-policy project or lock binding mismatch"
    else:
        error = "application-policy source binding mismatch"
    with pytest.raises(ValueError, match=error):
        draft_admission_manifest(project, root, template_path, policy_document, output)
    assert not output.exists()
