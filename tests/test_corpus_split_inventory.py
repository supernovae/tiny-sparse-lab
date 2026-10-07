"""Pre-build split inventory uses the same document IDs as the native parser."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_corpus_record_admission import _fixture

from sparselab.corpus import split_inventory
from sparselab.corpus.pipeline import _records_for_file
from sparselab.corpus.project import SourceDeclaration
from sparselab.corpus.rights import resolve_file_rights
from sparselab.training.manifest import canonical_json


def test_verified_snapshot_inventory_matches_parser_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, snapshot, raw, source = _fixture(tmp_path)
    snapshot_dir = tmp_path / source.id / snapshot["snapshot_sha256"]
    (snapshot_dir / "manifest.json").write_text(json.dumps(snapshot))
    project = SimpleNamespace(
        config=SimpleNamespace(id="inventory-fixture"), sources=[source]
    )
    calls = []

    def verified(candidate: object, root: Path) -> dict:
        calls.append((candidate, root))
        return {
            "sources": {
                source.id: {
                    "snapshot_path": str(snapshot_dir),
                    "snapshot_sha256": snapshot["snapshot_sha256"],
                }
            }
        }

    monkeypatch.setattr(split_inventory, "verify_acquisition", verified)
    output = tmp_path / "inventory" / "ids.jsonl"
    report = split_inventory.write_split_inventory(project, tmp_path, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    expected = _records_for_file(
        raw,
        snapshot["files"][0]["path"],
        source,
        snapshot["snapshot_sha256"],
        file_rights=resolve_file_rights(
            source.rights, snapshot["files"][0]["path"], raw
        ),
    )
    assert calls == [(project, tmp_path.resolve())]
    assert report["documents"] == 2
    assert [row["document_id"] for row in rows] == [
        document["document_id"] for document, _ in expected
    ]
    assert [row["family_hint"] for row in rows] == [
        "gutenberg-work:100",
        "gutenberg-work:101",
    ]
    assert all("text" not in row for row in rows)
    with pytest.raises(ValueError, match="new path"):
        split_inventory.write_split_inventory(project, tmp_path, output)


def test_nested_wikimedia_locator_groups_page_revisions_and_binds_row_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, snapshot, raw, source = _fixture(tmp_path)
    source = source.model_copy(
        update={
            "canonical_uri": "https://huggingface.co/datasets/common-pile/wikimedia_filtered"
        }
    )
    snapshot["declaration"]["canonical_uri"] = source.canonical_uri
    rows = [json.loads(line) for line in raw.splitlines()]
    for index, row in enumerate(rows):
        row["metadata"]["url"] = (
            "https://en.wikipedia.org/wiki/Same_Page"
            if index == 0
            else "https://en.wikipedia.org/w/index.php?title=Same_Page&oldid=2"
        )
        original = {
            key: value for key, value in row.items() if key != "_sparselab_source"
        }
        digest = hashlib.sha256(canonical_json(original)).hexdigest()
        row["_sparselab_source"]["source_row_sha256"] = digest
        snapshot["retrieval"]["shards"][0]["selected_rows"][index][
            "source_row_sha256"
        ] = digest
    updated = b"".join(canonical_json(row) + b"\n" for row in rows)
    snapshot["files"][0]["sha256"] = hashlib.sha256(updated).hexdigest()
    snapshot["files"][0]["size"] = len(updated)
    snapshot_dir = tmp_path / source.id / snapshot["snapshot_sha256"]
    (snapshot_dir / "files" / snapshot["files"][0]["path"]).write_bytes(updated)
    (snapshot_dir / "manifest.json").write_text(json.dumps(snapshot))
    project = SimpleNamespace(
        config=SimpleNamespace(id="wiki-inventory"), sources=[source]
    )

    def verified(_: object, __: Path) -> dict:
        return {
            "sources": {
                source.id: {
                    "snapshot_path": str(snapshot_dir),
                    "snapshot_sha256": snapshot["snapshot_sha256"],
                }
            }
        }

    monkeypatch.setattr(split_inventory, "verify_acquisition", verified)
    output = tmp_path / "wiki-inventory.jsonl"
    split_inventory.write_split_inventory(project, tmp_path, output)
    inventory = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["family_hint"] for row in inventory] == [
        "wikimedia-page:en.wikipedia.org:same page"
    ] * 2
    assert [row["source_row_index"] for row in inventory] == [0, 1]

    snapshot["retrieval"]["shards"][0]["selected_rows"][1]["source_row_sha256"] = (
        "0" * 64
    )
    (snapshot_dir / "manifest.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="row provenance mismatch"):
        split_inventory.write_split_inventory(
            project, tmp_path, tmp_path / "wiki-tampered.jsonl"
        )


def test_git_sections_share_file_family() -> None:
    source = SourceDeclaration.model_validate(
        {
            "schema_version": 2,
            "id": "incident_docs",
            "kind": "git",
            "canonical_uri": "https://github.com/example/docs",
            "revision": "a" * 40,
            "license": "Apache-2.0",
            "rights": {
                "training_eligibility": "review_required",
                "redistribution_mode": "review_required",
            },
            "domains": ["incident_response"],
            "document_kinds": ["markdown"],
            "source_family": "incident_docs",
            "acquisition": {
                "include": ["docs/*.md"],
                "max_bytes": 1000,
            },
        }
    )
    raw = b"# Detect\nInvestigate.\n# Mitigate\nRoll back.\n"
    rights = resolve_file_rights(source.rights, "docs/incident.md", raw)
    sections = _records_for_file(
        raw, "docs/incident.md", source, "a" * 64, file_rights=rights
    )
    assert len(sections) == 2
    assert {
        split_inventory._family_hint(source, document, {}) for document, _ in sections
    } == {"git-file:incident_docs:docs/incident.md"}


def test_wikimedia_page_hint_accepts_pinned_component_locator() -> None:
    assert (
        split_inventory._wiki_locator_hint("https://wikipedia.com/wiki/Amhara_people")
        == "wikimedia-page:wikipedia.com:amhara people"
    )
    assert (
        split_inventory._wiki_locator_hint(
            "https://en.wikipedia.org/wiki/Amhara_people"
        )
        == "wikimedia-page:en.wikipedia.org:amhara people"
    )
    assert (
        split_inventory._wiki_locator_hint(
            "https://wikipedia.com.attacker.example/wiki/Amhara_people"
        )
        is None
    )
