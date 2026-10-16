"""Offline checks for admission-aware deterministic family split freezing."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sparselab.corpus import split_freeze
from sparselab.corpus.mixture import _inventory as verify_mixture_inventory
from sparselab.corpus.project import SplitDeclaration
from sparselab.training.manifest import canonical_json, sha256_file


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family_count=10):
    sources = [SimpleNamespace(id=source) for source in ("books", "pages", "incidents")]
    project_root = tmp_path / "project"
    project_root.mkdir()
    admission = {"schema_version": 2, "sources": []}
    admission_bytes = canonical_json(admission)
    (project_root / "admission.json").write_bytes(admission_bytes)
    project = SimpleNamespace(
        config=SimpleNamespace(id="freeze-fixture"),
        root=project_root,
        release=SimpleNamespace(
            record_admission=SimpleNamespace(
                path="admission.json",
                sha256=hashlib.sha256(admission_bytes).hexdigest(),
            )
        ),
        sources=sources,
    )
    inventory = tmp_path / "inventory.jsonl"
    rows = []
    decisions = {}
    snapshots = {}
    for source in sources:
        snapshot = tmp_path / source.id
        snapshot.mkdir()
        (snapshot / "manifest.json").write_text("{}")
        snapshots[source.id] = {
            "snapshot_path": str(snapshot),
            "snapshot_sha256": "a" * 64,
        }
        for index in range(family_count):
            path = (
                f"docs/{index}.md"
                if source.id == "incidents"
                else f"{source.id}.sample.jsonl"
            )
            rows.append(
                {
                    "document_id": f"{source.id}-{index}",
                    "source_id": source.id,
                    "source_location": path
                    + (
                        "#lines=1-2"
                        if source.id == "incidents"
                        else f"#row={index + 1}"
                    ),
                    "source_row_index": None if source.id == "incidents" else index,
                    "family_hint": f"{source.id}-family:{index}",
                    "content_sha256": hashlib.sha256(
                        f"{source.id}-{index}".encode()
                    ).hexdigest(),
                }
            )
            decisions[source.id, path] = (
                {"decision": "qualify"}
                if source.id == "incidents"
                else {
                    "decisions": {
                        i: {"decision": "qualify"} for i in range(family_count)
                    }
                }
            )
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    clusters_path = tmp_path / "clusters.json"
    clusters = {
        "schema_version": 1,
        "inventory_sha256": sha256_file(inventory),
        "seed": "reviewed-seed-v1",
        "reviewer": "Fixture reviewer",
        "reviewed_on": "2026-10-07",
        "source_strata": {
            "books": "general_prose",
            "pages": "explanatory_prose",
            "incidents": "incident_response_docs",
        },
        "merges": [],
    }
    clusters_path.write_bytes(canonical_json(clusters))
    monkeypatch.setattr(
        split_freeze,
        "write_split_inventory",
        lambda *_: {"sha256": sha256_file(inventory)},
    )
    monkeypatch.setattr(
        split_freeze,
        "verify_acquisition",
        lambda *_: {"sources": snapshots},
    )
    monkeypatch.setattr(split_freeze, "source_declaration_payload", lambda _: {})
    monkeypatch.setattr(split_freeze, "verify_record_admission", lambda *_: decisions)
    return project, inventory, clusters_path, rows, decisions


def test_freeze_splits_is_deterministic_and_validates_integer_family_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters, rows, _ = _fixture(tmp_path, monkeypatch)
    first = tmp_path / "splits-first.yaml"
    second = tmp_path / "splits-second.yaml"
    report = split_freeze.freeze_splits(project, tmp_path, inventory, clusters, first)
    split_freeze.freeze_splits(project, tmp_path, inventory, clusters, second)
    assert first.read_bytes() == second.read_bytes()
    declaration = yaml.safe_load(first.read_text())
    SplitDeclaration.model_validate(declaration)
    assert set(declaration["assignments"]) == {row["document_id"] for row in rows}
    assert report["eligible_documents"] == 30
    assert all(
        counts
        == {
            "families": 10,
            "train": 8,
            "validation": 1,
            "test": 1,
            "fractions": {
                "train": "8/10",
                "validation": "1/10",
                "test": "1/10",
            },
        }
        for counts in report["family_counts"].values()
    )
    candidates = [
        json.loads(line)
        for line in Path(report["family_candidates"]).read_text().splitlines()
    ]
    assert len(candidates) == 30
    assert set(json.loads(Path(report["receipt"]).read_text())) == {
        "schema_version",
        "split_sha256",
        "family_candidate_sha256",
        "inventory_sha256",
        "admission_sha256",
        "cluster_sha256",
        "family_counts",
    }
    assert all(
        set(row) == {"document_id", "family_id", "split", "stratum", "content_sha256"}
        for row in candidates
    )


def test_freeze_splits_reads_embedded_unicode_separators_as_one_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters, rows, _ = _fixture(tmp_path, monkeypatch)
    rows[0]["family_hint"] += "\u0085\u2028\u2029"
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    declaration = json.loads(clusters.read_text())
    declaration["inventory_sha256"] = sha256_file(inventory)
    clusters.write_bytes(canonical_json(declaration))
    output = tmp_path / "splits-unicode.yaml"
    report = split_freeze.freeze_splits(project, tmp_path, inventory, clusters, output)
    assert report["eligible_documents"] == 30
    assert len(yaml.safe_load(output.read_text())["assignments"]) == 30


def test_freeze_splits_preserves_authenticated_prior_families_with_new_document_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters_path, rows, _ = _fixture(tmp_path, monkeypatch)
    release = tmp_path / ("a" * 64)
    release.mkdir()
    previous = []
    old_docs = []
    strata = {
        "books": "general_prose",
        "pages": "explanatory_prose",
        "incidents": "incident_response_docs",
    }
    expected = {}
    for row in rows:
        index = int(row["document_id"].rsplit("-", 1)[1])
        if index >= 3:
            continue
        split = ("test", "validation", "train")[index]
        old_id = "old-" + row["document_id"]
        old_docs.append(
            {
                "document_id": old_id,
                "drop_reason": None,
                "split": split,
                "content_sha256": row["content_sha256"],
                "domains": [strata[row["source_id"]]],
            }
        )
        previous.append(
            {
                "document_id": old_id,
                "family_id": row["family_hint"],
                "split": split,
                "stratum": strata[row["source_id"]],
                "content_sha256": row["content_sha256"],
            }
        )
        expected[row["document_id"]] = split
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in old_docs)
    )
    prior_path = tmp_path / "prior-families.jsonl"
    prior_path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in previous))
    clusters = json.loads(clusters_path.read_text())
    clusters["prior_family_inventory"] = {
        "path": str(prior_path),
        "sha256": sha256_file(prior_path),
        "release_path": str(release),
    }
    clusters_path.write_bytes(canonical_json(clusters))
    monkeypatch.setattr(
        split_freeze,
        "verify_release",
        lambda *a, **kw: {"release_id": release.name},
    )
    output = tmp_path / "preserved.yaml"
    report = split_freeze.freeze_splits(
        project, tmp_path, inventory, clusters_path, output
    )
    actual = yaml.safe_load(output.read_text())["assignments"]
    assert {key: actual[key] for key in expected} == expected
    assert json.loads(Path(report["receipt"]).read_text())[
        "prior_family_inventory_sha256"
    ] == sha256_file(prior_path)


def test_prior_family_inventory_changed_bytes_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters_path, _, _ = _fixture(tmp_path, monkeypatch)
    prior = tmp_path / "prior.jsonl"
    prior.write_text("{}\n")
    clusters = json.loads(clusters_path.read_text())
    clusters["prior_family_inventory"] = {
        "path": str(prior),
        "sha256": "0" * 64,
        "release_path": str(tmp_path / ("a" * 64)),
    }
    clusters_path.write_bytes(canonical_json(clusters))
    with pytest.raises(ValueError, match="digest mismatch"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters_path, tmp_path / "rejected.yaml"
        )


def test_prior_content_cannot_be_renamed_into_another_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters_path, rows, _ = _fixture(tmp_path, monkeypatch)
    source = rows[0]
    release = tmp_path / ("b" * 64)
    release.mkdir()
    old_id = "old-" + source["document_id"]
    (release / "documents.jsonl").write_bytes(
        canonical_json(
            {
                "document_id": old_id,
                "drop_reason": None,
                "split": "test",
                "content_sha256": source["content_sha256"],
                "domains": ["general_prose"],
            }
        )
        + b"\n"
    )
    prior = tmp_path / "old-families.jsonl"
    prior.write_bytes(
        canonical_json(
            {
                "document_id": old_id,
                "family_id": source["family_hint"],
                "split": "test",
                "stratum": "general_prose",
                "content_sha256": source["content_sha256"],
            }
        )
        + b"\n"
    )
    clusters = json.loads(clusters_path.read_text())
    clusters["prior_family_inventory"] = {
        "path": str(prior),
        "sha256": sha256_file(prior),
        "release_path": str(release),
    }
    clusters["merges"] = [
        {
            "family_id": "renamed-family",
            "member_hints": [source["family_hint"], rows[1]["family_hint"]],
        }
    ]
    clusters_path.write_bytes(canonical_json(clusters))
    monkeypatch.setattr(
        split_freeze,
        "verify_release",
        lambda *a, **kw: {"release_id": release.name},
    )
    with pytest.raises(ValueError, match="changed reviewed family"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters_path, tmp_path / "rejected.yaml"
        )


def test_exact_content_alias_uses_accepted_family_id_and_held_out_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters_path, rows, decisions = _fixture(tmp_path, monkeypatch)
    accepted, alias = rows[0], rows[1]
    alias["content_sha256"] = accepted["content_sha256"]
    extra = dict(rows[2])
    extra.update(
        document_id="books-10",
        source_row_index=10,
        family_hint="books-family:10",
        content_sha256=hashlib.sha256(b"books-10").hexdigest(),
    )
    rows.append(extra)
    decisions["books", "books.sample.jsonl"]["decisions"][10] = {"decision": "qualify"}
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    release = tmp_path / ("c" * 64)
    release.mkdir()
    old_id = "old-" + accepted["document_id"]
    (release / "documents.jsonl").write_bytes(
        canonical_json(
            {
                "document_id": old_id,
                "drop_reason": None,
                "split": "test",
                "content_sha256": accepted["content_sha256"],
                "domains": ["general_prose"],
            }
        )
        + b"\n"
    )
    prior = tmp_path / "accepted-families.jsonl"
    prior.write_bytes(
        canonical_json(
            {
                "document_id": old_id,
                "family_id": accepted["family_hint"],
                "split": "test",
                "stratum": "general_prose",
                "content_sha256": accepted["content_sha256"],
            }
        )
        + b"\n"
    )
    clusters = json.loads(clusters_path.read_text())
    clusters["inventory_sha256"] = sha256_file(inventory)
    clusters["prior_family_inventory"] = {
        "path": str(prior),
        "sha256": sha256_file(prior),
        "release_path": str(release),
    }
    clusters["merges"] = [
        {
            "family_id": accepted["family_hint"],
            "member_hints": [accepted["family_hint"], alias["family_hint"]],
        }
    ]
    clusters_path.write_bytes(canonical_json(clusters))
    monkeypatch.setattr(
        split_freeze,
        "verify_release",
        lambda *a, **kw: {"release_id": release.name},
    )
    output = tmp_path / "aliases-preserved.yaml"
    report = split_freeze.freeze_splits(
        project, tmp_path, inventory, clusters_path, output
    )
    assignments = yaml.safe_load(output.read_text())["assignments"]
    assert assignments[accepted["document_id"]] == "test"
    assert assignments[alias["document_id"]] == "test"
    candidates = {
        row["document_id"]: row
        for row in (
            json.loads(line)
            for line in Path(report["family_candidates"]).read_text().splitlines()
        )
    }
    assert candidates[accepted["document_id"]]["family_id"] == accepted["family_hint"]
    assert candidates[alias["document_id"]]["family_id"] == accepted["family_hint"]


def test_finalized_family_inventory_matches_kept_release_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters, _, _ = _fixture(tmp_path, monkeypatch)
    split_path = tmp_path / "splits.yaml"
    report = split_freeze.freeze_splits(
        project, tmp_path, inventory, clusters, split_path
    )
    release = tmp_path / "release"
    release.mkdir()
    candidates = [
        json.loads(line)
        for line in Path(report["family_candidates"]).read_text().splitlines()
    ]
    source_by_stratum = {
        "general_prose": "books",
        "explanatory_prose": "pages",
        "incident_response_docs": "incidents",
    }
    documents = {
        row["document_id"]: {
            "document_id": row["document_id"],
            "source_id": source_by_stratum[row["stratum"]],
            "domains": [row["stratum"]],
            "split": row["split"],
            "content_sha256": row["content_sha256"],
            "drop_reason": "duplicate" if index == 0 else None,
        }
        for index, row in enumerate(candidates)
    }
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in documents.values())
    )
    monkeypatch.setattr(
        split_freeze,
        "verify_release",
        lambda _: {
            "release_id": "fixture-release",
            "build_identity": {"split": yaml.safe_load(split_path.read_text())},
        },
    )
    final = tmp_path / "final-family-inventory.jsonl"
    result = split_freeze.finalize_family_inventory(release, split_path, final)
    assert result["kept_documents"] == 29
    kept = {key: doc for key, doc in documents.items() if doc["drop_reason"] is None}
    assert set(
        verify_mixture_inventory(
            final, kept, {v: k for k, v in source_by_stratum.items()}
        )
    ) == set(kept)


def test_freeze_splits_stops_on_missing_decision_unresolved_family_and_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters_path, rows, decisions = _fixture(tmp_path, monkeypatch)
    decisions.pop(("incidents", "docs/0.md"))
    with pytest.raises(ValueError, match="lacks file admission"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters_path, tmp_path / "missing.yaml"
        )
    decisions["incidents", "docs/0.md"] = {"decision": "qualify"}
    rows[0]["family_hint"] = "unresolved-family:books-0"
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    clusters = json.loads(clusters_path.read_text())
    clusters["inventory_sha256"] = sha256_file(inventory)
    clusters_path.write_bytes(canonical_json(clusters))
    with pytest.raises(ValueError, match="unresolved family"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters_path, tmp_path / "unresolved.yaml"
        )


def test_cross_split_duplicate_qualifying_content_fails_but_quarantine_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, inventory, clusters_path, rows, decisions = _fixture(tmp_path, monkeypatch)
    first = tmp_path / "baseline.yaml"
    split_freeze.freeze_splits(project, tmp_path, inventory, clusters_path, first)
    assignments = yaml.safe_load(first.read_text())["assignments"]
    train = next(row for row in rows if assignments[row["document_id"]] == "train")
    heldout = next(
        row
        for row in rows
        if row["source_id"] == "books"
        and assignments[row["document_id"]] == "validation"
    )
    heldout["content_sha256"] = train["content_sha256"]
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    clusters = json.loads(clusters_path.read_text())
    clusters["inventory_sha256"] = sha256_file(inventory)
    clusters_path.write_bytes(canonical_json(clusters))
    with pytest.raises(ValueError, match="cross-split exact content duplicate"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters_path, tmp_path / "duplicate.yaml"
        )
    source_id = heldout["source_id"]
    path = heldout["source_location"].split("#", 1)[0]
    sibling = {
        **heldout,
        "document_id": "books-sibling",
        "source_location": path + "#row=11",
        "source_row_index": 10,
        "content_sha256": hashlib.sha256(b"sibling").hexdigest(),
    }
    rows.append(sibling)
    decisions[source_id, path]["decisions"][10] = {"decision": "qualify"}
    decisions[source_id, path]["decisions"][heldout["source_row_index"]] = {
        "decision": "quarantine"
    }
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    clusters["inventory_sha256"] = sha256_file(inventory)
    clusters_path.write_bytes(canonical_json(clusters))
    split_freeze.freeze_splits(
        project, tmp_path, inventory, clusters_path, tmp_path / "quarantined.yaml"
    )


@pytest.mark.parametrize("publication", [1, 2, 3])
def test_interrupted_bundle_requires_explicit_identical_recovery(
    tmp_path, monkeypatch, publication
):
    project, inventory, clusters, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "nested" / "splits.yaml"
    publish = split_freeze._publish_new
    calls = 0

    def interrupted(path, data, **kwargs):
        nonlocal calls
        publish(path, data, **kwargs)
        calls += 1
        if calls == publication:
            raise OSError("injected post-publication failure")

    monkeypatch.setattr(split_freeze, "_publish_new", interrupted)
    with pytest.raises(OSError, match="post-publication"):
        split_freeze.freeze_splits(project, tmp_path, inventory, clusters, output)
    before = {p: (p.read_bytes(), p.stat().st_ino) for p in output.parent.iterdir()}
    if publication < 3:
        with pytest.raises(
            ValueError, match="incomplete or mismatched frozen split bundle"
        ):
            split_freeze.verify_split_bundle(output)
    else:
        assert split_freeze.verify_split_bundle(output)
    with pytest.raises(ValueError, match="already exists"):
        split_freeze.freeze_splits(project, tmp_path, inventory, clusters, output)
    monkeypatch.setattr(split_freeze, "_publish_new", publish)
    for _ in range(2):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters, output, recover=True
        )
        assert split_freeze.verify_split_bundle(output)
        assert all(
            (p.read_bytes(), p.stat().st_ino) == value for p, value in before.items()
        )


@pytest.mark.parametrize(
    "member", ["splits.yaml", "splits.family-candidates.jsonl", "splits.receipt.json"]
)
def test_recovery_rejects_conflict_before_publishing_anything(
    tmp_path, monkeypatch, member
):
    project, inventory, clusters, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "splits.yaml"
    conflict = tmp_path / member
    conflict.write_bytes(b"conflicting evidence\n")
    with pytest.raises(ValueError, match="conflicting"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters, output, recover=True
        )
    assert conflict.read_bytes() == b"conflicting evidence\n"
    assert {p.name for p in tmp_path.glob("splits*")} == {member}


@pytest.mark.parametrize("publication", [1, 2, 3])
def test_directory_sync_failure_retains_evidence_and_recovery_resyncs(
    tmp_path, monkeypatch, publication
):
    project, inventory, clusters, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "new" / "nested" / "splits.yaml"
    sync = split_freeze._sync_directory
    seen = []
    calls = 0

    def fail(path):
        nonlocal calls
        seen.append(path)
        if path == output.parent:
            calls += 1
            if calls == publication:
                raise OSError("directory sync failed; publication uncertain")
        sync(path)

    monkeypatch.setattr(split_freeze, "_sync_directory", fail)
    with pytest.raises(OSError, match="uncertain"):
        split_freeze.freeze_splits(project, tmp_path, inventory, clusters, output)
    before = {p: p.read_bytes() for p in output.parent.iterdir()}
    assert len(before) == publication
    seen.clear()
    monkeypatch.setattr(
        split_freeze, "_sync_directory", lambda p: (seen.append(p), sync(p))
    )
    split_freeze.freeze_splits(
        project, tmp_path, inventory, clusters, output, recover=True
    )
    assert all(p.read_bytes() == data for p, data in before.items())
    assert set(seen) == {output.parent, output.parent.parent, tmp_path}
    assert split_freeze.verify_split_bundle(output)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_receipt",
        "missing_both",
        "bad_receipt",
        "truncated_receipt",
        "bad_split",
        "bad_candidates",
        "symlink",
    ],
)
def test_project_consumer_refuses_incomplete_or_mismatched_bundle(
    tmp_path, monkeypatch, damage
):
    from sparselab.corpus.project import load_project

    project, inventory, clusters, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "splits.yaml"
    result = split_freeze.freeze_splits(project, tmp_path, inventory, clusters, output)
    receipt = Path(result["receipt"])
    candidates = Path(result["family_candidates"])
    if damage.startswith("missing"):
        receipt.unlink()
        if damage == "missing_both":
            candidates.unlink()
    elif damage == "bad_receipt":
        receipt.write_text("[]")
    elif damage == "truncated_receipt":
        content = json.loads(receipt.read_bytes())
        del content["cluster_sha256"]
        receipt.write_bytes(canonical_json(content))
    elif damage == "bad_split":
        output.write_bytes(output.read_bytes() + b"# changed\n")
    elif damage == "symlink":
        saved = tmp_path / "saved"
        candidates.rename(saved)
        candidates.symlink_to(saved)
    else:
        candidates.write_text("{}\n")
    # Real project loader, with checked-in source/release declarations.
    source = Path("corpora/devmind-sample-v0/corpus.yaml")
    for folder in ("sources", "transforms"):
        shutil.copytree(source.parent / folder, tmp_path / folder)
    config = tmp_path / "corpus.yaml"
    config.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="incomplete or mismatched"):
        load_project(config)
    monkeypatch.setattr(split_freeze, "verify_release", lambda _: {})
    with pytest.raises(ValueError, match="incomplete or mismatched"):
        split_freeze.finalize_family_inventory(
            tmp_path, output, tmp_path / "final.jsonl"
        )
    assert not (tmp_path / "final.jsonl").exists()


def test_legacy_complete_receipt_remains_readable(tmp_path, monkeypatch):
    project, inventory, clusters, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "splits.yaml"
    result = split_freeze.freeze_splits(project, tmp_path, inventory, clusters, output)
    # Synthetic pre-marker schema-1 fixture only; no real evidence migration.
    output.write_bytes(output.read_bytes().removeprefix(split_freeze._FREEZE_MARKER))
    receipt_path = Path(result["receipt"])
    receipt = json.loads(receipt_path.read_bytes())
    receipt["split_sha256"] = sha256_file(output)
    receipt_path.write_bytes(canonical_json(receipt) + b"\n")
    assert split_freeze.verify_split_bundle(output, required=False) == receipt


def test_recovery_revalidates_inputs_before_completing_subset(tmp_path, monkeypatch):
    project, inventory, clusters, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "splits.yaml"
    publish = split_freeze._publish_new

    def interrupt(path, data, **kwargs):
        publish(path, data, **kwargs)
        raise OSError("interrupted")

    monkeypatch.setattr(split_freeze, "_publish_new", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        split_freeze.freeze_splits(project, tmp_path, inventory, clusters, output)
    candidate = output.with_name("splits.family-candidates.jsonl")
    before = candidate.read_bytes()
    (project.root / "admission.json").write_text("{}")
    with pytest.raises(ValueError, match="admission SHA-256 mismatch"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters, output, recover=True
        )
    assert candidate.read_bytes() == before
    assert not output.exists()
    assert not output.with_suffix(".receipt.json").exists()


def test_publication_race_never_clobbers_conflicting_evidence(tmp_path, monkeypatch):
    output = tmp_path / "raced.yaml"
    link = split_freeze.os.link

    def race(source, destination):
        destination.write_bytes(b"other writer")
        link(source, destination)

    monkeypatch.setattr(split_freeze.os, "link", race)
    with pytest.raises(FileExistsError):
        split_freeze._publish_new(output, b"our bytes")
    assert output.read_bytes() == b"other writer"
    assert list(tmp_path.iterdir()) == [output]


def test_recovery_rejects_identical_symlink(tmp_path, monkeypatch):
    project, inventory, clusters, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "splits.yaml"
    result = split_freeze.freeze_splits(project, tmp_path, inventory, clusters, output)
    candidate = Path(result["family_candidates"])
    saved = tmp_path / "saved-candidate"
    candidate.rename(saved)
    candidate.symlink_to(saved)
    before = saved.read_bytes()
    with pytest.raises(ValueError, match="conflicting"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters, output, recover=True
        )
    assert candidate.is_symlink()
    assert saved.read_bytes() == before


@pytest.mark.parametrize("recover", [False, True])
def test_traverse_only_ancestor_does_not_block_publication_or_recovery(
    tmp_path, monkeypatch, recover
):
    locked = tmp_path / "traverse-only"
    root = locked / "work"
    root.mkdir(parents=True)
    project, inventory, clusters, _, _ = _fixture(root, monkeypatch)
    output = root / "new" / "nested" / "splits.yaml"
    original_open = split_freeze.os.open
    sync = split_freeze._sync_directory
    synced = []
    fail_root = recover

    def bounded_open(path, flags, *args, **kwargs):
        # Deterministic even on root/ACL/platform setups where chmod is advisory.
        if flags & split_freeze.os.O_DIRECTORY:
            assert Path(path).is_relative_to(root), "opened ancestor above work root"
        return original_open(path, flags, *args, **kwargs)

    def directory_sync(path):
        if path == root and fail_root:
            raise OSError("injected work-root directory sync failure")
        sync(path)
        synced.append(path)

    monkeypatch.setattr(split_freeze.os, "open", bounded_open)
    monkeypatch.setattr(split_freeze, "_sync_directory", directory_sync)
    locked.chmod(0o311)
    try:
        before = {}
        if recover:
            with pytest.raises(OSError, match="work-root directory sync"):
                split_freeze.freeze_splits(project, root, inventory, clusters, output)
            before = {
                p: (p.read_bytes(), p.stat().st_ino) for p in output.parent.iterdir()
            }
            assert len(before) == 1  # First member survived a failed ancestor sync.
            fail_root = False
            synced.clear()
        split_freeze.freeze_splits(
            project, root, inventory, clusters, output, recover=recover
        )
        assert split_freeze.verify_split_bundle(output)
        assert set(synced) == {output.parent, output.parent.parent, root}
        assert all(
            (p.read_bytes(), p.stat().st_ino) == value for p, value in before.items()
        )
    finally:
        locked.chmod(0o700)  # Only this disposable fixture's permissions.


def test_standalone_publication_syncs_created_parents_only(tmp_path, monkeypatch):
    sync = split_freeze._sync_directory
    synced = []
    monkeypatch.setattr(
        split_freeze, "_sync_directory", lambda p: (synced.append(p), sync(p))
    )
    output = tmp_path / "new" / "nested" / "family.jsonl"
    split_freeze._publish_new(output, b"fixture\n")
    assert synced == [output.parent, output.parent.parent, tmp_path]
    assert output.read_bytes() == b"fixture\n"
    assert output.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("failure", ["missing", "unreadable"])
def test_yaml_read_errors_have_uniform_bundle_refusal(
    tmp_path, monkeypatch, required, failure
):
    output = tmp_path / "splits.yaml"
    if failure == "unreadable":
        output.write_bytes(split_freeze._FREEZE_MARKER)
        read_bytes = Path.read_bytes

        def denied(path):
            if path == output:
                raise PermissionError("injected YAML read denial")
            return read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", denied)
    with pytest.raises(
        ValueError, match="^incomplete or mismatched frozen split bundle$"
    ) as error:
        split_freeze.verify_split_bundle(output, required=required)
    assert isinstance(error.value.__cause__, OSError)


def test_public_cli_recovers_identical_bundle_and_emits_json(
    tmp_path, monkeypatch, capsys
):
    from sparselab.cli.main import build_parser
    from sparselab.corpus import project as project_module
    from sparselab.training.attempt_commands import classify_attempt_command

    project, inventory, clusters, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "splits.yaml"
    publish = split_freeze._publish_new

    def interrupted(path, data, **kwargs):
        publish(path, data, **kwargs)
        raise OSError("injected first publication failure")

    with monkeypatch.context() as fault:
        fault.setattr(split_freeze, "_publish_new", interrupted)
        with pytest.raises(OSError, match="first publication"):
            split_freeze.freeze_splits(project, tmp_path, inventory, clusters, output)
    monkeypatch.setattr(project_module, "load_project", lambda _: project)
    words = [
        "--work-dir",
        str(tmp_path),
        "corpus",
        "freeze-splits",
        "fixture-project.yaml",
        "--inventory",
        str(inventory),
        "--clusters",
        str(clusters),
        "--output",
        str(output),
        "--json",
    ]
    parser = build_parser()
    ordinary = parser.parse_args(words)
    with pytest.raises(ValueError, match="already exists"):
        ordinary.handler(ordinary)
    recovery = parser.parse_args([*words, "--recover"])
    recovery.handler(recovery)
    report = json.loads(capsys.readouterr().out)
    assert report["output"] == str(output)
    assert report["sha256"] == sha256_file(output)
    assert report["eligible_documents"] == 30
    assert split_freeze.verify_split_bundle(output)
    # Pin the documented policy boundary; do not broaden the attempt allowlist.
    argv = ["uv", "run", "--locked", "--no-sync", "sparselab", *words]
    assert classify_attempt_command(argv)
    with pytest.raises(
        ValueError, match="unreviewed or duplicate native option: --recover"
    ):
        classify_attempt_command([*argv, "--recover"])


def test_reviewed_98_1_1_split_and_fresh_strata_are_identity_bound(
    tmp_path, monkeypatch
):
    project, inventory, clusters, _, _ = _fixture(
        tmp_path, monkeypatch, family_count=100
    )
    declaration = json.loads(clusters.read_text())
    declaration["split_weights"] = {"train": 98, "validation": 1, "test": 1}
    declaration["source_strata"] = {
        "books": "real_english",
        "pages": "synthetic",
        "incidents": "code",
    }
    clusters.write_bytes(canonical_json(declaration))
    output = tmp_path / "splits.yaml"
    result = split_freeze.freeze_splits(project, tmp_path, inventory, clusters, output)
    receipt = split_freeze.verify_split_bundle(output)
    assert receipt["split_weights"] == declaration["split_weights"]
    assert receipt["cluster_sha256"] == sha256_file(clusters)
    assert set(receipt["family_counts"]) == {"real_english", "synthetic", "code"}
    for counts in receipt["family_counts"].values():
        assert [counts[key] for key in ("train", "validation", "test")] == [98, 1, 1]
    first = output.read_bytes()
    again = tmp_path / "same.yaml"
    split_freeze.freeze_splits(project, tmp_path, inventory, clusters, again)
    assert again.read_bytes() == first
    declaration["split_weights"] = {"train": 96, "validation": 2, "test": 2}
    clusters.write_bytes(canonical_json(declaration))
    changed = tmp_path / "changed.yaml"
    split_freeze.freeze_splits(project, tmp_path, inventory, clusters, changed)
    assert changed.read_bytes() != first
    assert (
        split_freeze.verify_split_bundle(changed)["cluster_sha256"]
        != receipt["cluster_sha256"]
    )
    assert output.read_bytes() == first
    assert result["eligible_documents"] == 300


@pytest.mark.parametrize(
    "weights",
    [
        None,
        {},
        {"train": 98, "validation": 1},
        {"train": 98, "validation": 1, "test": 0},
        {"train": 98, "validation": 1, "test": True},
        {"train": 98, "validation": 1, "test": 1.0},
        {"train": 10000, "validation": 1, "test": 1},
    ],
)
def test_invalid_reviewed_split_weights_refuse_before_publication(
    tmp_path, monkeypatch, weights
):
    project, inventory, clusters, _, _ = _fixture(tmp_path, monkeypatch)
    raw = json.loads(clusters.read_text())
    raw["split_weights"] = weights
    clusters.write_bytes(canonical_json(raw))
    with pytest.raises(ValueError, match="split_weights"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters, tmp_path / "splits.yaml"
        )
    assert not (tmp_path / "splits.yaml").exists()


def test_small_inventory_does_not_silently_round_away_one_percent_holdouts(
    tmp_path, monkeypatch
):
    project, inventory, clusters, _, _ = _fixture(tmp_path, monkeypatch)
    raw = json.loads(clusters.read_text())
    raw["split_weights"] = {"train": 98, "validation": 1, "test": 1}
    clusters.write_bytes(canonical_json(raw))
    with pytest.raises(ValueError, match="too few eligible families"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters, tmp_path / "splits.yaml"
        )
