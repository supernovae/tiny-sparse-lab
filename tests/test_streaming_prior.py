"""Small regression cases for a retained release's additive split anchors."""

import hashlib
import json

import pytest

from sparselab.corpus.split_runs import rows, write_rows
from sparselab.corpus.streaming_prior import (
    _index,
    check_content,
    check_merges,
    pin_families,
    prior_indexes,
)
from sparselab.corpus.streaming_split_freeze import _allocation_phase, _join_phase


def _record(document, family, split, content, stratum="code"):
    return {
        "document_id": document,
        "family_id": family,
        "split": split,
        "stratum": stratum,
        "content_sha256": content,
    }


def test_prior_indexes_bind_old_release_and_approved_family_clusters(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    release = tmp_path / "corpora" / "fixture" / "builds" / "release"
    release.mkdir(parents=True)
    inventory = release / "family-inventory.jsonl"
    inventory.write_text(
        json.dumps(_record("old", "component", "train", "digest")) + "\n"
    )
    (release / "manifest.json").write_text("{}")
    (release / "release-publication.json").write_text("{}")
    clusters = tmp_path / "clusters.json"
    clusters.write_text(
        json.dumps(
            {"merges": [{"family_id": "component", "member_hints": ["old-a", "old-b"]}]}
        )
    )
    cluster_sha = hashlib.sha256(clusters.read_bytes()).hexdigest()
    split_receipt = tmp_path / "frozen.receipt.json"
    split_receipt.write_text(json.dumps({"cluster_sha256": cluster_sha}))
    inventory_sha = hashlib.sha256(inventory.read_bytes()).hexdigest()
    manifest = {
        "release_id": "release-id",
        "corpus_id": "fixture",
        "files": {"family-inventory.jsonl": {"sha256": inventory_sha}},
        "build_identity": {
            "completed_preparation": {"request": {"inputs": {str(split_receipt): []}}}
        },
    }
    monkeypatch.setattr(
        "sparselab.corpus.completed_release.verify_retained_release", lambda _: manifest
    )
    binding = {
        "path": str(inventory),
        "sha256": inventory_sha,
        "release_path": str(release),
        "family_clusters": {"path": str(clusters), "sha256": cluster_sha},
    }
    result, evidence = prior_indexes(tmp_path, binding)
    assert result["aliases"] == {"old-a": "component", "old-b": "component"}
    assert result["prior_counts"] == {"code": {"train": 1}}
    assert evidence["release_id"] == "release-id"
    split_receipt.write_text(json.dumps({"cluster_sha256": "different"}))
    with pytest.raises(ValueError, match="authenticated split receipt"):
        prior_indexes(tmp_path, binding)


def test_incremental_anchors_preserve_old_splits_and_count_new_families(tmp_path):
    inventory = tmp_path / "old.jsonl"
    inventory.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                _record("old-1", "family-001", "validation", "content-1"),
                _record("old-2", "family-002", "test", "content-2"),
                _record("old-3", "family-003", "train", "content-3"),
            ]
        )
    )
    digest = hashlib.sha256(inventory.read_bytes()).hexdigest()
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    prior = _index(prior_dir, inventory, digest)
    qualified = tmp_path / "qualified.jsonl"
    write_rows(qualified, [[1, "family-001", "code", "new-1", "content-1"]])
    content_dir = tmp_path / "content-check"
    content_dir.mkdir()
    assert check_content(content_dir, {"qualified": str(qualified)}, prior)["checked"]

    family_file = tmp_path / "new-families.jsonl"
    write_rows(family_file, (["code", f"{i:03}", f"family-{i:03}"] for i in range(100)))
    pin_dir = tmp_path / "pin"
    pin_dir.mkdir()
    pinned = pin_families(
        pin_dir, {"families": str(family_file), "counts": {"code": 100}}, prior
    )
    allocation_dir = tmp_path / "allocate"
    allocation_dir.mkdir()
    result = _allocation_phase(allocation_dir, pinned, (98, 1, 1), {"code"})
    allocations = {
        family: split for family, _stratum, split in rows(result["allocations"])
    }
    assert allocations["family-001"] == "validation"
    assert allocations["family-002"] == "test"
    assert allocations["family-003"] == "train"
    assert list(allocations.values()).count("train") == 98
    assert list(allocations.values()).count("validation") == 1
    assert list(allocations.values()).count("test") == 1


def test_incremental_anchors_old_content_in_another_family(tmp_path):
    inventory = tmp_path / "old.jsonl"
    inventory.write_text(
        json.dumps(_record("old", "family-a", "validation", "digest")) + "\n"
    )
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    prior = _index(
        prior_dir, inventory, hashlib.sha256(inventory.read_bytes()).hexdigest()
    )
    qualified = tmp_path / "qualified.jsonl"
    write_rows(qualified, [[1, "family-b", "code", "new", "digest"]])
    content_dir = tmp_path / "content-check"
    content_dir.mkdir()
    content = check_content(content_dir, {"qualified": str(qualified)}, prior)
    assert list(rows(content["anchors"])) == [["family-b", "code", "validation"]]
    family_file = tmp_path / "new-families.jsonl"
    write_rows(family_file, [["code", "rank", "family-b"]])
    pin_dir = tmp_path / "pin"
    pin_dir.mkdir()
    pinned = pin_families(
        pin_dir, {"families": str(family_file), "counts": {"code": 1}}, prior, content
    )
    assert list(rows(pinned["families"])) == [
        ["code", "rank", "family-b", "validation"]
    ]
    assert pinned["extra_pinned"] == {"code": {"validation": 1}}
    assert pinned["overlapping"] == {}


def test_incremental_refuses_family_linked_to_conflicting_old_splits(tmp_path):
    inventory = tmp_path / "old.jsonl"
    inventory.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                _record("old-a", "family-a", "train", "digest-a"),
                _record("old-b", "family-b", "validation", "digest-b"),
            ]
        )
    )
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    prior = _index(
        prior_dir, inventory, hashlib.sha256(inventory.read_bytes()).hexdigest()
    )
    qualified = tmp_path / "qualified.jsonl"
    write_rows(
        qualified,
        [
            [1, "new-family", "code", "new-a", "digest-a"],
            [1, "new-family", "code", "new-b", "digest-b"],
        ],
    )
    content_dir = tmp_path / "content-check"
    content_dir.mkdir()
    with pytest.raises(ValueError, match="conflicting retained splits"):
        check_content(content_dir, {"qualified": str(qualified)}, prior)


def test_new_content_anchor_counts_toward_combined_quota(tmp_path):
    inventory = tmp_path / "old.jsonl"
    old = [
        _record(
            f"old-{i}",
            f"old-family-{i:03}",
            "train" if i < 98 else ("validation" if i == 98 else "test"),
            f"content-{i}",
        )
        for i in range(100)
    ]
    inventory.write_text("".join(json.dumps(row) + "\n" for row in old))
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    prior = _index(
        prior_dir, inventory, hashlib.sha256(inventory.read_bytes()).hexdigest()
    )
    qualified = tmp_path / "qualified.jsonl"
    write_rows(qualified, [[1, "new-family-000", "code", "new", "content-98"]])
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    content = check_content(content_dir, {"qualified": str(qualified)}, prior)
    family_file = tmp_path / "families.jsonl"
    write_rows(
        family_file, (["code", f"{i:03}", f"new-family-{i:03}"] for i in range(100))
    )
    pin_dir = tmp_path / "pin"
    pin_dir.mkdir()
    pinned = pin_families(
        pin_dir, {"families": str(family_file), "counts": {"code": 100}}, prior, content
    )
    allocation_dir = tmp_path / "allocate"
    allocation_dir.mkdir()
    result = _allocation_phase(allocation_dir, pinned, (98, 1, 1), {"code"})
    allocations = {family: split for family, _, split in rows(result["allocations"])}
    assert allocations["new-family-000"] == "validation"
    assert result["family_counts"]["code"]["families"] == 200
    assert [
        list(allocations.values()).count(split)
        for split in ("train", "validation", "test")
    ] == [98, 1, 1]


def test_incremental_allocation_counts_unique_prior_and_new_families(tmp_path):
    inventory = tmp_path / "old.jsonl"
    old = [
        _record(
            f"old-{i}",
            f"family-{i:03}",
            "train" if i < 98 else ("validation" if i == 98 else "test"),
            f"content-{i}",
        )
        for i in range(100)
    ]
    inventory.write_text("".join(json.dumps(row) + "\n" for row in old + [old[0]]))
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    prior = _index(
        prior_dir, inventory, hashlib.sha256(inventory.read_bytes()).hexdigest()
    )
    family_file = tmp_path / "new-families.jsonl"
    write_rows(
        family_file, (["code", f"{i:03}", f"family-{i:03}"] for i in range(3, 103))
    )
    pin_dir = tmp_path / "pin"
    pin_dir.mkdir()
    pinned = pin_families(
        pin_dir, {"families": str(family_file), "counts": {"code": 100}}, prior
    )
    allocation_dir = tmp_path / "allocate"
    allocation_dir.mkdir()
    result = _allocation_phase(allocation_dir, pinned, (98, 1, 1), {"code"})
    allocations = {
        family: split for family, _stratum, split in rows(result["allocations"])
    }
    assert allocations["family-098"] == "validation"
    assert allocations["family-099"] == "test"
    assert result["family_counts"]["code"]["families"] == 103
    combined = {row["family_id"]: row["split"] for row in old}
    combined.update(allocations)
    assert [
        list(combined.values()).count(split)
        for split in ("train", "validation", "test")
    ] == [101, 1, 1]


def test_retained_inventory_digest_change_is_refused(tmp_path):
    inventory = tmp_path / "old.jsonl"
    inventory.write_text(
        json.dumps(_record("old", "family-a", "train", "digest")) + "\n"
    )
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _index(prior_dir, inventory, "0" * 64)


def test_reviewed_alias_anchors_new_rows_and_cannot_be_renamed(tmp_path):
    inventory = tmp_path / "inventory.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    write_rows(
        inventory, [["source", "sample", 0, "doc", "content", "old-hint", "code"]]
    )
    write_rows(decisions, [["source", "sample", 0, "qualify"]])
    directory = tmp_path / "join"
    directory.mkdir()
    joined = _join_phase(
        directory,
        {"inventory": str(inventory)},
        {"decisions": str(decisions)},
        {},
        {"old-hint": "component"},
    )
    assert list(rows(joined["qualified"])) == [
        [1, "component", "code", "doc", "content"]
    ]
    prior = tmp_path / "families.jsonl"
    write_rows(prior, [["component", "code", "validation"]])
    directory = tmp_path / "merges"
    directory.mkdir()
    with pytest.raises(ValueError, match="changes a prior reviewed family"):
        check_merges(
            directory,
            {"families": str(prior), "aliases": {"old-hint": "component"}},
            {"old-hint": "another-component"},
        )
