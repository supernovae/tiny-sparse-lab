"""Small read-only fixtures for the parameterized protected-lineage audit."""

import json
from pathlib import Path

import pytest

from sparselab.cli.main import build_parser
from sparselab.corpus import protected_lineage
from sparselab.training.manifest import sha256_file


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    for release in (old, new):
        (release / "manifest.json").write_text("{}\n")
    prior = tmp_path / "old-families.jsonl"
    candidate = tmp_path / "new-families.jsonl"
    prior_row = {
        "document_id": "old-heldout",
        "family_id": "protected-family",
        "split": "test",
        "stratum": "general_prose",
        "content_sha256": "a" * 64,
    }
    new_row = {**prior_row, "document_id": "new-heldout"}
    train = {
        "document_id": "new-train",
        "family_id": "train-family",
        "split": "train",
        "stratum": "general_prose",
        "content_sha256": "b" * 64,
    }
    _jsonl(prior, [prior_row])
    _jsonl(candidate, [new_row, train])
    _jsonl(
        old / "documents.jsonl",
        [{**prior_row, "domains": ["general_prose"], "drop_reason": None}],
    )
    _jsonl(
        new / "documents.jsonl",
        [
            {**row, "domains": ["general_prose"], "drop_reason": None}
            for row in (new_row, train)
        ],
    )
    monkeypatch.setattr(
        protected_lineage,
        "verify_release",
        lambda path: {
            "release_id": path.name,
            "files": {
                "documents.jsonl": {"sha256": sha256_file(path / "documents.jsonl")}
            },
        },
    )
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "release_id": "old",
                "release_manifest_sha256": sha256_file(old / "manifest.json"),
                "documents_sha256": sha256_file(old / "documents.jsonl"),
                "loss_slices": [prior_row],
            }
        )
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "content_sha256": "c" * 64,
                "release_id": "old",
                "family_inventory_sha256": sha256_file(prior),
                "items": [
                    {
                        "split": "test",
                        "parent_document_ids": ["old-heldout"],
                        "parent_family": "protected-family",
                    }
                ],
                "chunks": [{"document_id": "old-heldout"}],
            }
        )
    )
    return old, new, prior, candidate, profile, suite, tmp_path / "receipt.json"


def test_protected_lineage_passes_and_binds_inputs(lineage):
    result = protected_lineage.audit_protected_lineage(*lineage)
    assert result["status"] == "PASS"
    assert result["retained_protected_content_hashes"] == 1
    assert not result["new_train_content_collisions"]
    assert json.loads(lineage[-1].read_text()) == result
    with pytest.raises(FileExistsError):
        protected_lineage.audit_protected_lineage(*lineage)


@pytest.mark.parametrize("change", ["family", "content", "assignment"])
def test_protected_lineage_preserves_blocked_receipt(lineage, change):
    old, new, prior, candidate, profile, suite, output = lineage
    rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    if change == "family":
        rows[1]["family_id"] = "protected-family"
    elif change == "content":
        rows[1]["content_sha256"] = "a" * 64
    else:
        rows[0]["family_id"] = "changed-family"
    _jsonl(candidate, rows)
    _jsonl(
        new / "documents.jsonl",
        [{**row, "domains": ["general_prose"], "drop_reason": None} for row in rows],
    )
    result = protected_lineage.audit_protected_lineage(
        old, new, prior, candidate, profile, suite, output
    )
    assert result["status"] == "BLOCKED"
    assert output.exists()


def test_profile_substitution_and_duplicate_inventory_fail(lineage):
    old, new, prior, candidate, profile, suite, output = lineage
    payload = json.loads(profile.read_text())
    payload["loss_slices"] = [{"document_id": "other"}]
    profile.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fixed profile"):
        protected_lineage.audit_protected_lineage(*lineage)
    assert not output.exists()
    payload["loss_slices"] = [json.loads(prior.read_text())]
    profile.write_text(json.dumps(payload))
    candidate.write_text(candidate.read_text() * 2)
    with pytest.raises(ValueError, match="inventory"):
        protected_lineage.audit_protected_lineage(
            old, new, prior, candidate, profile, suite, output
        )


def test_lineage_cli_is_read_only_and_parameterized():
    args = build_parser().parse_args(
        [
            "corpus",
            "audit-protected-lineage",
            "--prior-release",
            "/old",
            "--candidate-release",
            "/new",
            "--prior-inventory",
            "/old.jsonl",
            "--candidate-inventory",
            "/new.jsonl",
            "--profile",
            "/profile.json",
            "--suite",
            "/suite.json",
            "--output",
            "/receipt.json",
        ]
    )
    assert args.corpus_command == "audit-protected-lineage"
    assert args.prior_release == "/old"
    assert args.output == "/receipt.json"
