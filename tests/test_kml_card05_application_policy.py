"""Offline binding checks for the new application of unchanged source rules."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sparselab.corpus.acquisition import declaration_sha256
from sparselab.corpus.project import load_project
from sparselab.training.manifest import sha256_file

BASE = Path(__file__).resolve().parents[1] / "experiments/research/kernel-memory-lab"
EXPECTED_SNAPSHOTS = {
    "kml_scale_pagerduty": "e040f260758cb00dd18e2a8a7b66a2e5d74455cbd26e1527304a52a81d984830",
    "kml_scale_project_gutenberg": "db2ce3b1af503bb9444cc83f8e8fd1a72e603463ff0724ae41385f5d19627a1e",
    "kml_scale_scoutflo": "574e06a4a327a3e25c04782c829b7015b8ba541da1d8e8f66ae3e4f1f7dfda0a",
    "kml_scale_wikimedia": "77a7dbfb6d74c88fa69d7515e306cd9e725ff1b3e0eb45f3d05eb41caaca1bcb",
}


def _check_template_rights(template: dict, historical: dict) -> None:
    assert template["sources"] == historical["sources"]


def test_card05_application_policy_retains_rights_and_expanded_selection() -> None:
    document_path = BASE / "card05-base/application-policy-v1.json"
    template = json.loads(
        (BASE / "card05-base/application-template-v1.json").read_text()
    )
    document = json.loads(document_path.read_text())
    historical = json.loads(
        (BASE / "corpus-scale/admission-policy-template-v1.json").read_text()
    )
    _check_template_rights(template, historical)
    assert template["policy_id"] == document["policy_id"]
    assert template["policy_sha256"] == sha256_file(document_path)
    assert historical["policy_sha256"] == sha256_file(
        BASE / "CARD03_SCALE_SOURCE_POLICY_V1.md"
    )
    assert document["accepted_general_policy"]["sha256"] == sha256_file(
        BASE / "CARD03_SOURCE_ADMISSION_POLICY_V1.md"
    )
    assert document["accepted_scale_source_policy"]["sha256"] == sha256_file(
        BASE / "CARD03_SCALE_SOURCE_POLICY_V1.md"
    )
    assert document["unchanged_rights_template"]["sha256"] == sha256_file(
        BASE / "corpus-scale/admission-policy-template-v1.json"
    )
    assert document["spot_audit"]["protocol_sha256"] == sha256_file(
        BASE / "CARD03_SCALE_SPOT_AUDIT.md"
    )
    assert document["spot_audit"]["seed"] == "kml-card03-scale-spot-audit-v1"
    assert document["spot_audit"]["clear_screen_counts"] == {
        "kml_scale_project_gutenberg": 12,
        "kml_scale_wikimedia": 24,
        "kml_scale_pagerduty": 8,
        "kml_scale_scoutflo": 24,
    }
    project = load_project(BASE / "card05-base/project-acquire.yaml")
    assert document["project"]["file_sha256"] == sha256_file(
        BASE / "card05-base/project-acquire.yaml"
    )
    assert (
        document["project"]["project_sha256"]
        == template["application_binding"]["project_sha256"]
    )
    assert (
        document["acquisition_lock_sha256"]
        == template["application_binding"]["acquisition_lock_sha256"]
    )
    assert document["acquisition_lock_sha256"] == (
        "fc8c7322aba8153663ad7df0cb8c447cd3a13ac39bf319a1968d9114268eb989"
    )
    sources = {row["source_id"]: row for row in document["sources"]}
    assert set(sources) == set(EXPECTED_SNAPSHOTS)
    for source in project.sources:
        row = sources[source.id]
        binding = template["application_binding"]["sources"][source.id]
        assert row["declaration_sha256"] == declaration_sha256(source)
        assert binding == {
            "declaration_sha256": row["declaration_sha256"],
            "snapshot_sha256": EXPECTED_SNAPSHOTS[source.id],
        }
        if source.kind == "huggingface_dataset":
            shard = source.acquisition.bounded_shards[0]
            assert row["selection"]["shard_path"] == shard.path
            assert row["selection"]["shard_sha256"] == shard.expected_sha256
            assert row["selection"]["hash_modulus"] == 16
            assert row["selection"]["hash_remainders"] == [0, 1]
            assert (
                row["selection"]["selected_rows"]
                == {
                    "kml_scale_project_gutenberg": 314,
                    "kml_scale_wikimedia": 51007,
                }[source.id]
            )
        else:
            assert row["selection"]["tree_oid"] == source.acquisition.tree_oid


def test_changed_rights_object_is_not_a_compatible_application() -> None:
    template = json.loads(
        (BASE / "card05-base/application-template-v1.json").read_text()
    )
    historical = json.loads(
        (BASE / "corpus-scale/admission-policy-template-v1.json").read_text()
    )
    altered = copy.deepcopy(template)
    altered["sources"][0]["rights"]["redistribution_mode"] = "review_required"
    with pytest.raises(AssertionError):
        _check_template_rights(altered, historical)
