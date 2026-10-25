"""Known overlap witnesses fail early; the complete native leakage check stays."""

import copy
import os

import pytest
from test_corpus_split_freeze import _assignments, _fixture

from sparselab.corpus import split_freeze
from sparselab.corpus.split_merge_provenance import (
    cluster_semantics_sha256,
    exact_content_family_id,
    reviewed_component_merges,
)
from sparselab.training.manifest import canonical_json, sha256_file


def _review(tmp_path, monkeypatch, *, merged=True, unresolved_second=False):
    project, inventory, clusters_path, rows, _ = _fixture(
        tmp_path, monkeypatch, family_count=11
    )
    rows[1]["content_sha256"] = rows[0]["content_sha256"]
    if unresolved_second:
        rows[3]["content_sha256"] = rows[2]["content_sha256"]
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    import json

    clusters = json.loads(clusters_path.read_text())
    clusters["inventory_sha256"] = sha256_file(inventory)
    component = {
        "stratum": "general_prose",
        "families": [rows[0]["family_hint"], rows[1]["family_hint"]],
    }
    if merged:
        clusters["merges"] = reviewed_component_merges([component])
    members = [{**row, "stratum": "general_prose"} for row in rows[:2]]
    data = {
        "cross_family_groups": 1,
        "components": [component],
        "groups": [
            {
                "content_sha256": rows[0]["content_sha256"],
                "members": members,
                "families": [
                    {"stratum": "general_prose", "family_hint": f}
                    for f in component["families"]
                ],
            }
        ],
    }
    if unresolved_second:
        component = {
            "stratum": "general_prose",
            "families": [row["family_hint"] for row in rows[2:4]],
        }
        data["components"].append(component)
        data["groups"].append(
            {
                "content_sha256": rows[2]["content_sha256"],
                "members": [{**row, "stratum": "general_prose"} for row in rows[2:4]],
                "families": [
                    {"stratum": "general_prose", "family_hint": hint}
                    for hint in component["families"]
                ],
            }
        )
        data["cross_family_groups"] = 2
    summary = {
        "cross_family_groups": len(data["groups"]),
        "cross_stratum_groups": 0,
        "inputs": {
            "inventory.jsonl": clusters["inventory_sha256"],
            "project/admission.json": project.release.record_admission.sha256,
        },
    }

    def put(name, value):
        path = tmp_path / name
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json(value))
        return {"path": name, "sha256": sha256_file(path)}

    provenance = {
        "kind": "reviewed_exact_content_components",
        "cluster_semantics_sha256": cluster_semantics_sha256(clusters),
        "diagnostic_summary": put("diagnostic-summary.json", summary),
        "diagnostic_components": put("diagnostic-components.json", data),
    }
    clusters["merge_provenance"] = put("merge-provenance.json", provenance)
    clusters_path.write_bytes(canonical_json(clusters))
    return project, inventory, clusters_path, rows, clusters


def test_component_identity_is_deterministic_without_changing_strata():
    components = [
        {"stratum": "code", "families": ["repo-c", "repo-a", "repo-b"]},
        {"stratum": "synthetic", "families": ["prompt-b", "prompt-a"]},
    ]
    reverse = copy.deepcopy(components[::-1])
    for component in reverse:
        component["families"].reverse()
    assert reviewed_component_merges(components) == reviewed_component_merges(reverse)
    assert exact_content_family_id("code", ["a", "b"]) != exact_content_family_id(
        "synthetic", ["a", "b"]
    )


def test_unresolved_reviewed_component_stops_before_source_work(tmp_path, monkeypatch):
    project, inventory, clusters, _, _ = _review(tmp_path, monkeypatch, merged=False)

    def no_work(*args, **kwargs):
        raise AssertionError("known overlap reached expensive source processing")

    monkeypatch.setattr(split_freeze, "write_split_inventory", no_work)
    monkeypatch.setattr(split_freeze, "verify_acquisition", no_work)
    with pytest.raises(ValueError, match="component is unresolved.*books row"):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters, tmp_path / "split.yaml"
        )


def test_unresolved_witness_does_not_name_an_already_merged_component(
    tmp_path, monkeypatch
):
    project, inventory, clusters, rows, _ = _review(
        tmp_path, monkeypatch, unresolved_second=True
    )
    with pytest.raises(ValueError, match=rows[2]["content_sha256"]) as failure:
        split_freeze.freeze_splits(
            project, tmp_path, inventory, clusters, tmp_path / "split.yaml"
        )
    assert rows[0]["content_sha256"] not in str(failure.value)


def test_reviewed_component_preserves_documents_and_coassigns_duplicates(
    tmp_path, monkeypatch
):
    project, inventory, clusters, rows, _ = _review(tmp_path, monkeypatch)
    output = tmp_path / "split.yaml"
    result = split_freeze.freeze_splits(project, tmp_path, inventory, clusters, output)
    assignments = _assignments(output)
    assert result["eligible_documents"] == len(rows) == 33
    assert assignments[rows[0]["document_id"]] == assignments[rows[1]["document_id"]]
    assert result["family_counts"]["general_prose"]["families"] == 10
    assert result["inventory_completion"] == "cold_verification"


@pytest.mark.parametrize("changed", ["seed", "stratum", "admission"])
def test_stale_review_refuses_before_source_work(tmp_path, monkeypatch, changed):
    project, inventory, path, _, clusters = _review(tmp_path, monkeypatch)
    if changed == "seed":
        clusters["seed"] = "a-different-seed"
    elif changed == "stratum":
        clusters["source_strata"]["books"] = "a_different_stratum"
    else:
        project.release.record_admission.sha256 = "0" * 64
    path.write_bytes(canonical_json(clusters))

    def no_work(*args, **kwargs):
        raise AssertionError("stale review reached expensive work")

    monkeypatch.setattr(split_freeze, "write_split_inventory", no_work)
    with pytest.raises(
        ValueError, match="split semantics|stale inventory or admission"
    ):
        split_freeze.freeze_splits(
            project, tmp_path, inventory, path, tmp_path / "split.yaml"
        )
