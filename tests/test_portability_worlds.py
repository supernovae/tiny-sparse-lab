"""Deterministic and leakage-audited Engram portability world contracts."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sparselab.data.toy_worlds import (
    StructuredKey,
    frozen_encoders,
    materialize_portability_worlds,
)
from sparselab.engram.semantic import SemanticRetriever


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_smoke_worlds_are_replayable_balanced_and_split_safe(tmp_path: Path) -> None:
    root = tmp_path / "worlds"
    manifest_path = materialize_portability_worlds(root, seed=20260925, scale="smoke")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["format"] == "sparselab-portability-worlds"
    assert manifest["training_fact_count"] == 80
    assert manifest["recipient_memory_fact_count"] == len(
        manifest["recipient_adapter_fact_ids"]
    )
    assert manifest["tokenizer"]["vocab_size"] == 260
    assert manifest["structured_key_dimension"] == 32
    assert manifest["structured_value_dimension"] == 8
    assert len(manifest["worlds"]) == 10
    assert len(manifest["semantic_packs"]) == 9

    producer_rows = [
        row
        for source in sorted((root / "producers").glob("*.jsonl"))
        for row in _jsonl(source)
    ]
    producer_ids = {str(row["id"]) for row in producer_rows}
    assert len(producer_rows) == 40 * len(manifest["worlds"])
    assert len(producer_ids) == len(producer_rows)
    assert all(row["license"] == "CC0-1.0" for row in producer_rows)
    assert all(row["source"] == "generated" for row in producer_rows)

    queries_by_split = {
        split: _jsonl(root / f"{name}_queries.jsonl")
        for split, name in (
            ("train", "training"),
            ("development", "development"),
            ("final", "final"),
        )
    }
    scores_by_split = {
        split: _jsonl(root / f"{name}_scores.jsonl")
        for split, name in (
            ("train", "training"),
            ("development", "development"),
            ("final", "final"),
        )
    }
    query_ids = {
        str(row["case_id"]) for rows in queries_by_split.values() for row in rows
    }
    score_ids = {
        str(row["case_id"]) for rows in scores_by_split.values() for row in rows
    }
    assert producer_ids.isdisjoint(query_ids)
    assert query_ids == score_ids
    assert all(
        "expected_answer" not in row and "expected_path" not in row
        for rows in queries_by_split.values()
        for row in rows
    )
    for rows in queries_by_split.values():
        assert all(len(str(row["lookup_suffix"]).encode("ascii")) == 32 for row in rows)
        assert all(
            str(row["prompt"]).endswith(str(row["lookup_suffix"])) for row in rows
        )

    template_sets = [
        {str(row["template"]) for row in queries_by_split[split]}
        for split in ("train", "development", "final")
    ]
    alias_sets = [
        {
            str(row["alias"])
            for row in queries_by_split[split]
            if row["alias"] is not None
        }
        for split in ("train", "development", "final")
    ]
    for left in range(3):
        for right in range(left + 1, 3):
            assert template_sets[left].isdisjoint(template_sets[right])
            assert alias_sets[left].isdisjoint(alias_sets[right])

    adapter_rows = _jsonl(root / "recipient_adapter_examples.jsonl")
    trained_fact_ids = {
        str(row["fact_id"]) for row in adapter_rows if row["fact_id"] is not None
    }
    assert trained_fact_ids == set(manifest["recipient_adapter_fact_ids"])
    training_pack = manifest["semantic_packs"]["training"]
    assert set(training_pack["record_ids"]) == trained_fact_ids
    replacement_ids = {
        str(row["id"])
        for row in producer_rows
        if row["partition"] in {"development", "final"}
    }
    assert replacement_ids.isdisjoint(trained_fact_ids)
    assert all(
        row["permitted_consumer_role"] == "recipient_adapter_training"
        for row in adapter_rows
    )

    facts_by_id: dict[str, dict[str, object]] = {
        str(row["id"]): row for row in producer_rows
    }
    for world in manifest["worlds"]:
        if world["partition"] != "train":
            continue
        eligible = [
            facts_by_id[str(fact_id)] for fact_id in world["eligible_adapter_fact_ids"]
        ]
        expected_values = set("ABCDEFGH" if int(world["slot"]) % 2 == 0 else "IJKLMNOP")
        assert len([row for row in eligible if row["relation"] == "next"]) == 8
        assert len([row for row in eligible if row["relation"] == "label"]) == 8
        assert all(row["value"] in expected_values for row in eligible)
        for eligible_row in eligible:
            roles = eligible_row["permitted_consumer_roles"]
            assert isinstance(roles, list)
            assert "recipient_adapter_training" in roles
    rows_by_world: dict[str, list[dict[str, object]]] = {}
    for row in producer_rows:
        rows_by_world.setdefault(str(row["world_id"]), []).append(row)
    worlds_by_slot: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for world in manifest["worlds"]:
        replacement = world["replacement"]
        if replacement is not None:
            worlds_by_slot.setdefault(
                (str(world["partition"]), int(world["slot"])), {}
            )[str(replacement)] = world
    for pair in worlds_by_slot.values():
        first, second = pair["a"], pair["b"]
        first_ids = first["producer_fact_ids"]
        second_ids = second["producer_fact_ids"]
        assert isinstance(first_ids, list) and isinstance(second_ids, list)
        assert set(first_ids).isdisjoint(second_ids)
        first_keys = {
            (str(row["subject"]), str(row["relation"]))
            for row in rows_by_world[str(first["world_id"])]
        }
        second_keys = {
            (str(row["subject"]), str(row["relation"]))
            for row in rows_by_world[str(second["world_id"])]
        }
        assert first_keys == second_keys
        first_next = first["next"]
        second_next = second["next"]
        assert isinstance(first_next, dict) and isinstance(second_next, dict)
        for hops in (1, 2, 3):
            left = "A"
            right = "A"
            for _ in range(hops):
                left = str(first_next[left])
                right = str(second_next[right])
            assert left != right
        key_encoder, value_encoder = frozen_encoders()
        slot = int(first["slot"])
        nonce = int(manifest["key_nonce_by_slot_node_relation"][f"{slot}|A|next"])
        query = torch.from_numpy(
            key_encoder.encode(StructuredKey(f"w{slot:02d}.A~{nonce}", "next"))
        )
        for world in (first, second):
            world_id = str(world["world_id"])
            world_next = world["next"]
            assert isinstance(world_next, dict)
            descriptor = manifest["semantic_packs"][world_id]
            retriever = SemanticRetriever.from_pack(
                root / str(descriptor["path"]),
                expected_pack_id=str(descriptor["pack_id"]),
            )
            outcome = retriever.retrieve(
                query,
                key_encoder=key_encoder.identity,
                min_score=3.5,
            )
            assert outcome.status == "hit"
            expected = torch.from_numpy(value_encoder.encode(str(world_next["A"])))
            torch.testing.assert_close(outcome.hits[0].value_vector, expected)

    representation = json.loads(
        (root / "representation_manifest.json").read_text(encoding="utf-8")
    )
    assert representation["tokenizer_sha256"] == manifest["tokenizer"]["sha256"]
    assert all(row["semantic_row_id"] for row in representation["records"])
    assert all(
        row["omission_reason"] == "replacement-world lexical training forbidden"
        for row in representation["records"]
        if row["partition"] != "train"
    )
    assert (
        materialize_portability_worlds(root, seed=20260925, scale="smoke")
        == manifest_path
    )


def test_world_manifest_fails_closed_on_changed_replay(tmp_path: Path) -> None:
    root = tmp_path / "worlds"
    materialize_portability_worlds(root, seed=17, scale="smoke")
    try:
        materialize_portability_worlds(root, seed=41, scale="smoke")
    except FileExistsError:
        pass
    else:
        raise AssertionError(
            "a different data seed must not overwrite an existing world"
        )
