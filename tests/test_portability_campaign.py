from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sparselab.data.allocation import (
    OWNER_LEXICAL,
    OWNER_NEURAL,
    OWNER_SEMANTIC,
    load_allocation_manifest,
)
from sparselab.data.portability_worlds import materialize_portability_worlds
from sparselab.data.tokenizer import load_tokenizer
from sparselab.engram.packs import verify_pack
from sparselab.research.portability_campaign import (
    build_portability_allocation,
    build_portability_memory_assets,
)
from sparselab.research.portability_runner import _same_prediction
from sparselab.training.manifest import source_identity


def test_memory_asset_compiler_builds_token_byte_and_semantic_replacements(
    tmp_path: Path,
) -> None:
    world_manifest_path = materialize_portability_worlds(
        tmp_path / "worlds", seed=17, scale="smoke"
    )
    asset_root = tmp_path / "assets"
    asset_manifest_path = build_portability_memory_assets(world_manifest_path, asset_root)
    manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))

    assert manifest["format"] == "sparselab-portability-memory-assets"
    assert manifest["compiler"] == "direct-structured-memory-compiler-v1"
    assert set(manifest["representations"]) == {"token", "byte", "semantic"}

    token_training = manifest["representations"]["token"]["real"]["training"]
    assert token_training["addressing"]["tokenizer_sha256"]
    assert token_training["tensor_sha256"]
    byte_training = manifest["representations"]["byte"]["real"]["training"]
    assert byte_training["addressing"]["hashing"] == "poly257-terminal-v1"
    assert byte_training["tensor_sha256"]

    worlds = manifest["representations"]["semantic"]["real"]
    development = [name for name in worlds if name.startswith("development-")]
    assert len(development) == 4
    pack_ids = set()
    for world_id in development:
        descriptor = worlds[world_id]
        pack_path = asset_root / descriptor["artifact"]["path"]
        report = verify_pack(pack_path, expected_pack_id=descriptor["pack_id"])
        assert report.valid
        pack_ids.add(descriptor["pack_id"])
    assert len(pack_ids) == len(development)

    addresses = json.loads((asset_root / "address_observations.json").read_text())
    assert addresses["addressing"]["byte_order"] == 32
    assert addresses["addressing"]["byte"] == "raw-utf8-poly257-terminal-v1"
    assert all(
        row["cross_key_collision_count"] == 0
        for row in addresses["records"]
        if row["scope"] == "training"
    )
    assert build_portability_memory_assets(world_manifest_path, asset_root) == asset_manifest_path
    asset_manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_portability_memory_assets(world_manifest_path, asset_root)


def test_portability_allocation_aligns_owned_targets_and_structured_queries(
    tmp_path: Path,
) -> None:
    world_manifest_path = materialize_portability_worlds(
        tmp_path / "worlds", seed=41, scale="smoke"
    )
    world_manifest = json.loads(world_manifest_path.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(
        world_manifest_path.parent / world_manifest["tokenizer"]["path"]
    )
    allocation_tokenizer_sha256 = hashlib.sha256(
        tokenizer.to_str().encode("utf-8")
    ).hexdigest()
    asset_root = tmp_path / "assets"
    asset_manifest_path = build_portability_memory_assets(world_manifest_path, asset_root)
    assets = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    training_pack = (
        asset_root
        / assets["representations"]["semantic"]["real"]["training"]["artifact"]["path"]
    )

    semantic_root = tmp_path / "allocation-semantic"
    semantic_input = build_portability_allocation(
        world_manifest_path,
        semantic_root,
        representation="semantic",
        condition="adapter-tuned",
        semantic_pack_path=training_pack,
    )
    semantic_manifest = load_allocation_manifest(
        semantic_input["allocation_manifest_path"],
        source_identity_sha256=str(source_identity()["sha256"]),
        tokenizer_sha256=allocation_tokenizer_sha256,
    )
    semantic_sides = semantic_manifest.split(
        "train", token_count=semantic_input["train_token_count"]
    )
    semantic_owner = semantic_sides.owner
    semantic_queries = semantic_sides.semantic_queries
    semantic_mask = semantic_sides.semantic_mask
    assert semantic_queries is not None
    assert semantic_mask is not None
    assert int((semantic_owner == OWNER_SEMANTIC).sum()) > 0
    assert int(semantic_mask.sum()) == int((semantic_owner == OWNER_SEMANTIC).sum())
    assert semantic_queries.shape[1] == 32

    lexical_root = tmp_path / "allocation-token"
    lexical_input = build_portability_allocation(
        world_manifest_path,
        lexical_root,
        representation="token",
        condition="adapter-tuned",
        semantic_pack_path=None,
    )
    lexical_manifest = load_allocation_manifest(
        lexical_input["allocation_manifest_path"],
        source_identity_sha256=str(source_identity()["sha256"]),
        tokenizer_sha256=allocation_tokenizer_sha256,
    )
    lexical_sides = lexical_manifest.split(
        "train", token_count=lexical_input["train_token_count"]
    )
    lexical_owner = lexical_sides.owner
    lexical_queries = lexical_sides.semantic_queries
    lexical_mask = lexical_sides.semantic_mask
    assert int((lexical_owner == OWNER_LEXICAL).sum()) > 0
    assert lexical_queries is None
    assert lexical_mask is None

    disabled_root = tmp_path / "allocation-disabled"
    disabled = build_portability_allocation(
        world_manifest_path,
        disabled_root,
        representation="semantic",
        condition="disabled",
        semantic_pack_path=None,
    )
    disabled_manifest = load_allocation_manifest(
        disabled["allocation_manifest_path"],
        source_identity_sha256=str(source_identity()["sha256"]),
        tokenizer_sha256=allocation_tokenizer_sha256,
    )
    disabled_sides = disabled_manifest.split(
        "train", token_count=disabled["train_token_count"]
    )
    disabled_owner = disabled_sides.owner
    disabled_queries = disabled_sides.semantic_queries
    disabled_mask = disabled_sides.semantic_mask
    assert int((disabled_owner == OWNER_NEURAL).sum()) == disabled_owner.size
    assert disabled_queries is None
    assert disabled_mask is None


def test_swap_reproducibility_requires_exact_answer_and_path() -> None:
    first = {"predicted_answer": ":", "predicted_path": ["A"]}
    same_path_different_answer = {"predicted_answer": "?", "predicted_path": ["A"]}
    same_answer_different_path = {
        "predicted_answer": "G",
        "predicted_path": ["A", "D", "G"],
    }
    exact_match = {"predicted_answer": "G", "predicted_path": ["A", "H", "G"]}

    assert not _same_prediction(first, same_path_different_answer)
    assert not _same_prediction(exact_match, same_answer_different_path)
    assert _same_prediction(exact_match, dict(exact_match))
