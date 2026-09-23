from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
import torch
from pydantic import ValidationError
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from sparselab.config.loading import load_config
from sparselab.config.models import AttentionConfig
from sparselab.data.tokenizer import SPECIAL_TOKENS, load_tokenizer
from sparselab.engram.packs import compile_pack, verify_pack
from sparselab.evaluation.generation import _prompt_byte_addresses
from sparselab.experiments.study import plan_study
from sparselab.research.catalog import (
    ResearchEntry,
    list_lessons,
    list_research,
    load_profiles,
    load_research,
)
from sparselab.research.probe import probe_model
from sparselab.research.scaffold import scaffold_lesson, scaffold_research
from sparselab.training.manifest import canonical_json


def _write_byte_tokenizer(path: Path) -> None:
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token in sorted(ByteLevel.alphabet()):
        vocabulary.setdefault(token, len(vocabulary))
    tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _rows(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return cast(list[dict[str, object]], value)


def _shape(value: object) -> list[int]:
    assert isinstance(value, list)
    return cast(list[int], value)


def _weight_rows(value: object) -> list[list[float]]:
    assert isinstance(value, list)
    assert all(isinstance(row, list) for row in value)
    return cast(list[list[float]], value)


def test_packaged_catalog_profiles_and_strict_versions() -> None:
    entries = list_research()
    lessons = list_lessons()

    assert {entry.id for entry in entries} == {
        "engram-ffn-substitution-v1",
        "engram-mla-compression-v1",
        "engram-moe-capacity-v1",
        "engram-placement-v1",
        "engram-sparse-budget-v1",
        "lexical-memory-heavy-v1",
    }
    assert len(lessons) == 10
    profiles = load_profiles().scales
    assert set(profiles) == {"smoke", "nano", "micro", "tiny"}
    assert [profiles[name].hidden_dim for name in profiles] == [64, 128, 320, 512]

    invalid = entries[0].model_dump(mode="json")
    invalid["version"] = True
    with pytest.raises(ValidationError, match="version must be integer 1"):
        ResearchEntry.model_validate(invalid)


def test_catalog_rejects_duplicate_json_and_lists_unknown_ids(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"format":"sparselab-research-entry","version":1,"version":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_research(duplicate)

    with pytest.raises(ValueError, match="available IDs") as error:
        load_research("not-a-research-entry")
    assert "engram-ffn-substitution-v1" in str(error.value)


def test_research_scaffold_binds_configs_without_preparing_or_overwriting(
    tmp_path: Path,
) -> None:
    output = scaffold_research(
        "engram-ffn-substitution-v1", tmp_path / "study", scale="smoke", data="offline"
    )
    plan = plan_study(output / "study.yaml")
    metadata = json.loads((output / "research.json").read_text(encoding="utf-8"))

    assert len(plan.expanded) == 18
    assert len(plan.pairs) == 21
    assert {item.config.seed for item in plan.expanded} == {17, 41, 73}
    assert len({item["config_sha256"] for item in metadata["coordinates"]}) == 18
    for item in metadata["inputs"]:
        content = (output / item["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == item["sha256"]
    unsigned = {
        key: value for key, value in metadata.items() if key != "research_sha256"
    }
    assert (
        hashlib.sha256(canonical_json(unsigned)).hexdigest()
        == metadata["research_sha256"]
    )
    assert not (output / "artifacts").exists()
    assert not (output / "runs").exists()

    marker = output / "keep.txt"
    marker.write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        scaffold_research("engram-ffn-substitution-v1", output, scale="smoke")
    assert marker.read_text(encoding="utf-8") == "user data"


def test_research_scaffold_rebases_paths_from_symlinked_destination(
    tmp_path: Path,
) -> None:
    alias = tmp_path.parent / f"{tmp_path.name}-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    output = scaffold_research(
        "engram-ffn-substitution-v1", alias / "study", scale="smoke", data="offline"
    )
    config_path = next((output / "configs").glob("*.yaml"))
    config = load_config(config_path)
    root = output.resolve()

    assert config.tokenizer.path == root / "artifacts/tokenizer/tokenizer.json"
    assert config.dataset.cache_dir == root / "artifacts/data"
    assert config.logging.root_dir == root / "runs"


def test_initialized_probe_bounds_and_reports_mechanisms(tmp_path: Path) -> None:
    lesson = scaffold_lesson("dense", tmp_path / "lesson", scale="smoke")
    config = load_config(lesson / "model.yaml")
    config = config.model_copy(
        update={"model": config.model.model_copy(update={"vocab_size": 260})}
    )
    tokenizer_path = config.tokenizer.path
    _write_byte_tokenizer(tokenizer_path)
    prompt = "a " * 12
    rng_state = torch.random.get_rng_state().clone()
    dense = probe_model(config, prompt)
    assert dense["execution"] == "cpu-fp32-reference"
    assert dense["logits_shape"] == [1, 24, config.model.vocab_size]
    assert set(_mapping(dense["source_identity"])) == {"algorithm", "sha256"}

    moe_model = config.model.model_copy(
        update={
            "ffn": "moe",
            "num_experts": 4,
            "experts_per_token": 2,
            "router_aux_loss_coefficient": 0.01,
        }
    )
    moe = probe_model(config.model_copy(update={"model": moe_model}), prompt)
    route = _mapping(_rows(moe["routing_samples"])[0])
    experts = _mapping(route["selected_experts"])
    weights = _mapping(route["selected_weights"])
    assert _shape(experts["sample_shape"]) == [8, 2]
    assert _shape(weights["sample_shape"]) == [8, 2]
    assert all(abs(sum(row) - 1.0) < 1e-6 for row in _weight_rows(weights["values"]))

    sparse_config = config.model_copy(
        update={
            "attention": AttentionConfig(
                kind="block_sparse", block_size=4, selected_blocks=2
            )
        }
    )
    sparse_result = _rows(probe_model(sparse_config, prompt)["sparse_samples"])[0]
    sparse = _mapping(sparse_result["selected_blocks"])
    assert _shape(sparse["sample_shape"]) == [8]
    assert sparse["truncated"] is True

    mla = probe_model(
        config.model_copy(
            update={"attention": AttentionConfig(kind="mla", latent_dim=32)}
        ),
        prompt,
    )
    kv_widths = []
    for layer in _rows(mla["layers"]):
        name = layer["name"]
        if isinstance(name, str) and name.endswith("kv_down"):
            kv_widths.append(_shape(layer["output_shape"])[-1])
    assert kv_widths == [32, 32]

    byte_model = config.model.model_copy(
        update={
            "memory": "byte",
            "memory_table_size": 257,
            "memory_ngram_size": 3,
            "memory_dim": 8,
        }
    )
    byte_config = config.model_copy(update={"model": byte_model})
    unicode_prompt = "café"
    byte_result = probe_model(byte_config, unicode_prompt)
    tokenizer = load_tokenizer(tokenizer_path)
    expected = _prompt_byte_addresses(
        tokenizer,
        unicode_prompt,
        tokenizer.encode(unicode_prompt, add_special_tokens=False).ids,
        257,
        3,
    )
    addresses = _mapping(
        _mapping(byte_result["memory_lookup_addresses"])["memory.table"]
    )
    assert addresses["values"] == expected[:8]
    input_addresses = _mapping(
        _mapping(byte_result["memory_lookup_addresses"])["input.byte_addresses"]
    )
    assert input_addresses["values"] == [expected[:8]]
    assert torch.equal(rng_state, torch.random.get_rng_state())
    json.dumps(byte_result)

    with pytest.raises(ValueError, match="128-token bound"):
        probe_model(config, "a" * 129)
    assert torch.equal(rng_state, torch.random.get_rng_state())


def test_engrampack_lesson_emits_compile_ready_jsonl(tmp_path: Path) -> None:
    lesson = scaffold_lesson("engrampack", tmp_path / "engrampack")
    source = lesson / "records.jsonl"
    records = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert {record["id"] for record in records} == {
        "tutorial-atlas-maps-to-beacon",
        "tutorial-beacon-maps-to-amber",
        "tutorial-cipher-maps-to-delta",
    }

    pack = tmp_path / "compiled.enpack"
    manifest = compile_pack(
        source,
        pack,
        name="tutorial-map",
        namespace="tutorial",
        default_license="CC0-1.0",
        source_name="original-tutorial-records",
        created_at="2026-09-23T00:00:00Z",
    )
    assert manifest.record_count == 3
    assert verify_pack(pack).valid
