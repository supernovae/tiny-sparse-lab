from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sparselab.config.loading import load_config
from sparselab.config.migrate import migrate_file
from sparselab.config.models import ModelConfig, RunConfig


def _v1() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "migration-test",
        "seed": 7,
        "device": "cpu",
        "model": {
            "vocab_size": 512,
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "ffn_dim": 32,
            "max_seq_len": 16,
        },
        "tokenizer": {"path": "assets/tokenizer.json"},
        "dataset": {
            "source": "synthetic",
            "cache_dir": "cache",
            "train_max_documents": 2,
            "validation_max_documents": 2,
            "train_max_tokens": 64,
            "validation_max_tokens": 64,
        },
        "training": {"batch_size": 1, "seq_len": 8, "max_steps": 20, "max_tokens": 80},
        "logging": {"root_dir": "runs", "checkpoint_every_steps": 10},
    }


def test_v1_loading_gives_migration_guidance(tmp_path: Path) -> None:
    source = tmp_path / "legacy.yaml"
    source.write_text(yaml.safe_dump(_v1()), encoding="utf-8")
    with pytest.raises(ValueError, match="sparselab config migrate"):
        load_config(source)


def test_migration_rebases_relative_paths_for_moved_output(tmp_path: Path) -> None:
    source = tmp_path / "input" / "legacy.yaml"
    source.parent.mkdir()
    source.write_text(yaml.safe_dump(_v1(), sort_keys=False), encoding="utf-8")
    output = tmp_path / "published" / "nested" / "config.yaml"

    migrate_file(source, output)
    config = load_config(output)

    assert config.tokenizer.path == source.parent / "assets/tokenizer.json"
    assert config.dataset.cache_dir == source.parent / "cache"
    assert config.logging.root_dir == source.parent / "runs"


def test_invalid_migration_does_not_publish_output(tmp_path: Path) -> None:
    source = tmp_path / "invalid.yaml"
    invalid = _v1()
    invalid["training"] = {
        "batch_size": 1,
        "seq_len": 32,
        "max_steps": 20,
        "max_tokens": 80,
    }
    source.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    output = tmp_path / "output.yaml"

    with pytest.raises(ValueError, match="migrated config is invalid"):
        migrate_file(source, output)
    assert not output.exists()


def test_memory_injection_serialization_preserves_legacy_final_config() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "context_study_dense_s17_b24k.yaml"
    )
    explicit_final = config.model_copy(
        update={
            "model": config.model.model_copy(update={"memory_injection": "final"})
        }
    )

    for current in (config, explicit_final):
        assert "memory_injection" not in current.model_dump(mode="python")["model"]
        assert "memory_injection" not in current.model_dump(mode="json")["model"]
        assert "memory_hash_heads" in current.model_dump(mode="json")["model"]
        assert "memory_injection" not in current.model_dump_json()
        assert RunConfig.model_validate(current.model_dump(mode="json")).model.memory_injection == "final"

    assert config.model_dump(mode="python") == explicit_final.model_dump(mode="python")
    assert config.model_dump(mode="json") == explicit_final.model_dump(mode="json")
    assert config.model_dump_json() == explicit_final.model_dump_json()

    enabled = {
        **config.model.model_dump(mode="python"),
        "memory": "ngram",
        "memory_table_size": 31,
        "memory_ngram_size": 3,
        "memory_dim": 8,
        "memory_injection": "embedding",
    }
    embedded = config.model_copy(update={"model": ModelConfig(**enabled)})
    assert embedded.model_dump(mode="python")["model"]["memory_injection"] == "embedding"
    assert embedded.model_dump(mode="json")["model"]["memory_injection"] == "embedding"
    assert (
        RunConfig.model_validate_json(embedded.model_dump_json()).model.memory_injection
        == "embedding"
    )


@pytest.mark.parametrize("value", [None, "middle", 2, [1]])
def test_memory_injection_rejects_unsupported_values(value: object) -> None:
    settings = {
        "vocab_size": 512,
        "hidden_dim": 16,
        "num_layers": 1,
        "num_heads": 2,
        "ffn_dim": 32,
        "max_seq_len": 16,
        "memory": "ngram",
        "memory_table_size": 31,
        "memory_ngram_size": 3,
        "memory_dim": 8,
        "memory_injection": value,
    }
    with pytest.raises(ValueError):
        ModelConfig(**settings)


def test_embedding_memory_injection_requires_enabled_memory() -> None:
    with pytest.raises(ValueError, match="requires enabled memory"):
        ModelConfig(
            vocab_size=512,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=16,
            memory_injection="embedding",
        )
