from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sparselab.config.loading import load_config
from sparselab.config.migrate import migrate_file


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
