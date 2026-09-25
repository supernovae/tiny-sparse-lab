from __future__ import annotations

import json
from pathlib import Path

from sparselab.config.models import DatasetConfig, TokenizerTrainConfig
from sparselab.data import tokenizer as tokenizer_module


def test_tokenizer_acquisition_does_not_request_past_document_bound(
    monkeypatch, tmp_path: Path
) -> None:
    requested = 0

    def documents(_config: DatasetConfig, _split: str):
        nonlocal requested
        for value in ("one", "must not be requested"):
            requested += 1
            yield value

    monkeypatch.setattr(tokenizer_module, "iter_documents", documents)
    config = TokenizerTrainConfig(
        schema_version=1,
        vocab_size=260,
        min_frequency=1,
        max_documents=1,
        output_dir=tmp_path / "tokenizer",
        dataset=DatasetConfig(
            source="synthetic",
            cache_dir=tmp_path / "cache",
            train_max_documents=1,
            validation_max_documents=1,
            train_max_tokens=16,
            validation_max_tokens=16,
        ),
    )

    tokenizer_module.train_tokenizer(config)
    assert requested == 1


def test_tokenizer_training_respects_utf8_byte_budget(
    monkeypatch, tmp_path: Path
) -> None:
    requested = 0

    def documents(_config: DatasetConfig, _split: str):
        nonlocal requested
        for value in ("abc", "defgh", "must not be requested"):
            requested += 1
            yield value

    monkeypatch.setattr(tokenizer_module, "iter_documents", documents)
    config = TokenizerTrainConfig(
        schema_version=1,
        vocab_size=260,
        min_frequency=1,
        max_documents=10,
        output_dir=tmp_path / "tokenizer",
        dataset=DatasetConfig(
            source="synthetic",
            cache_dir=tmp_path / "cache",
            train_max_documents=10,
            validation_max_documents=1,
            train_max_tokens=6,
            validation_max_tokens=16,
        ),
    )

    path = tokenizer_module.train_tokenizer(config)
    manifest = json.loads(
        path.with_name("tokenizer_manifest.json").read_text(encoding="utf-8")
    )

    assert requested == 2
    assert manifest["selected_documents"] == 2
    assert manifest["selected_input_bytes_utf8"] == 6
    assert manifest["training_contract"]["input_byte_budget_utf8"] == 6


def test_tokenizer_byte_budget_does_not_split_utf8_codepoint(
    monkeypatch, tmp_path: Path
) -> None:
    requested = 0

    def documents(_config: DatasetConfig, _split: str):
        nonlocal requested
        for value in ("ok", "éx", "must not be requested"):
            requested += 1
            yield value

    monkeypatch.setattr(tokenizer_module, "iter_documents", documents)
    config = TokenizerTrainConfig(
        schema_version=1,
        vocab_size=260,
        min_frequency=1,
        max_documents=10,
        output_dir=tmp_path / "tokenizer",
        dataset=DatasetConfig(
            source="synthetic",
            cache_dir=tmp_path / "cache",
            train_max_documents=10,
            validation_max_documents=1,
            train_max_tokens=3,
            validation_max_tokens=16,
        ),
    )

    path = tokenizer_module.train_tokenizer(config)
    manifest = json.loads(
        path.with_name("tokenizer_manifest.json").read_text(encoding="utf-8")
    )

    assert requested == 2
    assert manifest["selected_documents"] == 1
    assert manifest["selected_input_bytes_utf8"] == 2
