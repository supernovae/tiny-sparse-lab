from __future__ import annotations

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
