"""Bounded source document iterators."""

from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset

from sparselab.config.models import DatasetConfig
from sparselab.data.synthetic import iter_synthetic

TINYSTORIES_DATASET = "roneneldan/TinyStories"
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"


def iter_documents(config: DatasetConfig, split: str) -> Iterator[str]:
    """Yield source documents in stable stream order, without implicit fallbacks."""
    if split not in {"train", "validation"}:
        raise ValueError(f"unknown split {split!r}")
    if config.source == "synthetic":
        yield from iter_synthetic(config.synthetic_seed, split)
        return
    dataset = load_dataset(
        TINYSTORIES_DATASET,
        name="default",
        split=split,
        revision=config.revision,
        streaming=True,
        cache_dir=str(config.cache_dir),
    )
    for record in dataset:
        if "text" not in record or not isinstance(record["text"], str):
            raise ValueError("TinyStories stream record has missing or non-string text")
        yield record["text"]
