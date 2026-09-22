from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from sparselab.config.models import DatasetConfig
from sparselab.data import datasets


class Stream:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records
        self.skipped = 0

    def skip(self, count: int) -> Stream:
        self.skipped = count
        return self

    def __iter__(self) -> Iterator[dict[str, object]]:
        return iter(self.records[self.skipped :])


def remote_config(tmp_path: Path, source: str) -> DatasetConfig:
    return DatasetConfig(
        source=source,
        revision="pinned-revision",
        dataset_config="sample",
        cache_dir=tmp_path,
        train_max_documents=2,
        validation_max_documents=1,
        train_max_tokens=32,
        validation_max_tokens=16,
    )


@pytest.mark.parametrize("source", ("fineweb_edu", "cosmopedia"))
def test_train_only_remote_sources_use_disjoint_validation_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str
) -> None:
    calls: list[dict[str, object]] = []

    def load(dataset_name: str, **kwargs: object) -> Stream:
        calls.append({"dataset_name": dataset_name, **kwargs})
        return Stream([{"text": "zero"}, {"text": "one"}, {"text": "two"}])

    monkeypatch.setattr(datasets, "load_dataset", load)
    config = remote_config(tmp_path, source)
    assert list(datasets.iter_documents(config, "train")) == ["zero", "one", "two"]
    assert list(datasets.iter_documents(config, "validation")) == ["two"]
    assert all(call["split"] == "train" for call in calls)
    assert all(call["streaming"] is True for call in calls)


def test_remote_sources_require_subset_and_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="revision"):
        DatasetConfig(
            source="fineweb_edu",
            dataset_config="sample",
            cache_dir=tmp_path,
            train_max_documents=1,
            validation_max_documents=1,
            train_max_tokens=1,
            validation_max_tokens=1,
        )
