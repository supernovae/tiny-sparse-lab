"""Cold token measurement uses the tokenizer's adjacent frozen export."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab.corpus import token_denominator_identity as identity
from sparselab.training.manifest import sha256_file


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    release = tmp_path / ("a" * 64)
    release.mkdir()
    (release / "manifest.json").write_text("{}\n")
    export = tmp_path / "export"
    tokenizer = export / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("fixture tokenizer")
    (export / "tokenizer.yaml").write_text("fixture config")
    metadata = {
        "sha256": sha256_file(tokenizer),
        "source": "local_text",
        "revision": release.name,
        "vocab_size": 32768,
        "corpus_export": {"export_sha256": "b" * 64},
    }
    tokenizer.with_name("tokenizer_manifest.json").write_text(json.dumps(metadata))
    dataset = SimpleNamespace(
        corpus_release_path=release,
        corpus_export_path=export,
        source="local_text",
        revision=release.name,
    )
    config = SimpleNamespace(
        output_dir=tokenizer.parent, dataset=dataset, vocab_size=32768
    )
    monkeypatch.setattr(identity, "load_tokenizer_config", lambda _: config)
    monkeypatch.setattr(
        identity, "verify_release", lambda _: {"release_id": release.name}
    )
    return release, tokenizer, metadata, dataset


def test_cold_measurement_passes_matching_export_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, tokenizer, metadata, dataset = _fixture(tmp_path, monkeypatch)
    observed = []

    def verify(_path, **kwargs):
        observed.append(kwargs["dataset"])
        return metadata

    monkeypatch.setattr(identity, "verify_tokenizer_artifact", verify)
    manifest, _ = identity._authenticate_inputs(release, tokenizer)
    assert manifest["release_id"] == release.name
    assert observed == [dataset]


def test_cold_measurement_rejects_foreign_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, tokenizer, metadata, dataset = _fixture(tmp_path, monkeypatch)
    dataset.corpus_export_path = tmp_path / "foreign"
    with pytest.raises(ValueError, match="export configuration differs"):
        identity._dataset_for_tokenizer(release, tokenizer, metadata)
