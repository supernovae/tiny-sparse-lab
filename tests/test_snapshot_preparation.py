"""Offline generic snapshots use the native frozen tokenizer and chunk journal."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from dataset_fixtures import offline_hub_identity

from sparselab.config.loading import load_config
from sparselab.config.models import DatasetConfig, RunConfig, TokenizerTrainConfig
from sparselab.data import packing, sources
from sparselab.data.encoding import PreparationEncoder
from sparselab.data.preparation_chunks import PreparationChunks
from sparselab.data.tokenizer import (
    load_tokenizer,
    train_tokenizer,
    verify_tokenizer_artifact,
)
from sparselab.training.manifest import sha256_file


def _inventory(root: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in root.iterdir() if path.is_file()}


@pytest.fixture
def snapshot_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    revision = "a" * 40
    monkeypatch.setattr(sources, "_hub_identity", offline_hub_identity)
    originals = {}

    def make_snapshot(name):
        def stream(source, split, cache, **kw):
            mapped = source.splits[split]
            for index in range(8 if split == "train" else 4):
                yield {
                    source.text_field: (
                        f"{name} {mapped} record {index}: "
                        "the fox crossed a quiet forest."
                    )
                }

        monkeypatch.setattr(sources, "_stream", stream)
        declaration = tmp_path / f"{name}.yaml"
        declaration.write_text(
            yaml.safe_dump(
                {
                    "repo_id": "fixture/preparation",
                    "revision": revision,
                    "config": "default",
                    "splits": {"train": "training", "validation": "heldout"},
                    "text_field": "body",
                    "attribution": "offline fixture",
                    "license": "MIT",
                    "selection": {"mode": "exhaustion"},
                    "resources": {
                        "max_source_records": 100,
                        "max_text_bytes": 100000,
                        "max_record_bytes": 10000,
                        "max_work_bytes": 10000000,
                        "min_free_bytes": 0,
                    },
                }
            )
        )
        lock = sources.lock_source(
            declaration, tmp_path / f"{name}.lock.json", tmp_path / "hub"
        )
        manifest = sources.snapshot_source(lock, tmp_path / name, tmp_path / "hub")
        originals[manifest.parent] = _inventory(manifest.parent)
        return DatasetConfig(
            source="snapshot",
            revision=revision,
            license="MIT",
            dataset_config="default",
            cache_dir=tmp_path / "cache",
            source_manifest_path=manifest,
            train_path=manifest.parent / "train.jsonl",
            validation_path=manifest.parent / "validation.jsonl",
            train_max_documents=8,
            validation_max_documents=4,
            train_max_tokens=100000,
            validation_max_tokens=100000,
        )

    dataset = make_snapshot("original")
    training = TokenizerTrainConfig(
        schema_version=1,
        vocab_size=260,
        min_frequency=1,
        max_documents=8,
        output_dir=tmp_path / "tokenizer",
        dataset=dataset,
    )
    tokenizer_path = train_tokenizer(training)
    base = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    )
    run = RunConfig.model_validate(
        {
            **base.model_dump(mode="python"),
            "dataset": dataset,
            "tokenizer": {"path": tokenizer_path},
            "model": {**base.model.model_dump(), "vocab_size": 260},
            "logging": {**base.logging.model_dump(), "root_dir": tmp_path / "runs"},
        }
    )
    yield SimpleNamespace(
        run=run,
        tokenizer=load_tokenizer(tokenizer_path),
        training=training,
        make_snapshot=make_snapshot,
        root=tmp_path,
    )
    # Every success/failure/resume path must preserve the source publication exactly.
    for root, inventory in originals.items():
        assert _inventory(root) == inventory
        sources.verify_snapshot(root / "manifest.json")


def _small_chunks(monkeypatch):
    original = PreparationChunks.__init__

    def initialize(self, *args, **kwargs):
        kwargs["record_limit"] = 2
        original(self, *args, **kwargs)

    monkeypatch.setattr(PreparationChunks, "__init__", initialize)


@pytest.mark.parametrize("tamper", [False, True])
def test_preparation_resumes_sealed_native_chunks(
    snapshot_fixture, monkeypatch, tamper
):
    fixture = snapshot_fixture
    _small_chunks(monkeypatch)
    encode = PreparationEncoder.encode
    calls = 0

    def interrupt(self, batch):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("interrupted after sealed chunks")
        return encode(self, batch)

    monkeypatch.setattr(PreparationEncoder, "encode", interrupt)
    with pytest.raises(RuntimeError, match="interrupted after sealed chunks"):
        packing.prepare_data(
            fixture.run, fixture.tokenizer, tokenizer_batch_documents=2
        )
    (staging,) = fixture.run.dataset.cache_dir.glob("*.tmp")
    receipts = sorted(staging.glob("train-*.receipt.json"))
    assert receipts
    committed = sum(
        json.loads(path.read_text())["acquired_documents"] for path in receipts
    )
    assert 0 < committed < 8
    original_receipts = {
        path.name: (path.stat().st_ino, path.read_bytes()) for path in receipts
    }
    if tamper:
        raw = next(staging.glob("train-*.ids.raw"))
        with raw.open("r+b") as handle:
            handle.write(b"\xff")
    replayed = []
    checked_receipts = set()
    validate = PreparationChunks._validate_receipt

    def validate_existing(self, split, index, payload):
        name = f"{split}-{index:06d}.receipt.json"
        if name in original_receipts:
            path = self.root / name
            assert (path.stat().st_ino, path.read_bytes()) == original_receipts[name]
            checked_receipts.add(name)
        return validate(self, split, index, payload)

    def record(self, batch):
        replayed.extend(batch)
        return encode(self, batch)

    monkeypatch.setattr(PreparationChunks, "_validate_receipt", validate_existing)
    monkeypatch.setattr(PreparationEncoder, "encode", record)
    if tamper:
        with pytest.raises(ValueError, match="chunk|digest|changed"):
            packing.prepare_data(
                fixture.run, fixture.tokenizer, tokenizer_batch_documents=2
            )
        assert replayed == []
        return
    prepared = packing.prepare_data(
        fixture.run, fixture.tokenizer, tokenizer_batch_documents=2
    )
    assert checked_receipts == set(original_receipts)
    texts = list(sources.iter_snapshot(fixture.run.dataset, "train"))
    heldout = list(sources.iter_snapshot(fixture.run.dataset, "validation"))
    assert replayed == texts[committed:] + heldout
    eos = fixture.tokenizer.token_to_id("<eos>")
    expected = [
        token
        for text in texts
        for token in [
            *fixture.tokenizer.encode(text, add_special_tokens=False).ids,
            eos,
        ]
    ]
    assert np.array_equal(prepared.train, np.asarray(expected, dtype=np.int32))
    assert prepared.manifest["train"]["retained_documents"] == 8
    assert prepared.manifest["train"]["truncated_documents"] == 0
    assert not staging.exists()


def test_snapshot_packing_fails_before_partial_document_publication(snapshot_fixture):
    fixture = snapshot_fixture
    first = next(sources.iter_snapshot(fixture.run.dataset, "train"))
    cap = len(fixture.tokenizer.encode(first, add_special_tokens=False).ids) + 2
    cfg = fixture.run.model_copy(
        update={
            "dataset": fixture.run.dataset.model_copy(update={"train_max_tokens": cap})
        }
    )
    with pytest.raises(ValueError, match="token cap would truncate selected story 2"):
        packing.prepare_data(cfg, fixture.tokenizer, tokenizer_batch_documents=1)
    assert not list(cfg.dataset.cache_dir.glob("*/manifest.json"))
    assert not list(cfg.dataset.cache_dir.glob("*/train.npy"))


def test_snapshot_packing_consumes_bounded_batches_and_returns_mmaps(
    snapshot_fixture, monkeypatch
):
    fixture = snapshot_fixture
    streamed = encoded = peak_pending = 0
    iterate = sources.iter_snapshot
    encode = PreparationEncoder.encode

    def guarded(config, split):
        nonlocal streamed, peak_pending
        for text in iterate(config, split):
            streamed += 1
            peak_pending = max(peak_pending, streamed - encoded)
            assert streamed - encoded <= 3, (
                "snapshot materialized before bounded encoding"
            )
            yield text

    def observe(self, batch):
        nonlocal encoded
        assert 1 <= len(batch) <= 2
        assert sum(len(text.encode()) for text in batch) <= 512
        result = encode(self, batch)
        encoded += len(batch)
        return result

    def eager_collector(*args, **kwargs):
        pytest.fail("generic snapshot used corpus-sized scalar collector")

    monkeypatch.setattr(sources, "iter_snapshot", guarded)
    monkeypatch.setattr(PreparationEncoder, "encode", observe)
    monkeypatch.setattr(packing, "_collect", eager_collector)
    prepared = packing.prepare_data(
        fixture.run,
        fixture.tokenizer,
        tokenizer_batch_documents=2,
        tokenizer_batch_source_bytes=512,
    )
    assert streamed == encoded == 12
    assert peak_pending <= 3
    assert isinstance(prepared.train, np.memmap)
    assert isinstance(prepared.validation, np.memmap)
    assert prepared.train_supervision is None
    assert prepared.manifest["supervision"] == {"kind": "all_tokens"}


def test_frozen_tokenizer_rejects_other_verified_snapshot(snapshot_fixture):
    fixture = snapshot_fixture
    path = fixture.run.tokenizer.path
    before = _inventory(path.parent)
    verify_tokenizer_artifact(
        path,
        source="snapshot",
        revision=fixture.run.dataset.revision,
        vocab_size=260,
        dataset=fixture.run.dataset,
    )
    other = fixture.make_snapshot("different")
    sources.verify_snapshot(other)
    with pytest.raises(ValueError, match="tokenizer snapshot identity mismatch"):
        verify_tokenizer_artifact(
            path,
            source="snapshot",
            revision=other.revision,
            vocab_size=260,
            dataset=other,
        )
    with pytest.raises(
        FileExistsError, match="different or unverifiable training provenance"
    ):
        train_tokenizer(fixture.training.model_copy(update={"dataset": other}))
    assert _inventory(path.parent) == before


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("binding", ["config_path", "supplied_object"])
def test_preparation_rejects_valid_tokenizer_from_other_snapshot(
    snapshot_fixture, cached, binding
):
    fixture = snapshot_fixture
    if cached:
        packing.prepare_data(fixture.run, fixture.tokenizer)
    cache = fixture.run.dataset.cache_dir
    before = {
        str(path.relative_to(cache)): sha256_file(path)
        for path in cache.rglob("*")
        if path.is_file() and path.name != ".cleanup.lock"
    }
    other = fixture.make_snapshot("other-tokenizer-source")
    # A byte-only vocabulary can be identical across snapshots. For the object
    # guard, use one additional learned entry to guarantee different model bytes.
    vocab_size = 261 if binding == "supplied_object" else 260
    path = train_tokenizer(
        fixture.training.model_copy(
            update={
                "dataset": other,
                "output_dir": fixture.root / "other-tokenizer",
                "vocab_size": vocab_size,
            }
        )
    )
    verify_tokenizer_artifact(
        path,
        source="snapshot",
        revision=other.revision,
        vocab_size=vocab_size,
        dataset=other,
    )
    supplied = load_tokenizer(path)
    cfg = fixture.run
    if binding == "config_path":
        cfg = cfg.model_copy(
            update={"tokenizer": cfg.tokenizer.model_copy(update={"path": path})}
        )
        reason = "tokenizer snapshot identity mismatch"
    else:
        reason = "supplied tokenizer differs from verified snapshot tokenizer"
    with pytest.raises(ValueError, match=reason):
        packing.prepare_data(cfg, supplied)
    assert {
        str(path.relative_to(cache)): sha256_file(path)
        for path in cache.rglob("*")
        if path.is_file() and path.name != ".cleanup.lock"
    } == before


def test_tokenizer_byte_cap_selects_whole_documents(snapshot_fixture):
    fixture = snapshot_fixture
    first = next(sources.iter_snapshot(fixture.run.dataset, "train"))
    size = len(first.encode())
    cfg = fixture.training.model_copy(
        update={
            "dataset": fixture.run.dataset.model_copy(
                update={"train_max_tokens": size + 1}
            ),
            "output_dir": fixture.root / "bounded-tokenizer",
        }
    )
    path = train_tokenizer(cfg)
    manifest = json.loads(path.with_name("tokenizer_manifest.json").read_text())
    contract = manifest["training_contract"]
    assert contract["selected_input_bytes_utf8"] == size
    assert contract["acquired_documents"] == 2
    assert contract["stop_reason"] == "byte_limit"
    assert manifest["selected_documents"] == 1
    assert manifest["source_manifest_sha256"] == sha256_file(
        cfg.dataset.source_manifest_path
    )


def test_relocated_snapshot_preserves_config_and_tokenizer_identity(snapshot_fixture):
    import shutil

    from sparselab.training.manifest import config_sha256

    fixture = snapshot_fixture
    moved = fixture.root / "relocated"
    shutil.copytree(fixture.run.dataset.source_manifest_path.parent, moved)
    dataset = fixture.run.dataset.model_copy(
        update={
            key: moved / getattr(fixture.run.dataset, key).name
            for key in ("source_manifest_path", "train_path", "validation_path")
        }
    )
    relocated = fixture.run.model_copy(update={"dataset": dataset})
    assert sources.verify_snapshot(dataset) == sources.verify_snapshot(
        fixture.run.dataset
    )
    assert config_sha256(relocated.model_dump(mode="json")) == config_sha256(
        fixture.run.model_dump(mode="json")
    )
    verify_tokenizer_artifact(
        fixture.run.tokenizer.path,
        source="snapshot",
        revision=dataset.revision,
        vocab_size=260,
        dataset=dataset,
    )


def test_legacy_local_stories_config_hash_is_unchanged():
    from sparselab.training.manifest import config_sha256

    # Pinned with HEAD's pre-snapshot config_sha256, not the implementation under test.
    payload = {
        "schema_version": 2,
        "name": "legacy-story-identity",
        "dataset": {
            "source": "local_stories",
            "source_manifest_path": "/frozen/story/manifest.json",
            "train_path": "/frozen/story/train.jsonl",
            "validation_path": "/frozen/story/validation.jsonl",
            "cache_dir": "/cache",
            "revision": "a" * 40,
        },
        "training": {"seq_len": 16, "max_steps": 4, "max_tokens": 128},
    }
    expected = "f13b7425749949e099f5453e4e57bb16a59696b6fcbeaffe56f470ea5bb985ca"
    assert config_sha256(payload) == expected
    payload["dataset"]["source_manifest_path"] = "/relocated/manifest.json"
    assert config_sha256(payload) != expected
