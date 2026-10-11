from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
import yaml
from dataset_fixtures import offline_hub_identity
from test_training import config

from sparselab.config.loading import load_config
from sparselab.data.coverage import (
    _block_counts,
    _source_coverage,
    coverage,
    derive_budget,
    register_parser,
)
from sparselab.data.packing import TokenBlockDataset, prepare_data
from sparselab.data.tokenizer import load_tokenizer


@pytest.fixture
def prepared(tmp_path):
    cfg = config(tmp_path)
    data = prepare_data(cfg, load_tokenizer(cfg.tokenizer.path))
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return cfg, data, path


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("length", [1, 3, 5, 18, 23])
def test_counts_match_native_blocks(masked, length, monkeypatch):
    import sparselab.data.coverage as module

    monkeypatch.setattr(module, "_CHUNK_TOKENS", 8)
    ids = np.arange(length, dtype=np.int32)
    mask = (ids % 7 == 0) if masked else None
    ds = TokenBlockDataset(ids, 4, supervision=mask)
    result = _block_counts(ids, mask, 4)
    assert result["usable_blocks"] == len(ds)
    assert result["supervised_targets"] == sum(
        int(np.count_nonzero(ds.numpy_block(i)[1] != -100)) for i in range(len(ds))
    )
    assert result["dropped_tail_tokens"] == (length - 1) % 4
    assert result["complete_blocks"] * 4 + result["dropped_tail_tokens"] + 1 == length


def test_coverage_optional_config_and_bounded_budget(prepared, tmp_path):
    cfg, data, path = prepared
    report = coverage(data.root)
    assert report["splits"]["train"]["blocks"] is None
    assert not report["splits"]["train"]["source_coverage"]["full"]
    with pytest.raises(ValueError, match="full source coverage"):
        derive_budget(path, data.root, 2, tmp_path / "full.yaml")
    out = tmp_path / "budget.yaml"
    budget = derive_budget(path, data.root, 3, out, require_full=False)
    native = TokenBlockDataset(data.train, cfg.training.seq_len)
    batch = cfg.training.micro_batch_size * cfg.training.gradient_accumulation
    assert budget["max_steps"] == (3 * len(native) + batch - 1) // batch
    assert budget["max_tokens"] == 3 * len(native) * cfg.training.seq_len
    loaded = load_config(out)
    assert loaded.dataset == cfg.dataset
    assert loaded.optimizer == cfg.optimizer
    assert loaded.training.max_tokens == budget["max_tokens"]
    with pytest.raises(FileExistsError):
        derive_budget(path, data.root, 1, out, require_full=False)


@pytest.mark.parametrize("passes", [0, -1, True, 1.5, "2"])
def test_invalid_passes_rejected_before_io(tmp_path, passes):
    with pytest.raises(ValueError, match="positive integer"):
        derive_budget(tmp_path / "absent", tmp_path, passes, tmp_path / "out")


def test_tampered_packed_data_rejected(prepared):
    _, data, _ = prepared
    with (data.root / "train.npy").open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    with pytest.raises(ValueError):
        coverage(data.root)


def test_wrong_config_and_source_identity_rejected(prepared, monkeypatch):
    import sparselab.data.coverage as module

    cfg, data, _ = prepared
    changed = cfg.model_copy(
        update={"dataset": cfg.dataset.model_copy(update={"synthetic_seed": 456})}
    )
    with pytest.raises(ValueError, match="binding differs"):
        coverage(data.root, changed)
    monkeypatch.setattr(module, "source_identity", lambda: {"sha256": "0" * 64})
    with pytest.raises(ValueError, match="source identity"):
        coverage(data.root, cfg)
    assert coverage(data.root)["config_verified"] is False


def test_source_exhaustion_requires_complete_preparation():
    snapshot = {
        "format": "sparselab-dataset-snapshot-v1",
        "splits": {
            "train": {
                "source_records_consumed": 11,
                "count": 8,
                "duplicate": 3,
                "source_exhausted": True,
                "stop_reason": "source_exhausted",
            }
        },
    }
    packed = {
        "acquired_documents": 8,
        "retained_documents": 8,
        "skipped_documents": 0,
        "truncated_documents": 0,
    }
    assert _source_coverage(snapshot, "train", packed)["full"]
    for key, value in (
        ("truncated_documents", 1),
        ("retained_documents", 7),
        ("acquired_documents", 7),
        ("skipped_documents", 1),
    ):
        assert not _source_coverage(snapshot, "train", {**packed, key: value})["full"]
    for reason in ("document_cap", "token_cap", "interrupted", None):
        snapshot["splits"]["train"]["stop_reason"] = reason
        assert not _source_coverage(snapshot, "train", packed)["full"]


def test_masked_budget_and_short_final_update(tmp_path):
    cfg = config(tmp_path)
    cfg = cfg.model_copy(
        update={"optimizer": cfg.optimizer.model_copy(update={"warmup_steps": 0})}
    )
    for split in ("train", "validation"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "format_version": 2,
                        "loss_mode": "assistant_only",
                        "messages": [
                            {"role": "user", "content": f"{split} question {i}"},
                            {"role": "assistant", "content": "fox " * (i + 1)},
                        ],
                    }
                )
                + "\n"
                for i in range(5)
            )
        )
    cfg = cfg.model_copy(
        update={
            "dataset": cfg.dataset.model_copy(
                update={
                    "source": "local_chat",
                    "license": "fixture",
                    "train_path": tmp_path / "train.jsonl",
                    "validation_path": tmp_path / "validation.jsonl",
                }
            ),
            "training": cfg.training.model_copy(
                update={"micro_batch_size": 1, "gradient_accumulation": 3}
            ),
        }
    )
    data = prepare_data(cfg, load_tokenizer(cfg.tokenizer.path))
    ds = TokenBlockDataset(
        data.train, cfg.training.seq_len, supervision=data.train_supervision
    )
    # Choose a batch that guarantees a short last update, without changing inputs.
    cfg = cfg.model_copy(
        update={
            "training": cfg.training.model_copy(
                update={"gradient_accumulation": len(ds) + 1}
            )
        }
    )
    path = tmp_path / "source.yaml"
    raw = cfg.model_dump(mode="json")
    raw["tokenizer"]["path"] = str(cfg.tokenizer.path.relative_to(tmp_path))
    path.write_text(yaml.safe_dump(raw))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = derive_budget(path, data.root, 2, elsewhere / "budget.yaml", False)
    assert result["max_tokens"] == 2 * sum(
        int(np.count_nonzero(ds.numpy_block(i)[1] != -100)) for i in range(len(ds))
    )
    assert result["max_steps"] == 2
    assert result["final_update_blocks"] == len(ds) - 1
    written = load_config(elsewhere / "budget.yaml")
    assert written.tokenizer.path == cfg.tokenizer.path
    assert written.dataset.train_path == cfg.dataset.train_path
    (tmp_path / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="binding differs"):
        coverage(data.root, cfg)


def test_cli_text_json_and_failure(prepared, capsys, tmp_path):
    _, data, path = prepared
    capsys.readouterr()
    parser = argparse.ArgumentParser()
    register_parser(parser.add_subparsers(required=True))
    args = parser.parse_args(
        ["coverage", "--prepared-root", str(data.root), "--config", str(path), "--json"]
    )
    args.handler(args)
    assert json.loads(capsys.readouterr().out)["config_verified"]
    args.json = False
    args.handler(args)
    assert "supervised_targets=" in capsys.readouterr().out
    args = parser.parse_args(
        [
            "budget",
            "--prepared-root",
            str(data.root),
            "--config",
            str(path),
            "--passes",
            "1",
            "--output",
            str(tmp_path / "out.yaml"),
            "--json",
        ]
    )
    with pytest.raises(SystemExit):
        args.handler(args)
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    args.require_full = False
    args.handler(args)
    assert json.loads(capsys.readouterr().out)["transition"] == "fresh"


def test_symlink_output_rejected(tmp_path):
    out = tmp_path / "out.yaml"
    out.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        derive_budget(tmp_path / "missing-config", tmp_path, 1, out)


@pytest.mark.parametrize("interrupted", [False, True])
def test_native_snapshot_full_coverage_and_resume(tmp_path, monkeypatch, interrupted):
    from sparselab.config.models import TokenizerTrainConfig
    from sparselab.data import sources
    from sparselab.data.tokenizer import train_tokenizer

    cfg = config(tmp_path)
    declaration = {
        "repo_id": "fixture/stories",
        "revision": "a" * 40,
        "config": "default",
        "splits": {"train": "train", "validation": "validation"},
        "text_field": "text",
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
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump(declaration))
    monkeypatch.setattr(
        sources,
        "_hub_identity",
        offline_hub_identity,
    )
    lock = sources.lock_source(source, tmp_path / "lock.json", tmp_path / "hub")
    should_interrupt = interrupted

    def stream(source, split, cache, **_options):
        nonlocal should_interrupt
        yield {"text": f"{split} story fox " * 20}
        if should_interrupt:
            should_interrupt = False
            raise RuntimeError("fixture interruption")
        yield {"text": None}
        yield {"text": f"{split} second tale " * 20}

    monkeypatch.setattr(sources, "_stream", stream)
    destination = tmp_path / "snapshot"
    if interrupted:
        with pytest.raises(RuntimeError, match="fixture interruption"):
            sources.snapshot_source(lock, destination, tmp_path / "hub")
    sources.snapshot_source(lock, destination, tmp_path / "hub", resume=interrupted)
    cfg = cfg.model_copy(
        update={
            "dataset": cfg.dataset.model_copy(
                update={
                    "source": "snapshot",
                    "revision": declaration["revision"],
                    "license": "MIT",
                    "dataset_config": "default",
                    "source_manifest_path": destination / "manifest.json",
                    "train_path": destination / "train.jsonl",
                    "validation_path": destination / "validation.jsonl",
                    "train_max_documents": 2,
                    "validation_max_documents": 2,
                }
            )
        }
    )
    tokenizer_path = train_tokenizer(
        TokenizerTrainConfig(
            schema_version=1,
            vocab_size=260,
            min_frequency=1,
            max_documents=2,
            output_dir=tmp_path / "snapshot-tokenizer",
            dataset=cfg.dataset,
        )
    )
    cfg = cfg.model_copy(
        update={
            "tokenizer": cfg.tokenizer.model_copy(update={"path": tokenizer_path}),
            "model": cfg.model.model_copy(update={"vocab_size": 260}),
        }
    )
    data = prepare_data(cfg, load_tokenizer(cfg.tokenizer.path))
    report = coverage(data.root)
    for split in ("train", "validation"):
        row = report["splits"][split]["source_coverage"]
        assert row["full"]
        assert row["acquired_records"] == 3
        assert row["retained_records"] == 2
        assert row["excluded_records"] == 1
        assert row["exclusions"]["null"] == 1
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    result = derive_budget(path, data.root, 2, tmp_path / "budget.yaml")
    assert result["require_full"]
    moved = cfg.model_copy(
        update={
            "dataset": cfg.dataset.model_copy(
                update={"train_path": tmp_path / "wrong.jsonl"}
            )
        }
    )
    with pytest.raises(ValueError, match="snapshot source paths/provenance"):
        coverage(data.root, moved)
    assert (
        result["max_tokens"]
        == 2
        * len(TokenBlockDataset(data.train, cfg.training.seq_len))
        * cfg.training.seq_len
    )
    # Packed signature alone cannot substitute for the embedded source binding.
    from sparselab.data.coverage import _digest
    from sparselab.training.manifest import canonical_json

    manifest = json.loads((data.root / "manifest.json").read_text())
    manifest["dataset_snapshot"]["source_exhausted"] = False
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = _digest(manifest)
    (data.root / "manifest.json").write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match="snapshot source binding"):
        coverage(data.root)


@pytest.mark.parametrize(
    ("training", "match"),
    [
        ({"micro_batch_size": 100000}, "warmup_steps"),
        ({"neural_loss_weight": 0.5}, "weighted objectives"),
    ],
    ids=["incompatible-schedule", "unsupported-objective"],
)
def test_invalid_budget_derivation_leaves_no_output(
    prepared, tmp_path, training, match
):
    cfg, data, path = prepared
    cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update=training)})
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    out = tmp_path / "invalid.yaml"
    with pytest.raises(ValueError, match=match):
        derive_budget(path, data.root, 1, out, False)
    assert not out.exists()
