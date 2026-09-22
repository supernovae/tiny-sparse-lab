"""Immutable contiguous-EOS data packing and deterministic block traversal."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from sparselab.config.models import DatasetConfig, RunConfig
from sparselab.data.datasets import iter_documents

PACKING_VERSION = "contiguous-eos-v1"


@dataclass(frozen=True)
class PreparedData:
    root: Path
    train: np.ndarray
    validation: np.ndarray
    manifest: dict[str, object]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collect(
    config: DatasetConfig, tokenizer: Tokenizer, split: str
) -> tuple[np.ndarray, dict[str, int]]:
    max_documents = (
        config.train_max_documents
        if split == "train"
        else config.validation_max_documents
    )
    max_tokens = (
        config.train_max_tokens if split == "train" else config.validation_max_tokens
    )
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("tokenizer has no <eos> special token")
    values: list[int] = []
    stats = {
        "acquired_documents": 0,
        "retained_documents": 0,
        "skipped_documents": 0,
        "truncated_documents": 0,
    }
    for document in iter_documents(config, split):
        if stats["acquired_documents"] >= max_documents or len(values) >= max_tokens:
            break
        stats["acquired_documents"] += 1
        if not document:
            stats["skipped_documents"] += 1
            continue
        text_ids = tokenizer.encode(document, add_special_tokens=False).ids
        remaining = max_tokens - len(values)
        if remaining == 0:
            break
        if len(text_ids) + 1 > remaining:
            selected = text_ids[: remaining - 1] + [eos]
            stats["truncated_documents"] += 1
        else:
            selected = text_ids + [eos]
        if selected:
            values.extend(selected)
            stats["retained_documents"] += 1
    return np.asarray(values, dtype=np.int32), stats


def _atomic_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, values, allow_pickle=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def prepare_data(config: RunConfig, tokenizer: Tokenizer) -> PreparedData:
    """Prepare bounded data once; blocks may cross EOS document boundaries by design."""
    root = (
        config.dataset.cache_dir
        / hashlib.sha256(
            json.dumps(config.dataset.model_dump(mode="json"), sort_keys=True).encode()
            + json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
        ).hexdigest()[:16]
    )
    manifest_path = root / "manifest.json"
    train_path, validation_path = root / "train.npy", root / "validation.npy"
    if manifest_path.is_file() and train_path.is_file() and validation_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return PreparedData(
            root,
            np.load(train_path, mmap_mode="r"),
            np.load(validation_path, mmap_mode="r"),
            manifest,
        )

    temporary_root = root.with_name(root.name + ".tmp")
    if temporary_root.exists():
        raise RuntimeError(f"incomplete prepared-data sibling exists: {temporary_root}")
    temporary_root.mkdir(parents=True, exist_ok=False)
    train, train_stats = _collect(config.dataset, tokenizer, "train")
    validation, validation_stats = _collect(config.dataset, tokenizer, "validation")
    if (
        len(train) < config.training.seq_len + 1
        or len(validation) < config.training.seq_len + 1
    ):
        raise ValueError("prepared split lacks a full next-token block")
    _atomic_array(temporary_root / "train.npy", train)
    _atomic_array(temporary_root / "validation.npy", validation)
    manifest = {
        "packing_version": PACKING_VERSION,
        "source": config.dataset.source,
        "revision": config.dataset.revision,
        "license": "CDLA-Sharing-1.0"
        if config.dataset.source == "tinystories"
        else "synthetic fixture",
        "tokenizer_sha256": hashlib.sha256(
            json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
        ).hexdigest(),
        "settings_sha256": hashlib.sha256(
            json.dumps(config.dataset.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest(),
        "train": {
            **train_stats,
            "tokens": len(train),
            "sha256": _sha256(temporary_root / "train.npy"),
        },
        "validation": {
            **validation_stats,
            "tokens": len(validation),
            "sha256": _sha256(temporary_root / "validation.npy"),
        },
    }
    (temporary_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root.replace(root)
    return PreparedData(
        root,
        np.load(train_path, mmap_mode="r"),
        np.load(validation_path, mmap_mode="r"),
        manifest,
    )


class TokenBlockDataset:
    def __init__(self, ids: np.ndarray, seq_len: int) -> None:
        self.ids, self.seq_len = ids, seq_len
        self.num_blocks = (len(ids) - 1) // seq_len

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.seq_len
        values = torch.from_numpy(
            np.asarray(self.ids[start : start + self.seq_len + 1], dtype=np.int64)
        )
        return values[:-1], values[1:]


@dataclass(frozen=True)
class BatchCursor:
    epoch: int = 0
    next_block: int = 0


def epoch_order(num_blocks: int, seed: int, epoch: int) -> torch.Tensor:
    return torch.randperm(
        num_blocks, generator=torch.Generator().manual_seed(seed + epoch)
    )
