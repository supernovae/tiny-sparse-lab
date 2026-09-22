"""Immutable contiguous-EOS packing with optional causal byte-address arrays."""

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
from sparselab.data.byte_hash import table_address
from sparselab.data.datasets import iter_documents

PACKING_VERSION = "contiguous-eos-v2"


@dataclass(frozen=True)
class PreparedData:
    root: Path
    train: np.ndarray
    validation: np.ndarray
    train_byte_addresses: np.ndarray | None
    validation_byte_addresses: np.ndarray | None
    manifest: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect(
    config: DatasetConfig,
    tokenizer: Tokenizer,
    split: str,
    *,
    byte_table_size: int | None = None,
    byte_ngram_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, int]]:
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
    byte_addresses: list[int] | None = [] if byte_table_size is not None else None
    stats = {
        "acquired_documents": 0,
        "retained_documents": 0,
        "skipped_documents": 0,
        "truncated_documents": 0,
    }
    documents = iter(iter_documents(config, split))
    while stats["acquired_documents"] < max_documents and len(values) < max_tokens:
        try:
            document = next(documents)
        except StopIteration:
            break
        stats["acquired_documents"] += 1
        if not document:
            stats["skipped_documents"] += 1
            continue
        encoding = tokenizer.encode(document, add_special_tokens=False)
        remaining = max_tokens - len(values)
        if remaining == 0:
            break
        selected_count = min(len(encoding.ids), remaining - 1)
        selected = encoding.ids[:selected_count] + [eos]
        if selected_count < len(encoding.ids):
            stats["truncated_documents"] += 1
        if byte_addresses is not None:
            assert byte_ngram_size is not None
            for _, end in encoding.offsets[:selected_count]:
                prefix = document[:end].encode("utf-8")
                byte_addresses.append(
                    table_address(prefix[-byte_ngram_size:], byte_table_size)
                )
            byte_addresses.append(0)
        values.extend(selected)
        stats["retained_documents"] += 1
    return (
        np.asarray(values, dtype=np.int32),
        None if byte_addresses is None else np.asarray(byte_addresses, dtype=np.int32),
        stats,
    )


def _atomic_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, values, allow_pickle=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def prepare_data(config: RunConfig, tokenizer: Tokenizer) -> PreparedData:
    """Prepare immutable IDs and, for byte memory, causal raw-UTF-8 suffix addresses."""
    root = (
        config.dataset.cache_dir
        / hashlib.sha256(
            json.dumps(config.dataset.model_dump(mode="json"), sort_keys=True).encode()
            + json.dumps(config.model.model_dump(mode="json"), sort_keys=True).encode()
            + json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
        ).hexdigest()[:16]
    )
    manifest_path = root / "manifest.json"
    train_path, validation_path = root / "train.npy", root / "validation.npy"
    byte_enabled = config.model.memory in {"byte", "portable"}
    train_byte_path, validation_byte_path = (
        root / "train_byte_addresses.npy",
        root / "validation_byte_addresses.npy",
    )
    if (
        manifest_path.is_file()
        and train_path.is_file()
        and validation_path.is_file()
        and (
            not byte_enabled
            or (train_byte_path.is_file() and validation_byte_path.is_file())
        )
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return PreparedData(
            root,
            np.load(train_path, mmap_mode="r"),
            np.load(validation_path, mmap_mode="r"),
            np.load(train_byte_path, mmap_mode="r") if byte_enabled else None,
            np.load(validation_byte_path, mmap_mode="r") if byte_enabled else None,
            manifest,
        )
    temporary_root = root.with_name(root.name + ".tmp")
    if temporary_root.exists():
        raise RuntimeError(f"incomplete prepared-data sibling exists: {temporary_root}")
    temporary_root.mkdir(parents=True, exist_ok=False)
    settings = (
        {
            "byte_table_size": config.model.memory_table_size,
            "byte_ngram_size": config.model.memory_ngram_size,
        }
        if byte_enabled
        else {}
    )
    train, train_byte, train_stats = _collect(
        config.dataset, tokenizer, "train", **settings
    )
    validation, validation_byte, validation_stats = _collect(
        config.dataset, tokenizer, "validation", **settings
    )
    if (
        len(train) < config.training.seq_len + 1
        or len(validation) < config.training.seq_len + 1
    ):
        raise ValueError("prepared split lacks a full next-token block")
    _atomic_array(temporary_root / "train.npy", train)
    _atomic_array(temporary_root / "validation.npy", validation)
    if byte_enabled:
        assert train_byte is not None and validation_byte is not None
        _atomic_array(temporary_root / "train_byte_addresses.npy", train_byte)
        _atomic_array(temporary_root / "validation_byte_addresses.npy", validation_byte)
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
        "byte_addressing": None
        if not byte_enabled
        else {
            "kind": "raw-utf8-suffix-v1",
            "table_size": config.model.memory_table_size,
            "ngram_size": config.model.memory_ngram_size,
            "train_sha256": _sha256(temporary_root / "train_byte_addresses.npy"),
            "validation_sha256": _sha256(
                temporary_root / "validation_byte_addresses.npy"
            ),
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
        np.load(train_byte_path, mmap_mode="r") if byte_enabled else None,
        np.load(validation_byte_path, mmap_mode="r") if byte_enabled else None,
        manifest,
    )


class TokenBlockDataset:
    def __init__(
        self, ids: np.ndarray, seq_len: int, byte_addresses: np.ndarray | None = None
    ) -> None:
        self.ids, self.seq_len, self.byte_addresses = ids, seq_len, byte_addresses
        self.num_blocks = (len(ids) - 1) // seq_len
        if byte_addresses is not None and len(byte_addresses) != len(ids):
            raise ValueError("byte address array must align with packed IDs")

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(
        self, index: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        start = index * self.seq_len
        values = torch.from_numpy(
            np.asarray(self.ids[start : start + self.seq_len + 1], dtype=np.int64)
        )
        if self.byte_addresses is None:
            return values[:-1], values[1:]
        addresses = torch.from_numpy(
            np.asarray(
                self.byte_addresses[start : start + self.seq_len], dtype=np.int64
            )
        )
        return values[:-1], values[1:], addresses


@dataclass(frozen=True)
class BatchCursor:
    epoch: int = 0
    next_block: int = 0


def epoch_order(num_blocks: int, seed: int, epoch: int) -> torch.Tensor:
    return torch.randperm(
        num_blocks, generator=torch.Generator().manual_seed(seed + epoch)
    )
