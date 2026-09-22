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
from sparselab.data.byte_hash import table_address, token_bytes
from sparselab.data.datasets import iter_documents
from sparselab.training.manifest import canonical_json, sha256_file, source_identity

PACKING_VERSION = "contiguous-eos-v3"


@dataclass(frozen=True)
class PreparedData:
    root: Path
    train: np.ndarray
    validation: np.ndarray
    train_byte_addresses: np.ndarray | None
    validation_byte_addresses: np.ndarray | None
    manifest: dict[str, object]


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _tokenizer_sha256(tokenizer: Tokenizer) -> str:
    """Hash the complete tokenizer, not merely its vocabulary."""
    return hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()


def _cache_is_valid(
    manifest: dict[str, object],
    train_path: Path,
    validation_path: Path,
    train_byte_path: Path,
    validation_byte_path: Path,
    *,
    byte_enabled: bool,
) -> bool:
    try:
        train = manifest["train"]
        validation = manifest["validation"]
        if not isinstance(train, dict) or not isinstance(validation, dict):
            return False
        if (
            manifest.get("packing_version") != PACKING_VERSION
            or _sha256(train_path) != train.get("sha256")
            or _sha256(validation_path) != validation.get("sha256")
        ):
            return False
        if byte_enabled:
            byte = manifest.get("byte_addressing")
            return (
                isinstance(byte, dict)
                and _sha256(train_byte_path) == byte.get("train_sha256")
                and _sha256(validation_byte_path) == byte.get("validation_sha256")
            )
        return manifest.get("byte_addressing") is None
    except (OSError, KeyError, TypeError):
        return False


def _encoded_token_bytes(
    tokenizer: Tokenizer, ids: list[int], document: str
) -> list[bytes]:
    pieces = [token_bytes(tokenizer, token_id) for token_id in ids]
    if b"".join(pieces) != document.encode("utf-8"):
        raise ValueError("tokenizer token bytes do not reconstruct the source document")
    return pieces


def _local_chat_identity(config: DatasetConfig) -> dict[str, str] | None:
    if config.source != "local_chat":
        return None
    paths = (
        getattr(config, "train_path", None),
        getattr(config, "validation_path", None),
    )
    if not all(isinstance(path, Path) and path.is_file() for path in paths):
        raise ValueError("local_chat requires readable train_path and validation_path")
    return {
        "train_sha256": sha256_file(paths[0]),
        "validation_sha256": sha256_file(paths[1]),
    }


def _assert_local_chat_disjoint(config: DatasetConfig) -> None:
    if config.source != "local_chat":
        return
    train = {
        hashlib.sha256(document.encode("utf-8")).digest()
        for document in iter_documents(config, "train")
    }
    overlap = next(
        (
            document
            for document in iter_documents(config, "validation")
            if hashlib.sha256(document.encode("utf-8")).digest() in train
        ),
        None,
    )
    if overlap is not None:
        raise ValueError(
            "local_chat train and validation contain an overlapping conversation"
        )


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
            prefix = bytearray()
            for token_bytes in _encoded_token_bytes(tokenizer, encoding.ids, document)[
                :selected_count
            ]:
                prefix.extend(token_bytes)
                byte_addresses.append(
                    table_address(bytes(prefix[-byte_ngram_size:]), byte_table_size)
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
    local_chat = _local_chat_identity(config.dataset)
    cache_identity = {
        "dataset": config.dataset.model_dump(mode="json"),
        "local_chat_source": local_chat,
        "model": config.model.model_dump(mode="json"),
        "packing_version": PACKING_VERSION,
        "source_identity_sha256": source_identity()["sha256"],
        "tokenizer_sha256": _tokenizer_sha256(tokenizer),
    }
    _assert_local_chat_disjoint(config.dataset)
    root = (
        config.dataset.cache_dir
        / hashlib.sha256(canonical_json(cache_identity)).hexdigest()[:16]
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
        if _cache_is_valid(
            manifest,
            train_path,
            validation_path,
            train_byte_path,
            validation_byte_path,
            byte_enabled=byte_enabled,
        ):
            return PreparedData(
                root,
                np.load(train_path, mmap_mode="r"),
                np.load(validation_path, mmap_mode="r"),
                np.load(train_byte_path, mmap_mode="r") if byte_enabled else None,
                np.load(validation_byte_path, mmap_mode="r") if byte_enabled else None,
                manifest,
            )
        raise ValueError(f"prepared-data cache integrity check failed: {root}")
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
        "license": (
            getattr(config.dataset, "license", None)
            if config.dataset.source == "local_chat"
            else "CDLA-Sharing-1.0"
            if config.dataset.source == "tinystories"
            else "MIT synthetic chat recall fixture"
            if config.dataset.source == "chat_recall"
            else "synthetic instruction reference"
            if config.dataset.source == "instruction_reference"
            else "synthetic associative recall reference"
            if config.dataset.source == "engram_recall"
            else "dataset-specific license/source metadata unavailable"
        ),
        "tokenizer_sha256": _tokenizer_sha256(tokenizer),
        "settings_sha256": hashlib.sha256(canonical_json(cache_identity)).hexdigest(),
        "source_identity_sha256": cache_identity["source_identity_sha256"],
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
