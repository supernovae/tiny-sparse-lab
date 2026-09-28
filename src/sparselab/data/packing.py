"""Immutable contiguous-EOS packing with optional causal byte-address arrays."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from sparselab.config.models import DatasetConfig, RunConfig
from sparselab.corpus.export import verify_release_export
from sparselab.data.allocation import AllocationManifest, load_allocation_manifest
from sparselab.data.byte_hash import table_address, token_bytes
from sparselab.data.conversations import (
    RenderedConversation,
    iter_rendered_conversations,
)
from sparselab.data.datasets import iter_documents
from sparselab.data.local_stories import verify_snapshot
from sparselab.progress import progress_phase
from sparselab.training.manifest import canonical_json, sha256_file, source_identity
from sparselab.workdir import ensure_work_dir
from sparselab.workspace_cleanup import campaign_lock, mark_prepared_cache

PACKING_VERSION = "contiguous-eos-v5"
HISTORICAL_PACKING_VERSION = "contiguous-eos-v4"


@dataclass(frozen=True)
class PreparedData:
    root: Path
    train: np.ndarray
    validation: np.ndarray
    train_supervision: np.ndarray | None
    validation_supervision: np.ndarray | None
    train_byte_addresses: np.ndarray | None
    validation_byte_addresses: np.ndarray | None
    train_owner_ids: np.ndarray | None
    validation_owner_ids: np.ndarray | None
    train_semantic_queries: np.ndarray | None
    validation_semantic_queries: np.ndarray | None
    train_semantic_mask: np.ndarray | None
    validation_semantic_mask: np.ndarray | None
    allocation: AllocationManifest | None
    manifest: dict[str, object]


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _tokenizer_sha256(tokenizer: Tokenizer) -> str:
    """Hash the complete tokenizer, not merely its vocabulary."""
    return hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()


def _array_metadata(
    path: Path,
    *,
    dtype: np.dtype[np.generic] | type[np.generic] = np.int32,
    dimensions: int = 1,
) -> dict[str, object]:
    size_bytes = path.stat().st_size
    with progress_phase(
        f"data_artifact_hash_{path.name}",
        completed_work=0,
        total_work=size_bytes,
        unit="bytes",
        raw_counters={"artifact": path.name, "artifact_bytes": size_bytes},
    ) as progress:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != dimensions or values.dtype != dtype:
            raise ValueError(f"packed array has unexpected shape or dtype: {path}")
        digest = _sha256(path)
        progress.update(
            completed_work=size_bytes,
            total_work=size_bytes,
            unit="bytes",
            raw_counters={"artifact": path.name, "artifact_bytes": size_bytes},
        )
        return {
            "dtype": values.dtype.name,
            "shape": list(values.shape),
            "tokens": int(values.shape[0]),
            "sha256": digest,
        }


def _cache_is_valid(
    manifest: dict[str, object],
    cache_identity: dict[str, object],
    train_path: Path,
    validation_path: Path,
    train_supervision_path: Path,
    validation_supervision_path: Path,
    train_byte_path: Path,
    validation_byte_path: Path,
    *,
    byte_enabled: bool,
) -> bool:
    try:
        payload = dict(manifest)
        digest = payload.pop("manifest_sha256")
        train = payload["train"]
        validation = payload["validation"]
        version = payload.get("packing_version")
        if (
            not isinstance(digest, str)
            or hashlib.sha256(canonical_json(payload)).hexdigest() != digest
            or not isinstance(train, dict)
            or not isinstance(validation, dict)
            or version not in {PACKING_VERSION, HISTORICAL_PACKING_VERSION}
            or payload.get("cache_identity") != cache_identity
            or payload.get("settings_sha256")
            != hashlib.sha256(canonical_json(cache_identity)).hexdigest()
            or any(
                train.get(key) != value
                for key, value in _array_metadata(train_path).items()
            )
            or any(
                validation.get(key) != value
                for key, value in _array_metadata(validation_path).items()
            )
        ):
            return False
        if version == PACKING_VERSION:
            supervision = payload.get("supervision")
            if not (
                isinstance(supervision, dict)
                and supervision.get("kind") == "token-loss-mask-v1"
                and isinstance(supervision.get("train"), dict)
                and isinstance(supervision.get("validation"), dict)
                and supervision["train"].get("shape") == train.get("shape")
                and supervision["validation"].get("shape") == validation.get("shape")
                and all(
                    supervision["train"].get(key) == value
                    for key, value in _array_metadata(
                        train_supervision_path, dtype=np.dtype(bool)
                    ).items()
                )
                and all(
                    supervision["validation"].get(key) == value
                    for key, value in _array_metadata(
                        validation_supervision_path, dtype=np.dtype(bool)
                    ).items()
                )
            ):
                return False
        elif payload.get("supervision") is not None:
            return False
        if byte_enabled:
            byte = payload.get("byte_addressing")
            return (
                isinstance(byte, dict)
                and byte.get("kind") == "raw-utf8-suffix-v1"
                and isinstance(byte.get("train"), dict)
                and isinstance(byte.get("validation"), dict)
                and byte["train"].get("shape") == train.get("shape")
                and byte["validation"].get("shape") == validation.get("shape")
                and byte.get("table_size")
                == cache_identity["packing"]["memory_table_size"]
                and byte.get("ngram_size")
                == cache_identity["packing"]["memory_ngram_size"]
                and all(
                    byte["train"].get(key) == value
                    for key, value in _array_metadata(train_byte_path).items()
                )
                and all(
                    byte["validation"].get(key) == value
                    for key, value in _array_metadata(validation_byte_path).items()
                )
            )
        return payload.get("byte_addressing") is None
    except AttributeError, OSError, KeyError, TypeError, ValueError:
        return False


def load_prepared_data(
    root: Path,
    *,
    byte_enabled: bool,
    expected_identity: dict[str, object] | None = None,
) -> PreparedData:
    """Open a verified immutable cache or run-owned copy without reacquiring data."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"prepared-data manifest must be an object: {root}")
    identity = (
        expected_identity
        if expected_identity is not None
        else manifest.get("cache_identity")
    )
    train, validation = root / "train.npy", root / "validation.npy"
    train_supervision, validation_supervision = (
        root / "train_supervision.npy",
        root / "validation_supervision.npy",
    )
    train_byte, validation_byte = (
        root / "train_byte_addresses.npy",
        root / "validation_byte_addresses.npy",
    )
    if not isinstance(identity, dict) or not _cache_is_valid(
        manifest,
        identity,
        train,
        validation,
        train_supervision,
        validation_supervision,
        train_byte,
        validation_byte,
        byte_enabled=byte_enabled,
    ):
        raise ValueError(f"prepared-data cache integrity check failed: {root}")
    has_supervision = manifest.get("packing_version") == PACKING_VERSION
    allocation_enabled = isinstance(manifest.get("allocation"), dict)
    paths = [
        root / name
        for name in (
            "train_owner_ids.npy",
            "validation_owner_ids.npy",
            "train_semantic_queries.npy",
            "validation_semantic_queries.npy",
            "train_semantic_mask.npy",
            "validation_semantic_mask.npy",
        )
    ]
    allocation_arrays: tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ] = (None, None, None, None, None, None)
    if allocation_enabled:
        allocation_metadata = manifest["allocation"]
        assert isinstance(allocation_metadata, dict)
        semantic = allocation_metadata.get("semantic")
        if allocation_metadata.get("format") != "sparselab-prepared-allocation-v1":
            raise ValueError("prepared allocation metadata is malformed")
        expected_paths = paths[:2] if semantic is None else paths
        if not all(path.is_file() for path in expected_paths) or (
            semantic is None and any(path.exists() for path in paths[2:])
        ):
            raise ValueError("prepared allocation sidecars are missing or unexpected")
        loaded: list[np.ndarray] = []
        for split, owner_path, query_path, mask_path in (
            ("train", paths[0], paths[2], paths[4]),
            ("validation", paths[1], paths[3], paths[5]),
        ):
            split_metadata = allocation_metadata.get(split)
            if not isinstance(split_metadata, dict) or not isinstance(
                split_metadata.get("owner"), dict
            ):
                raise TypeError("prepared allocation metadata is malformed")
            packed_ids = np.load(
                train if split == "train" else validation,
                mmap_mode="r",
                allow_pickle=False,
            )
            owner = np.load(owner_path, mmap_mode="r", allow_pickle=False)
            if (
                owner.ndim != 1
                or owner.shape != packed_ids.shape
                or any(
                    split_metadata["owner"].get(key) != value
                    for key, value in _array_metadata(
                        owner_path, dtype=np.dtype(np.uint8)
                    ).items()
                )
                or np.any(owner > 3)
            ):
                raise ValueError("prepared allocation owner sidecar integrity failed")
            loaded.append(owner)
            if semantic is None:
                continue
            query_metadata = split_metadata.get("semantic_queries")
            mask_metadata = split_metadata.get("semantic_mask")
            if not isinstance(query_metadata, dict) or not isinstance(
                mask_metadata, dict
            ):
                raise TypeError("prepared semantic allocation metadata is malformed")
            query = np.load(query_path, mmap_mode="r", allow_pickle=False)
            mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
            if (
                any(
                    query_metadata.get(key) != value
                    for key, value in _array_metadata(
                        query_path, dtype=np.dtype(np.float32), dimensions=2
                    ).items()
                )
                or any(
                    mask_metadata.get(key) != value
                    for key, value in _array_metadata(
                        mask_path, dtype=np.dtype(bool)
                    ).items()
                )
                or query.ndim != 2
                or query.shape[0] != len(owner)
                or query.shape[1] <= 0
                or mask.shape != owner.shape
                or mask[-1]
                or np.any(mask[:-1] & ~np.isin(owner[1:], (2, 3)))
            ):
                raise ValueError(
                    "prepared semantic allocation sidecar integrity failed"
                )
            loaded.extend((query, mask))
        if semantic is None:
            allocation_arrays = (loaded[0], loaded[1], None, None, None, None)
        else:
            allocation_arrays = (
                loaded[0],
                loaded[3],
                loaded[1],
                loaded[4],
                loaded[2],
                loaded[5],
            )
    return PreparedData(
        root,
        np.load(train, mmap_mode="r", allow_pickle=False),
        np.load(validation, mmap_mode="r", allow_pickle=False),
        np.load(train_supervision, mmap_mode="r", allow_pickle=False)
        if has_supervision
        else None,
        np.load(validation_supervision, mmap_mode="r", allow_pickle=False)
        if has_supervision
        else None,
        np.load(train_byte, mmap_mode="r", allow_pickle=False)
        if byte_enabled
        else None,
        np.load(validation_byte, mmap_mode="r", allow_pickle=False)
        if byte_enabled
        else None,
        *allocation_arrays,
        None,
        manifest,
    )


def _encoded_token_bytes(
    tokenizer: Tokenizer, ids: list[int], document: str
) -> list[bytes]:
    pieces = [token_bytes(tokenizer, token_id) for token_id in ids]
    if b"".join(pieces) != document.encode("utf-8"):
        raise ValueError("tokenizer token bytes do not reconstruct the source document")
    return pieces


def _local_chat_identity(config: DatasetConfig) -> dict[str, str] | None:
    if config.source not in {"local_chat", "local_text"}:
        return None
    paths = (
        getattr(config, "train_path", None),
        getattr(config, "validation_path", None),
    )
    if not all(isinstance(path, Path) and path.is_file() for path in paths):
        raise ValueError(
            f"{config.source} requires readable train_path and validation_path"
        )
    return {
        "train_sha256": sha256_file(paths[0]),
        "validation_sha256": sha256_file(paths[1]),
    }


def _assert_local_chat_disjoint(config: DatasetConfig) -> None:
    if config.source not in {"local_chat", "local_text"}:
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


def _assert_local_chat_supervision_consistent(config: DatasetConfig) -> None:
    if config.source != "local_chat":
        return
    assert config.train_path is not None and config.validation_path is not None
    train_modes = {
        item.loss_mode for item in iter_rendered_conversations(config.train_path)
    }
    validation_modes = {
        item.loss_mode for item in iter_rendered_conversations(config.validation_path)
    }
    if (
        len(train_modes) != 1
        or len(validation_modes) != 1
        or train_modes != validation_modes
    ):
        raise ValueError(
            "local_chat training and validation must use one matching supervision mode"
        )


def _supervision_for_encoding(
    document: RenderedConversation | None,
    offsets: list[tuple[int, int]],
    token_count: int,
) -> list[bool]:
    if document is None or document.loss_mode == "all_tokens":
        return [True] * token_count
    spans = document.supervision_spans
    # A token crossing a role boundary cannot be partially supervised. Spans
    # include the assistant's leading separator space, but never role markers.
    return [
        any(
            start < end and span_start <= start and end <= span_end
            for span_start, span_end in spans
        )
        for start, end in offsets[:token_count]
    ]


def _source_documents(
    config: DatasetConfig, split: str
) -> Iterator[RenderedConversation]:
    if config.source == "local_chat":
        path = config.train_path if split == "train" else config.validation_path
        assert path is not None
        yield from iter_rendered_conversations(path)
        return
    for text in iter_documents(config, split):
        yield RenderedConversation(text, ((0, len(text)),), "all_tokens")


def _collect(
    config: DatasetConfig,
    tokenizer: Tokenizer,
    split: str,
    *,
    byte_table_size: int | None = None,
    byte_ngram_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, int]]:
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
    supervision: list[bool] = []
    byte_addresses: list[int] | None = [] if byte_table_size is not None else None
    stats = {
        "acquired_documents": 0,
        "retained_documents": 0,
        "skipped_documents": 0,
        "truncated_documents": 0,
    }
    documents = iter(_source_documents(config, split))
    source_wait_seconds = 0.0
    encoding_seconds = 0.0
    source_bytes = 0
    encoded_tokens = 0
    source_digest = hashlib.sha256()
    started = time.monotonic()
    with progress_phase(
        f"data_{split}_document_iteration_and_collection",
        completed_work=0,
        unit="documents",
        raw_counters={**stats, "source_bytes": 0, "encoded_tokens": 0},
    ) as progress:
        while stats["acquired_documents"] < max_documents and len(values) < max_tokens:
            read_started = time.monotonic()
            try:
                rendered = next(documents)
            except StopIteration:
                break
            finally:
                source_wait_seconds += time.monotonic() - read_started
            stats["acquired_documents"] += 1
            document = rendered.text
            if not document:
                stats["skipped_documents"] += 1
                continue
            encoded_bytes = document.encode("utf-8")
            source_bytes += len(encoded_bytes)
            source_digest.update(encoded_bytes)
            source_digest.update(b"\0")
            encode_started = time.monotonic()
            encoding = tokenizer.encode(document, add_special_tokens=False)
            encoding_seconds += time.monotonic() - encode_started
            encoded_tokens += len(encoding.ids)
            remaining = max_tokens - len(values)
            if remaining <= 1:
                break
            selected_count = min(len(encoding.ids), remaining - 1)
            selected = encoding.ids[:selected_count] + [eos]
            selected_supervision = _supervision_for_encoding(
                rendered, encoding.offsets, selected_count
            )
            selected_supervision.append(
                rendered.loss_mode == "all_tokens"
                or (
                    selected_count == len(encoding.ids)
                    and bool(rendered.supervision_spans)
                )
            )
            if selected_count < len(encoding.ids):
                stats["truncated_documents"] += 1
            if byte_addresses is not None:
                assert byte_ngram_size is not None
                prefix = bytearray()
                for token_piece in _encoded_token_bytes(
                    tokenizer, encoding.ids, document
                )[:selected_count]:
                    prefix.extend(token_piece)
                    byte_addresses.append(
                        table_address(bytes(prefix[-byte_ngram_size:]), byte_table_size)
                    )
                byte_addresses.append(0)
            values.extend(selected)
            supervision.extend(selected_supervision)
            stats["retained_documents"] += 1
            if stats["acquired_documents"] % 500 == 0:
                progress.update(
                    completed_work=stats["acquired_documents"],
                    unit="documents",
                    raw_counters={
                        **stats,
                        "source_bytes": source_bytes,
                        "encoded_tokens": encoded_tokens,
                        "retained_tokens": len(values),
                    },
                )
        elapsed = max(time.monotonic() - started, 1e-9)
        construction_seconds = max(
            0.0, elapsed - source_wait_seconds - encoding_seconds
        )
        progress.update(
            completed_work=stats["acquired_documents"],
            unit="documents",
            raw_counters={
                **stats,
                "source_bytes": source_bytes,
                "encoded_tokens": encoded_tokens,
                "source_content_sha256": source_digest.hexdigest(),
                "encoded_tokens_per_second": encoded_tokens
                / max(encoding_seconds, 1e-9),
                "construction_tokens_per_second": len(values)
                / max(construction_seconds, 1e-9),
                "output_tokens": len(values),
                "source_iteration_seconds": source_wait_seconds,
                "tokenizer_encoding_seconds": encoding_seconds,
                "python_array_construction_seconds": construction_seconds,
                "documents_per_second": stats["acquired_documents"] / elapsed,
                "source_bytes_per_second": source_bytes / elapsed,
                "output_tokens_per_second": len(values) / elapsed,
            },
        )
    return (
        np.asarray(values, dtype=np.int32),
        np.asarray(supervision, dtype=bool),
        None if byte_addresses is None else np.asarray(byte_addresses, dtype=np.int32),
        stats,
    )


class _ArraySpool:
    """Bounded in-memory chunks with a disk-backed, exactly sized final array."""

    def __init__(self, path: Path, dtype: np.dtype) -> None:
        self.path = path
        self.dtype = np.dtype(dtype)
        self.raw_path = path.with_suffix(".raw")
        self.raw = self.raw_path.open("xb")
        self.buffer: list[int | bool] = []
        self.count = 0

    def append(self, values: list[int] | list[bool]) -> None:
        self.buffer.extend(values)
        self.count += len(values)
        if len(self.buffer) >= 65_536:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.raw.write(np.asarray(self.buffer, dtype=self.dtype).tobytes())
            self.buffer.clear()

    def finish(self) -> None:
        self.flush()
        self.raw.flush()
        os.fsync(self.raw.fileno())
        self.raw.close()
        temporary = self.path.with_name(self.path.stem + ".tmp.npy")
        with temporary.open("xb") as output, self.raw_path.open("rb") as source:
            np.lib.format.write_array_header_1_0(
                output,
                {
                    "descr": np.lib.format.dtype_to_descr(self.dtype),
                    "fortran_order": False,
                    "shape": (self.count,),
                },
            )
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(self.path)
        self.raw_path.unlink()


def _collect_streaming(
    config: DatasetConfig,
    tokenizer: Tokenizer,
    split: str,
    root: Path,
    *,
    selected_documents: int,
    byte_table_size: int | None = None,
    byte_ngram_size: int | None = None,
) -> dict[str, int]:
    """Pack every selected story, refusing a partial document or short source."""
    max_tokens = (
        config.train_max_tokens if split == "train" else config.validation_max_tokens
    )
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("tokenizer has no <eos> special token")
    ids = _ArraySpool(root / f"{split}.npy", np.dtype(np.int32))
    supervision = _ArraySpool(root / f"{split}_supervision.npy", np.dtype(bool))
    byte_addresses = (
        _ArraySpool(root / f"{split}_byte_addresses.npy", np.dtype(np.int32))
        if byte_table_size is not None
        else None
    )
    stats = {
        "acquired_documents": 0,
        "retained_documents": 0,
        "skipped_documents": 0,
        "truncated_documents": 0,
    }
    with progress_phase(
        f"data_{split}_document_iteration_and_collection",
        completed_work=0,
        total_work=selected_documents,
        unit="documents",
        raw_counters=stats,
    ) as progress:
        documents = iter(_source_documents(config, split))
        while stats["retained_documents"] < selected_documents:
            try:
                rendered = next(documents)
            except StopIteration:
                break
            stats["acquired_documents"] += 1
            document = rendered.text
            if not document:
                stats["skipped_documents"] += 1
                continue
            encoding = tokenizer.encode(document, add_special_tokens=False)
            selected = encoding.ids + [eos]
            if ids.count + len(selected) > max_tokens:
                raise ValueError(
                    f"{split} token cap would truncate selected story "
                    f"{stats['acquired_documents']}: {ids.count + len(selected)} "
                    f"> {max_tokens}"
                )
            selected_supervision = _supervision_for_encoding(
                rendered, encoding.offsets, len(encoding.ids)
            )
            selected_supervision.append(
                rendered.loss_mode == "all_tokens" or bool(rendered.supervision_spans)
            )
            if byte_addresses is not None:
                assert byte_ngram_size is not None and byte_table_size is not None
                prefix = bytearray()
                addresses: list[int] = []
                for piece in _encoded_token_bytes(tokenizer, encoding.ids, document):
                    prefix.extend(piece)
                    addresses.append(
                        table_address(bytes(prefix[-byte_ngram_size:]), byte_table_size)
                    )
                addresses.append(0)
                byte_addresses.append(addresses)
            ids.append(selected)
            supervision.append(selected_supervision)
            stats["retained_documents"] += 1
            if stats["acquired_documents"] % 500 == 0:
                progress.update(
                    completed_work=stats["retained_documents"],
                    total_work=selected_documents,
                    unit="documents",
                    raw_counters={
                        **stats,
                        "output_tokens": ids.count,
                        "peak_host_rss_bytes": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss
                        * 1024,
                    },
                )
        if stats["retained_documents"] != selected_documents:
            raise ValueError(
                f"{split} snapshot has only {stats['retained_documents']} "
                f"selected distinct stories; required {selected_documents}"
            )
        for spool in (ids, supervision, byte_addresses):
            if spool is not None:
                spool.finish()
        stats["output_tokens"] = ids.count
        stats["artifact_bytes"] = sum(
            spool.path.stat().st_size
            for spool in (ids, supervision, byte_addresses)
            if spool is not None
        )
        stats["artifact_files"] = 2 + (byte_addresses is not None)
        stats["peak_host_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        progress.update(
            completed_work=selected_documents,
            total_work=selected_documents,
            unit="documents",
            raw_counters=stats,
        )
    return stats


def _atomic_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npy")
    with progress_phase(
        f"data_array_write_fsync_{path.name}",
        completed_work=0,
        unit="bytes",
        raw_counters={"artifact": path.name},
    ) as progress:
        np.save(temporary, values, allow_pickle=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
        output_bytes = path.stat().st_size
        progress.update(
            completed_work=output_bytes,
            total_work=output_bytes,
            unit="bytes",
            raw_counters={"artifact": path.name, "artifact_bytes": output_bytes},
        )


def _prepare_data(config: RunConfig, tokenizer: Tokenizer) -> PreparedData:
    """Prepare immutable IDs and causal sidecars with an optional allocation."""
    local_stories = (
        verify_snapshot(config.dataset)
        if config.dataset.source == "local_stories"
        else None
    )
    corpus_export = (
        verify_release_export(config.dataset)
        if config.dataset.corpus_release_path is not None
        else None
    )
    if (
        corpus_export is not None
        and corpus_export["vocab_size"] != config.model.vocab_size
    ):
        raise ValueError("corpus export vocabulary size does not match run model")
    local_chat = _local_chat_identity(config.dataset)
    source_digest = source_identity()["sha256"]
    allocation = None
    if config.dataset.allocation_manifest_path is not None:
        if config.dataset.source != "local_chat" or local_chat is None:
            raise ValueError("allocation manifests require a local_chat dataset")
        allocation = load_allocation_manifest(
            config.dataset.allocation_manifest_path,
            source_identity_sha256=source_digest,
            tokenizer_sha256=_tokenizer_sha256(tokenizer),
        )
        corpus = allocation.payload["corpus"]
        assert isinstance(corpus, dict)
        if (
            corpus["train_jsonl_sha256"] != local_chat["train_sha256"]
            or corpus["validation_jsonl_sha256"] != local_chat["validation_sha256"]
        ):
            raise ValueError(
                "allocation manifest corpus digests do not match local_chat"
            )
    cache_identity = {
        "dataset": {
            key: value
            for key, value in config.dataset.model_dump(mode="json").items()
            if key
            not in {
                "cache_dir",
                "train_path",
                "validation_path",
                "source_manifest_path",
                "corpus_release_path",
                "corpus_export_path",
            }
        },
        "local_chat_source": local_chat,
        **({"corpus_export": corpus_export} if corpus_export is not None else {}),
        **(
            {
                "local_stories_source_sha256": hashlib.sha256(
                    canonical_json(local_stories)
                ).hexdigest()
            }
            if local_stories is not None
            else {}
        ),
        "allocation_manifest_sha256": None if allocation is None else allocation.sha256,
        "packing": {
            "memory": config.model.memory,
            "memory_table_size": config.model.memory_table_size,
            "memory_ngram_size": config.model.memory_ngram_size,
        },
        "packing_version": PACKING_VERSION,
        "source_identity_sha256": source_digest,
        "tokenizer_sha256": _tokenizer_sha256(tokenizer),
    }
    root = (
        config.dataset.cache_dir
        / hashlib.sha256(canonical_json(cache_identity)).hexdigest()[:16]
    )
    manifest_path = root / "manifest.json"
    train_path, validation_path = root / "train.npy", root / "validation.npy"
    train_supervision_path, validation_supervision_path = (
        root / "train_supervision.npy",
        root / "validation_supervision.npy",
    )
    byte_enabled = config.model.memory in {"byte", "portable"}
    train_byte_path, validation_byte_path = (
        root / "train_byte_addresses.npy",
        root / "validation_byte_addresses.npy",
    )
    if (
        manifest_path.is_file()
        and train_path.is_file()
        and validation_path.is_file()
        and train_supervision_path.is_file()
        and validation_supervision_path.is_file()
        and (
            not byte_enabled
            or (train_byte_path.is_file() and validation_byte_path.is_file())
        )
    ):
        with progress_phase(
            "data_cache_validation",
            completed_work=0,
            total_work=1,
            unit="cache_verifications",
            raw_counters={"cache_reused": True, "verification_only": True},
        ) as progress:
            cached = load_prepared_data(
                root, byte_enabled=byte_enabled, expected_identity=cache_identity
            )
            progress.update(
                completed_work=1,
                total_work=1,
                unit="cache_verifications",
                raw_counters={"cache_reused": True, "verification_only": True},
            )
            return cached
    _assert_local_chat_disjoint(config.dataset)
    _assert_local_chat_supervision_consistent(config.dataset)
    from sparselab.workspace_preflight import (
        check_storage,
        projected_data_bytes,
        require_storage,
    )

    require_storage([check_storage(root, projected_bytes=projected_data_bytes(config))])
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
    if config.dataset.source in {"local_stories", "local_text"}:
        if config.dataset.source == "local_stories":
            assert local_stories is not None
        train_stats = _collect_streaming(
            config.dataset,
            tokenizer,
            "train",
            temporary_root,
            selected_documents=(
                min(
                    config.dataset.train_max_documents,
                    local_stories["splits"]["train"]["count"],
                )
                if local_stories is not None
                else config.dataset.train_max_documents
            ),
            **settings,
        )
        validation_stats = _collect_streaming(
            config.dataset,
            tokenizer,
            "validation",
            temporary_root,
            selected_documents=(
                min(
                    config.dataset.validation_max_documents,
                    local_stories["splits"]["validation"]["count"],
                )
                if local_stories is not None
                else config.dataset.validation_max_documents
            ),
            **settings,
        )
        train = np.load(temporary_root / "train.npy", mmap_mode="r", allow_pickle=False)
        validation = np.load(
            temporary_root / "validation.npy", mmap_mode="r", allow_pickle=False
        )
    else:
        train, train_supervision, train_byte, train_stats = _collect(
            config.dataset, tokenizer, "train", **settings
        )
        validation, validation_supervision, validation_byte, validation_stats = (
            _collect(config.dataset, tokenizer, "validation", **settings)
        )
    if (
        len(train) < config.training.seq_len + 1
        or len(validation) < config.training.seq_len + 1
    ):
        raise ValueError("prepared split lacks a full next-token block")
    allocation_sides = None
    if allocation is not None:
        train_sides = allocation.split("train", token_count=len(train))
        validation_sides = allocation.split("validation", token_count=len(validation))
        allocation_sides = (train_sides, validation_sides)
        for name, values in (
            ("train_owner_ids.npy", train_sides.owner),
            ("validation_owner_ids.npy", validation_sides.owner),
            ("train_semantic_queries.npy", train_sides.semantic_queries),
            ("validation_semantic_queries.npy", validation_sides.semantic_queries),
            ("train_semantic_mask.npy", train_sides.semantic_mask),
            ("validation_semantic_mask.npy", validation_sides.semantic_mask),
        ):
            if values is not None:
                _atomic_array(temporary_root / name, values)
    if config.dataset.source not in {"local_stories", "local_text"}:
        _atomic_array(temporary_root / "train.npy", train)
        _atomic_array(temporary_root / "validation.npy", validation)
        _atomic_array(temporary_root / "train_supervision.npy", train_supervision)
        _atomic_array(
            temporary_root / "validation_supervision.npy", validation_supervision
        )
        if byte_enabled:
            assert train_byte is not None and validation_byte is not None
            _atomic_array(temporary_root / "train_byte_addresses.npy", train_byte)
            _atomic_array(
                temporary_root / "validation_byte_addresses.npy", validation_byte
            )
    attributions = {
        "tinystories": {
            "license": "CDLA-Sharing-1.0",
            "source_attribution": "roneneldan/TinyStories",
        },
        "fineweb_edu": {
            "license": "ODC-By-1.0; Common Crawl Terms of Use",
            "source_attribution": "HuggingFaceFW/fineweb-edu",
        },
        "cosmopedia": {
            "license": "Apache-2.0",
            "source_attribution": "HuggingFaceTB/cosmopedia dataset card",
        },
        "local_chat": {
            "license": config.dataset.license,
            "source_attribution": "user-provided local_chat",
        },
        "local_text": {
            "license": config.dataset.license,
            "source_attribution": "frozen corpus local_text export",
        },
        "local_stories": {
            "license": "CDLA-Sharing-1.0",
            "source_attribution": "roneneldan/TinyStories pinned local snapshot",
        },
        "synthetic": {
            "license": "synthetic fixture",
            "source_attribution": "sparselab synthetic",
        },
        "instruction_reference": {
            "license": "synthetic fixture",
            "source_attribution": "sparselab instruction_reference",
        },
        "chat_recall": {
            "license": "synthetic fixture",
            "source_attribution": "sparselab chat_recall",
        },
        "engram_recall": {
            "license": "synthetic fixture",
            "source_attribution": "sparselab engram_recall",
        },
        "withheld_facts": {
            "license": "project fixture",
            "source_attribution": "sparselab withheld_facts",
        },
    }
    manifest = {
        "packing_version": PACKING_VERSION,
        "cache_identity": cache_identity,
        "source": config.dataset.source,
        "revision": config.dataset.revision,
        "dataset_config": config.dataset.dataset_config,
        **attributions[config.dataset.source],
        **({"corpus_export": corpus_export} if corpus_export is not None else {}),
        "tokenizer_sha256": _tokenizer_sha256(tokenizer),
        "settings_sha256": hashlib.sha256(canonical_json(cache_identity)).hexdigest(),
        "source_identity_sha256": cache_identity["source_identity_sha256"],
        "train": {
            **train_stats,
            **_array_metadata(temporary_root / "train.npy"),
        },
        "validation": {
            **validation_stats,
            **_array_metadata(temporary_root / "validation.npy"),
        },
        "supervision": {
            "kind": "token-loss-mask-v1",
            "train": _array_metadata(
                temporary_root / "train_supervision.npy", dtype=np.dtype(bool)
            ),
            "validation": _array_metadata(
                temporary_root / "validation_supervision.npy", dtype=np.dtype(bool)
            ),
        },
        "byte_addressing": None
        if not byte_enabled
        else {
            "kind": "raw-utf8-suffix-v1",
            "table_size": config.model.memory_table_size,
            "ngram_size": config.model.memory_ngram_size,
            "train": _array_metadata(temporary_root / "train_byte_addresses.npy"),
            "validation": _array_metadata(
                temporary_root / "validation_byte_addresses.npy"
            ),
        },
        "allocation": None
        if allocation is None
        else {
            "format": "sparselab-prepared-allocation-v1",
            "manifest_sha256": allocation.sha256,
            "owner_codes": {"neural": 0, "lexical": 1, "semantic": 2, "hybrid": 3},
            "resource_regime": allocation.payload.get("resource_regime"),
            "ownership_profile": allocation.payload.get("ownership_profile"),
            "semantic": allocation.semantic,
            "train": {
                "owner": _array_metadata(
                    temporary_root / "train_owner_ids.npy", dtype=np.dtype(np.uint8)
                ),
                **(
                    {}
                    if allocation_sides is None
                    or allocation_sides[0].semantic_queries is None
                    else {
                        "semantic_queries": _array_metadata(
                            temporary_root / "train_semantic_queries.npy",
                            dtype=np.dtype(np.float32),
                            dimensions=2,
                        ),
                        "semantic_mask": _array_metadata(
                            temporary_root / "train_semantic_mask.npy",
                            dtype=np.dtype(bool),
                        ),
                    }
                ),
            },
            "validation": {
                "owner": _array_metadata(
                    temporary_root / "validation_owner_ids.npy",
                    dtype=np.dtype(np.uint8),
                ),
                **(
                    {}
                    if allocation_sides is None
                    or allocation_sides[1].semantic_queries is None
                    else {
                        "semantic_queries": _array_metadata(
                            temporary_root / "validation_semantic_queries.npy",
                            dtype=np.dtype(np.float32),
                            dimensions=2,
                        ),
                        "semantic_mask": _array_metadata(
                            temporary_root / "validation_semantic_mask.npy",
                            dtype=np.dtype(bool),
                        ),
                    }
                ),
            },
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    manifest_path = temporary_root / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json(manifest) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root.replace(root)
    directory_fd = os.open(root.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return load_prepared_data(
        root, byte_enabled=byte_enabled, expected_identity=cache_identity
    )


def prepare_data(config: RunConfig, tokenizer: Tokenizer) -> PreparedData:
    """Prepare data and mark only new caches owned by the selected workspace."""
    workspace = ensure_work_dir()
    base = config.dataset.cache_dir
    if (
        base.is_symlink()
        or not base.resolve().is_relative_to(workspace)
        or base.resolve() == workspace
    ):
        return _prepare_data(config, tokenizer)
    with campaign_lock(workspace):
        existing = {child.name for child in base.iterdir()} if base.is_dir() else set()
        prepared = _prepare_data(config, tokenizer)
        if prepared.root.name not in existing:
            mark_prepared_cache(workspace, prepared.root)
        return prepared


class TokenBlockDataset:
    def __init__(
        self,
        ids: np.ndarray,
        seq_len: int,
        byte_addresses: np.ndarray | None = None,
        supervision: np.ndarray | None = None,
        owner_ids: np.ndarray | None = None,
        semantic_queries: np.ndarray | None = None,
        semantic_mask: np.ndarray | None = None,
    ) -> None:
        if ids.ndim != 1 or ids.dtype != np.dtype(np.int32):
            raise ValueError("packed IDs must be one-dimensional int32")
        if supervision is not None and (
            supervision.ndim != 1
            or supervision.dtype != np.dtype(bool)
            or len(supervision) != len(ids)
        ):
            raise ValueError(
                "supervision mask must be one-dimensional bool aligned with IDs"
            )
        if byte_addresses is not None and len(byte_addresses) != len(ids):
            raise ValueError("byte address array must align with packed IDs")
        if owner_ids is not None and (
            owner_ids.ndim != 1
            or owner_ids.dtype != np.dtype(np.uint8)
            or len(owner_ids) != len(ids)
        ):
            raise ValueError("owner IDs must be one-dimensional uint8 aligned with IDs")
        if semantic_queries is not None and (
            semantic_queries.ndim != 2
            or semantic_queries.dtype != np.dtype(np.float32)
            or semantic_queries.shape[0] != len(ids)
            or semantic_mask is None
        ):
            raise ValueError(
                "semantic queries must be float32 [tokens,key_dim] with mask"
            )
        if semantic_mask is not None and (
            semantic_mask.ndim != 1
            or semantic_mask.dtype != np.dtype(bool)
            or len(semantic_mask) != len(ids)
        ):
            raise ValueError("semantic query mask must be bool aligned with IDs")
        self.ids, self.seq_len = ids, seq_len
        self.byte_addresses, self.supervision = byte_addresses, supervision
        self.owner_ids = owner_ids
        self.semantic_queries, self.semantic_mask = semantic_queries, semantic_mask
        candidates = (len(ids) - 1) // seq_len
        self.block_indices = (
            np.arange(candidates, dtype=np.int64)
            if supervision is None
            else np.asarray(
                [
                    block
                    for block in range(candidates)
                    if bool(
                        supervision[
                            block * seq_len + 1 : (block + 1) * seq_len + 1
                        ].any()
                    )
                ],
                dtype=np.int64,
            )
        )

    def __len__(self) -> int:
        return len(self.block_indices)

    def numpy_block(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        block = int(self.block_indices[index])
        start = block * self.seq_len
        values = np.asarray(self.ids[start : start + self.seq_len + 1], dtype=np.int64)
        targets = values[1:].copy()
        if self.supervision is not None:
            targets[~self.supervision[start + 1 : start + self.seq_len + 1]] = -100
        addresses = (
            None
            if self.byte_addresses is None
            else np.asarray(
                self.byte_addresses[start : start + self.seq_len], dtype=np.int64
            )
        )
        return values[:-1], targets, addresses

    def numpy_microblock(
        self, index: int
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        inputs, targets, addresses = self.numpy_block(index)
        start = int(self.block_indices[index]) * self.seq_len
        owners = (
            None
            if self.owner_ids is None
            else np.asarray(
                self.owner_ids[start + 1 : start + self.seq_len + 1], dtype=np.uint8
            )
        )
        queries = (
            None
            if self.semantic_queries is None
            else np.asarray(
                self.semantic_queries[start : start + self.seq_len], dtype=np.float32
            )
        )
        mask = (
            None
            if self.semantic_mask is None
            else np.asarray(
                self.semantic_mask[start : start + self.seq_len], dtype=bool
            )
        )
        return inputs, targets, addresses, owners, queries, mask

    def __getitem__(
        self, index: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        inputs, targets, addresses = self.numpy_block(index)
        if addresses is None:
            return torch.from_numpy(inputs), torch.from_numpy(targets)
        return (
            torch.from_numpy(inputs),
            torch.from_numpy(targets),
            torch.from_numpy(addresses),
        )


@dataclass(frozen=True)
class BatchCursor:
    epoch: int = 0
    next_block: int = 0


def epoch_order(num_blocks: int, seed: int, epoch: int) -> torch.Tensor:
    return torch.randperm(
        num_blocks, generator=torch.Generator().manual_seed(seed + epoch)
    )
