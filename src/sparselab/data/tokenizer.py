"""Portable ByteLevel BPE tokenizer artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from sparselab.config.models import DatasetConfig, TokenizerTrainConfig
from sparselab.corpus.export import verify_release_export
from sparselab.data.datasets import iter_documents
from sparselab.data.local_stories import verify_snapshot
from sparselab.progress import progress_phase

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_tokenizer(path: Path) -> Tokenizer:
    if not path.is_file():
        raise FileNotFoundError(
            f"tokenizer artifact missing: {path}; run `sparselab tokenizer train CONFIG`"
        )
    return Tokenizer.from_file(str(path))


def verify_tokenizer_artifact(
    path: Path,
    *,
    source: str,
    revision: str | None,
    vocab_size: int,
    dataset: DatasetConfig | None = None,
) -> dict[str, object]:
    """Fail closed on an incomplete or mismatched tokenizer artifact."""
    manifest_path = path.with_name("tokenizer_manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"complete tokenizer artifact required at {path}; finish `sparselab tokenizer train CONFIG` first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
        or manifest.get("source") != source
        or manifest.get("revision") != revision
        or manifest.get("vocab_size") != vocab_size
    ):
        raise ValueError(f"tokenizer artifact provenance or digest mismatch: {path}")
    if source == "local_stories":
        if dataset is None:
            raise ValueError(
                "local_stories tokenizer verification requires dataset configuration"
            )
        assert dataset.source_manifest_path is not None
        verify_snapshot(dataset)
        snapshot_digest = hashlib.sha256(
            dataset.source_manifest_path.read_bytes()
        ).hexdigest()
        if manifest.get("source_manifest_sha256") != snapshot_digest:
            raise ValueError("tokenizer snapshot identity mismatch")
    if dataset is not None and dataset.corpus_release_path is not None:
        binding = verify_release_export(dataset)
        if (
            manifest.get("corpus_export") != binding
            or manifest.get("training_contract", {}).get("corpus_export") != binding
        ):
            raise ValueError("tokenizer corpus export identity mismatch")
    elif manifest.get("corpus_export") is not None:
        raise ValueError(
            "corpus tokenizer verification requires frozen export configuration"
        )
    return manifest


def _bounded_documents(
    config: TokenizerTrainConfig,
    stats: dict[str, object],
    *,
    whole_documents: bool = False,
):
    """Replay the same train prefix without retaining corpus strings in memory."""
    selected = acquired = byte_count = 0
    digest = hashlib.sha256()
    documents = iter(iter_documents(config.dataset, "train"))
    budget = config.dataset.train_max_tokens
    stop_reason = "source_exhausted"
    while acquired < config.max_documents and byte_count < budget:
        try:
            document = next(documents)
        except StopIteration:
            break
        acquired += 1
        if not document:
            continue
        encoded = document.encode("utf-8")
        remaining = budget - byte_count
        if len(encoded) > remaining:
            if whole_documents:
                stop_reason = "byte_limit"
                break
            document = encoded[:remaining].decode("utf-8", errors="ignore")
            encoded = document.encode("utf-8")
            truncated = True
        else:
            truncated = False
        if document:
            digest.update(encoded + b"\0")
            byte_count += len(encoded)
            selected += 1
            yield document
        if truncated:
            stop_reason = "byte_limit"
            break
    if stop_reason == "source_exhausted":
        if acquired >= config.max_documents:
            stop_reason = "document_limit"
        elif byte_count >= budget:
            stop_reason = "byte_limit"
    stats.update(
        acquired=acquired,
        selected=selected,
        bytes=byte_count,
        digest=digest.hexdigest(),
        stop_reason=stop_reason,
    )


def _snapshot_digest(config: TokenizerTrainConfig) -> str | None:
    if config.dataset.source != "local_stories":
        return None
    assert config.dataset.source_manifest_path is not None
    verify_snapshot(config.dataset)
    return hashlib.sha256(config.dataset.source_manifest_path.read_bytes()).hexdigest()


def train_tokenizer(config: TokenizerTrainConfig) -> Path:
    """Train BPE from a bounded train-only UTF-8 byte prefix."""
    from sparselab.workspace_preflight import require_storage, tokenizer_storage_checks

    require_storage(tokenizer_storage_checks(config))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output = config.output_dir
    json_path = output / "tokenizer.json"
    manifest_path = output / "tokenizer_manifest.json"

    input_byte_budget = config.dataset.train_max_tokens
    corpus_export = (
        verify_release_export(config.dataset)
        if config.dataset.corpus_release_path is not None
        else None
    )
    if corpus_export is not None and corpus_export["vocab_size"] != config.vocab_size:
        raise ValueError(
            "corpus export vocabulary size does not match tokenizer request"
        )
    source_manifest_sha256 = _snapshot_digest(config)
    whole_documents = config.dataset.source in {"local_stories", "local_text"}
    stats: dict[str, object] = {}
    source_started = time.monotonic()
    with progress_phase(
        "tokenizer_dataset_initialization_and_source_iteration",
        completed_work=0,
        unit="documents",
        raw_counters={
            "documents_acquired": 0,
            "documents_retained": 0,
            "source_bytes": 0,
        },
    ) as progress:
        for _ in _bounded_documents(config, stats, whole_documents=whole_documents):
            pass
        selected = int(stats["selected"])
        selected_input_bytes = int(stats["bytes"])
        acquired = int(stats["acquired"])
        progress.update(
            completed_work=acquired,
            unit="documents",
            raw_counters={
                "documents_acquired": acquired,
                "documents_retained": selected,
                "source_bytes": selected_input_bytes,
                "source_bytes_per_second": selected_input_bytes
                / max(time.monotonic() - source_started, 1e-9),
            },
        )
    if not selected:
        raise ValueError("tokenizer training selected no non-empty documents")
    training_contract = {
        "vocab_size": config.vocab_size,
        "min_frequency": config.min_frequency,
        "source": config.dataset.source,
        "revision": config.dataset.revision,
        "input_byte_budget_utf8": input_byte_budget,
        "selected_input_bytes_utf8": selected_input_bytes,
        "max_documents": config.max_documents,
        "stop_reason": stats["stop_reason"],
        "acquired_documents": acquired,
        "content_digest_sha256": stats["digest"],
        "source_manifest_sha256": source_manifest_sha256,
        **({"corpus_export": corpus_export} if corpus_export is not None else {}),
    }
    if json_path.exists() or manifest_path.exists():
        if json_path.is_file() and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get(
                "training_contract"
            ) == training_contract and hashlib.sha256(
                json_path.read_bytes()
            ).hexdigest() == manifest.get("sha256"):
                return json_path
        raise FileExistsError(
            f"tokenizer output at {output} has different or unverifiable training provenance; "
            "use a new output directory, or reference the existing tokenizer explicitly"
        )

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
    )
    bpe_started = time.monotonic()
    with progress_phase(
        "tokenizer_bpe_training",
        completed_work=0,
        total_work=1,
        unit="trainer_calls",
        raw_counters={
            "documents": selected,
            "source_bytes": selected_input_bytes,
        },
    ) as progress:
        replay_stats: dict[str, object] = {}
        tokenizer.train_from_iterator(
            _bounded_documents(config, replay_stats, whole_documents=whole_documents),
            trainer=trainer,
            length=selected,
        )
        if replay_stats != stats:
            raise ValueError(
                "tokenizer training source changed between prefix verification and BPE fit"
            )
        bpe_elapsed = max(time.monotonic() - bpe_started, 1e-9)
        progress.update(
            completed_work=1,
            total_work=1,
            unit="trainer_calls",
            raw_counters={
                "documents": selected,
                "source_bytes": selected_input_bytes,
                "bpe_input_bytes_per_second": selected_input_bytes / bpe_elapsed,
            },
        )
    actual = tokenizer.get_vocab_size()
    if actual != config.vocab_size:
        raise ValueError(
            f"BPE produced {actual} entries, requested {config.vocab_size}; increase document budget"
        )
    if [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] != [0, 1, 2, 3]:
        raise RuntimeError("tokenizer special token IDs are not the required 0..3")
    with progress_phase(
        "tokenizer_artifact_serialization",
        completed_work=0,
        unit="bytes",
        raw_counters={"artifact": "tokenizer.json"},
    ) as progress:
        output.mkdir(parents=True, exist_ok=False)
        temporary = output / "tokenizer.json.tmp"
        tokenizer.save(str(temporary))
        temporary.replace(json_path)
        artifact_bytes = json_path.stat().st_size
        progress.update(
            completed_work=artifact_bytes,
            total_work=artifact_bytes,
            unit="bytes",
            raw_counters={
                "artifact": "tokenizer.json",
                "artifact_bytes": artifact_bytes,
            },
        )
    with progress_phase(
        "tokenizer_artifact_hash",
        completed_work=0,
        unit="bytes",
        raw_counters={"artifact": "tokenizer.json"},
    ) as progress:
        content = json_path.read_bytes()
        content_sha256 = hashlib.sha256(content).hexdigest()
        progress.update(
            completed_work=len(content),
            total_work=len(content),
            unit="bytes",
            raw_counters={
                "artifact": "tokenizer.json",
                "artifact_bytes": len(content),
            },
        )
    with progress_phase(
        "tokenizer_manifest_serialization",
        completed_work=0,
        unit="bytes",
        raw_counters={"artifact": "tokenizer_manifest.json"},
    ) as progress:
        _atomic_json(
            manifest_path,
            {
                "sha256": content_sha256,
                "training_contract": training_contract,
                "license": config.dataset.license
                if config.dataset.source
                in {"local_chat", "local_text", "local_stories"}
                else None,
                "requested_vocab_size": config.vocab_size,
                **(
                    {"bpe_fit_seconds": bpe_elapsed}
                    if config.dataset.source == "local_stories"
                    else {}
                ),
                "vocab_size": actual,
                "special_ids": {
                    token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS
                },
                "normalizer": None,
                "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
                "decoder": "ByteLevel",
                "source": config.dataset.source,
                "revision": config.dataset.revision,
                "split": "train",
                "selected_documents": selected,
                "input_byte_budget_utf8": input_byte_budget,
                "selected_input_bytes_utf8": selected_input_bytes,
                "content_digest_sha256": stats["digest"],
                "source_manifest_sha256": source_manifest_sha256,
                **(
                    {"corpus_export": corpus_export}
                    if corpus_export is not None
                    else {}
                ),
                "tokenizers_version": __import__("tokenizers").__version__,
            },
        )
        manifest_bytes = manifest_path.stat().st_size
        progress.update(
            completed_work=manifest_bytes,
            total_work=manifest_bytes,
            unit="bytes",
            raw_counters={
                "artifact": "tokenizer_manifest.json",
                "artifact_bytes": manifest_bytes,
            },
        )
    return json_path
