"""Portable ByteLevel BPE tokenizer artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from sparselab.config.models import TokenizerTrainConfig
from sparselab.data.datasets import iter_documents
from sparselab.data.progress import heartbeat

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
    path: Path, *, source: str, revision: str | None, vocab_size: int
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
    return manifest


def train_tokenizer(config: TokenizerTrainConfig) -> Path:
    """Train BPE from a bounded train-only UTF-8 byte prefix."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output = config.output_dir
    json_path = output / "tokenizer.json"
    manifest_path = output / "tokenizer_manifest.json"

    selected: list[str] = []
    digest = hashlib.sha256()
    documents = iter(iter_documents(config.dataset, "train"))
    input_byte_budget = config.dataset.train_max_tokens
    selected_input_bytes = 0
    acquired = 0
    source_started = time.monotonic()
    with heartbeat("tokenizer_dataset_initialization_and_source_iteration"):
        while (
            acquired < config.max_documents and selected_input_bytes < input_byte_budget
        ):
            try:
                document = next(documents)
            except StopIteration:
                break
            acquired += 1
            if not document:
                continue
            encoded = document.encode("utf-8")
            remaining = input_byte_budget - selected_input_bytes
            truncated = len(encoded) > remaining
            if truncated:
                document = encoded[:remaining].decode("utf-8", errors="ignore")
                encoded = document.encode("utf-8")
            if document:
                selected.append(document)
                digest.update(encoded)
                digest.update(b"\0")
                selected_input_bytes += len(encoded)
            if truncated:
                break
    source_elapsed = max(time.monotonic() - source_started, 1e-9)
    print(
        json.dumps(
            {
                "preprocessing": "tokenizer_source_selection",
                "documents_acquired": acquired,
                "documents_retained": len(selected),
                "source_bytes": selected_input_bytes,
                "source_bytes_per_second": selected_input_bytes / source_elapsed,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
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
        "content_digest_sha256": digest.hexdigest(),
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
    with heartbeat("tokenizer_bpe_training"):
        tokenizer.train_from_iterator(selected, trainer=trainer)
    actual = tokenizer.get_vocab_size()
    print(
        json.dumps(
            {
                "preprocessing": "tokenizer_bpe_training",
                "documents": len(selected),
                "source_bytes": selected_input_bytes,
                "bpe_input_bytes_per_second": selected_input_bytes
                / max(time.monotonic() - bpe_started, 1e-9),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    if actual != config.vocab_size:
        raise ValueError(
            f"BPE produced {actual} entries, requested {config.vocab_size}; increase document budget"
        )
    if [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] != [0, 1, 2, 3]:
        raise RuntimeError("tokenizer special token IDs are not the required 0..3")
    with heartbeat("tokenizer_artifact_serialization_and_hash"):
        output.mkdir(parents=True, exist_ok=False)
        temporary = output / "tokenizer.json.tmp"
        tokenizer.save(str(temporary))
        temporary.replace(json_path)
        content = json_path.read_bytes()
    with heartbeat("tokenizer_manifest_serialization_and_hash"):
        _atomic_json(
            manifest_path,
            {
                "sha256": hashlib.sha256(content).hexdigest(),
                "training_contract": training_contract,
                "license": config.dataset.license
                if config.dataset.source == "local_chat"
                else None,
                "requested_vocab_size": config.vocab_size,
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
                "selected_documents": len(selected),
                "input_byte_budget_utf8": input_byte_budget,
                "selected_input_bytes_utf8": selected_input_bytes,
                "content_digest_sha256": digest.hexdigest(),
                "tokenizers_version": __import__("tokenizers").__version__,
            },
        )
    return json_path
