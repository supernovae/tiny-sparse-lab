"""Portable ByteLevel BPE tokenizer artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from sparselab.config.models import TokenizerTrainConfig
from sparselab.data.datasets import iter_documents

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


def train_tokenizer(config: TokenizerTrainConfig) -> Path:
    """Train BPE on a bounded train-only prefix and publish an immutable artifact."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output = config.output_dir
    json_path = output / "tokenizer.json"
    manifest_path = output / "tokenizer_manifest.json"

    selected: list[str] = []
    digest = hashlib.sha256()
    for document in iter_documents(config.dataset, "train"):
        if len(selected) >= config.max_documents:
            break
        if not document:
            continue
        selected.append(document)
        digest.update(document.encode("utf-8"))
        digest.update(b"\0")
    if not selected:
        raise ValueError("tokenizer training selected no non-empty documents")
    training_contract = {
        "vocab_size": config.vocab_size,
        "min_frequency": config.min_frequency,
        "source": config.dataset.source,
        "revision": config.dataset.revision,
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
    tokenizer.train_from_iterator(selected, trainer=trainer)
    actual = tokenizer.get_vocab_size()
    if actual != config.vocab_size:
        raise ValueError(
            f"BPE produced {actual} entries, requested {config.vocab_size}; increase document budget"
        )
    if [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] != [0, 1, 2, 3]:
        raise RuntimeError("tokenizer special token IDs are not the required 0..3")

    output.mkdir(parents=True, exist_ok=False)
    temporary = output / "tokenizer.json.tmp"
    tokenizer.save(str(temporary))
    temporary.replace(json_path)
    content = json_path.read_bytes()
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
            "content_digest_sha256": digest.hexdigest(),
            "tokenizers_version": __import__("tokenizers").__version__,
        },
    )
    return json_path
