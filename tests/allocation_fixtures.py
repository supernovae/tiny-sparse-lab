"""Generated wiring inputs for the existing fixed-shape allocation contract.

No personal corpus is read or copied. The legacy schema/card names and 24/8/12
shape are required by the current builder, not a benchmark or quality claim.
"""

import json
from pathlib import Path, PurePosixPath

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from sparselab.data.tokenizer import SPECIAL_TOKENS


def write_allocation_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    cards = root / "cards"
    cards.mkdir()
    audit = []
    for split, count, card_name in (
        ("train", 24, "acquisition"),
        ("development", 8, "development"),
        ("frozen_evaluation", 12, "frozen"),
    ):
        records, cases = [], []
        for line in range(1, count + 1):
            path = f"fixture/{split}/Item{line:02d}.txt"
            # Preserve case and composed Unicode literally, not via normalization.
            if line == 1:
                path = f"fixture/{split}/Café.txt"
            operation = "name" if line % 2 else "with_suffix"
            argument = None if line % 2 else ".BIN"
            label = (
                PurePosixPath(path).name
                if argument is None
                else str(PurePosixPath(path).with_suffix(argument))
            )
            prompt = f"Apply {operation} to {path}."
            if argument is not None:
                prompt += f" Argument: {argument}."
            records.append(
                {
                    "format_version": 2,
                    "loss_mode": "assistant_only",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": label},
                    ],
                }
            )
            audit.append(
                {
                    "split": split,
                    "line": line,
                    "matches_oracle": True,
                    "semantic_key": {
                        "path": path,
                        "operation": operation,
                        "argument": argument,
                    },
                    "rendered_assistant_label": label,
                }
            )
            if line <= 2:
                cases.append(
                    {
                        "identifier": f"fixture-{operation.replace('_', '-')}-{line}",
                        "prompt": prompt,
                        "expected": label,
                        "kind": "custom",
                    }
                )
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )
        card = {
            "format": "capability_card_v2",
            "name": f"path-domain-{card_name}-v1",
            "version": 2,
            "hypothesis": "Synthetic allocation wiring only.",
            "scorer": "literal_full_answer_exact_v1",
            "cases": cases,
            "generation": {
                "max_new_tokens": 8,
                "temperature": 0.0,
                "top_k": 0,
                "seed": 0,
                "stop_sequences": ["\nUser:", "\nSystem:", "\nAssistant:"],
            },
            "scoring": {
                "normalization": "trim_outer_whitespace",
                "match": "full_answer",
            },
            "controls": {"uniform_chance": 0.0, "majority_answer": 0.0},
            "limitations": "Generated schema fixture; no model or scientific evidence.",
        }
        (cards / f"path-domain-{card_name}-v1.json").write_text(
            json.dumps(card), encoding="utf-8"
        )
    (root / "oracle_audit.json").write_text(
        json.dumps(
            {
                "format": "path_domain_oracle_audit_v1",
                "records": audit,
            }
        ),
        encoding="utf-8",
    )
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "format": "path_domain_provenance_v1",
                "dataset_name": "path_domain_v1",
                "dataset_version": "1.1.0",
                "record_format": {"loss_mode": "assistant_only"},
                "splits": {
                    "training": {"path": "train.jsonl"},
                    "development": {"path": "development.jsonl"},
                },
                "license": {"spdx": "MIT"},
                "oracle": {"audit": "oracle_audit.json"},
                "fixture_scope": "Generated test records, not the published personal corpus.",
            }
        ),
        encoding="utf-8",
    )
    # Fixed byte alphabet: no tokenizer training, merges or external artifact reads.
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    vocabulary.update(
        {
            token: index + len(SPECIAL_TOKENS)
            for index, token in enumerate(sorted(ByteLevel.alphabet()))
        }
    )
    tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    return root, tokenizer_path
