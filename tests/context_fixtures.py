"""Tiny declarative composition contract; no scientific curriculum or training."""

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit


def write_composition_fixture(root: Path) -> tuple[Path, Tokenizer]:
    answers = ("amber", "ivory", "onyx", "pearl", "cobalt", "sienna", "topaz", "violet")
    cases = []
    for pair in range(4):
        edges = (
            f"start{pair} maps to middle{pair}. middle{pair} maps to {answers[2 * pair]}. "
            f"other{pair} maps to next{pair}. next{pair} maps to {answers[2 * pair + 1]}."
        )
        for side, start in enumerate((f"start{pair}", f"other{pair}")):
            cases.append(
                {
                    "identifier": f"two-hop-{pair + 1}{'ab'[side]}",
                    "prompt": f"{edges} Follow two links from {start}. Reply with only the destination.",
                    "expected": answers[2 * pair + side],
                    "kind": "custom",
                }
            )
    payload = {
        "format": "capability_card_v2",
        "name": "neutral-composition-fixture",
        "version": 1,
        "hypothesis": "Fixture checks graph and card wiring only.",
        "scorer": "normalized_full_answer_exact_v1",
        "cases": cases,
        "generation": {
            "max_new_tokens": 8,
            "temperature": 0.0,
            "top_k": 0,
            "seed": 0,
            "stop_sequences": ["\nUser:", "\nSystem:", "\nAssistant:"],
        },
        "scoring": {
            "normalization": "unicode_nfc_casefold_trim_collapse_whitespace",
            "match": "full_answer",
        },
        "controls": {"uniform_chance": 0.5, "majority_answer": 0.125},
        "limitations": "Synthetic parser fixture; no model quality evidence.",
    }
    path = root / "composition.card.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    words = sorted({word for case in cases for word in case["prompt"].split()})
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, **{word: i + 1 for i, word in enumerate(words)}},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return path, tokenizer
