"""Offline declaration and one-shot guards for the matched-prompt diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _tool():
    path = Path(__file__).parents[1] / "tools/kernel-memory-lab/run-card05-matched-diagnostic.py"
    spec = importlib.util.spec_from_file_location("kml_matched_diagnostic", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool):
        assert not add_special_tokens
        return type("Encoded", (), {"ids": list(range(len(text)))})()


def _declaration(module):
    return {
        "format": "kml-card05-matched-diagnostic-v1",
        "pairs": [
            {
                "stratum": stratum,
                "split": "test",
                "source_record_id": f"{index:064x}",
                "prose_prompt": f"Passage {index}.",
                "question_prompt": f"Context:\nPassage {index}.\nQuestion: What number?\nAnswer:",
                "gold_answer": str(index),
            }
            for index, stratum in enumerate(module.PAIR_STRATA)
        ],
    }


def test_exact_eight_pairs_and_two_preselected_parity_cases() -> None:
    module = _tool()
    jobs = module.validate_declaration(_declaration(module), _Tokenizer())
    assert len(jobs) == 18
    assert jobs[:2] == [(0, "prose", "Passage 0.", True),
                        (0, "question", "Context:\nPassage 0.\nQuestion: What number?\nAnswer:", True)]
    assert jobs[-2:] == [(0, "prose", "Passage 0.", False),
                         (0, "question", "Context:\nPassage 0.\nQuestion: What number?\nAnswer:", False)]


@pytest.mark.parametrize("change", ["count", "stratum", "split", "context", "long"])
def test_invalid_declarations_fail_before_inference(change: str) -> None:
    module = _tool()
    declaration = _declaration(module)
    if change == "count":
        declaration["pairs"].pop()
    elif change == "stratum":
        declaration["pairs"][0]["stratum"] = "wrong"
    elif change == "split":
        declaration["pairs"][0]["split"] = "train"
    elif change == "context":
        declaration["pairs"][0]["question_prompt"] = "Context:\nother\nQuestion: X\nAnswer:"
    else:
        declaration["pairs"][0]["prose_prompt"] = "x" * 961
        declaration["pairs"][0]["question_prompt"] = (
            "Context:\n" + "x" * 961 + "\nQuestion: X\nAnswer:"
        )
    with pytest.raises(ValueError):
        module.validate_declaration(declaration, _Tokenizer())


def test_existing_output_rejects_replay_without_loading_model(tmp_path: Path) -> None:
    module = _tool()
    output = tmp_path / "generations.jsonl"
    output.touch()
    with pytest.raises(FileExistsError):
        module.run(tmp_path / "missing.json", output, "invalid", tmp_path, tmp_path)
