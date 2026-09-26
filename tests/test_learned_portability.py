"""Learned-portability evaluator protocol boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparselab.evaluation.inference import InferenceRun
from sparselab.evaluation.learned_portability import evaluate_learned_portability
from sparselab.model.memory import ByteAddressMemory
from sparselab.data.learned_portability import materialize_learned_portability_data
from sparselab.data.tokenizer import load_tokenizer


class _AlwaysA(torch.nn.Module):
    def __init__(self, vocab_size: int, answer_id: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            memory="byte", memory_table_size=65521, memory_ngram_size=32, max_seq_len=128
        )
        self.memory = ByteAddressMemory(2, 65521, 2)
        self.vocab_size = vocab_size
        self.answer_id = answer_id

    def forward(self, input_ids: torch.Tensor, *, byte_addresses: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros((*input_ids.shape, 2), device=input_ids.device)
        self.memory(hidden, byte_addresses)
        logits = torch.zeros((*input_ids.shape, self.vocab_size), device=input_ids.device)
        logits[..., self.answer_id] = 1.0
        return logits



def _fixture(root: Path) -> tuple[InferenceRun, Path]:
    data_root = root / "data"
    manifest_path = materialize_learned_portability_data(data_root, fact_count=128)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_path = data_root / manifest["tokenizer"]["path"]
    tokenizer = load_tokenizer(tokenizer_path)
    answer_id = tokenizer.token_to_id("A")
    assert answer_id is not None
    model = _AlwaysA(tokenizer.get_vocab_size(), answer_id)
    loaded = InferenceRun(
        data_root,
        None,
        model,
        tokenizer,
        torch.device("cpu"),
        {"tokenizer_sha256": manifest["tokenizer"]["sha256"]},
    )
    return loaded, manifest_path

def test_scores_full_vocabulary_symbol_and_preserves_inference_state(tmp_path: Path) -> None:
    loaded, manifest = _fixture(tmp_path)
    loaded.model.train()
    before = torch.get_rng_state()
    result = evaluate_learned_portability(
        loaded, manifest, partitions=("held_out",), wording="final_report"
    )
    metrics = result["metrics"]
    assert metrics["count"] == 64
    assert metrics["correct"] == 2
    assert metrics["accuracy"] == 2 / 64
    assert isinstance(metrics["answer_nll"], float) and metrics["answer_nll"] > 0.0
    assert {row["predicted_answer"] for row in result["results"]} == {"A"}
    assert loaded.model.training
    assert torch.equal(before, torch.get_rng_state())

@pytest.mark.parametrize("special", ("<pad>", "<bos>", "<eos>", "<unk>"))
def test_control_token_argmax_is_scored_as_an_incorrect_prediction(
    special: str, tmp_path: Path
) -> None:
    loaded, manifest = _fixture(tmp_path)
    token_id = loaded.tokenizer.token_to_id(special)
    assert token_id is not None
    loaded.model.answer_id = token_id

    result = evaluate_learned_portability(
        loaded, manifest, partitions=("held_out",), wording="final_report"
    )

    assert result["metrics"]["count"] == 64
    assert result["metrics"]["correct"] == 0
    assert result["metrics"]["accuracy"] == 0.0
    assert {row["predicted_token_id"] for row in result["results"]} == {token_id}


def test_rejects_tampered_query_file_before_scoring(tmp_path: Path) -> None:
    loaded, manifest_path = _fixture(tmp_path)
    query_path = manifest_path.parent / "queries.jsonl"
    row = json.loads(query_path.read_text(encoding="utf-8").splitlines()[0])
    row["address"] += 1
    query_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file integrity"):
        evaluate_learned_portability(
            loaded, manifest_path, partitions=("held_out",), wording="final_report"
        )
