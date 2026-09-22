from __future__ import annotations

import torch

from sparselab.config.models import AttentionConfig, ModelConfig
from sparselab.model.memory import TokenNgramMemory
from sparselab.model.transformer import DenseLM


def test_ngram_address_does_not_read_future_ids() -> None:
    memory = TokenNgramMemory(hidden_dim=4, table_size=97, ngram_size=3, value_dim=3)
    left = torch.tensor([[4, 8, 15, 16, 23]])
    right = left.clone()
    right[:, 3:] = torch.tensor([42, 99])
    assert torch.equal(memory.addresses(left)[:, :3], memory.addresses(right)[:, :3])


def test_disabled_memory_is_absent_and_enabled_memory_runs_backward() -> None:
    plain = ModelConfig(
        vocab_size=512,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        max_seq_len=16,
    )
    assert DenseLM(plain, AttentionConfig()).memory is None
    config = plain.model_copy(
        update={
            "memory": "ngram",
            "memory_table_size": 31,
            "memory_ngram_size": 3,
            "memory_dim": 7,
        }
    )
    model = DenseLM(config, AttentionConfig())
    logits = model(torch.randint(0, 512, (2, 8)))
    logits.mean().backward()
    assert model.memory is not None
    assert model.memory.last_diagnostics is not None
