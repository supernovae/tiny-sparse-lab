from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from sparselab.config.migrate import migrate_v1
from sparselab.config.models import AttentionConfig, ModelConfig, RunConfig
from sparselab.data.byte_hash import table_address
from sparselab.data.packing import prepare_data
from sparselab.evaluation.chat import chat_turn
from sparselab.evaluation.generation import _prompt_byte_addresses, generate
from sparselab.model.transformer import DenseLM


def _tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train_from_iterator(
        ["hello world\nhi\nUser: next", "é"],
        BpeTrainer(
            vocab_size=260,
            initial_alphabet=ByteLevel.alphabet(),
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        ),
    )
    return tokenizer


class _ScriptedModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        scripts: list[dict[int, float]],
        *,
        memory: str = "none",
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            memory=memory, memory_table_size=97, memory_ngram_size=3
        )
        self.vocab_size = tokenizer.get_vocab_size()
        self.scripts = scripts
        self.calls = 0
        self.byte_addresses: list[torch.Tensor | None] = []

    def forward(
        self, input_ids: torch.Tensor, *, byte_addresses: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.byte_addresses.append(
            None if byte_addresses is None else byte_addresses.cpu()
        )
        scores = torch.full((1, input_ids.shape[1], self.vocab_size), -100.0)
        for token_id, score in self.scripts[
            min(self.calls, len(self.scripts) - 1)
        ].items():
            scores[0, -1, token_id] = score
        self.calls += 1
        return scores.to(input_ids.device)


def _ids(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def test_generation_rejects_invalid_sampling_ranges_before_model_execution() -> None:
    tokenizer = _tokenizer()
    model = _ScriptedModel(tokenizer, [{0: 1}])

    with pytest.raises(ValueError, match="max_seq_len"):
        generate(model, tokenizer, "hello", 0, 1, torch.device("cpu"))
    with pytest.raises(ValueError, match="top_k"):
        generate(model, tokenizer, "hello", 8, 1, torch.device("cpu"), top_k=-1)
    with pytest.raises(ValueError, match="requested completion"):
        generate(
            model,
            tokenizer,
            "hello",
            len(_ids(tokenizer, "hello")),
            1,
            torch.device("cpu"),
            strict_context=True,
        )
    assert model.calls == 0


def test_generation_defaults_to_greedy_suppresses_special_outputs_and_restores_mode() -> (
    None
):
    tokenizer = _tokenizer()
    pad = tokenizer.token_to_id("<pad>")
    eos = tokenizer.token_to_id("<eos>")
    token = _ids(tokenizer, "h")[0]
    assert pad is not None and eos is not None
    model = _ScriptedModel(tokenizer, [{pad: 20, token: 10}, {eos: 20}])
    model.train()

    completion = generate(model, tokenizer, "prompt", 16, 2, torch.device("cpu"))

    assert completion == "prompth"
    assert model.training


def test_sampling_is_seeded_locally_without_mutating_global_rng() -> None:
    tokenizer = _tokenizer()
    first, second = _ids(tokenizer, "hi")
    model = _ScriptedModel(tokenizer, [{first: 0, second: 0}])
    global_before = torch.get_rng_state()

    first_completion = generate(
        model, tokenizer, "", 16, 1, torch.device("cpu"), temperature=1.0, seed=43
    )
    second_completion = generate(
        model, tokenizer, "", 16, 1, torch.device("cpu"), temperature=1.0, seed=43
    )

    assert first_completion == second_completion
    assert torch.equal(torch.get_rng_state(), global_before)


def test_generation_stops_on_eos_without_echoing_new_prompt_text() -> None:
    tokenizer = _tokenizer()
    eos = tokenizer.token_to_id("<eos>")
    assert eos is not None
    model = _ScriptedModel(tokenizer, [{eos: 10}])

    assert (
        generate(
            model, tokenizer, "User: hello\n\nAssistant:", 32, 4, torch.device("cpu")
        )
        == "User: hello\n\nAssistant:"
    )


def test_byte_memory_uses_raw_leading_space_and_unicode_bytes_for_causal_addresses() -> (
    None
):
    tokenizer = _tokenizer()
    prompt_ids = _ids(tokenizer, " é")
    assert len(prompt_ids) == 3  # Space plus UTF-8's two BPE byte symbols.
    output = _ids(tokenizer, "!")[0]
    eos = tokenizer.token_to_id("<eos>")
    assert eos is not None
    model = _ScriptedModel(tokenizer, [{output: 10}, {eos: 10}], memory="byte")

    generate(model, tokenizer, " é", 3, 2, torch.device("cpu"))

    expected_prompt = [
        table_address(b" ", 97),
        table_address(b" \xc3", 97),
        table_address(b" \xc3\xa9", 97),
    ]
    assert model.byte_addresses[0].tolist() == [expected_prompt]
    assert model.byte_addresses[1].tolist() == [
        [expected_prompt[1], expected_prompt[2], table_address(b"\xc3\xa9!", 97)]
    ]


def test_prepare_data_unicode_addresses_match_generation_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = _tokenizer()
    document = " leading é"
    config = RunConfig.model_validate(
        migrate_v1(
            {
                "schema_version": 1,
                "name": "byte-parity",
                "seed": 1,
                "device": "cpu",
                "model": {
                    "vocab_size": tokenizer.get_vocab_size(),
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 2,
                    "ffn_dim": 32,
                    "max_seq_len": 16,
                    "memory": "byte",
                    "memory_table_size": 97,
                    "memory_ngram_size": 3,
                    "memory_dim": 8,
                },
                "tokenizer": {"path": tmp_path / "tokenizer.json"},
                "dataset": {
                    "source": "synthetic",
                    "cache_dir": tmp_path / "cache",
                    "train_max_documents": 1,
                    "validation_max_documents": 1,
                    "train_max_tokens": 64,
                    "validation_max_tokens": 64,
                },
                "training": {
                    "batch_size": 1,
                    "seq_len": 8,
                    "max_steps": 1,
                    "max_tokens": 8,
                },
                "optimizer": {
                    "learning_rate": 0.001,
                    "min_learning_rate": 0.0001,
                    "warmup_steps": 0,
                },
                "logging": {"root_dir": tmp_path / "runs", "checkpoint_every_steps": 1},
            }
        )
    )
    monkeypatch.setattr(
        "sparselab.data.packing.iter_documents",
        lambda _config, _split: iter([document]),
    )

    prepared = prepare_data(config, tokenizer)
    eos = tokenizer.token_to_id("<eos>")
    assert eos is not None and prepared.train_byte_addresses is not None
    end = list(prepared.train).index(eos)
    ids = prepared.train[:end].tolist()
    assert prepared.train_byte_addresses[:end].tolist() == _prompt_byte_addresses(
        tokenizer, document, ids, 97, 3
    )


def test_chat_stops_generation_when_another_role_boundary_is_decoded() -> None:
    tokenizer = _tokenizer()
    reply_ids = _ids(tokenizer, " hi\nUser:")
    eos = tokenizer.token_to_id("<eos>")
    assert eos is not None
    model = _ScriptedModel(
        tokenizer, [{token_id: 10} for token_id in reply_ids] + [{eos: 10}]
    )

    prompt, reply = chat_turn(
        model, tokenizer, [], "hello", 64, len(reply_ids) + 1, torch.device("cpu")
    )

    assert prompt == "User: hello\n\nAssistant:"
    assert reply == "hi"
    assert model.calls == len(reply_ids)


def test_real_tiny_model_forward_is_usable_for_generation() -> None:
    tokenizer = _tokenizer()
    model = DenseLM(
        ModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_seq_len=16,
        ),
        AttentionConfig(),
    )
    model.train()

    completion = generate(model, tokenizer, "hello", 16, 1, torch.device("cpu"))

    assert completion.startswith("hello")
    assert model.training
