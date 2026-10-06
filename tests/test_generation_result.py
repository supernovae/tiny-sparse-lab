from __future__ import annotations

import weakref
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from sparselab.evaluation.generation import (
    GenerationCancelled,
    generate,
    generate_result,
    generate_with_token_ids,
)


@pytest.fixture
def tokenizer() -> Tokenizer:
    result = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "hello": 4,
                "one": 5,
                "two": 6,
            },
            unk_token="<unk>",
        )
    )
    result.pre_tokenizer = Whitespace()
    return result


class Cache:
    def __init__(self, length: int, capacity: int) -> None:
        self.length = length
        self.capacity = capacity


class ScriptedModel(torch.nn.Module):
    def __init__(self, tokens: list[int], cancellation: Event | None = None) -> None:
        super().__init__()
        self.config = SimpleNamespace(memory="none")
        self.tokens = tokens
        self.calls = 0
        self.cancellation = cancellation
        self.cancel_on_call = 1
        self.caches: list[weakref.ReferenceType[Cache]] = []

    def forward(self, ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        scores = torch.full((1, ids.shape[1], 7), -100.0)
        scores[0, -1, self.tokens[min(self.calls, len(self.tokens) - 1)]] = 10
        self.calls += 1
        if self.cancellation is not None and self.calls == self.cancel_on_call:
            self.cancellation.set()
        return scores

    def forward_cached(
        self,
        ids: torch.Tensor,
        *,
        cache: Cache | None = None,
        cache_capacity: int = 0,
        **kwargs: object,
    ) -> tuple[torch.Tensor, Cache]:
        if cache is None:
            cache = Cache(ids.shape[1], cache_capacity)
            self.caches.append(weakref.ref(cache))
        else:
            cache.length += ids.shape[1]
        return self.forward(ids), cache


@pytest.mark.parametrize(
    "tokens,limit,stops,text,ids,count,reason",
    [
        ([2], 4, (), "hello", [], 1, "stop"),
        ([5, 2], 4, (), "helloone", [5], 2, "stop"),
        ([5, 6], 2, (), "helloone two", [5, 6], 2, "length"),
        ([5, 6], 2, ("two",), "helloone ", [5, 6], 2, "stop"),
        ([5], 0, (), "hello", [], 0, "length"),
    ],
)
def test_result_accounts_for_sampled_tokens(
    tokenizer: Tokenizer,
    tokens: list[int],
    limit: int,
    stops: tuple[str, ...],
    text: str,
    ids: list[int],
    count: int,
    reason: str,
) -> None:
    model = ScriptedModel(tokens)
    result = generate_result(
        model, tokenizer, "hello", 16, limit, torch.device("cpu"), stop_sequences=stops
    )
    assert result.text == text
    assert result.token_ids == ids
    assert result.prompt_tokens == 1
    assert result.completion_tokens == count
    assert result.finish_reason == reason
    assert model.training
    assert all(cache() is None for cache in model.caches)
    assert generate_with_token_ids(
        ScriptedModel(tokens),
        tokenizer,
        "hello",
        16,
        limit,
        torch.device("cpu"),
        stop_sequences=stops,
    ) == (text, ids)


def test_empty_prompt_counts_bos(tokenizer: Tokenizer) -> None:
    result = generate_result(
        ScriptedModel([2]), tokenizer, "", 16, 1, torch.device("cpu")
    )
    assert result.prompt_tokens == 1
    assert result.completion_tokens == 1
    assert result.text == ""


@pytest.mark.parametrize(
    "entrypoint", [generate, generate_with_token_ids, generate_result]
)
@pytest.mark.parametrize("callable_signal", [False, True])
def test_pre_cancelled_request_does_not_execute(
    tokenizer: Tokenizer,
    entrypoint: object,
    callable_signal: bool,
) -> None:
    event = Event()
    event.set()
    model = ScriptedModel([5]).eval()
    with pytest.raises(GenerationCancelled, match="cancelled"):
        entrypoint(
            model,
            tokenizer,
            "hello",
            16,
            4,
            torch.device("cpu"),
            cancellation=event.is_set if callable_signal else event,
        )
    assert model.calls == 0
    assert not model.training


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("cancel_on_call", [1, 2])
def test_cancelled_forward_restores_mode_and_discards_cache(
    tokenizer: Tokenizer,
    use_cache: bool,
    cancel_on_call: int,
) -> None:
    event = Event()
    model = ScriptedModel([5], event)
    model.cancel_on_call = cancel_on_call
    with pytest.raises(GenerationCancelled) as error:
        generate_result(
            model,
            tokenizer,
            "hello",
            16,
            4,
            torch.device("cpu"),
            cancellation=event,
            use_cache=use_cache,
        )
    assert (
        error.value.__traceback__ is not None
    )  # Keep the traceback alive during checks.
    assert model.calls == cancel_on_call
    assert model.training
    assert all(cache() is None for cache in model.caches)
    model.cancellation = None
    event.clear()
    result = generate_result(
        model,
        tokenizer,
        "hello",
        16,
        2,
        torch.device("cpu"),
        cancellation=event,
        use_cache=use_cache,
    )
    assert result.token_ids == [5, 5]
    assert result.finish_reason == "length"
    assert all(cache() is None for cache in model.caches)
