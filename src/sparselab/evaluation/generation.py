from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from tokenizers import Tokenizer

from sparselab.data.byte_hash import table_address, token_bytes


def _addresses_from_ids(
    tokenizer: Tokenizer, ids: Sequence[int], table_size: int, ngram_size: int
) -> list[int]:
    source = bytearray()
    addresses: list[int] = []
    for token_id in ids:
        source.extend(token_bytes(tokenizer, token_id))
        addresses.append(table_address(bytes(source[-ngram_size:]), table_size))
    return addresses


def _prompt_byte_addresses(
    tokenizer: Tokenizer, prompt: str, ids: list[int], table_size: int, ngram_size: int
) -> list[int]:
    """Build causal addresses from raw tokenizer bytes, including split Unicode tokens."""
    if not prompt:
        return [0]
    encoding = tokenizer.encode(prompt, add_special_tokens=False)
    if encoding.ids != ids:
        raise ValueError("prompt IDs must come from the supplied tokenizer")
    addresses = _addresses_from_ids(tokenizer, ids, table_size, ngram_size)
    if b"".join(token_bytes(tokenizer, token_id) for token_id in ids) != prompt.encode(
        "utf-8"
    ):
        raise ValueError("tokenizer byte encoding does not reproduce the prompt")
    return addresses


def _validate_generation_options(
    max_seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    stop_sequences: Sequence[str],
) -> None:
    if (
        isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, int)
        or max_seq_len <= 0
    ):
        raise ValueError("max_seq_len must be a positive integer")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 0
    ):
        raise ValueError("max_new_tokens must be a non-negative integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be finite and non-negative")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if isinstance(stop_sequences, str) or any(
        not isinstance(stop, str) or not stop for stop in stop_sequences
    ):
        raise ValueError("stop_sequences must contain only non-empty strings")


def _next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    blocked_ids: set[int],
    generator: torch.Generator,
) -> int:
    if not torch.isfinite(logits).all():
        raise ValueError("generation logits must be finite")
    scores = logits.float().clone()
    for token_id in blocked_ids:
        if 0 <= token_id < scores.numel():
            scores[token_id] = -torch.inf
    if not torch.isfinite(scores).any():
        raise ValueError("generation has no permitted output tokens")
    if temperature == 0:
        return int(scores.argmax())
    scores.div_(temperature)
    if top_k:
        limit = min(top_k, scores.numel())
        cutoff = torch.topk(scores, limit).values[-1]
        scores[scores < cutoff] = -torch.inf
    probabilities = torch.softmax(scores, dim=0).cpu()
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
    device: torch.device,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    seed: int = 0,
    stop_sequences: Sequence[str] = (),
    strict_context: bool = False,
) -> str:
    """Continue ``prompt`` using greedy or locally seeded sampled decoding."""
    _validate_generation_options(
        max_seq_len, max_new_tokens, temperature, top_k, seed, stop_sequences
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    byte_memory = model.config.memory in {"byte", "portable"}
    if not ids:
        bos = tokenizer.token_to_id("<bos>")
        if bos is None:
            raise ValueError("tokenizer has no <bos> token")
        ids = [bos]
    if strict_context and len(ids) + max_new_tokens > max_seq_len:
        raise ValueError("prompt and requested completion exceed max_seq_len")

    source_bytes = bytearray(prompt.encode("utf-8")) if byte_memory else None
    addresses = (
        _prompt_byte_addresses(
            tokenizer,
            prompt,
            ids,
            model.config.memory_table_size,
            model.config.memory_ngram_size,
        )
        if byte_memory and prompt
        else [0]
        if byte_memory
        else None
    )
    blocked_ids = {
        token_id
        for name in ("<pad>", "<bos>", "<unk>")
        if (token_id := tokenizer.token_to_id(name)) is not None
    }
    eos = tokenizer.token_to_id("<eos>")
    generated: list[int] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                x = torch.tensor([ids[-max_seq_len:]], device=device)
                byte_addresses = (
                    torch.tensor([addresses[-max_seq_len:]], device=device)
                    if addresses is not None
                    else None
                )
                token = _next_token(
                    model(x, byte_addresses=byte_addresses)[0, -1],
                    temperature=temperature,
                    top_k=top_k,
                    blocked_ids=blocked_ids,
                    generator=generator,
                )
                if token == eos:
                    break
                generated.append(token)
                ids.append(token)
                if addresses is not None:
                    assert source_bytes is not None
                    source_bytes.extend(token_bytes(tokenizer, token))
                    addresses.append(
                        table_address(
                            bytes(source_bytes[-model.config.memory_ngram_size :]),
                            model.config.memory_table_size,
                        )
                    )
                if stop_sequences:
                    decoded = tokenizer.decode(generated, skip_special_tokens=True)
                    if any(stop in decoded for stop in stop_sequences):
                        break
    finally:
        model.train(was_training)

    completion = tokenizer.decode(generated, skip_special_tokens=True)
    for stop in stop_sequences:
        completion = completion.split(stop, 1)[0]
    return prompt + completion
