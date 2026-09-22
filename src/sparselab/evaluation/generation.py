from __future__ import annotations

import torch
from tokenizers import Tokenizer

from sparselab.data.byte_hash import table_address


def _prompt_byte_addresses(
    tokenizer: Tokenizer, prompt: str, ids: list[int], table_size: int, ngram_size: int
) -> list[int]:
    if not prompt:
        return [0]
    encoding = tokenizer.encode(prompt, add_special_tokens=False)
    if encoding.ids != ids:
        raise ValueError("prompt IDs must come from the supplied tokenizer")
    return [
        table_address(prompt[:end].encode("utf-8")[-ngram_size:], table_size)
        for _, end in encoding.offsets
    ]


def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    byte_memory = model.config.memory in {"byte", "portable"}
    if not ids:
        bos = tokenizer.token_to_id("<bos>")
        if bos is None:
            raise ValueError("tokenizer has no <bos> token")
        ids = [bos]
    source_bytes = bytearray(prompt.encode("utf-8"))
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
    eos = tokenizer.token_to_id("<eos>")
    generated = []
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-max_seq_len:]], device=device)
            byte_addresses = (
                torch.tensor([addresses[-max_seq_len:]], device=device)
                if addresses is not None
                else None
            )
            token = int(model(x, byte_addresses=byte_addresses)[0, -1].argmax())
            if token == eos:
                break
            generated.append(token)
            ids.append(token)
            if addresses is not None:
                piece = tokenizer.decode([token], skip_special_tokens=True).encode(
                    "utf-8"
                )
                source_bytes.extend(piece)
                addresses.append(
                    table_address(
                        bytes(source_bytes[-model.config.memory_ngram_size :]),
                        model.config.memory_table_size,
                    )
                )
    return prompt + tokenizer.decode(generated, skip_special_tokens=True)
