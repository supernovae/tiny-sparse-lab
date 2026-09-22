from __future__ import annotations

import torch
from tokenizers import Tokenizer


def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids or [
        tokenizer.token_to_id("<bos>")
    ]
    eos = tokenizer.token_to_id("<eos>")
    generated = []
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-max_seq_len:]], device=device)
            token = int(model(x)[0, -1].argmax())
            if token == eos:
                break
            generated.append(token)
            ids.append(token)
    return prompt + tokenizer.decode(generated, skip_special_tokens=True)
