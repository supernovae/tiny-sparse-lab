"""A lightweight conversational wrapper around local greedy generation."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer

from sparselab.evaluation.generation import generate


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError(f"unsupported chat role: {self.role}")
        if self.role == "user" and not self.content.strip():
            raise ValueError("user chat messages must not be blank")


def format_chat_prompt(
    history: Sequence[ChatMessage], message: str, *, system: str | None = None
) -> str:
    """Render an explicit plain-text transcript without tokenizer-specific tokens."""
    if not message.strip():
        raise ValueError("chat message must not be blank")
    lines: list[str] = []
    if system:
        lines.extend((f"System: {system.strip()}", ""))
    for item in history:
        lines.extend((f"{item.role.title()}: {item.content.strip()}", ""))
    lines.extend((f"User: {message.strip()}", "", "Assistant:"))
    return "\n".join(lines)


def assistant_reply(completion: str, prompt: str) -> str:
    """Extract one assistant turn and stop if the model starts another role."""
    if not completion.startswith(prompt):
        raise ValueError("completion does not begin with its chat prompt")
    reply = completion.removeprefix(prompt)
    for marker in ("\nUser:", "\nSystem:", "\nAssistant:"):
        reply = reply.split(marker, 1)[0]
    return reply.strip()


def chat_turn(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    history: Sequence[ChatMessage],
    message: str,
    max_seq_len: int,
    max_new_tokens: int,
    device: torch.device,
    *,
    system: str | None = None,
) -> tuple[str, str]:
    """Generate one local greedy assistant turn and return its prompt and reply."""
    prompt = format_chat_prompt(history, message, system=system)
    completion = generate(
        model, tokenizer, prompt, max_seq_len, max_new_tokens, device
    )
    return prompt, assistant_reply(completion, prompt)
