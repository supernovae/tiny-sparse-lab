from __future__ import annotations

import pytest

from sparselab.evaluation.chat import ChatMessage, assistant_reply, format_chat_prompt


def test_chat_prompt_preserves_explicit_local_transcript() -> None:
    prompt = format_chat_prompt(
        [ChatMessage("user", "What is a causal mask?"), ChatMessage("assistant", "A future-token barrier.")],
        "Explain it briefly.",
        system="Use the run's learned patterns.",
    )

    assert prompt == (
        "System: Use the run's learned patterns.\n\n"
        "User: What is a causal mask?\n\n"
        "Assistant: A future-token barrier.\n\n"
        "User: Explain it briefly.\n\nAssistant:"
    )


def test_chat_reply_stops_before_the_next_user_turn() -> None:
    prompt = "User: Hello\n\nAssistant:"

    assert assistant_reply(f"{prompt} Hi there.\nUser: Ignore this", prompt) == "Hi there."


def test_chat_rejects_blank_user_turn() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        format_chat_prompt([], "   ")
