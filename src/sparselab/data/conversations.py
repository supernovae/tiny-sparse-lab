"""Strict streaming reader for licensed local chat corpora."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sparselab.evaluation.chat import ChatMessage, format_chat_prompt


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _error(path: Path, line_number: int, detail: str) -> ValueError:
    return ValueError(f"{path}:{line_number}: {detail}")


def _conversation_document(record: object, path: Path, line_number: int) -> str:
    if not isinstance(record, dict) or set(record) != {"messages"}:
        raise _error(path, line_number, "record must contain only messages")
    messages = record["messages"]
    if not isinstance(messages, list):
        raise _error(path, line_number, "messages must be an array")

    parsed: list[ChatMessage] = []
    system: str | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise _error(
                path, line_number, f"message {index} must contain only role and content"
            )
        role, content = message["role"], message["content"]
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise _error(path, line_number, f"message {index} has unsupported role")
        if not isinstance(content, str) or not content.strip():
            raise _error(
                path, line_number, f"message {index} content must not be blank"
            )
        if role == "system":
            if index != 0:
                raise _error(path, line_number, "system message is allowed only first")
            system = content
        else:
            parsed.append(ChatMessage(role, content))

    if len(parsed) < 2 or len(parsed) % 2:
        raise _error(
            path, line_number, "conversation requires complete user/assistant pairs"
        )
    for index in range(0, len(parsed), 2):
        if parsed[index].role != "user" or parsed[index + 1].role != "assistant":
            raise _error(
                path, line_number, "messages must alternate user then assistant"
            )

    history: list[ChatMessage] = []
    document = ""
    for index in range(0, len(parsed), 2):
        user, assistant = parsed[index : index + 2]
        prompt = format_chat_prompt(history, user.content, system=system)
        if document and not prompt.startswith(document):
            raise AssertionError("chat transcript rendering is not append-only")
        document = f"{prompt} {assistant.content.strip()}"
        history.extend((user, assistant))
    return document


def iter_conversations(path: Path) -> Iterator[str]:
    """Yield validated JSONL conversations rendered in the chat training format.

    Each yielded document ends with its final assistant answer; packers append the
    document EOS token.  The file is consumed line-by-line rather than materialized.
    """
    with path.open("r", encoding="utf-8", newline="") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line, object_pairs_hook=_object_without_duplicates)
            except (json.JSONDecodeError, ValueError) as error:
                raise _error(path, line_number, f"invalid JSON: {error}") from error
            yield _conversation_document(record, path, line_number)
