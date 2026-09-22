from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparselab.data.conversations import iter_conversations


def _write_jsonl(path: Path, records: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_iter_conversations_streams_chat_format_with_system_and_multiple_turns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "licensed.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Second question"},
                    {"role": "assistant", "content": "Second answer"},
                ]
            }
        ],
    )

    assert list(iter_conversations(path)) == [
        (
            "System: Be concise.\n\n"
            "User: First question\n\nAssistant: First answer\n\n"
            "User: Second question\n\nAssistant: Second answer"
        )
    ]


@pytest.mark.parametrize(
    "record",
    [
        {"messages": []},
        {"messages": [{"role": "user", "content": "orphan"}]},
        {
            "messages": [
                {"role": "assistant", "content": "wrong"},
                {"role": "user", "content": "wrong"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "q", "extra": 1},
                {"role": "assistant", "content": "a"},
            ]
        },
        {"unexpected": []},
        {
            "messages": [
                {"role": [], "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        },
    ],
)
def test_iter_conversations_rejects_invalid_schema_and_turns(
    tmp_path: Path, record: object
) -> None:
    path = tmp_path / "invalid.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises(ValueError, match=r"invalid\.jsonl:1"):
        list(iter_conversations(path))


def test_iter_conversations_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    path.write_text('{"messages": [], "messages": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        list(iter_conversations(path))
