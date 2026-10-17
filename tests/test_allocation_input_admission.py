"""Explicit external task-input routing; the scientific builder is never run."""

from pathlib import Path

import pytest

from sparselab.cli import main as cli
from sparselab.data import allocation_tasks

FLAGS = (
    "tokenizer",
    "train-jsonl",
    "validation-jsonl",
    "oracle-audit",
    "provenance",
    "cards-dir",
)


@pytest.mark.parametrize("missing", FLAGS)
def test_allocation_missing_input_refuses_before_builder(
    tmp_path, monkeypatch, missing
):
    def forbidden(*args, **kwargs):
        pytest.fail("missing inputs must refuse before domain work")

    monkeypatch.setattr(allocation_tasks, "build_memory_allocation", forbidden)
    output = tmp_path / "not-created"
    command = [
        "research",
        "tasks",
        "build",
        "memory-allocation",
        "--output",
        str(output),
    ]
    for flag in FLAGS:
        if flag != missing:
            command.extend(["--" + flag, str(tmp_path / flag)])
    args = cli.build_parser().parse_args(command)
    with pytest.raises(ValueError, match="requires explicit inputs: --" + missing):
        args.handler(args)
    assert not output.exists()


def test_allocation_forwards_explicit_paths_unchanged(tmp_path, monkeypatch):
    observed = []

    def capture(output, **inputs):
        observed.append((output, inputs))
        return output / "manifest.json"

    monkeypatch.setattr(allocation_tasks, "build_memory_allocation", capture)
    output = tmp_path / "not-created"
    command = [
        "research",
        "tasks",
        "build",
        "memory-allocation",
        "--output",
        str(output),
    ]
    for flag in FLAGS:
        command.extend(["--" + flag, str(tmp_path / flag)])
    args = cli.build_parser().parse_args(command)
    args.handler(args)
    assert observed == [
        (
            output,
            dict(
                zip(
                    (
                        "tokenizer_path",
                        "train_path",
                        "validation_path",
                        "audit_path",
                        "provenance_path",
                        "cards_dir",
                    ),
                    (Path(tmp_path / flag) for flag in FLAGS),
                    strict=True,
                )
            ),
        )
    ]
    assert not output.exists()
