"""Research surface and generated instructions, validated without execution."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import pytest

from sparselab.cli import entry
from sparselab.cli import main as application
from sparselab.research.catalog import load_research
from sparselab.research.scaffold import _study_readme


def _commands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    return next(
        action.choices
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )


def test_portability_surface_removes_byte_wrapper_without_alias(tmp_path: Path) -> None:
    parser = application.build_parser(tmp_path / "work")
    portability = _commands(_commands(_commands(parser)["research"])["portability"])
    assert set(portability) == {"build", "plan", "run", "continue", "report"}
    assert not hasattr(application, "_research_portability_probe_byte")
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("route", [entry.main, application.main])
def test_removed_byte_route_fails_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route
) -> None:
    def refuse_execution(*args, **kwargs):
        raise AssertionError("removed command reached execution")

    monkeypatch.setattr(application, "_prepare_runtime_command", refuse_execution)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "--work-dir",
            str(tmp_path / "work"),
            "research",
            "portability",
            "probe-byte",
            "é",
        ],
    )
    with pytest.raises(SystemExit) as error:
        route()
    assert error.value.code == 2
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        "research portability build --output absent",
        "research portability plan --campaign-root absent",
        (
            "research portability run --campaign-root absent --recipient width32 "
            "--representation byte --condition adapter-tuned --seed 17"
        ),
        "research portability continue --campaign-root absent --max-arms 1",
        "research portability report --campaign-root absent",
        "reference pythia trajectory absent.yaml --output absent.json",
        "research tasks build math-identities --output absent",
        (
            "research corpus mine --train-jsonl absent.jsonl --tokenizer absent.json "
            "--table-size 263 --memory-dim 8 --ngram-orders 3 --output absent.json"
        ),
        "learn list",
        "learn describe byte-engram",
        "learn scaffold byte-engram --output absent",
        'learn probe absent.yaml --prompt "é"',
        "research scaffold engram-ffn-substitution-v1 --output absent",
        "study plan absent.yaml",
        "study submit absent.yaml",
        "study collect absent.yaml absent.json",
        "study report absent.yaml absent.json --evidence absent.json --output absent",
    ],
)
def test_retained_research_contracts_parse_without_inputs_or_dispatch(
    tmp_path: Path, arguments: str
) -> None:
    args = application.build_parser(tmp_path / "work").parse_args(
        shlex.split(arguments)
    )
    assert callable(args.handler)
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("data,backend", [("offline", "cpu"), ("tinystories", "cuda")])
def test_study_instructions_have_one_preparation_and_native_dispatch(
    tmp_path: Path, data: str, backend: str
) -> None:
    readme = _study_readme(
        load_research("engram-ffn-substitution-v1"),
        "smoke",
        data,
        backend,
        "default",
        [{"coordinate": {"seed": 17}, "config_sha256": "a" * 64}],
        1,
        0,
        0,
        str(tmp_path / "work"),
    )
    assert readme.count("sparselab tokenizer train") == 1
    assert readme.count("sparselab data prepare") == 1
    assert "sparselab learn probe" not in readme
    assert "lesson-paths.md" in readme
    assert "iteration.md#check-the-actual-proposed-configuration" in readme
    assert (
        readme.index("`inspect`")
        < readme.index("`workspace preflight`")
        < readme.index("`stage` it with `--through smoke`")
        < readme.index("sparselab train")
    )
    assert "a fresh output under\n`$WORK/staging/`" in readme
    assert "a disposable pilot on the configured runtime" in readme
    assert readme.index("sparselab data prepare") < readme.index("sparselab train")
    if data == "tinystories":
        assert (
            readme.index("sparselab data lock")
            < readme.index("sparselab data snapshot")
            < readme.index("sparselab tokenizer train")
        )
    parser = application.build_parser(tmp_path / "work")
    # Parse every generated native example; no command handler is called.
    for line in readme.splitlines():
        if " sparselab " in line:
            command = line.split(" sparselab ", 1)[1]
            assert callable(parser.parse_args(shlex.split(command)).handler)
    assert not (tmp_path / "work").exists()
