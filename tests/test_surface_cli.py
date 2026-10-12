"""Model-free work-root selection and subprocess handoff regressions."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab.cli import main as cli
from sparselab.cli import surface


@pytest.fixture
def work_roots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("SPARSELAB_WORK_DIR", raising=False)
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", None)
    return tmp_path


@pytest.mark.parametrize("spelling", ["split", "equals"])
@pytest.mark.parametrize(
    "selection", ["prefix", "local", "environment", "default", "both"]
)
def test_surface_chat_work_root_through_parsers_and_launch(
    work_roots, monkeypatch, selection, spelling
):
    root = work_roots
    prefix = Path("prefix") if selection in {"prefix", "both"} else None
    local = Path("local") if selection in {"local", "both"} else None
    if selection != "default":
        monkeypatch.setenv("SPARSELAB_WORK_DIR", str(root / "environment"))
    selected = local or prefix
    fallback = "data/sparselab" if selection == "default" else "environment"
    expected = root / (str(selected) if selected else fallback)

    def flag(value):
        if value is None:
            return []
        return (
            ["--work-dir", str(value)]
            if spelling == "split"
            else [f"--work-dir={value}"]
        )

    argv = [
        *flag(prefix),
        "surface",
        "chat",
        *flag(local),
        "--cell",
        "first=missing/checkpoint-a",
        "--cell",
        "second=missing/checkpoint-b",
        "--backend",
        "cpu",
        "--seed",
        "19",
        "--port",
        "8509",
    ]
    built = []
    original_build = cli.build_parser

    def build(work_dir):
        built.append(work_dir)
        parser = original_build(work_dir)
        # The final parser preserves the prefix unless the local flag is explicit.
        assert parser.parse_args(argv).work_dir == (str(local) if local else prefix)
        assert not expected.exists()
        return parser

    launched = []

    def launch(cmd, *, check):
        assert check is True
        launched.append(cmd)
        assert os.environ["SPARSELAB_WORK_DIR"] == str(expected)
        assert all(
            os.environ[name] == str(expected / "scratch")
            for name in ("TMPDIR", "TEMP", "TMP")
        )
        with tempfile.TemporaryDirectory() as scratch:
            assert Path(scratch).parent == expected / "scratch"

    monkeypatch.setattr(cli, "build_parser", build)
    # Replace only the surface module's subprocess reference; storage checks still
    # use their real read-only Git query. No app or checkpoint is opened.
    monkeypatch.setattr(surface, "subprocess", SimpleNamespace(run=launch))
    monkeypatch.setattr(cli, "set_token_file", lambda path: None)
    monkeypatch.setattr(sys, "argv", ["sparselab", *argv])
    cli.main()

    assert built == [prefix]  # The preliminary parser sees only global options.
    assert launched == [
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(
                Path(surface.__file__).resolve().parents[1] / "dashboard/surface_app.py"
            ),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8509",
            "--server.headless",
            "true",
            "--",
            "--mode",
            "chat",
            "--cell",
            "first=missing/checkpoint-a",
            "--cell",
            "second=missing/checkpoint-b",
            "--backend",
            "cpu",
            "--seed",
            "19",
            "--work-dir",
            str(expected),
        ]
    ]
    assert list(expected.iterdir()) == [expected / "scratch"]
    assert list((expected / "scratch").iterdir()) == []
    for unused in {
        root / "prefix",
        root / "local",
        root / "environment",
        root / "data/sparselab",
    } - {expected}:
        assert not unused.exists()


@pytest.mark.parametrize("prefix", [[], ["--work-dir", "parent"]])
@pytest.mark.parametrize("separator", [[], ["--"]])
@pytest.mark.parametrize(
    "child_flags", [["--work-dir", "child"], ["--work-dir=child"], ["--work-dir"]]
)
def test_preliminary_parser_does_not_consume_remainder(
    work_roots, monkeypatch, prefix, separator, child_flags
):
    argv = [
        *prefix,
        "monitor",
        "--policy",
        "missing.yaml",
        "--workspace",
        "workspace",
        "--log-dir",
        "logs",
        *separator,
        "child-command",
        *child_flags,
    ]
    original_build = cli.build_parser
    built = []

    def build(work_dir):
        built.append(work_dir)
        return original_build(work_dir)

    class Parsed(Exception):
        pass

    def inspect(args):
        assert args.work_dir == (Path("parent") if prefix else None)
        assert args.monitored_command == [*separator, "child-command", *child_flags]
        raise Parsed

    monkeypatch.setattr(cli, "build_parser", build)
    monkeypatch.setattr(cli, "_prepare_runtime_command", inspect)
    monkeypatch.setattr(sys, "argv", ["sparselab", *argv])
    with pytest.raises(Parsed):
        cli.main()
    assert built == [Path("parent") if prefix else None]
    assert list(work_roots.iterdir()) == []


@pytest.mark.parametrize("token_first", [False, True])
def test_preliminary_parser_handles_other_global_option_value(
    work_roots, monkeypatch, token_first
):
    token = ["--hf-token-file", "unused-token-file"]
    work = ["--work-dir", "parent"]
    argv = [*(token + work if token_first else work + token), "eval", "missing-run"]
    original_build = cli.build_parser
    built = []

    def build(work_dir):
        built.append(work_dir)
        parser = original_build(work_dir)
        args = parser.parse_args(argv)
        assert args.work_dir == Path("parent")
        assert args.runs_dir == str(work_roots / "parent/runs")
        assert args.hf_token_file == Path("unused-token-file")
        raise Parsed

    class Parsed(Exception):
        pass

    monkeypatch.setattr(cli, "build_parser", build)
    monkeypatch.setattr(sys, "argv", ["sparselab", *argv])
    with pytest.raises(Parsed):
        cli.main()
    assert built == [Path("parent")]
    assert list(work_roots.iterdir()) == []
