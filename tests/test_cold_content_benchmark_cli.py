"""Argument routing only: no hashing, fixture creation or host observation."""

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1] / "benchmarks/cold_content_verification.py"
    )
    spec = importlib.util.spec_from_file_location("cold_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("missing", ["--output", "--fixture-root", "--prepared-root"])
def test_benchmark_requires_explicit_input_and_storage(tmp_path, monkeypatch, missing):
    module = _module()
    args = {
        "--output": tmp_path / "result.json",
        "--fixture-root": tmp_path / "fixture",
        "--prepared-root": tmp_path / "prepared",
    }
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            *[
                part
                for key, path in args.items()
                if key != missing
                for part in (key, str(path))
            ],
        ],
    )

    def forbidden(*args):
        pytest.fail("missing inputs must refuse before benchmark execution")

    monkeypatch.setattr(module, "benchmark", forbidden)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


def test_benchmark_routes_only_explicit_paths(tmp_path, monkeypatch):
    module = _module()
    output, fixture, prepared = (
        tmp_path / name for name in ("result.json", "fixture", "prepared")
    )
    seen = []

    def capture(*args):
        seen.append(args)
        return {"samples": []}

    monkeypatch.setattr(module, "benchmark", capture)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "--output",
            str(output),
            "--fixture-root",
            str(fixture),
            "--prepared-root",
            str(prepared),
        ],
    )
    module.main()
    assert seen == [(output, fixture, 3, 180, prepared)]
    assert json.loads(output.read_text()) == {"samples": []}
    assert not fixture.exists()
    assert not prepared.exists()
