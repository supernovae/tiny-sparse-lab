from __future__ import annotations

import json
import shutil
import sys

import pytest
import torch
from test_training import config

from sparselab.cli.main import main
from sparselab.evaluation.inference import load_run
from sparselab.runtime import validate_runtime
from sparselab.training.trainer import train


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("inference")
    original = config(root)
    small = original.model_copy(
        update={
            "training": original.training.model_copy(update={"max_steps": 2}),
            "optimizer": original.optimizer.model_copy(update={"warmup_steps": 0}),
        }
    )
    train(small, run_id="original")
    return small


def test_relocated_run_does_not_need_external_tokenizer(trained_run, tmp_path):
    original = load_run("original", trained_run.logging.root_dir)
    shutil.copytree(original.run, tmp_path / "moved")
    external = trained_run.tokenizer.path
    original_bytes = external.read_bytes()
    external.write_text("not a tokenizer")
    try:
        moved = load_run("moved", tmp_path)
        tokens = torch.tensor([moved.tokenizer.encode("hello").ids])
        with torch.no_grad():
            assert torch.equal(original.model(tokens), moved.model(tokens))
        assert (
            original.identity["checkpoint_sha256"]
            == moved.identity["checkpoint_sha256"]
        )
    finally:
        external.write_bytes(original_bytes)


def test_inference_rejects_tampered_tokenizer(trained_run, tmp_path):
    shutil.copytree(trained_run.logging.root_dir / "original", tmp_path / "corrupt")
    (tmp_path / "corrupt/tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact integrity"):
        load_run("corrupt", tmp_path)


def test_selected_checkpoint_identity_is_not_latest(trained_run, tmp_path):
    shutil.copytree(trained_run.logging.root_dir / "original", tmp_path / "history")
    generations = sorted((tmp_path / "history/checkpoints").glob("step_*"))
    initial = load_run("history", tmp_path, str(generations[0]))
    final = load_run("history", tmp_path)
    assert initial.identity["step"] == 0
    assert final.identity["step"] == 2
    assert initial.identity["checkpoint_sha256"] != final.identity["checkpoint_sha256"]
    pointer = tmp_path / "history/checkpoints/latest.json"
    stale = json.loads(pointer.read_text())
    stale["relative_path"] = generations[0].name
    pointer.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="pointer digest"):
        load_run("history", tmp_path)
    # A frozen generation selection remains independently usable.
    again = load_run("history", tmp_path, str(generations[0]))
    assert again.identity == initial.identity


def test_unimplemented_precision_is_not_silently_fp32(trained_run):
    unsupported = trained_run.model_copy(
        update={"runtime": trained_run.runtime.model_copy(update={"precision": "bf16"})}
    )
    with pytest.raises(ValueError, match="FP32 only"):
        validate_runtime(unsupported)


def _chat_cli(monkeypatch, capsys, cwd, *options):
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "chat",
            "original",
            "--message",
            "hello",
            "--max-new-tokens",
            "1",
            "--json",
            *options,
        ],
    )
    main()
    return json.loads(capsys.readouterr().out)


def test_chat_finds_project_run_from_source_subdirectory(
    trained_run, tmp_path, monkeypatch, capsys
):
    (tmp_path / "pyproject.toml").write_text("")
    source = tmp_path / "src"
    source.mkdir()
    shutil.copytree(
        trained_run.logging.root_dir / "original", tmp_path / "runs/original"
    )

    from_root = _chat_cli(monkeypatch, capsys, tmp_path)
    from_source = _chat_cli(monkeypatch, capsys, source)

    assert from_source == from_root


def test_chat_explicit_run_directory_remains_cwd_relative(
    trained_run, tmp_path, monkeypatch, capsys
):
    (tmp_path / "pyproject.toml").write_text("")
    source = tmp_path / "src"
    source.mkdir()
    shutil.copytree(
        trained_run.logging.root_dir / "original", tmp_path / "runs/original"
    )

    # An explicitly missing directory must not silently select the project run.
    with pytest.raises(FileNotFoundError):
        _chat_cli(monkeypatch, capsys, source, "--runs-dir", "runs")
    response = _chat_cli(monkeypatch, capsys, source, "--runs-dir", "../runs")
    pointer = json.loads(
        (tmp_path / "runs/original/checkpoints/latest.json").read_text()
    )

    assert response["identity"]["checkpoint_sha256"] == pointer["manifest_sha256"]


def test_chat_outside_a_project_uses_local_runs(
    trained_run, tmp_path, monkeypatch, capsys
):
    shutil.copytree(
        trained_run.logging.root_dir / "original", tmp_path / "runs/original"
    )

    response = _chat_cli(monkeypatch, capsys, tmp_path)
    pointer = json.loads(
        (tmp_path / "runs/original/checkpoints/latest.json").read_text()
    )

    assert response["identity"]["checkpoint_sha256"] == pointer["manifest_sha256"]
