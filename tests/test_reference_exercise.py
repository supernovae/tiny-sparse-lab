from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparselab.cli.main import build_parser
from sparselab.evaluation import reference_exercise as exercise


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        del add_special_tokens
        return SimpleNamespace(ids=list(text.encode("utf-8")))

    def token_to_id(self, token: str) -> int | None:
        return {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}.get(token)


class _Model:
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": torch.ones(1)}


def _run() -> SimpleNamespace:
    config = SimpleNamespace(
        model=SimpleNamespace(
            max_seq_len=256,
            ffn="dense",
            tie_embeddings=True,
            memory="none",
            model_dump=lambda **_: {"ffn": "dense", "tie_embeddings": True},
        ),
        attention=SimpleNamespace(kind="dense"),
        training=SimpleNamespace(
            micro_batch_size=8, gradient_accumulation=1, seq_len=128
        ),
    )
    return SimpleNamespace(
        run=Path("/tmp/fake-run"),
        config=config,
        model=_Model(),
        tokenizer=_Tokenizer(),
        device=torch.device("cpu"),
        identity={
            "run_id": "seed42",
            "checkpoint_sha256": "a" * 64,
            "checkpoint_relative_path": "checkpoints/generation-1",
            "source_identity_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "data_sha256": {"train": "d" * 64, "validation": "e" * 64},
            "step": 1,
            "tokens_seen": 1024,
            "config": {},
        },
        engine=None,
        evaluate=lambda: {"loss": 4.0, "perplexity": 54.6, "valid_targets": 128},
    )


def test_exercise_requires_explicit_generation_not_pointer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit immutable"):
        exercise.exercise_checkpoint(
            "seed42", tmp_path / "latest.json", runs_dir=tmp_path
        )


def test_cli_requires_checkpoint() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["model", "exercise", "seed42", "--output", "observation.json"]
        )


def test_cli_writes_observation_from_explicit_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _run()
    monkeypatch.setattr(exercise, "load_run", lambda *args, **kwargs: run)
    monkeypatch.setattr(
        exercise,
        "generate",
        lambda _model, _tokenizer, prompt, *_args, **_kwargs: prompt + " done",
    )
    monkeypatch.setattr(
        exercise,
        "evaluate_capability",
        lambda card, *_args, **_kwargs: {"card": card.name, "valid": False},
    )
    checkpoint = tmp_path / "run" / "checkpoints" / "generation-1"
    monkeypatch.setattr(run, "run", checkpoint.parent.parent)
    output = tmp_path / "observation.json"
    args = build_parser().parse_args(
        [
            "model",
            "exercise",
            "seed42",
            "--checkpoint",
            str(checkpoint),
            "--runs-dir",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )
    args.handler(args)
    assert output.is_file()
    assert json.loads(output.read_text())["checkpoint"]["checkpoint_sha256"] == "a" * 64
    assert str(output) in capsys.readouterr().out


def test_diagnostics_report_completion_repetition_and_special_tokens() -> None:
    diagnostics = exercise._diagnostics(_run(), "ignored prompt", "abcabcabc")
    assert diagnostics["repeated_trigram_excess"] == 4
    assert diagnostics["contains_eos"] is False
    assert diagnostics["contains_bos"] is False
    assert diagnostics["token_count"] == len("abcabcabc")


def test_observation_binds_checkpoint_and_is_content_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run()
    monkeypatch.setattr(exercise, "load_run", lambda *args, **kwargs: run)
    monkeypatch.setattr(
        exercise,
        "generate",
        lambda _model, _tokenizer, prompt, *_args, **_kwargs: prompt + " done",
    )
    monkeypatch.setattr(
        exercise,
        "evaluate_capability",
        lambda card, *_args, **_kwargs: {"card": card.name, "valid": False},
    )
    checkpoint = tmp_path / "run" / "checkpoints" / "generation-1"
    monkeypatch.setattr(run, "run", checkpoint.parent.parent)

    first = exercise.exercise_checkpoint("seed42", checkpoint, runs_dir=tmp_path)
    second = exercise.exercise_checkpoint("seed42", checkpoint, runs_dir=tmp_path)
    assert first["identity"] == second["identity"]
    assert first["checkpoint"]["checkpoint_sha256"] == "a" * 64
    assert all(
        group["status"] in {"PASS", "NOT_APPLICABLE", "UNAVAILABLE"}
        for group in first["groups"].values()
    )
    assert first["groups"]["capability_cards"]["status"] == "NOT_APPLICABLE"
    assert all(
        item["applicability"] == "OUT_OF_DOMAIN"
        for item in first["groups"]["capability_cards"]["details"]["outcomes"]
    )
    assert first["promotion_eligible"] is True
    incomplete = dict(first["groups"])
    incomplete.pop("held_out_lm")
    assert exercise.promotion_gate(incomplete) is False
    assert exercise.promotion_gate(first["groups"]) is True
    path = exercise.write_observation(first, tmp_path / "observations")
    assert json.loads(path.read_text()) == first


def test_generation_errors_are_not_recorded_as_empty_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run()
    monkeypatch.setattr(exercise, "load_run", lambda *args, **kwargs: run)
    monkeypatch.setattr(
        exercise,
        "generate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )
    monkeypatch.setattr(
        exercise,
        "evaluate_capability",
        lambda card, *_args, **_kwargs: {"card": card.name, "valid": False},
    )
    checkpoint = tmp_path / "run" / "checkpoints" / "generation-1"
    monkeypatch.setattr(run, "run", checkpoint.parent.parent)
    observation = exercise.exercise_checkpoint("seed42", checkpoint, runs_dir=tmp_path)
    generation = observation["groups"]["fixed_prompt_panel"]
    assert generation["status"] == "FAIL"
    assert generation["details"]["cases"][0]["output"] is None
    assert "decode failed" in generation["details"]["cases"][0]["error"]


def test_missing_telemetry_is_unavailable_not_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run()
    monkeypatch.setattr(exercise, "load_run", lambda *args, **kwargs: run)
    monkeypatch.setattr(
        exercise,
        "generate",
        lambda _model, _tokenizer, prompt, *_args, **_kwargs: prompt + " done",
    )
    monkeypatch.setattr(
        exercise,
        "evaluate_capability",
        lambda card, *_args, **_kwargs: {"card": card.name, "valid": False},
    )
    checkpoint = tmp_path / "run" / "checkpoints" / "generation-1"
    monkeypatch.setattr(run, "run", checkpoint.parent.parent)
    observation = exercise.exercise_checkpoint("seed42", checkpoint, runs_dir=tmp_path)
    resource = observation["groups"]["resource_capture"]
    assert resource["status"] == "UNAVAILABLE"
    assert resource["details"]["optimizer_bytes"] is None
