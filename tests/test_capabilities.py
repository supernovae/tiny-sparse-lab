from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from sparselab.data.chat_recall import cases
from sparselab.evaluation.capabilities import (
    capability_card,
    compare_results,
    describe_capability_card,
    evaluate_capability,
    write_capability_result,
)


def _tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def _result(
    *,
    score: float,
    memory: str = "none",
    tokens: int = 768,
    seed: int = 42,
    tokenizer_sha: str = "tok",
    path: str = "/a",
) -> dict[str, object]:
    config = {
        "name": "either-name",
        "seed": seed,
        "model": {
            "memory": memory,
            "memory_table_size": 0 if memory == "none" else 31,
            "hidden_dim": 64,
        },
        "dataset": {"cache_dir": path, "source": "chat_recall"},
        "tokenizer": {"path": f"{path}/tokenizer.json"},
        "training": {"max_steps": 12},
        "logging": {"root_dir": path},
        "checkpoint": {"every_steps": 3},
    }
    passed = score == 1.0
    return {
        "card": "chat-alias-recall-v1",
        "card_digest": "card",
        "valid": True,
        "score": score,
        "results": [{"id": "one", "passed": passed}, {"id": "two", "passed": False}],
        "identity": {
            "checkpoint_sha256": f"checkpoint-{memory}",
            "step": 12,
            "tokens_seen": tokens,
            "config": config,
            "tokenizer_sha256": tokenizer_sha,
            "data_sha256": {"train": "train", "validation": "validation"},
            "source_identity_sha256": "source",
            "runtime": {
                "engine": "pytorch",
                "backend": "cpu",
                "device_name": "CPU",
                "framework_version": "x",
                "os": "Darwin",
                "precision": "fp32",
            },
            "parameter_inventory": {
                "total": 10,
                "trainable": 10,
                "active_per_token": 10,
            },
        },
    }


def test_chat_alias_card_uses_held_out_chat_wording() -> None:
    train = cases("train")
    card = capability_card("chat-alias-recall-v1")

    assert card.version == 1
    assert {item.prompt for item in train}.isdisjoint(
        case.prompt for case in card.cases
    )
    assert {item.expected for item in train} >= {case.expected for case in card.cases}
    assert all("Memory record" not in case.prompt for case in card.cases)
    assert {case.identifier for case in card.cases} == {
        case.identifier for case in cases("test")
    }
    assert all(case.prompt.endswith("Assistant:") for case in card.cases)


def test_described_card_round_trips_and_user_schema_is_strict(tmp_path: Path) -> None:
    payload = describe_capability_card("chat-alias-recall-v1")
    path = tmp_path / "card.json"
    path.write_text(__import__("json").dumps(payload))

    assert capability_card(str(path)).digest == payload["digest"]
    payload["scorer"] = "python:evil"
    payload.pop("digest")
    path.write_text(__import__("json").dumps(payload))
    with pytest.raises(ValueError, match="only normalized_full_answer"):
        capability_card(str(path))


def test_custom_response_budget_cannot_be_silently_replaced(tmp_path: Path) -> None:
    payload = describe_capability_card("chat-alias-recall-v1")
    payload.pop("digest")
    payload["generation"]["max_new_tokens"] = 1000
    path = tmp_path / "long-response.json"
    path.write_text(__import__("json").dumps(payload))
    result = evaluate_capability(
        capability_card(str(path)),
        object(),
        _tokenizer(),
        256,
        __import__("torch").device("cpu"),
    )
    assert not result["valid"]
    assert result["score"] is None


def test_exact_full_answer_scoring_rejects_first_word_junk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = capability_card("chat-alias-recall-v1")
    monkeypatch.setattr(
        "sparselab.evaluation.capabilities.generate",
        lambda _model, _tokenizer, prompt, *_args, **_kwargs: (
            prompt + " lumen definitely"
        ),
    )
    result = evaluate_capability(
        card, object(), _tokenizer(), 256, __import__("torch").device("cpu")
    )

    assert result["score"] == 0
    assert not any(item["passed"] for item in result["results"])


def test_prompt_overflow_invalidates_evaluation_without_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = capability_card("chat-alias-recall-v1")
    monkeypatch.setattr(
        "sparselab.evaluation.capabilities.generate",
        lambda *_args, **_kwargs: pytest.fail("generation must not run"),
    )

    result = evaluate_capability(
        card, object(), _tokenizer(), 1, __import__("torch").device("cpu")
    )

    assert result["valid"] is False
    assert result["score"] is None
    assert result["failures"][0]["reason"] == "prompt_and_response_exceed_max_context"


def test_legacy_card_retains_its_explicit_first_word_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = capability_card("engram-recall-v1")
    monkeypatch.setattr(
        "sparselab.evaluation.capabilities.generate",
        lambda _model, _tokenizer, prompt, *_args, **_kwargs: (
            prompt + " lumen trailing text"
        ),
    )

    result = evaluate_capability(
        card, object(), _tokenizer(), 256, __import__("torch").device("cpu")
    )

    assert result["scorer"] == "first_normalized_word_exact_v1"
    assert result["passed"] == 1


def test_compare_requires_matched_identity_but_accepts_relocated_paths() -> None:
    base = _result(score=0.0, path="/old")
    variant = _result(score=1.0, memory="ngram", path="/new")

    comparison = compare_results(base, variant)

    assert comparison["score_delta"] == 1.0
    assert comparison["paired_gains"] == 1
    assert comparison["outcome"] == "supported_observation"
    with pytest.raises(ValueError, match="tokens_seen"):
        compare_results(base, _result(score=1.0, memory="ngram", tokens=767))
    with pytest.raises(ValueError):
        compare_results(base, _result(score=1.0, memory="ngram", seed=7))
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        compare_results(base, _result(score=1.0, memory="ngram", tokenizer_sha="other"))


def test_results_are_checkpoint_content_addressed_and_coexist(tmp_path: Path) -> None:
    one = {
        "card": "card",
        "card_digest": "card-digest",
        "checkpoint_sha256": "first",
        "score": 0.0,
    }
    two = {**one, "checkpoint_sha256": "second", "score": 1.0}

    first = write_capability_result(tmp_path, one)
    assert write_capability_result(tmp_path, one) == first
    second = write_capability_result(tmp_path, two)

    assert first != second
    assert first.exists() and second.exists()
