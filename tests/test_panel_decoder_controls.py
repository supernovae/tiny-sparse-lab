"""Mock-only panel controls; no fixture loads or trains a model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab.evaluation import panel


def _declaration(**decoder_fields: object) -> dict[str, object]:
    return {
        "generation_panel_version": 1,
        "id": "controls",
        "role": "descriptive_not_quality_gate",
        "prompts": ["short prompt"],
        "decoder": {
            "temperature": 0,
            "top_k": 0,
            "max_new_tokens": 2,
            "seed": 17,
            **decoder_fields,
        },
        "checkpoint_selection": "pinned synthetic test binding",
    }


def test_absent_controls_preserve_legacy_declaration_and_hash() -> None:
    source = _declaration()
    parsed = panel.GenerationPanel.model_validate(source)
    assert parsed.decoder.use_cache is True
    assert parsed.decoder.strict_context is False
    assert parsed.model_dump(mode="json") == source
    assert panel._digest(parsed.model_dump(mode="json")) == panel._digest(source)
    assert (
        panel.GenerationPanel.model_validate_json(
            parsed.model_dump_json()
        ).model_dump_json()
        == parsed.model_dump_json()
    )


def test_explicit_controls_are_serialized_and_distinguishable() -> None:
    legacy = panel.GenerationPanel.model_validate(_declaration())
    explicit_defaults = panel.GenerationPanel.model_validate(
        _declaration(use_cache=True, strict_context=False)
    )
    explicit_override = panel.GenerationPanel.model_validate(
        _declaration(use_cache=False, strict_context=True)
    )
    assert (
        explicit_defaults.model_dump(mode="json")["decoder"]
        == _declaration(use_cache=True, strict_context=False)["decoder"]
    )
    assert explicit_override.decoder.use_cache is False
    assert explicit_override.decoder.strict_context is True
    assert panel._digest(legacy.model_dump(mode="json")) != panel._digest(
        explicit_defaults.model_dump(mode="json")
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_cache", 1),
        ("use_cache", "false"),
        ("strict_context", 0),
        ("strict_context", "true"),
    ],
)
def test_controls_reject_coercion(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        panel.GenerationPanel.model_validate(_declaration(**{field: value}))


def test_run_panel_forwards_explicit_and_legacy_controls_without_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    index_path = tmp_path / "index.json"
    index_path.write_text("{}")
    runtime = {"engine": "mock", "backend": "cpu"}
    index = {
        "run": str(run),
        "run_id": "mock-run",
        "checkpoint": "checkpoints/selected.json",
        "checkpoint_sha256": "a" * 64,
        "index_sha256": "b" * 64,
        "evaluation_runtime": {"observed": runtime},
    }
    loaded = SimpleNamespace(
        run=run,
        identity={
            "checkpoint_sha256": index["checkpoint_sha256"],
            "checkpoint_relative_path": index["checkpoint"],
            "runtime": runtime,
        },
        model=object(),
        tokenizer=object(),
        config=SimpleNamespace(training=SimpleNamespace(seq_len=32)),
        device="cpu",
        engine=object(),
    )
    monkeypatch.setattr(panel, "verify_evaluation_index", lambda *_a, **_kw: index)
    monkeypatch.setattr(panel, "load_run", lambda *_a, **_kw: loaded)
    monkeypatch.setattr(
        panel, "verify_panel_result", lambda path, **_kw: json.loads(path.read_text())
    )
    forwarded: list[tuple[bool, bool]] = []

    def mocked_generation(*_args: object, **kwargs: object) -> tuple[str, list[int]]:
        forwarded.append((kwargs["use_cache"], kwargs["strict_context"]))
        return "short promptok", [7]

    monkeypatch.setattr(panel, "generate_with_token_ids", mocked_generation)
    for name, controls in (
        ("legacy", {}),
        ("uncached", {"use_cache": False, "strict_context": True}),
    ):
        source = tmp_path / f"{name}.json"
        source.write_text(json.dumps(_declaration(**controls)))
        result = panel.run_panel(source, index_path)
        assert json.loads(result.read_text())["rows"][0]["status"] == "COMPLETED"
    assert forwarded == [(True, False), (False, True)]


def test_pair_parity_requires_exact_binding_and_reports_divergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = "short question"
    cached_path = tmp_path / "cached.json"
    uncached_path = tmp_path / "uncached.json"
    cached_path.write_text("mock")
    uncached_path.write_text("mock")
    shared = {
        "evaluation_index_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "identity": {"tokenizer_sha256": "c" * 64},
    }
    cache = {
        **shared,
        "record_sha256": "d" * 64,
        "declaration": {"decoder": _declaration()["decoder"]},
        "rows": [{"status": "COMPLETED", "prompt": prompt, "token_ids": [1, 2]}],
    }
    no_cache = {
        **shared,
        "record_sha256": "e" * 64,
        "declaration": {"decoder": _declaration(use_cache=False)["decoder"]},
        "rows": [{"status": "COMPLETED", "prompt": prompt, "token_ids": [1, 2]}],
    }
    records = {cached_path: cache, uncached_path: no_cache}
    monkeypatch.setattr(panel, "verify_panel_result", lambda source: records[source])
    manifest_path = tmp_path / "pairs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "panel_pair_version": 1,
                "frozen_content_sha256": "f" * 64,
                "pairs": [
                    {
                        "id": "pair_0",
                        "cached_index": 0,
                        "uncached_index": 0,
                        "item_content_sha256": "1" * 64,
                        "source_family_id": "family_0",
                        "gold_answer": "answer",
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    }
                ],
            }
        )
    )
    assert (
        panel.verify_panel_pair_parity(cached_path, uncached_path, manifest_path)[
            "pairs"
        ][0]["status"]
        == "MATCH"
    )
    no_cache["rows"][0]["token_ids"] = [1, 3]
    assert (
        panel.verify_panel_pair_parity(cached_path, uncached_path, manifest_path)[
            "pairs"
        ][0]["status"]
        == "DIVERGED"
    )
    no_cache["rows"][0]["status"] = "FAILED"
    assert (
        panel.verify_panel_pair_parity(cached_path, uncached_path, manifest_path)[
            "pairs"
        ][0]["status"]
        == "UNAVAILABLE"
    )
    no_cache["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different model inputs"):
        panel.verify_panel_pair_parity(cached_path, uncached_path, manifest_path)
