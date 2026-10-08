"""Zero-update, no-generation checks for the base-language scoring adapter."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparselab.corpus.mixture import MixtureDeclaration, _tokenizer
from sparselab.evaluation.fixed_slices import (
    BoundFixedSlices,
    bind_fixed_slices,
    score_fixed_slices,
    score_window,
)
from sparselab.training.manifest import sha256_file


class NextId(torch.nn.Module):
    def forward(self, x):
        logits = torch.full((*x.shape, 128), -8.0)
        next_ids = (x + 1) % 128
        return logits.scatter(-1, next_ids.unsqueeze(-1), 8.0)


def test_exact_shift_and_nonfinite_rejection():
    model = NextId()
    assert (
        score_window(
            model,
            [2, 3, 4, 5],
            first_target=1,
            target_count=3,
            device=torch.device("cpu"),
        )
        < 0.01
    )
    assert (
        score_window(
            model,
            [2, 3, 9, 5],
            first_target=1,
            target_count=3,
            device=torch.device("cpu"),
        )
        > 10
    )
    with pytest.raises(ValueError, match="target positions"):
        score_window(
            model, [2, 3], first_target=0, target_count=1, device=torch.device("cpu")
        )


def _scoring_bound():
    slices = []
    ids = {}
    for split in ("validation", "test"):
        for stratum in ("general_prose", "explanatory_prose", "incident_response_docs"):
            for number in range(4):
                ident = f"{split}-{stratum}-{number}"
                slices.append(
                    {
                        "id": ident,
                        "split": split,
                        "stratum": stratum,
                        "document_id": ident,
                        "start_token": 0,
                    }
                )
                ids[ident] = [i % 128 for i in range(257)]
    return BoundFixedSlices({"loss_slices": slices}, "a" * 64, None, ids, {}, {})


def test_scoring_coverage_and_pre_forward_budget():
    model = NextId()
    bound = _scoring_bound()
    with pytest.raises(ValueError, match="cap exhausted"):
        score_fixed_slices(bound, model, torch.device("cpu"), "validation", 3083)
    assert model.training  # State restored even on cap exhaustion.
    result = score_fixed_slices(bound, model, torch.device("cpu"), "validation", 3084)
    assert (result["forward_input_positions"], result["scored_targets"]) == (3084, 3072)
    assert len(result["items"]) == 12
    assert set(result["per_stratum_loss"]) == {
        "general_prose",
        "explanatory_prose",
        "incident_response_docs",
    }


class TinyTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(ids=[ord(char) % 128 for char in text])

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join(chr(item) for item in ids)


def _profile_fixture(tmp_path: Path, monkeypatch):
    release = tmp_path / "corpora" / "fixture" / "releases" / "release"
    release.mkdir(parents=True)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}")
    tokenizer_config = tmp_path / "tokenizer.yaml"
    tokenizer_config.write_text("fixture")
    rows = []
    family = []
    slices = []
    pairs = []
    prompts = []
    for split in ("validation", "test"):
        for stratum in ("general_prose", "explanatory_prose", "incident_response_docs"):
            for number in range(4):
                ident = f"{split}-{stratum}-{number}"
                rows.append(
                    {
                        "document_id": ident,
                        "content_sha256": "c" * 64,
                        "source_id": stratum,
                        "split": split,
                        "drop_reason": None,
                        "text": "a" * 300,
                    }
                )
                family.append(
                    {
                        "document_id": ident,
                        "content_sha256": "c" * 64,
                        "split": split,
                        "stratum": stratum,
                        "family_id": ident,
                    }
                )
                slices.append(
                    {
                        "id": ident,
                        "document_id": ident,
                        "content_sha256": "c" * 64,
                        "source_id": stratum,
                        "split": split,
                        "stratum": stratum,
                        "family_id": ident,
                        "document_tokens": 300,
                        "start_token": 0,
                        "input_tokens": 257,
                        "scored_targets": 256,
                    }
                )
    by_key = {
        (row["split"], row["stratum"]): [
            r
            for r in slices
            if r["split"] == row["split"] and r["stratum"] == row["stratum"]
        ]
        for row in slices
    }
    for row in slices:
        choices = by_key[(row["split"], row["stratum"])]
        decoy = choices[(choices.index(row) + 1) % 4]
        pairs.append(
            {
                "slice_id": row["id"],
                "context_start_token": 0,
                "context_tokens": 64,
                "true_next_tokens": 32,
                "decoy_document_id": decoy["document_id"],
                "decoy_start_token": 0,
                "decoy_tokens": 32,
            }
        )
    for row in slices:
        if (
            row["stratum"] in {"general_prose", "explanatory_prose"}
            and len(prompts) < 8
        ):
            prompts.append(
                {
                    "slice_id": row["id"],
                    "prompt_start_token": 0,
                    "prompt_tokens": 64,
                    "max_new_tokens": 64,
                }
            )
    (release / "manifest.json").write_text("{}")
    (release / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    inventory_path = tmp_path / "families.jsonl"
    inventory_path.write_text("".join(json.dumps(row) + "\n" for row in family))
    profile = {
        "schema_version": 1,
        "release_id": "release",
        "release_manifest_sha256": sha256_file(release / "manifest.json"),
        "documents_sha256": sha256_file(release / "documents.jsonl"),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "tokenization": {
            "add_special_tokens": False,
            "padding": False,
            "truncation": False,
        },
        "loss_slices": slices,
        "utility_pairs": pairs,
        "continuations": prompts,
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    monkeypatch.setattr(
        "sparselab.evaluation.fixed_slices.verification_options", lambda *a, **kw: {}
    )
    monkeypatch.setattr(
        "sparselab.evaluation.fixed_slices.verify_release",
        lambda *a, **kw: {"release_id": "release"},
    )
    monkeypatch.setattr(
        "sparselab.evaluation.fixed_slices.load_tokenizer_config",
        lambda _: SimpleNamespace(
            output_dir=tmp_path,
            dataset=SimpleNamespace(source="local_text", revision="release"),
            vocab_size=128,
        ),
    )
    monkeypatch.setattr(
        "sparselab.evaluation.fixed_slices.verify_tokenizer_artifact",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "sparselab.evaluation.fixed_slices.load_tokenizer", lambda _: TinyTokenizer()
    )
    return profile_path, release, tokenizer_path, tokenizer_config, inventory_path


def test_binding_rejects_offset_and_decoy_substitution(tmp_path, monkeypatch):
    args = _profile_fixture(tmp_path, monkeypatch)
    assert len(bind_fixed_slices(*args).ids) == 24
    profile = json.loads(args[0].read_text())
    profile["loss_slices"][0]["start_token"] = 100
    args[0].write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="loss slice"):
        bind_fixed_slices(*args)
    profile["loss_slices"][0]["start_token"] = 0
    profile["utility_pairs"][0]["decoy_document_id"] = profile["loss_slices"][0][
        "document_id"
    ]
    args[0].write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="decoy"):
        bind_fixed_slices(*args)


@pytest.mark.parametrize("damage", ["duplicate", "omission", "changed_tokenizer"])
def test_binding_rejects_coverage_and_tokenizer_drift(tmp_path, monkeypatch, damage):
    args = _profile_fixture(tmp_path, monkeypatch)
    profile = json.loads(args[0].read_text())
    if damage == "duplicate":
        profile["loss_slices"][1]["id"] = profile["loss_slices"][0]["id"]
    elif damage == "omission":
        profile["loss_slices"].pop()
    else:
        args[2].write_text('{"changed":true}')
    args[0].write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="binding mismatch|24 unique"):
        bind_fixed_slices(*args)


def test_binding_requires_declared_profile_and_family_bytes(tmp_path, monkeypatch):
    args = _profile_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="profile digest"):
        bind_fixed_slices(*args, expected_profile_sha256="0" * 64)
    with pytest.raises(ValueError, match="family inventory digest"):
        bind_fixed_slices(*args, expected_family_sha256="0" * 64)
    with args[4].open("a") as stream:
        stream.write(args[4].read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate fixed family"):
        bind_fixed_slices(*args)


def test_mixture_legacy_absent_field_and_explicit_origin(tmp_path, monkeypatch):
    release = tmp_path / ("b" * 64)
    origin = tmp_path / ("a" * 64)
    origin.mkdir()
    config_path = tmp_path / "old.yaml"
    config_path.write_text("old")
    raw = {
        "schema_version": 1,
        "release_path": str(release),
        "tokenizer_config": str(config_path),
        "family_inventory": str(tmp_path / "family.jsonl"),
        "source_strata": {"source": "general_prose"},
        "target_quotas": {"general_prose": 100},
        "seed": 17,
        "max_exposures": 2,
    }
    legacy = MixtureDeclaration.model_validate(raw)
    assert legacy.tokenizer_origin_release_id is None
    assert "tokenizer_origin_release_id" not in legacy.model_dump(exclude_none=True)
    fit = SimpleNamespace(
        dataset=SimpleNamespace(
            source="local_text", corpus_release_path=origin, revision=origin.name
        ),
        output_dir=tmp_path,
        vocab_size=128,
    )
    monkeypatch.setattr("sparselab.corpus.mixture.load_tokenizer_config", lambda _: fit)
    monkeypatch.setattr(
        "sparselab.corpus.mixture.verify_tokenizer_artifact",
        lambda *args, **kwargs: {"sha256": "a" * 64},
    )
    monkeypatch.setattr("sparselab.corpus.mixture.load_tokenizer", lambda _: object())
    with pytest.raises(ValueError, match="not bound"):
        _tokenizer(legacy, release, release.name)
    reused = MixtureDeclaration.model_validate(
        {**raw, "tokenizer_origin_release_id": origin.name}
    )
    assert _tokenizer(reused, release, release.name)[1] == "a" * 64


def test_fixed_score_runtime_authorization_is_bound_without_loading_model(
    tmp_path, monkeypatch
):
    from sparselab.cli.main import _prepare_runtime_command

    profile = SimpleNamespace(
        python=tmp_path / "python", model_dump_json=lambda: '{"profile":"fixture"}'
    )
    marker = object()
    monkeypatch.setattr("sparselab.cli.main.load_runtime_profile", lambda _: profile)
    monkeypatch.setattr(
        "sparselab.cli.main._ensure_runtime_interpreter", lambda *a: None
    )
    monkeypatch.setattr(
        "sparselab.evaluation.inference.evaluation_config", lambda *a: object()
    )
    monkeypatch.setattr("sparselab.cli.main.authorize_profile", lambda *a: marker)
    args = Namespace(
        command="evaluation",
        evaluation_command="fixed-slices",
        fixed_command="score",
        runtime=None,
        runtime_profile=str(tmp_path / "profile.yaml"),
        run_id="fixture-run",
        runs_dir=str(tmp_path),
        checkpoint="checkpoints/step-1",
        backend="rocm",
    )
    _prepare_runtime_command(args)
    assert args.runtime_authorization is marker
    assert args.runtime_profile_python == profile.python


def test_declared_nontraining_forward_allocation_closes():
    root = Path(__file__).parents[1] / "experiments/research/kernel-memory-lab"
    budget = json.loads((root / "card05-base/evaluation-budget-v1.json").read_text())
    assert (
        sha256_file(root / "CARD05_BASE_EVALUATION_PROFILE_V1.json")
        == budget["profile_sha256"]
    )
    scored = (
        budget["new_validation_sweeps"]
        + budget["new_test_sweeps"]
        + budget["comparison_validation_sweeps"]
        + budget["comparison_test_sweeps"]
    ) * budget["loss_sweep_input_positions"] + budget["utility_sweeps"] * budget[
        "utility_sweep_input_positions"
    ]
    assert scored == budget["fixed_scoring_planned_input_positions"] == 52392
    assert scored <= budget["fixed_scoring_max_input_positions"] == 55000
    prompt = budget["prompt_input_ids_per_call"]
    requested = budget["max_requested_new_tokens_per_call"]
    full_prefix = sum(range(prompt, prompt + requested))
    assert full_prefix == budget["generation_full_prefix_max_input_positions_per_call"]
    assert (
        full_prefix * budget["generation_calls"]
        == budget["generation_reserved_input_positions"]
    )
    assert (
        budget["generation_calls"] * requested
        == budget["max_requested_new_tokens_total"]
    )
    assert (
        budget["fixed_scoring_max_input_positions"]
        + budget["operational_validation_max_input_positions"]
        + budget["generation_reserved_input_positions"]
        == budget["nontraining_reserved_input_positions"]
        <= budget["nontraining_max_input_positions"]
    )
