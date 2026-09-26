from __future__ import annotations

import json
from pathlib import Path

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.data.allocation_tasks import build_memory_allocation
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
from sparselab.evaluation.capabilities import compare_results, load_capability_card
from sparselab.experiments.analysis import build_research_analysis
from sparselab.experiments.charts import render_charts
from sparselab.research.catalog import load_recipe, load_research

ROOT = Path(__file__).resolve().parents[1]


def _run(allocation: str, weights: float) -> dict[str, object]:
    checkpoints = [
        {
            "identity": {
                "run_id": f"run-{allocation}",
                "checkpoint_sha256": "a" * 64,
                "checkpoint_relative_path": "checkpoints/step_2_gen_1",
                "step": 2,
                "tokens_seen": 32,
                "source_identity_sha256": "b" * 64,
            },
            "validation": {"loss": 0.9},
            "capabilities": {"retention": {"result": {"score": 0.2}}},
        },
        {
            "identity": {
                "run_id": f"run-{allocation}",
                "checkpoint_sha256": "c" * 64,
                "checkpoint_relative_path": "checkpoints/step_4_gen_2",
                "step": 4,
                "tokens_seen": 64,
                "source_identity_sha256": "b" * 64,
            },
            "validation": {"loss": 0.4},
            "capabilities": {"retention": {"result": {"score": 0.8}}},
        },
        {
            "identity": {
                "run_id": f"run-{allocation}",
                "checkpoint_sha256": "d" * 64,
                "checkpoint_relative_path": "checkpoints/step_6_gen_3",
                "step": 6,
                "tokens_seen": 96,
                "source_identity_sha256": "b" * 64,
            },
            "validation": {"loss": 0.8},
            "capabilities": {"retention": {"result": {"score": 0.3}}},
        },
    ]
    return {
        "run_id": f"run-{allocation}",
        "config_sha256": "e" * 64,
        "coordinate": {
            "seed": "s17",
            "ownership": allocation,
            "neural_loss_weight": f"w{int(weights * 100)}",
        },
        "endpoint_status": {"status": "complete", "step": 6, "tokens_seen": 96},
        "identity": {
            **checkpoints[-1]["identity"],
            "tokenizer_sha256": "f" * 64,
            "data_sha256": {"train": "1" * 64, "validation": "2" * 64},
            "runtime": {"engine": "pytorch", "backend": "cpu"},
            "training_runtime": {"device": "cpu"},
            "config": {"seed": 17},
        },
        "validation": {"loss": 0.8},
        "capabilities": {"retention": {"result": {"valid": True, "score": 0.3}}},
        "allocation": {
            "manifest_sha256": "3" * 64,
            "owner_codes": {"neural": 0, "lexical": 1, "semantic": 2, "hybrid": 3},
            "raw_tokens": 96,
            "valid_targets": 93,
            "weighted_neural_supervision_mass": 93 * weights,
            "semantic_engram_pack": {"sha256": "4" * 64},
        },
        "parameter_inventory": {
            "trainable": 1000,
            "memory_table": 100,
            "memory_adapter": 40,
            "active_per_token": 64,
        },
        "allocation_metrics": [
            {"name": "allocation/raw_tokens", "value": 96},
            {"name": "allocation/valid_targets", "value": 93},
            {
                "name": "allocation/weighted_neural_supervision_mass",
                "value": 93 * weights,
            },
        ],
        "checkpoint_observations": checkpoints,
    }


def _analysis(design: str) -> dict[str, object]:
    return build_research_analysis(
        [_run("n100", 1.0), _run("n50", 0.5), _run("n0", 0.0)],
        entry={
            "dependent_metrics": [
                {
                    "name": "held-out validation loss",
                    "kind": "measured",
                    "direction": "lower",
                },
                {
                    "name": "capability-card score",
                    "kind": "capability",
                    "direction": "higher",
                },
            ]
        },
        selection={
            "scale": "smoke",
            "data": "offline",
            "backend": "cpu",
            "design": design,
        },
        factorial_designs=[],
        design_axes={},
        quantities=[],
        cards=[{"reference": "retention"}],
    )


def test_memory_allocation_recipe_declares_all_weights_and_regimes() -> None:
    entry = load_research("memory-allocation-curve-v1")
    recipe = load_recipe(entry)

    assert set(recipe.designs) == {
        "default",
        "iso-total",
        "iso-active",
        "iso-token",
        "iso-flop",
    }
    for design in recipe.designs.values():
        ownership = design["smoke"].axes["ownership"]
        weights = design["smoke"].axes["neural_loss_weight"]
        assert [option.label for option in ownership] == [
            "n100",
            "n75",
            "n50",
            "n25",
            "n0",
        ]
        assert all(
            "dataset.allocation_manifest_path" in option.set for option in ownership
        )
        assert [option.label for option in weights] == [
            "w100",
            "w75",
            "w50",
            "w25",
            "w0",
        ]
        assert [option.set["training.neural_loss_weight"] for option in weights] == [
            1,
            0.75,
            0.5,
            0.25,
            0,
        ]
        for scale in design.values():
            assert scale.base_set["dataset.source"] == "local_chat"
            assert (
                scale.base_set["dataset.train_path"]
                == "../../data/path_domain_v1/train.jsonl"
            )
            assert (
                scale.base_set["dataset.validation_path"]
                == "../../data/path_domain_v1/development.jsonl"
            )
            assert scale.base_set["dataset.license"] == (
                "MIT; original synthetic path_domain_v1 corpus; "
                "see data/path_domain_v1/provenance.json"
            )
            assert (
                scale.base_set["tokenizer.path"]
                == "../../artifacts/tokenizer_path_domain_v1/tokenizer.json"
            )
            assert scale.base_set["model.vocab_size"] == 512
            assert scale.base_set["model.memory"] == "ngram"
            assert len(scale.comparisons) == 40

            assert scale.base_set["model.semantic_memory_dim"] == 8


def test_allocation_analysis_retains_task_checkpoint_points_and_nonmonotonicity() -> (
    None
):
    analysis = _analysis("default")
    curve = analysis["allocation_curve"]
    assert isinstance(curve, dict)
    rows = curve["rows"]
    assert isinstance(rows, list)
    first = rows[0]
    assert isinstance(first, dict)
    assert first["curve_id"] == "n100:w100"
    observations = first["checkpoint_observations"]
    assert [point["validation"]["loss"] for point in observations] == [0.9, 0.4, 0.8]
    assert first["tasks"] == [
        {"task": "retention", "evidence": {"result": {"valid": True, "score": 0.3}}}
    ]
    accounting_by_regime = first["resource_accounting"]
    assert accounting_by_regime["iso-neural"]["value"] == 860
    assert accounting_by_regime["iso-total"]["value"] == 1000
    assert accounting_by_regime["iso-active"]["value"] == 64
    assert accounting_by_regime["iso-token"]["raw_input_tokens"] == 96
    assert accounting_by_regime["iso-token"]["valid_supervised_targets"] == 93
    assert accounting_by_regime["iso-token"]["weighted_neural_supervision_mass"] == 93
    assert accounting_by_regime["iso-flop"]["estimated_parameter_proxy_flops"] == 36_864
    accounting = curve["accounting"]
    assert isinstance(accounting, dict)
    assert accounting["raw_input_tokens"] == "measured_count"
    assert "neural_loss_weight" in str(accounting["weighted_neural_supervision_mass"])
    assert "No smoothing" in str(curve["interpretation"])


def test_allocation_regimes_remain_distinct_and_flops_are_estimated() -> None:
    labels = {}
    for design in ("default", "iso-total", "iso-active", "iso-token", "iso-flop"):
        curve = _analysis(design)["allocation_curve"]
        assert isinstance(curve, dict)
        regime = curve["regime"]
        assert isinstance(regime, dict)
        labels[design] = regime
    assert labels["default"]["label"] == "iso-neural"
    assert labels["iso-total"]["comparison_denominator"] == "total trainable parameters"
    assert (
        labels["iso-active"]["comparison_denominator"] == "active parameters per token"
    )
    assert (
        labels["iso-token"]["comparison_denominator"]
        == "raw input tokens and valid supervised targets"
    )
    assert labels["iso-flop"]["cost_kind"] == "estimated"
    assert "never inferred" in str(labels["iso-flop"]["comparison_denominator"])
    charts = render_charts({"research_analysis": _analysis("iso-flop")})
    assert any(name.startswith("allocation-curve-") for name in charts)


def test_allocation_builder_binds_published_development_audit(tmp_path: Path) -> None:
    root = ROOT
    corpus = root / "data" / "path_domain_v1"
    tokenizer_config = load_tokenizer_config(
        root / "configs/path_domain_tokenizer_cpu.yaml"
    )
    tokenizer_config = tokenizer_config.model_copy(
        update={
            "output_dir": tmp_path / "path-domain-tokenizer",
            "dataset": tokenizer_config.dataset.model_copy(
                update={"cache_dir": tmp_path / "path-domain-tokenizer-cache"}
            ),
        }
    )
    tokenizer_path = train_tokenizer(tokenizer_config)
    bundle = tmp_path / "allocation"
    manifest_path = build_memory_allocation(
        bundle,
        tokenizer_path=tokenizer_path,
        train_path=corpus / "train.jsonl",
        validation_path=corpus / "development.jsonl",
        audit_path=corpus / "oracle_audit.json",
        provenance_path=corpus / "provenance.json",
        cards_dir=corpus / "cards",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(manifest["profiles"]) == 25
    development = manifest["profiles"]["iso-total/n50"]
    assert development["train_supervised_targets"] == sum(
        development["train_owner_targets"].values()
    )
    assert development["validation_supervised_targets"] == sum(
        development["validation_owner_targets"].values()
    )
    assert development["train_raw_tokens"] > development["train_supervised_targets"]
    assert (
        development["validation_raw_tokens"]
        > development["validation_supervised_targets"]
    )
    assert all(
        entry["case_count"] == entry["semantic_query_count"]
        for entry in manifest["cards"]
    )
    for card_name in (
        "path-domain-acquisition-v1.json",
        "path-domain-development-v1.json",
        "path-domain-frozen-v1.json",
    ):
        card = load_capability_card(bundle / "cards" / card_name)
        assert all(case.semantic_query is not None for case in card.cases)

    base = load_config(root / "configs/runtime_smoke_cpu.yaml")
    configured = base.model_copy(
        update={
            "tokenizer": base.tokenizer.model_copy(update={"path": tokenizer_path}),
            "model": base.model.model_copy(update={"semantic_memory_dim": 8}),
            "dataset": base.dataset.model_copy(
                update={
                    "cache_dir": tmp_path / "cache",
                    "train_max_documents": tokenizer_config.dataset.train_max_documents,
                    "validation_max_documents": tokenizer_config.dataset.validation_max_documents,
                    "train_max_tokens": tokenizer_config.dataset.train_max_tokens,
                    "validation_max_tokens": tokenizer_config.dataset.validation_max_tokens,
                    "source": "local_chat",
                    "train_path": corpus / "train.jsonl",
                    "validation_path": corpus / "development.jsonl",
                    "allocation_manifest_path": bundle
                    / "iso-total"
                    / "n50-owners-neural-lexical-semantic-hybrid.json",
                }
            ),
        }
    )
    prepared = prepare_data(configured, load_tokenizer(tokenizer_path))
    assert prepared.train_semantic_queries is not None
    assert prepared.validation_semantic_queries is not None
    assert prepared.train_semantic_queries.ndim == 2
    assert prepared.train_semantic_queries.shape[0] == len(prepared.train)
    assert prepared.validation_semantic_queries.shape[0] == len(prepared.validation)


def test_allocation_comparison_preserves_profile_name_not_local_path() -> None:
    def result(profile: str, score: float) -> dict[str, object]:
        return {
            "card_digest": "a" * 64,
            "evaluation_source_sha256": "b" * 64,
            "valid": True,
            "score": score,
            "identity": {
                "checkpoint_sha256": "c" * 64,
                "step": 8,
                "tokens_seen": 512,
                "config": {
                    "seed": 17,
                    "dataset": {
                        "allocation_manifest_path": (
                            f"/private/machine/allocations/{profile}-owners.json"
                        )
                    },
                },
                "tokenizer_sha256": "d" * 64,
                "data_sha256": {"train": "e" * 64, "validation": "f" * 64},
                "source_identity_sha256": "1" * 64,
                "runtime": {"engine": "pytorch", "backend": "cpu"},
            },
        }

    compared = compare_results(
        result("n100", 0.25),
        result("n75", 0.5),
        vary="custom",
        vary_fields=("dataset.allocation_manifest_path",),
    )

    assert compared["differences"]["dataset.allocation_manifest_path"] == {
        "base": "n100-owners.json",
        "variant": "n75-owners.json",
    }
    assert "/private" not in str(compared["differences"])
