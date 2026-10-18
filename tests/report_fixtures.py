"""Synthetic report wiring only: no measured model outcomes or execution."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from sparselab.config.loading import load_config
from sparselab.experiments.reporting import _historical_study_identity
from sparselab.experiments.study import _comparison_report, plan_study
from sparselab.training.manifest import canonical_json, config_sha256


def write_report_inputs(
    root: Path, *, single: bool = False, historical: bool = False
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    )
    (root / "base.yaml").write_text(
        yaml.safe_dump(base.model_dump(mode="json")), encoding="utf-8"
    )
    variants = [{"label": "base", "set": {}}]
    if not single:
        variants.append({"label": "wide", "set": {"model.ffn_dim": 48}})
    (root / "matrix.yaml").write_text(
        yaml.safe_dump(
            {
                "matrix_version": 1,
                "base_config": "base.yaml",
                "axes": {"size": variants},
            }
        ),
        encoding="utf-8",
    )
    comparisons = (
        []
        if single
        else [
            {
                "id": "width",
                "vary": "custom",
                "vary_fields": ["model.ffn_dim"],
                "baseline": {"size": "base"},
                "variant": {"size": "wide"},
            }
        ]
    )
    (root / "study.yaml").write_text(
        yaml.safe_dump(
            {
                "study_version": 1,
                "name": "neutral-report-fixture",
                "matrix": "matrix.yaml",
                "cards": ["chat-alias-recall-v1"],
                "comparisons": comparisons,
            }
        ),
        encoding="utf-8",
    )
    study = plan_study(root / "study.yaml")
    matrix_digest, study_digest = study.matrix_sha256, study.study_sha256
    if historical:
        matrix_digest, study_digest, _ = _historical_study_identity(study)
    cards = [
        {
            "reference": item.reference,
            "name": item.card.name,
            "digest": item.card.digest,
        }
        for item in study.cards
    ]
    submitted, records = [], []
    for index, planned in enumerate(study.expanded):
        config = planned.config.model_dump(mode="json")
        if historical:
            config["model"].pop("semantic_memory_dim")
            config["training"].pop("neural_loss_weight")
        run_id = f"synthetic-report-{index}"
        row = {
            "run_id": run_id,
            "coordinate": planned.coordinate,
            "config_sha256": config_sha256(config),
        }
        submitted.append(row)
        identity = {
            "run_id": run_id,
            "runtime": {
                "engine": "pytorch",
                "backend": "cpu",
                "precision": "fp32",
                "framework_version": "synthetic-fixture",
            },
            "config": config,
            "checkpoint_sha256": "a" * 64,
            "checkpoint_relative_path": "checkpoints/fixture",
            "step": planned.config.training.max_steps,
            "tokens_seen": planned.config.training.max_tokens,
            "tokenizer_sha256": "b" * 64,
            "source_identity_sha256": "c" * 64,
            "data_sha256": {"train": "d" * 64, "validation": "e" * 64},
        }
        capabilities = {}
        for item in study.cards:
            card = item.card
            # Deliberately wrong literal responses make re-signed scorer tampering observable.
            result = {
                "card": card.name,
                "card_digest": card.digest,
                "identity": identity,
                "valid": True,
                "case_count": len(card.cases),
                "passed": 0,
                "score": 0.0,
                "results": [
                    {
                        "id": case.identifier,
                        "prompt": case.prompt,
                        "expected": case.expected,
                        "response": "fixture-no-answer",
                        "passed": False,
                    }
                    for case in card.cases
                ],
            }
            capabilities[item.reference] = {
                "name": card.name,
                "digest": card.digest,
                "valid": True,
                "case_count": len(card.cases),
                "passed": 0,
                "score": 0.0,
                "result": result,
            }
        records.append(
            {
                **row,
                "identity": identity,
                "capabilities": capabilities,
                "validation": {"loss": 2.0 - index * 0.25},
            }
        )
    receipt = {
        "format": "sparselab-architecture-study-receipt",
        "schema_version": 1,
        "study_sha256": study_digest,
        "matrix_sha256": matrix_digest,
        "cards": cards,
        "runs": submitted,
    }
    receipt_bytes = canonical_json(receipt) + b"\n"
    (root / "receipt.json").write_bytes(receipt_bytes)
    evidence = {
        "format": "sparselab-architecture-study-report",
        "report_version": 1,
        "study_sha256": study_digest,
        "matrix_sha256": matrix_digest,
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "cards": cards,
        "runs": records,
        "comparisons": [
            _comparison_report(study, pair, records) for pair in study.pairs
        ],
    }
    evidence["report_sha256"] = hashlib.sha256(canonical_json(evidence)).hexdigest()
    (root / "collected-report.json").write_bytes(canonical_json(evidence) + b"\n")
    return root
