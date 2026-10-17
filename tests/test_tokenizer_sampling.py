"""Tiny offline byte sampling; mocked export authentication is explicit."""

import hashlib
import json

import pytest

from sparselab.config.models import (
    DatasetConfig,
    TokenizerSampling,
    TokenizerTrainConfig,
)
from sparselab.data import tokenizer as engine
from sparselab.data.tokenizer_sampling import documents
from sparselab.training.manifest import canonical_json


def fixture(tmp_path, monkeypatch):
    release = tmp_path / ("a" * 64)
    release.mkdir()
    (release / "lm").mkdir()
    rows = []
    for stratum, text in (("real", "abcdef"), ("synthetic", "é"), ("code", "x\n")):
        for index in range(4):
            content = text if index == 0 else text + str(index)
            rows.append(
                {
                    "document_id": f"{stratum}-{index}",
                    "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "text": content,
                    "split": "train",
                    "drop_reason": None,
                    "source_id": stratum,
                    "domains": [stratum],
                    "rights": {"training_eligibility": "eligible"},
                }
            )
    # Held-out rows deliberately have no text field: selection must not access it.
    rows.append({"document_id": "heldout", "split": "test", "drop_reason": None})
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in rows)
    )
    (release / "lm/train.lineage.jsonl").write_bytes(
        b"".join(
            canonical_json({"record_id": row["document_id"]}) + b"\n"
            for row in rows
            if row["split"] == "train"
        )
    )
    (release / "sources.json").write_text(
        json.dumps(
            [
                {"id": key, "rights_policy": {"training_eligibility": "eligible"}}
                for key in ("real", "synthetic", "code")
            ]
        )
    )
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text('{"text":"fixture"}\n')
    validation.write_text('{"text":"fixture"}\n')
    config = TokenizerTrainConfig(
        schema_version=1,
        vocab_size=260,
        min_frequency=1,
        max_documents=30,
        output_dir=tmp_path / "tokenizer",
        dataset=DatasetConfig(
            source="local_text",
            revision=release.name,
            license="MIT",
            cache_dir=tmp_path / "cache",
            train_path=train,
            validation_path=validation,
            train_max_documents=12,
            validation_max_documents=1,
            train_max_tokens=1000,
            validation_max_tokens=100,
            corpus_release_path=release,
            corpus_export_path=tmp_path / "export",
        ),
        sampling=TokenizerSampling(
            source_strata={key: key for key in ("real", "synthetic", "code")},
            byte_weights={"real": 60, "synthetic": 20, "code": 20},
            max_fit_bytes=100,
            audit_documents_per_stratum=1,
        ),
    )
    # Exercise the real trainer/manifest verifier while replacing only the
    # expensive release/export fixture authentication with a fixed tiny binding.
    binding = {"release_id": release.name, "vocab_size": 260}
    monkeypatch.setattr(
        engine, "verify_release_export", lambda *_args, **_kwargs: binding
    )
    return config, rows


def test_sampling_byte_shares_whole_utf8_and_disjoint_audit(tmp_path, monkeypatch):
    config, _ = fixture(tmp_path, monkeypatch)
    config = config.model_copy(
        update={"sampling": config.sampling.model_copy(update={"max_fit_bytes": 10})}
    )
    stats = {}
    assert list(documents(config, stats)) == ["abcdef", "é", "x\n"]
    assert stats["sampling"]["selected_bytes"] == {"real": 6, "synthetic": 2, "code": 2}
    path = engine.train_tokenizer(config)
    manifest = engine.verify_tokenizer_artifact(
        path,
        source="local_text",
        revision=config.dataset.revision,
        vocab_size=260,
        dataset=config.dataset,
    )
    assert manifest["input_byte_budget_utf8"] == 10
    assert manifest["sampling_audit"]["complete"]
    assert all(
        group["documents"] == 1 and group["roundtrip_failures"] == 0
        for group in manifest["sampling_audit"]["strata"].values()
    )
    assert engine.train_tokenizer(config) == path
    manifest["sampling_audit"]["selected_bytes"] += 1
    path.with_name("tokenizer_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="audit mismatch"):
        engine.verify_tokenizer_artifact(
            path,
            source="local_text",
            revision=config.dataset.revision,
            vocab_size=260,
            dataset=config.dataset,
        )


def test_unfilled_quota_and_audit_shortfall_are_explicit(tmp_path, monkeypatch):
    config, _ = fixture(tmp_path, monkeypatch)
    path = engine.train_tokenizer(config)
    manifest = json.loads(path.with_name("tokenizer_manifest.json").read_text())
    assert any(
        manifest["training_contract"]["sampling_result"]["unfilled_bytes"].values()
    )
    assert not manifest["sampling_audit"]["complete"]
    assert all(
        group["document_shortfall"] == 1
        for group in manifest["sampling_audit"]["strata"].values()
    )


@pytest.mark.parametrize(
    "change",
    [
        {"max_fit_bytes": 512 * 1024 * 1024 + 1},
        {"byte_weights": {"real": True}},
        {"source_strata": {}},
        {"max_fit_bytes": 0},
    ],
)
def test_sampling_refuses_invalid_or_unbounded_declarations(change):
    payload = {
        "source_strata": {"real": "real"},
        "byte_weights": {"real": 1},
        "max_fit_bytes": 100,
    }
    with pytest.raises(ValueError):
        TokenizerSampling.model_validate({**payload, **change})


def test_sampling_never_uses_unexported_train_document(tmp_path, monkeypatch):
    config, rows = fixture(tmp_path, monkeypatch)
    release = config.dataset.corpus_release_path
    allowed = [
        row
        for row in rows
        if row["split"] == "train" and not row["document_id"].endswith("-0")
    ]
    (release / "lm/train.lineage.jsonl").write_bytes(
        b"".join(
            canonical_json({"record_id": row["document_id"]}) + b"\n" for row in allowed
        )
    )
    stats = {}
    selected = list(documents(config, stats))
    assert "abcdef" not in selected and "é" not in selected and "x\n" not in selected


def test_changed_sampling_contract_refuses_existing_output(tmp_path, monkeypatch):
    config, _ = fixture(tmp_path, monkeypatch)
    engine.train_tokenizer(config)
    changed = config.model_copy(
        update={"sampling": config.sampling.model_copy(update={"max_fit_bytes": 90})}
    )
    with pytest.raises(FileExistsError, match="different or unverifiable"):
        engine.train_tokenizer(changed)


def test_sampling_from_cold_production_template_release(tmp_path):
    from pathlib import Path

    from test_kml_50m_preparation_sequence import _prior

    from sparselab.config.loading import load_tokenizer_config
    from sparselab.corpus.export import export_release

    root = tmp_path / "work"
    root.mkdir()
    release, _, strata = _prior(root)
    exported = export_release(
        release, "lm", Path("configs/runtime_smoke_cpu.yaml"), 260, root
    )
    config = load_tokenizer_config(exported / "tokenizer.yaml")
    sampling = TokenizerSampling(
        source_strata=strata,
        byte_weights=dict.fromkeys(strata.values(), 1),
        max_fit_bytes=min(6000, config.dataset.train_max_tokens),
        audit_documents_per_stratum=1,
    )
    config = TokenizerTrainConfig.model_validate(
        {**config.model_dump(mode="python"), "sampling": sampling}
    )
    path = engine.train_tokenizer(config)
    verified = engine.verify_tokenizer_artifact(
        path,
        source="local_text",
        revision=release.name,
        vocab_size=260,
        dataset=config.dataset,
    )
    assert verified["training_contract"]["corpus_export"]["release_id"] == release.name
    assert verified["sampling_audit"]["disjoint_from_fit"]
