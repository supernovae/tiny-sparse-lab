"""Offline deterministic source-mixture and fail-closed inventory checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sparselab.config.loading import load_config
from sparselab.config.models import DatasetConfig, RunConfig
from sparselab.corpus import mixture
from sparselab.data import packing
from sparselab.data.packing import (
    TokenBlockDataset,
    _collect_token_mixture,
    _prepare_data,
)
from sparselab.training.manifest import canonical_json


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    release = tmp_path / ("a" * 64)
    (release / "lm").mkdir(parents=True)
    docs = [
        {
            "document_id": "g1",
            "source_id": "gutenberg",
            "split": "train",
            "drop_reason": None,
            "content_sha256": "g" * 64,
            "text": "abcdefgh",
            "rights": {"training_eligibility": "eligible_with_obligations"},
        },
        {
            "document_id": "w1",
            "source_id": "wikimedia",
            "split": "train",
            "drop_reason": None,
            "content_sha256": "w" * 64,
            "text": "abcdef",
            "rights": {"training_eligibility": "eligible_with_obligations"},
        },
        {
            "document_id": "p1",
            "source_id": "pagerduty",
            "split": "train",
            "drop_reason": None,
            "content_sha256": "p" * 64,
            "text": "abc",
            "rights": {"training_eligibility": "eligible"},
        },
        {
            "document_id": "g2",
            "source_id": "gutenberg",
            "split": "validation",
            "drop_reason": None,
            "content_sha256": "h" * 64,
            "text": "heldout",
            "rights": {"training_eligibility": "eligible_with_obligations"},
        },
    ]
    for row in docs:
        row["domains"] = {
            "gutenberg": ["general"],
            "wikimedia": ["explanatory"],
            "pagerduty": ["incident"],
        }[row["source_id"]]
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in docs)
    )
    (release / "lm/train.lineage.jsonl").write_bytes(
        b"".join(
            canonical_json({"record_id": row["document_id"]}) + b"\n"
            for row in docs
            if row["split"] == "train"
        )
    )
    (release / "sources.json").write_text(
        json.dumps(
            [
                {
                    "id": source,
                    "explicit_training_restriction": "none_found",
                    "rights_policy": {"training_eligibility": "review_required"},
                }
                for source in ("gutenberg", "wikimedia", "pagerduty")
            ]
        )
    )
    (release / "manifest.json").write_text("{}")
    (release / "report.json").write_text(
        json.dumps(
            {"requested_mixture": {"general": 0.5, "explanatory": 0.3, "incident": 0.2}}
        )
    )
    monkeypatch.setattr(
        mixture, "verify_release", lambda path: {"release_id": release.name}
    )
    monkeypatch.setattr(
        mixture,
        "_tokenizer",
        lambda *args: (
            SimpleNamespace(
                encode=lambda text, **kwargs: SimpleNamespace(ids=list(text.encode())),
                token_to_id=lambda token: 3 if token == "<eos>" else None,
            ),
            "t" * 64,
        ),
    )
    inventory = tmp_path / "families.jsonl"
    strata = {
        "gutenberg": "general",
        "wikimedia": "explanatory",
        "pagerduty": "incident",
    }
    inventory.write_bytes(
        b"".join(
            canonical_json(
                {
                    "document_id": row["document_id"],
                    "family_id": "family-" + row["document_id"],
                    "split": row["split"],
                    "stratum": strata[row["source_id"]],
                    "content_sha256": row["content_sha256"],
                }
            )
            + b"\n"
            for row in docs
        )
    )
    declaration = tmp_path / "mixture.yaml"
    declaration.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "release_path": str(release),
                "tokenizer_config": str(tmp_path / "tokenizer.yaml"),
                "family_inventory": str(inventory),
                "source_strata": strata,
                "target_quotas": {"general": 10, "explanatory": 6, "incident": 4},
                "seed": 42,
                "max_exposures": 2,
            }
        )
    )
    return declaration, inventory, release


def test_card03_order_policy_uses_contract_document_hash() -> None:
    expected = hashlib.sha256(b"kml-card03-v1 | incident | document-1").hexdigest()
    assert mixture._rank(
        17, "incident", "family-a", "document-1", "kml-card03-v1"
    ) == expected
    assert mixture._rank(
        17, "incident", "family-b", "document-1", "kml-card03-v1"
    ) == expected


def test_exact_quotas_repeats_and_cold_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    assert receipt["actual_target_tokens"] == {
        "explanatory": 6,
        "general": 10,
        "incident": 4,
    }
    assert receipt["unique_target_positions"] == {
        "explanatory": 6,
        "general": 9,
        "incident": 4,
    }
    assert receipt["quota_shortfalls"] == {
        "general": 0,
        "explanatory": 0,
        "incident": 0,
    }
    rows = [
        json.loads(line)
        for line in (output / "train.tokens.jsonl").read_text().splitlines()
    ]
    assert {row["document_id"] for row in rows} == {"g1", "w1", "p1"}
    assert max(row["exposure"] for row in rows) == 2
    assert all(row["token_ids"][-1] == 3 for row in rows)
    assert all(len(row["token_ids"]) > row["content_token_end"] for row in rows)
    assert mixture.verify_mixture(declaration, output) == receipt


def test_accepted_65_25_10_target_shares_are_realized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, release = _fixture(tmp_path, monkeypatch)
    spec = yaml.safe_load(declaration.read_text())
    spec["target_quotas"] = {"general": 13, "explanatory": 5, "incident": 2}
    declaration.write_text(yaml.safe_dump(spec))
    (release / "report.json").write_text(
        json.dumps(
            {
                "requested_mixture": {
                    "general": 0.65,
                    "explanatory": 0.25,
                    "incident": 0.10,
                }
            }
        )
    )
    receipt = mixture.materialize_mixture(declaration, tmp_path / "accepted")
    assert receipt["actual_target_tokens"] == spec["target_quotas"]
    assert receipt["max_exposures"] == 2
    assert receipt["unique_target_positions"]["general"] >= 7
    assert receipt["unique_target_positions"]["explanatory"] >= 3
    assert receipt["unique_target_positions"]["incident"] >= 1


def test_shortfall_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    spec = yaml.safe_load(declaration.read_text())
    spec["target_quotas"]["incident"] = 9
    declaration.write_text(yaml.safe_dump(spec))
    (tmp_path / ("a" * 64) / "report.json").write_text(
        json.dumps(
            {
                "requested_mixture": {
                    "general": 10 / 25,
                    "explanatory": 6 / 25,
                    "incident": 9 / 25,
                }
            }
        )
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="quota impossible"):
        mixture.materialize_mixture(declaration, output)
    assert not output.exists()


def test_family_leak_and_rights_fail_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, inventory, release = _fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    rows[-1]["family_id"] = rows[0]["family_id"]
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match="family leaks"):
        mixture.materialize_mixture(declaration, tmp_path / "leak")
    rows[-1]["family_id"] = "other"
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    docs = [
        json.loads(line)
        for line in (release / "documents.jsonl").read_text().splitlines()
    ]
    docs[0]["rights"]["training_eligibility"] = "review_required"
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in docs)
    )
    with pytest.raises(ValueError, match="lacks admitted"):
        mixture.materialize_mixture(declaration, tmp_path / "rights")


def test_changed_export_and_inventory_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, inventory, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    mixture.materialize_mixture(declaration, output)
    with (output / "train.tokens.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        mixture.verify_mixture(declaration, output)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    rows[0]["content_sha256"] = "z" * 64
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match="disagrees"):
        mixture.materialize_mixture(declaration, tmp_path / "changed")


def test_heldout_row_in_train_view_and_wrong_share_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, release = _fixture(tmp_path, monkeypatch)
    with (release / "lm/train.lineage.jsonl").open("ab") as stream:
        stream.write(canonical_json({"record_id": "g2"}) + b"\n")
    with pytest.raises(ValueError, match="train LM view"):
        mixture.materialize_mixture(declaration, tmp_path / "leaked")
    (release / "lm/train.lineage.jsonl").write_bytes(
        b"".join(
            canonical_json({"record_id": doc_id}) + b"\n"
            for doc_id in ("g1", "w1", "p1")
        )
    )
    report = {
        "requested_mixture": {"general": 0.65, "explanatory": 0.25, "incident": 0.1}
    }
    (release / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="target quotas disagree"):
        mixture.materialize_mixture(declaration, tmp_path / "wrong_share")


def test_native_packing_supervises_exact_exported_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    config = SimpleNamespace(
        dataset=SimpleNamespace(train_path=output / "train.tokens.jsonl"),
        training=SimpleNamespace(seq_len=8),
        model=SimpleNamespace(vocab_size=260),
    )
    tokenizer = SimpleNamespace(
        token_to_id=lambda value: {"<bos>": 2, "<eos>": 3, "<pad>": 0}[value]
    )
    ids, mask, stats = _collect_token_mixture(config, tokenizer, receipt)
    assert ids[0] == 2 and not mask[0]
    assert stats["scheduled_target_positions"] == 20
    assert stats["supervised_target_positions"] == 20
    assert stats["masked_padding_positions"] == 4
    blocks = TokenBlockDataset(ids, 8, supervision=mask)
    targets = [blocks.numpy_block(index)[1] for index in range(len(blocks))]
    assert sum(int((block != -100).sum()) for block in targets) == 20
    assert sum(int((block == -100).sum()) for block in targets) == 4
    assert [value for block in targets for value in block if value != -100] == [
        value
        for row in mixture._rows(output / "train.tokens.jsonl")
        for value in row["token_ids"]
    ]


def test_native_packing_rejects_malformed_target_row(tmp_path: Path) -> None:
    path = tmp_path / "train.tokens.jsonl"
    path.write_text(json.dumps({"token_ids": [5, 5]}) + "\n")
    config = SimpleNamespace(
        dataset=SimpleNamespace(train_path=path),
        training=SimpleNamespace(seq_len=4),
        model=SimpleNamespace(vocab_size=260),
    )
    tokenizer = SimpleNamespace(
        token_to_id=lambda value: {"<bos>": 2, "<eos>": 3, "<pad>": 0}[value]
    )
    with pytest.raises(ValueError, match="invalid token mixture"):
        _collect_token_mixture(
            config,
            tokenizer,
            {
                "actual_target_tokens": {"x": 2},
                "records": 1,
                "train_tokens_sha256": "0" * 64,
            },
        )


def test_mixture_dataset_binds_train_validation_and_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, release = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    output.mkdir()
    train = output / "train.tokens.jsonl"
    train.write_text("{}\n")
    validation = release / "lm/validation.jsonl"
    validation.write_text('{"text":"heldout"}\n')
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    (fit_dir / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(
        mixture,
        "verify_mixture",
        lambda *args: {
            "release_id": release.name,
            "actual_target_tokens": {"x": 20},
            "records": 3,
        },
    )
    monkeypatch.setattr(
        mixture,
        "load_tokenizer_config",
        lambda *args: SimpleNamespace(output_dir=fit_dir, vocab_size=260),
    )
    from sparselab.corpus import export

    monkeypatch.setattr(export, "_licenses", lambda path: "local research")
    dataset = DatasetConfig(
        source="local_token_mixture",
        revision=release.name,
        cache_dir=tmp_path / "cache",
        train_path=train,
        validation_path=validation,
        train_max_documents=3,
        validation_max_documents=1,
        train_max_tokens=20,
        validation_max_tokens=9,
        license="local research",
        mixture_declaration_path=declaration,
        mixture_output_path=output,
    )
    assert (
        mixture.verify_mixture_dataset(dataset, fit_dir / "tokenizer.json", 260)[
            "records"
        ]
        == 3
    )
    wrong = dataset.model_copy(update={"validation_path": tmp_path / "other.jsonl"})
    with pytest.raises(ValueError, match="identity mismatch"):
        mixture.verify_mixture_dataset(wrong, fit_dir / "tokenizer.json", 260)


def test_native_prepared_cache_preserves_exact_mixture_supervision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, release = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    validation = release / "lm/validation.jsonl"
    validation.write_text('{"text":"heldout long enough for two blocks"}\n')
    dataset = DatasetConfig(
        source="local_token_mixture",
        revision=release.name,
        cache_dir=tmp_path / "cache",
        train_path=output / "train.tokens.jsonl",
        validation_path=validation,
        train_max_documents=receipt["records"],
        validation_max_documents=1,
        train_max_tokens=20,
        validation_max_tokens=100,
        license="local research",
        mixture_declaration_path=declaration,
        mixture_output_path=output,
    )
    base = load_config(Path("configs/runtime_smoke_cpu.yaml"))
    config = base.model_copy(
        update={
            "dataset": dataset,
            "model": base.model.model_copy(update={"vocab_size": 260}),
            "training": base.training.model_copy(update={"seq_len": 8}),
        }
    )
    config = RunConfig.model_validate(config.model_dump(mode="json"))
    tokenizer = SimpleNamespace(
        token_to_id=lambda value: {"<bos>": 2, "<eos>": 3, "<pad>": 0}[value],
        encode=lambda text, **kwargs: SimpleNamespace(
            ids=list(text.encode()), offsets=[(0, 0)] * len(text)
        ),
        to_str=lambda: "{}",
    )
    monkeypatch.setattr(mixture, "verify_mixture_dataset", lambda *args: receipt)
    monkeypatch.setattr(
        packing, "Tokenizer", SimpleNamespace(from_file=lambda path: tokenizer)
    )
    prepared = _prepare_data(config, tokenizer)
    assert prepared.manifest["train"]["supervised_target_positions"] == 20
    assert (
        prepared.manifest["train"]["supervised_target_positions_by_stratum"]
        == receipt["actual_target_tokens"]
    )
    assert prepared.manifest["supervision"]["kind"] == "token-loss-mask-v1"
    assert prepared.train_supervision is not None
    blocks = TokenBlockDataset(
        prepared.train, 8, supervision=prepared.train_supervision
    )
    assert (
        sum(int((blocks.numpy_block(i)[1] != -100).sum()) for i in range(len(blocks)))
        == 20
    )


def test_absent_mixture_fields_do_not_change_existing_config_payload() -> None:
    existing = load_config(Path("configs/runtime_smoke_cpu.yaml"))
    payload = existing.model_dump(mode="json")
    assert "mixture_declaration_path" not in payload["dataset"]
    assert "mixture_output_path" not in payload["dataset"]
