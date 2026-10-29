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
    assert (
        mixture._rank(17, "incident", "family-a", "document-1", "kml-card03-v1")
        == expected
    )
    assert (
        mixture._rank(17, "incident", "family-b", "document-1", "kml-card03-v1")
        == expected
    )


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
    (tmp_path / "pack").mkdir()
    ids, mask, stats = _collect_token_mixture(
        config, tokenizer, receipt, tmp_path / "pack"
    )
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


def test_yaml_mixture_consumer_is_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Release authentication and tokenizer encoding use the tiny synthetic fixture;
    # mixture materialization, cold replay, dataset binding and packing are real.
    assets = tmp_path / "source" / "nested"
    assets.mkdir(parents=True)
    declaration, _, release = _fixture(assets, monkeypatch)
    output = assets / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    (release / "lm/validation.jsonl").write_text('{"text":"heldout"}\n')
    fit_dir = assets / "fit"
    fit_dir.mkdir()
    (fit_dir / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(
        mixture,
        "load_tokenizer_config",
        lambda *args: SimpleNamespace(output_dir=fit_dir, vocab_size=260),
    )
    from sparselab.corpus import export

    monkeypatch.setattr(export, "_licenses", lambda path: "local research")
    raw = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    ).model_dump(mode="json")
    raw["model"]["vocab_size"] = 260
    raw["training"]["seq_len"] = 8
    raw["tokenizer"]["path"] = "fit/tokenizer.json"
    raw["dataset"] = {
        "source": "local_token_mixture",
        "revision": release.name,
        "cache_dir": "cache",
        "train_path": "output/train.tokens.jsonl",
        "validation_path": f"{release.name}/lm/validation.jsonl",
        "train_max_documents": receipt["records"],
        "validation_max_documents": 1,
        "train_max_tokens": 20,
        "validation_max_tokens": 9,
        "license": "local research",
        "mixture_declaration_path": "mixture.yaml",
        "mixture_output_path": "output",
    }
    source = assets / "run.yaml"
    source.write_text(yaml.safe_dump(raw))
    tokenizer = SimpleNamespace(
        token_to_id=lambda value: {"<bos>": 2, "<eos>": 3, "<pad>": 0}[value]
    )
    for cwd in (tmp_path / "cwd-a", tmp_path / "unrelated" / "cwd-b"):
        cwd.mkdir(parents=True)
        monkeypatch.chdir(cwd)
        config = load_config(source)
        verified = mixture.verify_mixture_dataset(
            config.dataset, config.tokenizer.path, config.model.vocab_size
        )
        assert verified == receipt
        (cwd / "pack").mkdir()
        ids, mask, stats = _collect_token_mixture(
            config, tokenizer, verified, cwd / "pack"
        )
        assert stats["supervised_target_positions"] == 20
        assert sum(mask) == 20
        assert len(ids) == len(mask)


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
            tmp_path,
        )


def test_tokenizer_accepts_retained_release_path_but_not_a_different_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_id = "a" * 64
    release = tmp_path / ("b" * 64)
    dataset = SimpleNamespace(
        source="local_text", corpus_release_path=release, revision=release_id
    )
    config = SimpleNamespace(
        dataset=dataset, output_dir=tmp_path / "fit", vocab_size=260
    )
    spec = SimpleNamespace(
        tokenizer_config=tmp_path / "config", tokenizer_origin_release_id=release_id
    )
    monkeypatch.setattr(mixture, "load_tokenizer_config", lambda path: config)
    monkeypatch.setattr(
        mixture,
        "verify_tokenizer_artifact",
        lambda *args, **kwargs: {"sha256": "t" * 64},
    )
    monkeypatch.setattr(mixture, "load_tokenizer", lambda path: "tokenizer")
    assert mixture._tokenizer(spec, release, release_id) == ("tokenizer", "t" * 64)
    dataset.revision = "c" * 64
    with pytest.raises(ValueError, match="not bound to this release"):
        mixture._tokenizer(spec, release, release_id)
    dataset.revision = release_id
    dataset.corpus_release_path = tmp_path / ("d" * 64)
    with pytest.raises(ValueError, match="not bound to this release"):
        mixture._tokenizer(spec, release, release_id)


@pytest.mark.parametrize("change_during_run", [False, True])
def test_mixture_records_and_rechecks_opt_in_tokenizer_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change_during_run: bool
) -> None:
    original_tokenizer = mixture._tokenizer
    declaration, _, release = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(mixture, "_tokenizer", original_tokenizer)
    (release / "manifest.json").write_text(json.dumps({"release_id": release.name}))
    data = SimpleNamespace(
        source="local_text", corpus_release_path=release, revision=release.name
    )
    config = SimpleNamespace(dataset=data, output_dir=tmp_path / "fit", vocab_size=260)
    monkeypatch.setattr(mixture, "load_tokenizer_config", lambda path: config)
    monkeypatch.setattr(
        mixture,
        "verify_tokenizer_artifact",
        lambda *args, **kwargs: pytest.fail("cold tokenizer replay must not run"),
    )
    monkeypatch.setattr(
        mixture,
        "load_tokenizer",
        lambda path: SimpleNamespace(
            encode=lambda text, **kwargs: SimpleNamespace(ids=list(text.encode())),
            token_to_id=lambda token: 3 if token == "<eos>" else None,
        ),
    )
    from sparselab.corpus import tokenizer_observation

    calls = []

    def observe(*args):
        calls.append(1)
        if change_during_run and len(calls) == 2:
            raise ValueError("observed input changed")
        return {"method": "post_run_observation", "tokenizer_sha256": "t" * 64}

    monkeypatch.setattr(tokenizer_observation, "verify_observed_tokenizer", observe)
    data = yaml.safe_load(declaration.read_text())
    names = (
        "job_spec",
        "session_spec",
        "job_report",
        "session_report",
        "container_report",
        "stdout",
        "stderr",
        "tokenizer_config",
        "tokenizer",
        "tokenizer_manifest",
        "export",
    )
    data["tokenizer_observation"] = {
        "method": "post_run_observation",
        "observed_at_utc": "2026-10-29T05:00:00Z",
        "files": {
            name: {"path": str(tmp_path / name), "sha256": "a" * 64} for name in names
        },
    }
    declaration.write_text(yaml.safe_dump(data))
    output = tmp_path / "observed-output"
    if change_during_run:
        with pytest.raises(ValueError, match="observed input changed"):
            mixture.materialize_mixture(declaration, output)
        assert not output.exists()
    else:
        receipt = mixture.materialize_mixture(declaration, output)
        assert receipt["tokenizer_observation"]["method"] == "post_run_observation"
    assert len(calls) == 2


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


def test_observed_mixture_consumer_reuses_export_validation_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, _, release = _fixture(tmp_path, monkeypatch)
    raw = yaml.safe_load(declaration.read_text())
    names = (
        "job_spec",
        "session_spec",
        "job_report",
        "session_report",
        "container_report",
        "stdout",
        "stderr",
        "tokenizer_config",
        "tokenizer",
        "tokenizer_manifest",
        "export",
    )
    raw["tokenizer_observation"] = {
        "method": "post_run_observation",
        "observed_at_utc": "2026-10-29T05:00:00Z",
        "files": {
            name: {"path": str(tmp_path / name), "sha256": "a" * 64} for name in names
        },
    }
    declaration.write_text(yaml.safe_dump(raw))
    output = tmp_path / "output"
    fit_dir = tmp_path / "fit"
    source = SimpleNamespace(
        license="local research", validation_max_documents=1, validation_max_tokens=9
    )
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
        lambda *args: SimpleNamespace(
            output_dir=fit_dir, vocab_size=260, dataset=source
        ),
    )
    from sparselab.corpus import export

    monkeypatch.setattr(export, "_licenses", lambda *args: pytest.fail("license scan"))
    monkeypatch.setattr(
        export, "_split_stats", lambda *args: pytest.fail("validation scan")
    )
    dataset = DatasetConfig(
        source="local_token_mixture",
        revision=release.name,
        cache_dir=tmp_path / "cache",
        train_path=output / "train.tokens.jsonl",
        validation_path=release / "lm/validation.jsonl",
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
    wrong = dataset.model_copy(update={"validation_max_tokens": 10})
    with pytest.raises(ValueError, match="validation split count or budget mismatch"):
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
    from sparselab.training.trainer import _check_training_storage

    monkeypatch.setattr(
        "sparselab.data.tokenizer.load_tokenizer", lambda path: tokenizer
    )
    _check_training_storage(config, prepared)
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


@pytest.mark.parametrize("exposures", [True, False, "1", 1.0, 0, 3])
def test_exposure_limit_requires_explicit_supported_integer(
    tmp_path, monkeypatch, exposures
):
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    raw = yaml.safe_load(declaration.read_text())
    raw["max_exposures"] = exposures
    with pytest.raises(ValueError):
        mixture.MixtureDeclaration.model_validate(raw)


def test_unique_export_refuses_repetition_with_measured_shortfall(
    tmp_path, monkeypatch
):
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    raw = yaml.safe_load(declaration.read_text())
    raw["max_exposures"] = 1
    declaration.write_text(yaml.safe_dump(raw))
    with pytest.raises(
        ValueError, match="9 unique, 10 required, max_exposures=1, shortfall=1"
    ):
        mixture.materialize_mixture(declaration, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".mixture-*"))


def test_unique_export_has_no_repeated_positions_and_cold_replays(
    tmp_path, monkeypatch
):
    declaration, _, release = _fixture(tmp_path, monkeypatch)
    raw = yaml.safe_load(declaration.read_text())
    raw["max_exposures"] = 1
    raw["target_quotas"] = {"general": 9, "explanatory": 6, "incident": 4}
    declaration.write_text(yaml.safe_dump(raw))
    (release / "report.json").write_text(
        json.dumps(
            {
                "requested_mixture": {
                    key: value / 19 for key, value in raw["target_quotas"].items()
                }
            }
        )
    )
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    assert (
        receipt["unique_target_positions"]
        == receipt["actual_target_tokens"]
        == raw["target_quotas"]
    )
    rows = [
        json.loads(line)
        for line in (output / "train.tokens.jsonl").read_text().splitlines()
    ]
    assert all(row["exposure"] == 1 for row in rows)
    assert len({row["document_id"] for row in rows}) == len(rows)
    assert mixture.verify_mixture(declaration, output) == receipt


_GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures/mixture_schedule_v1.json").read_text()
)


@pytest.mark.parametrize("golden", _GOLDEN)
def test_disk_mixture_matches_prechange_bytes_and_receipt(
    tmp_path, monkeypatch, golden
):
    # Golden bytes were produced by c5961c4, including duplicate and empty records.
    declaration, inventory, release = _fixture(tmp_path, monkeypatch)
    docs = list(mixture._rows(release / "documents.jsonl"))
    extra = [
        {**docs[0], "document_id": "g3"},
        {**docs[0], "document_id": "g4", "content_sha256": "z" * 64, "text": ""},
    ]
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in docs + extra)
    )
    with inventory.open("ab") as stream:
        for doc in extra:
            stream.write(
                canonical_json(
                    {
                        "document_id": doc["document_id"],
                        "family_id": "family-" + doc["document_id"],
                        "split": "train",
                        "stratum": "general",
                        "content_sha256": doc["content_sha256"],
                    }
                )
                + b"\n"
            )
    with (release / "lm/train.lineage.jsonl").open("ab") as stream:
        for doc in extra:
            stream.write(canonical_json({"record_id": doc["document_id"]}) + b"\n")
    spec = yaml.safe_load(declaration.read_text())
    spec.update(
        seed=golden["seed"],
        max_exposures=golden["max_exposures"],
        target_quotas=golden["receipt"]["requested_target_quotas"],
    )
    if golden["order_policy"]:
        spec["order_policy"] = golden["order_policy"]
    declaration.write_text(yaml.safe_dump(spec))
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    assert (output / "train.tokens.jsonl").read_bytes() == b"".join(
        canonical_json(row) + b"\n" for row in golden["tokens"]
    )
    expected = {
        **golden["receipt"],
        "declaration_sha256": hashlib.sha256(declaration.read_bytes()).hexdigest(),
        "family_inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
    }
    assert receipt == expected
    assert mixture.verify_mixture(declaration, output) == receipt
    assert sorted(path.name for path in output.iterdir()) == [
        "receipt.json",
        "train.tokens.jsonl",
    ]
    assert not list(tmp_path.glob(".mixture-*"))


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("missing", "cover every"),
        ("duplicate", "duplicate document"),
        ("unknown", "unknown"),
        ("content", "disagrees"),
        ("stratum", "disagrees"),
        ("split", "disagrees"),
    ],
)
def test_disk_inventory_refusals_publish_nothing(
    tmp_path, monkeypatch, mutation, error
):
    declaration, inventory, _ = _fixture(tmp_path, monkeypatch)
    rows = list(mixture._rows(inventory))
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    else:
        field, value = {
            "unknown": ("document_id", "absent"),
            "content": ("content_sha256", "z" * 64),
            "stratum": ("stratum", "incident"),
            "split": ("split", "validation"),
        }[mutation]
        rows[0][field] = value
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match=error):
        mixture.materialize_mixture(declaration, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".mixture-*"))


def test_materialization_actually_uses_disk_order_and_token_spool(
    tmp_path, monkeypatch
):
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    original = mixture.MixtureIndex
    observations = []

    class ObservedIndex(original):
        def encoded(self, stratum):
            assert self.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 4
            assert self.db.execute("SELECT COUNT(*) FROM encoded").fetchone()[0] > 0
            assert Path(self.tokens.name).stat().st_size > 0 or self.tokens.tell() > 0
            observations.append(stratum)
            yield from super().encoded(stratum)

    monkeypatch.setattr(mixture, "MixtureIndex", ObservedIndex)
    mixture.materialize_mixture(declaration, tmp_path / "output")
    assert set(observations) == {"general", "explanatory", "incident"}


def test_mixture_record_and_receipt_bounds_refuse(tmp_path, monkeypatch):
    from sparselab.corpus import mixture_index

    declaration, inventory, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(mixture_index, "MAX_ROW_BYTES", 10)
    with pytest.raises(ValueError, match="record-byte cap"):
        mixture.materialize_mixture(declaration, tmp_path / "row-output")
    assert not (tmp_path / "row-output").exists()
    monkeypatch.setattr(mixture_index, "MAX_ROW_BYTES", 64 * 1024**2)
    monkeypatch.setattr(mixture, "MAX_RECEIPT_BYTES", 10)
    with pytest.raises(ValueError, match="receipt-byte cap"):
        mixture.materialize_mixture(declaration, tmp_path / "receipt-output")
    output = tmp_path / "too-large"
    output.mkdir()
    (output / "receipt.json").write_text(" " * 11)
    with pytest.raises(ValueError, match="receipt-byte cap"):
        mixture.verify_mixture(declaration, output)
    assert not (tmp_path / "receipt-output").exists()
    assert inventory.exists()


def test_mixture_packing_uses_spools_not_schedule_sized_arrays(tmp_path, monkeypatch):
    import numpy as np

    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    root = tmp_path / "packed"
    root.mkdir()
    config = SimpleNamespace(
        dataset=SimpleNamespace(train_path=output / "train.tokens.jsonl"),
        training=SimpleNamespace(seq_len=8),
        model=SimpleNamespace(vocab_size=260),
    )
    tokenizer = SimpleNamespace(
        token_to_id=lambda token: {"<bos>": 2, "<eos>": 3, "<pad>": 0}[token]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("schedule-sized allocation")

    monkeypatch.setattr(packing.np, "full", forbidden)
    monkeypatch.setattr(packing.np, "zeros", forbidden)
    proofs = {}
    ids, mask, stats = packing._collect_token_mixture(
        config, tokenizer, receipt, root, proofs=proofs
    )
    assert isinstance(ids, np.memmap) and isinstance(mask, np.memmap)
    assert set(proofs) == {"train.npy", "train_supervision.npy"}
    assert stats["supervised_target_positions"] == 20
    assert (
        ids.tolist()
        == [2]
        + [
            token
            for row in mixture._rows(output / "train.tokens.jsonl")
            for token in row["token_ids"]
        ]
        + [0] * 4
    )
    assert mask.tolist() == [False] + [True] * 20 + [False] * 4
    assert not list(root.glob("*.raw"))


@pytest.mark.parametrize("max_tokens", [1, 7, 100])
def test_spooled_validation_is_byte_identical_to_legacy_collect(tmp_path, max_tokens):
    import numpy as np
    from tokenizers import Tokenizer, models, pre_tokenizers

    path = tmp_path / "validation.jsonl"
    path.write_text('{"text":"one two three"}\n{"text":"two three"}\n')
    tokenizer = Tokenizer(
        models.WordLevel(
            {"<unk>": 0, "<eos>": 1, "one": 2, "two": 3, "three": 4}, unk_token="<unk>"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    config = SimpleNamespace(
        source="local_token_mixture",
        validation_path=path,
        validation_max_documents=2,
        validation_max_tokens=max_tokens,
    )
    expected = packing._collect(config, tokenizer, "validation")
    proofs = {}
    actual = packing._collect(
        config, tokenizer, "validation", root=tmp_path, proofs=proofs
    )
    assert isinstance(actual[0], np.memmap)
    assert isinstance(actual[1], np.memmap)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]
    assert set(proofs) == {"validation.npy", "validation_supervision.npy"}


@pytest.mark.parametrize("changed", ["documents", "inventory", "lineage"])
def test_input_changes_during_offset_replay_never_publish(
    tmp_path, monkeypatch, changed
):
    declaration, inventory, release = _fixture(tmp_path, monkeypatch)
    target = {
        "documents": release / "documents.jsonl",
        "inventory": inventory,
        "lineage": release / "lm/train.lineage.jsonl",
    }[changed]
    original = mixture.MixtureIndex

    class ChangedIndex(original):
        def verify_inputs(self):
            with target.open("ab") as stream:
                stream.write(b"\n")
            return super().verify_inputs()

    monkeypatch.setattr(mixture, "MixtureIndex", ChangedIndex)
    with pytest.raises(ValueError, match="input changed"):
        mixture.materialize_mixture(declaration, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".mixture-*"))


@pytest.mark.parametrize(
    "kind,error", [("split", "content leaks"), ("stratum", "different strata")]
)
def test_disk_index_refuses_cross_document_content_conflicts(
    tmp_path, monkeypatch, kind, error
):
    declaration, inventory, release = _fixture(tmp_path, monkeypatch)
    docs = list(mixture._rows(release / "documents.jsonl"))
    families = list(mixture._rows(inventory))
    position = 3 if kind == "split" else 1
    docs[position]["content_sha256"] = docs[0]["content_sha256"]
    families[position]["content_sha256"] = docs[0]["content_sha256"]
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in docs)
    )
    inventory.write_bytes(b"".join(canonical_json(row) + b"\n" for row in families))
    with pytest.raises(ValueError, match=error):
        mixture.materialize_mixture(declaration, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_disk_index_requires_all_training_lineage(tmp_path, monkeypatch):
    declaration, _, release = _fixture(tmp_path, monkeypatch)
    (release / "lm/train.lineage.jsonl").write_bytes(
        canonical_json({"record_id": "g1"}) + b"\n"
    )
    with pytest.raises(ValueError, match="train LM view"):
        mixture.materialize_mixture(declaration, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_disk_source_and_encoded_row_limits_are_independently_enforced(
    tmp_path, monkeypatch
):
    from sparselab.corpus import mixture_index

    docs = tmp_path / "documents.jsonl"
    payload = (
        canonical_json(
            {
                "document_id": "one",
                "drop_reason": None,
                "split": "train",
                "content_sha256": "a" * 64,
                "source_id": "source",
            }
        )
        + b"\n"
    )
    docs.write_bytes(payload)
    scratch = tmp_path / "index"
    scratch.mkdir()
    monkeypatch.setattr(mixture_index, "MAX_ROW_BYTES", len(payload) - 1)
    with (
        pytest.raises(ValueError, match="record-byte cap"),
        mixture_index.MixtureIndex(scratch, docs),
    ):
        pass
    other = tmp_path / "encoded"
    other.mkdir()
    monkeypatch.setattr(mixture_index, "MAX_ROW_BYTES", len(payload))
    with mixture_index.MixtureIndex(other, docs) as index:
        with pytest.raises(ValueError, match="encoded record-byte cap"):
            index.append(
                {
                    "family_id": "family",
                    "stratum": "source",
                    "token_ids": list(range(200)),
                }
            )
        assert index.db.execute("SELECT COUNT(*) FROM encoded").fetchone()[0] == 0


def test_exclusion_metadata_limit_refuses_without_silent_truncation(
    tmp_path, monkeypatch
):
    declaration, inventory, release = _fixture(tmp_path, monkeypatch)
    docs = list(mixture._rows(release / "documents.jsonl"))
    duplicate = {**docs[0], "document_id": "duplicate"}
    with (release / "documents.jsonl").open("ab") as stream:
        stream.write(canonical_json(duplicate) + b"\n")
    with inventory.open("ab") as stream:
        stream.write(
            canonical_json(
                {
                    "document_id": "duplicate",
                    "family_id": "duplicate-family",
                    "split": "train",
                    "stratum": "general",
                    "content_sha256": docs[0]["content_sha256"],
                }
            )
            + b"\n"
        )
    with (release / "lm/train.lineage.jsonl").open("ab") as stream:
        stream.write(canonical_json({"record_id": "duplicate"}) + b"\n")
    monkeypatch.setattr(mixture, "MAX_RECEIPT_BYTES", 10)
    with pytest.raises(ValueError, match="exclusion receipt-byte cap"):
        mixture.materialize_mixture(declaration, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_disk_index_builds_empty_indexes_and_avoids_corpus_sized_temp_sorts(tmp_path):
    from sparselab.corpus.mixture_index import MixtureIndex

    docs = tmp_path / "documents.jsonl"
    docs.write_text("")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with MixtureIndex(scratch, docs) as index:
        indexes = {
            row[0]
            for row in index.db.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        assert {"ordered_candidates", "encoded_order", "encoded_families"} <= indexes
        statements = [
            (
                "SELECT offset,family FROM docs WHERE selected=1 AND stratum=? ORDER BY rank,id",
                ("train",),
            ),
            ("SELECT offset FROM encoded WHERE stratum=? ORDER BY ordinal", ("train",)),
            (
                "SELECT COUNT(*) FROM (SELECT DISTINCT family FROM encoded INDEXED BY encoded_families WHERE stratum=?)",
                ("train",),
            ),
        ]
        for sql, parameters in statements:
            plan = list(index.db.execute("EXPLAIN QUERY PLAN " + sql, parameters))
            assert not any("TEMP B-TREE" in row[3] for row in plan), plan
        assert index.family_count("train") == 0
