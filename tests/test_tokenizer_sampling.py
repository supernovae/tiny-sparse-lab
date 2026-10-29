"""Tiny offline byte sampling; mocked export authentication is explicit."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from sparselab import workspace_preflight as storage
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


def storage_fixture(tmp_path, monkeypatch, *, available_bytes):
    config, _ = fixture(tmp_path, monkeypatch)
    config = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(
                update={"train_max_tokens": 105_000_000_000}
            ),
            "sampling": config.sampling.model_copy(
                update={"max_fit_bytes": 512 * 1024 * 1024}
            ),
        }
    )
    monkeypatch.setattr(
        storage.os,
        "statvfs",
        lambda _path: SimpleNamespace(
            f_bavail=available_bytes,
            f_frsize=1,
            f_files=1_000_000,
            f_favail=1_000_000,
        ),
    )
    return config


def test_sampled_fit_storage_does_not_reserve_the_full_corpus(tmp_path, monkeypatch):
    config = storage_fixture(tmp_path, monkeypatch, available_bytes=3 * 1024**3)
    path = engine.train_tokenizer(config)
    manifest = json.loads(path.with_name("tokenizer_manifest.json").read_text())
    assert manifest["input_byte_budget_utf8"] == 512 * 1024 * 1024
    assert (
        sum(
            manifest["training_contract"]["sampling_result"]["requested_bytes"].values()
        )
        == manifest["input_byte_budget_utf8"]
    )


def test_sampled_fit_still_refuses_insufficient_storage(tmp_path, monkeypatch):
    config = storage_fixture(tmp_path, monkeypatch, available_bytes=1024**3)
    with pytest.raises(OSError, match="storage preflight failed"):
        engine.train_tokenizer(config)
    assert not config.output_dir.exists()


def test_unsampled_storage_keeps_the_dataset_bound(tmp_path, monkeypatch):
    config = storage_fixture(tmp_path, monkeypatch, available_bytes=3 * 1024**3)
    config = config.model_copy(update={"sampling": None})
    with pytest.raises(OSError, match="storage preflight failed"):
        storage.require_storage(storage.tokenizer_storage_checks(config))
    assert not config.output_dir.exists()


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


def write_rows(config, rows):
    release = config.dataset.corpus_release_path
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


def small(config, **changes):
    return config.model_copy(
        update={
            "sampling": config.sampling.model_copy(
                update={"max_fit_bytes": 10, **changes}
            )
        }
    )


def test_literal_special_token_text_publishes_and_cold_replays(tmp_path, monkeypatch):
    config, rows = fixture(tmp_path, monkeypatch)
    config = small(config)
    rows[1]["text"] = "see <unk> and <eos>"
    rows[1]["content_sha256"] = hashlib.sha256(rows[1]["text"].encode()).hexdigest()
    write_rows(config, rows)
    path = engine.train_tokenizer(config)
    manifest = engine.verify_tokenizer_artifact(
        path,
        source="local_text",
        revision=config.dataset.revision,
        vocab_size=260,
        dataset=config.dataset,
    )
    assert manifest["sampling_audit"]["special_tokens_as_text"]
    assert manifest["sampling_audit"]["strata"]["real"]["unknown_tokens"] == 0
    assert manifest["sampling_audit"]["strata"]["real"]["roundtrip_failures"] == 0
    tokenizer = engine.load_tokenizer(path)
    assert not tokenizer.encode_special_tokens
    assert tokenizer.token_to_id("<unk>") in tokenizer.encode("see <unk>").ids
    assert engine.train_tokenizer(config) == path


@pytest.mark.parametrize(
    "change",
    [
        {"source_strata": {}, "byte_weights": {}},
        {"byte_weights": {}},
        {"source_strata": {}},
        {"source_strata": {"real": "different"}},
        {"source_strata": {" ": "real"}},
        {"source_strata": {"real": " "}, "byte_weights": {" ": 1}},
        {"byte_weights": {"real": 10001}},
        {"byte_weights": {"real": 0}},
        {"audit_documents_per_stratum": 10001},
        {"max_audit_bytes": 512 * 1024 * 1024 + 1},
    ],
)
def test_sampling_declaration_guards_are_independent(change):
    payload = {
        "source_strata": {"real": "real"},
        "byte_weights": {"real": 1},
        "max_fit_bytes": 10,
    }
    with pytest.raises(ValueError):
        TokenizerSampling.model_validate({**payload, **change})
    TokenizerSampling.model_validate({**payload, "byte_weights": {"real": 10000}})


@pytest.mark.parametrize("change", ["source", "release", "export", "budget"])
def test_sampling_requires_authenticated_compatible_dataset(
    tmp_path, monkeypatch, change
):
    config, _ = fixture(tmp_path, monkeypatch)
    updates = {
        "source": {"source": "synthetic"},
        "release": {"corpus_release_path": None},
        "export": {"corpus_export_path": None},
        "budget": {"train_max_tokens": 99},
    }[change]
    # Check the tokenizer guard directly as well as public validation: the
    # dataset's earlier cross-field guards otherwise shadow missing bindings.
    dataset = config.dataset.model_copy(update=updates)
    with pytest.raises(ValueError, match="stratified sampling requires"):
        config.model_copy(update={"dataset": dataset}).authenticated_sampling()
    with pytest.raises(ValueError):
        TokenizerTrainConfig.model_validate(
            {**config.model_dump(mode="python"), "dataset": dataset}
        )


@pytest.mark.parametrize(
    "change", ["extra", "missing", "restricted", "row_rights", "domains"]
)
def test_sampling_refuses_source_or_document_policy_mismatch(
    tmp_path, monkeypatch, change
):
    config, rows = fixture(tmp_path, monkeypatch)
    if change in {"extra", "missing"}:
        mapping = dict(config.sampling.source_strata)
        if change == "extra":
            mapping["extra"] = "real"
        else:
            mapping.pop("real")
        config = config.model_copy(
            update={
                "sampling": config.sampling.model_copy(
                    update={"source_strata": mapping}
                )
            }
        )
    elif change == "restricted":
        path = config.dataset.corpus_release_path / "sources.json"
        sources = json.loads(path.read_text())
        sources[0]["explicit_training_restriction"] = "training forbidden"
        path.write_text(json.dumps(sources))
    else:
        rows[0]["rights"] = (
            {"training_eligibility": "review_required"}
            if change == "row_rights"
            else rows[0]["rights"]
        )
        if change == "domains":
            rows[0]["domains"] = ["different"]
        write_rows(config, rows)
    with pytest.raises(
        ValueError, match="source strata|rights path|ineligible TRAIN|document domains"
    ):
        engine.train_tokenizer(config)
    assert not config.output_dir.exists()


def test_scan_limit_and_whole_document_empty_stratum(tmp_path, monkeypatch):
    config, rows = fixture(tmp_path, monkeypatch)
    first = [rows[index] for index in (0, 4, 8)]
    remaining = [row for row in rows if row not in first]
    write_rows(config, first + remaining)
    limited = config.model_copy(update={"max_documents": 3})
    stats = {}
    assert list(documents(limited, stats)) == [row["text"] for row in first]
    assert stats["acquired"] == 3
    with pytest.raises(ValueError, match="no fitting TRAIN document"):
        engine.train_tokenizer(small(config, max_fit_bytes=1))
    assert not config.output_dir.exists()


def test_fit_and_audit_never_repeat_content(tmp_path, monkeypatch):
    from sparselab.data.tokenizer_sampling import audit

    config, rows = fixture(tmp_path, monkeypatch)
    duplicate = {**rows[1], "document_id": "duplicate-real"}
    write_rows(config, [*rows[:-1], duplicate, rows[-1]])
    stats = {}
    assert len(list(documents(config, stats))) == 12
    config = small(config, audit_documents_per_stratum=10)
    path = engine.train_tokenizer(config)
    report = audit(config, engine.load_tokenizer(path))
    assert report["strata"]["real"]["documents"] == 3
    assert report["strata"]["real"]["document_shortfall"] == 7


def test_audit_shared_byte_cap_and_quota_rounding(tmp_path, monkeypatch):
    from sparselab.data.tokenizer_sampling import quotas

    config, _ = fixture(tmp_path, monkeypatch)
    equal = small(config, byte_weights={"real": 1, "synthetic": 1, "code": 1})
    assert quotas(equal) == {"real": 3, "synthetic": 3, "code": 4}
    uneven = small(config, byte_weights={"real": 1, "synthetic": 2, "code": 3})
    assert quotas(uneven) == {"real": 2, "synthetic": 3, "code": 5}
    config = small(config, max_audit_bytes=3)
    path = engine.train_tokenizer(config)
    report = json.loads(path.with_name("tokenizer_manifest.json").read_text())[
        "sampling_audit"
    ]
    assert report["selected_bytes"] == 3
    assert report["selected_bytes"] <= report["max_bytes"]
    assert not report["complete"]


@pytest.mark.parametrize(
    "field",
    [
        "sampling_result",
        "content_digest_sha256",
        "selected_input_bytes_utf8",
        "reuse_audit",
    ],
)
def test_selection_contract_and_reuse_tampering_refuse(tmp_path, monkeypatch, field):
    config, _ = fixture(tmp_path, monkeypatch)
    config = small(config)
    path = engine.train_tokenizer(config)
    manifest_path = path.with_name("tokenizer_manifest.json")
    value = json.loads(manifest_path.read_text())
    if field == "reuse_audit":
        value["sampling_audit"]["selected_bytes"] += 1
    elif field == "sampling_result":
        value["training_contract"][field]["unfilled_bytes"]["real"] += 1
    elif field == "content_digest_sha256":
        value["training_contract"][field] = "0" * 64
    else:
        value["training_contract"][field] += 1
    manifest_path.write_text(json.dumps(value))
    with pytest.raises(
        ValueError,
        match="audit mismatch"
        if field == "reuse_audit"
        else "byte-selection identity mismatch",
    ):
        if field == "reuse_audit":
            engine.train_tokenizer(config)
        else:
            engine.verify_tokenizer_artifact(
                path,
                source="local_text",
                revision=config.dataset.revision,
                vocab_size=260,
                dataset=config.dataset,
            )


@pytest.mark.parametrize(
    "mode", ["stratum_unknown", "stratum_roundtrip", "probe_unknown", "probe_roundtrip"]
)
def test_audit_failure_counters_prevent_publication(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace

    from sparselab.data import tokenizer_sampling as sampling

    config, _ = fixture(tmp_path, monkeypatch)
    config = small(config)
    captured = {}
    original_audit = sampling.audit

    class BrokenAuditTokenizer:
        def __init__(self):
            self.values = {}

        def token_to_id(self, _token):
            return 0

        def encode(self, text):
            key = len(self.values) + 10
            target = (
                text.startswith("abcdef")
                if mode.startswith("stratum")
                else text.startswith("some_identifier42")
            )
            self.values[key] = (text, target)
            return SimpleNamespace(
                ids=[0, key] if target and mode.endswith("unknown") else [key]
            )

        def decode(self, ids, *, skip_special_tokens):
            assert not skip_special_tokens
            text, target = self.values[ids[-1]]
            return text[:-1] if target and mode.endswith("roundtrip") else text

    monkeypatch.setattr(
        sampling,
        "Tokenizer",
        SimpleNamespace(from_str=lambda _: BrokenAuditTokenizer()),
    )

    def capture(*args):
        report = original_audit(*args)
        captured.update(report)
        return report

    monkeypatch.setattr(sampling, "audit", capture)
    with pytest.raises(
        ValueError, match="audit found unknown tokens or failed exact roundtrip"
    ):
        engine.train_tokenizer(config)
    assert not config.output_dir.exists()
    if mode.startswith("stratum"):
        result = captured["strata"]["real"]
        assert result["unknown_tokens"] == (1 if mode.endswith("unknown") else 0)
        assert result["roundtrip_failures"] == (1 if mode.endswith("roundtrip") else 0)
        assert all(
            probe["roundtrip"] and probe["unknown_tokens"] == 0
            for probe in captured["fixed_probes"].values()
        )
    else:
        result = captured["fixed_probes"]["identifier"]
        assert result["unknown_tokens"] == (1 if mode.endswith("unknown") else 0)
        assert result["roundtrip"] == (not mode.endswith("roundtrip"))
        assert all(
            group["roundtrip_failures"] == group["unknown_tokens"] == 0
            for group in captured["strata"].values()
        )
