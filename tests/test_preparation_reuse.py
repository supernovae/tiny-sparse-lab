"""Bounded regressions for plan-independent preparation artifacts."""

import json
from types import SimpleNamespace

import pytest
import yaml
from test_corpus_mixture import _fixture

from sparselab.corpus import mixture
from sparselab.corpus.token_cache import TokenCache
from sparselab.data import packing


def test_failed_supply_and_changed_quota_reuse_encoding(tmp_path, monkeypatch):
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    original = mixture._tokenizer
    calls = []

    def counted(*args):
        tokenizer, digest = original(*args)
        encode = tokenizer.encode

        def record(text, **kwargs):
            calls.append(text)
            return encode(text, **kwargs)

        tokenizer.encode = record
        return tokenizer, digest

    monkeypatch.setattr(mixture, "_tokenizer", counted)
    raw = yaml.safe_load(declaration.read_bytes())
    raw["unique_train_content_floors"] = {
        "general": 100,
        "explanatory": 1,
        "incident": 1,
    }
    declaration.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="supply below floor"):
        mixture.materialize_mixture(declaration, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert calls
    measured = list(calls)
    raw.pop("unique_train_content_floors")
    declaration.write_text(yaml.safe_dump(raw))
    mixture.materialize_mixture(declaration, tmp_path / "complete")
    assert all(calls.count(text) == 1 for text in measured)
    before = list(calls)
    raw["target_quotas"] = {"general": 5, "explanatory": 3, "incident": 2}
    declaration.write_text(yaml.safe_dump(raw))
    mixture.materialize_mixture(declaration, tmp_path / "smaller-quota")
    assert calls == before


def test_normal_verification_never_replays(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)

    def forbidden(*args, **kwargs):
        pytest.fail("normal consumption replayed a schedule")

    assert mixture.materialize_mixture(declaration, output) == receipt
    monkeypatch.setattr(mixture, "materialize_mixture", forbidden)
    assert mixture.verify_mixture(declaration, output) == receipt
    changed = dict(receipt, records=receipt["records"] + 1)
    (output / "receipt.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="completion unavailable or changed"):
        mixture.verify_mixture(declaration, output)


def test_encoding_survives_interruption_and_only_new_content_encodes(tmp_path):
    calls = []
    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: (
            calls.append(text) or SimpleNamespace(ids=[ord(c) for c in text])
        )
    )
    with pytest.raises(KeyboardInterrupt), TokenCache(tmp_path, "a" * 64) as cache:
        assert cache.encode(tokenizer, "first")
        raise KeyboardInterrupt
    with TokenCache(tmp_path, "a" * 64) as cache:
        assert cache.encode(tokenizer, "first")
        assert cache.encode(tokenizer, "added")
    assert calls == ["first", "added"]
    with TokenCache(tmp_path, "b" * 64) as cache:
        cache.encode(tokenizer, "first")
    assert calls == ["first", "added", "first"]


def test_preparation_identity_excludes_unrelated_package_code(monkeypatch):
    original = packing._sha256
    before = packing.preparation_source_identity()
    seen = []

    def changed(path):
        seen.append(path)
        return original(path)

    monkeypatch.setattr(packing, "_sha256", changed)
    assert packing.preparation_source_identity() == before
    assert all(path.parent.name in {"data", "config"} for path in seen)
    monkeypatch.setattr(packing, "_sha256", lambda path: "a" * 64)
    assert packing.preparation_source_identity() != before


def test_mixture_packing_resumes_committed_masks_and_targets(tmp_path, monkeypatch):
    import numpy as np

    from sparselab.data import preparation_chunks
    from sparselab.data.preparation_chunks import PreparationChunks

    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    receipt = mixture.materialize_mixture(declaration, tmp_path / "output")
    config = SimpleNamespace(
        dataset=SimpleNamespace(train_path=tmp_path / "output/train.tokens.jsonl"),
        training=SimpleNamespace(seq_len=8),
        model=SimpleNamespace(vocab_size=260),
    )
    tokenizer = SimpleNamespace(
        token_to_id=lambda name: {"<bos>": 2, "<eos>": 3, "<pad>": 0}[name]
    )
    expected_root = tmp_path / "expected"
    expected_root.mkdir()
    expected = packing._collect_token_mixture(config, tokenizer, receipt, expected_root)
    output = tmp_path / "packed"
    output.mkdir()
    kwargs = {
        "cache_identity": {"receipt": receipt},
        "binding": {"fixture": "mixture"},
        "arrays": {"ids": np.dtype(np.int32), "supervision": np.dtype(bool)},
        "record_limit": 1,
    }

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(preparation_chunks, "_after_sealed_chunk", interrupt)
    with (
        pytest.raises(KeyboardInterrupt),
        PreparationChunks(tmp_path / "chunks", **kwargs) as owner,
    ):
        packing._collect_token_mixture(
            config, tokenizer, receipt, output, chunks=owner.open_split("train")
        )
    monkeypatch.setattr(preparation_chunks, "_after_sealed_chunk", None)
    with PreparationChunks(tmp_path / "chunks", **kwargs) as owner:
        chunks = owner.open_split("train")
        assert chunks.state["acquired_documents"] == 1
        appended = []
        append = chunks.append_record

        def record(index, values):
            appended.append(index)
            return append(index, values)

        monkeypatch.setattr(chunks, "append_record", record)
        actual = packing._collect_token_mixture(
            config, tokenizer, receipt, output, chunks=chunks
        )
    assert appended and min(appended) == 2
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def test_mixture_cli_returns_artifact_paths_without_document_receipt(
    tmp_path, monkeypatch, capsys
):
    from sparselab.corpus.cli import _handle

    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    _handle(
        SimpleNamespace(
            work_dir=str(tmp_path),
            corpus_command="materialize-mixture",
            declaration=str(declaration),
            output=str(output),
        )
    )
    raw = capsys.readouterr().out
    result = json.loads(raw)
    assert len(raw) < 2048
    assert result["receipt_path"] == str(output / "receipt.json")
    assert result["actual_target_tokens"] == {
        "explanatory": 6,
        "general": 10,
        "incident": 4,
    }
    assert "excluded" not in result
    assert "excluded" in json.loads((output / "receipt.json").read_bytes())


def test_mixture_authority_change_requires_explicit_reencoding_audit(
    tmp_path, monkeypatch
):
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    original_output = {p.name: p.read_bytes() for p in output.iterdir()}
    original_sha = mixture.sha256_file
    monkeypatch.setattr(
        mixture,
        "sha256_file",
        lambda path: "f" * 64 if path.name == "mixture.py" else original_sha(path),
    )
    with pytest.raises(ValueError, match="explicit audit"):
        mixture.verify_mixture(declaration, output)
    original = mixture._tokenizer
    encoded = []

    def counted(*args):
        tok, digest = original(*args)
        encode = tok.encode

        def record(text, **kwargs):
            encoded.append(text)
            return encode(text, **kwargs)

        tok.encode = record
        return tok, digest

    monkeypatch.setattr(mixture, "_tokenizer", counted)
    assert mixture.verify_mixture(declaration, output, audit=True) == receipt
    assert encoded
    before = list(encoded)
    assert mixture.verify_mixture(declaration, output) == receipt
    assert encoded == before
    assert {p.name: p.read_bytes() for p in output.iterdir()} == original_output


def test_legacy_mixture_audit_recovers_without_replacing_output(tmp_path, monkeypatch):
    declaration, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    receipt = mixture.materialize_mixture(declaration, output)
    for path in (tmp_path / "cache/admission-inspection").glob("native-mixture-*.json"):
        path.unlink()
    before = {
        p.name: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
        for p in output.iterdir()
    }
    with pytest.raises(ValueError, match="explicit audit"):
        mixture.verify_mixture(declaration, output)
    assert mixture.verify_mixture(declaration, output, audit=True) == receipt
    assert mixture.verify_mixture(declaration, output) == receipt
    assert {
        p.name: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
        for p in output.iterdir()
    } == before
