"""Authenticated source-token measurement over frozen releases and bounded fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from sparselab.campaign.policy import CorpusReadinessPolicy
from sparselab.config.loading import load_tokenizer_config
from sparselab.corpus.acquisition import acquire
from sparselab.corpus.export import export_release, verify_release_export
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import load_project
from sparselab.corpus.release import freeze
from sparselab.corpus.token_denominator import (
    _measure_source_domains,
    measure_source_tokens,
    read_source_token_receipt,
)
from sparselab.training.manifest import canonical_json, sha256_file


def _tokenizer(folder: Path) -> Path:
    folder.mkdir(parents=True)
    path = folder / "tokenizer.json"
    model = Tokenizer(
        WordLevel({"[UNK]": 0, "<eos>": 1, "hello": 2, "world": 3}, unk_token="[UNK]")
    )
    model.pre_tokenizer = Whitespace()
    model.post_processor = TemplateProcessing(
        single="$A <eos>", special_tokens=[("<eos>", 1)]
    )
    model.save(str(path))
    (folder / "tokenizer_manifest.json").write_text(
        json.dumps(
            {
                "source": "local_text",
                "revision": "fixture",
                "vocab_size": 4,
                "sha256": sha256_file(path),
            }
        )
    )
    return path


@pytest.fixture
def frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "state"))
    recipe = tmp_path / "project"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "examples/tiny-campaign", recipe
    )
    project = load_project(recipe / "corpus.yaml")
    workspace = tmp_path / "workspace"
    acquire(project, workspace, offline=False)
    release = freeze(build(project, workspace, offline=True), workspace)
    tokenizer = _tokenizer(tmp_path / "tokenizer")
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "min_unique_train_tokens_by_domain": {
                    "developer": 1,
                    "technical_docs": 1,
                    "absent": 0,
                }
            }
        )
    )
    return release, tokenizer, policy


@pytest.fixture
def cross_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, Path]:
    """Two real local releases and a prebuilt tokenizer; no fitting or model work."""
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "state"))
    recipe = tmp_path / "project"
    shutil.copytree(Path("corpora/devmind-sample-v0"), recipe)
    project = load_project(recipe / "corpus.yaml")
    workspace = tmp_path / "workspace"
    acquire(project, workspace, offline=False)
    origin = freeze(build(project, workspace, offline=True), workspace)
    policy = tmp_path / "cross-policy.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "min_unique_train_tokens_by_domain": {
                    name: 1
                    for name in (
                        "technical_docs",
                        "code",
                        "configs",
                        "systems_scenarios",
                    )
                }
            }
        )
    )
    exported = export_release(
        origin, "lm", Path("configs/runtime_smoke_cpu.yaml"), 300, workspace
    )
    config = load_tokenizer_config(exported / "tokenizer.yaml")
    tokenizer = config.output_dir / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    vocab = {"[UNK]": 0, "<eos>": 1, "hello": 2, "world": 3}
    vocab.update({f"unused-{index}": index for index in range(4, 300)})
    model = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    model.pre_tokenizer = Whitespace()
    model.save(str(tokenizer))
    binding = verify_release_export(config.dataset)
    (tokenizer.parent / "tokenizer_manifest.json").write_bytes(
        canonical_json(
            {
                "source": "local_text",
                "revision": origin.name,
                "vocab_size": 300,
                "sha256": sha256_file(tokenizer),
                "corpus_export": binding,
                "training_contract": {"corpus_export": binding},
            }
        )
        + b"\n"
    )
    specification = yaml.safe_load((recipe / "release.yaml").read_text())
    specification["normalizer"] = "normalizer-structure-v3"
    (recipe / "release.yaml").write_text(yaml.safe_dump(specification))
    project = load_project(recipe / "corpus.yaml")
    target = freeze(build(project, workspace, offline=True), workspace)
    assert target != origin
    inventory = tmp_path / "target-families.jsonl"
    from sparselab.corpus.jsonl_records import records_from_path

    with inventory.open("wb") as stream:
        for record in records_from_path(target / "documents.jsonl"):
            doc = record.value
            if doc["drop_reason"] is not None:
                continue
            stream.write(
                canonical_json(
                    {
                        "document_id": doc["document_id"],
                        "family_id": f"fixture-family:{doc['document_id']}",
                        "split": doc["split"],
                        "stratum": doc["domains"][0],
                        "content_sha256": doc["content_sha256"],
                    }
                )
                + b"\n"
            )
    return target, origin, tokenizer, policy, inventory


def _synthetic(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    release = tmp_path / "fixture-release"
    release.mkdir(parents=True)
    raw = b"\n" + b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    documents = release / "documents.jsonl"
    documents.write_bytes(raw)
    (release / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "documents.jsonl": {
                        "sha256": sha256_file(documents),
                        "size": len(raw),
                    }
                }
            }
        )
    )
    return release


def _row(
    text: str, domains: list[str], *, split: str = "train", drop: str | None = None
) -> dict[str, object]:
    return {
        "split": split,
        "drop_reason": drop,
        "domains": domains,
        "text": text,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def test_domain_overlap_dedup_raw_scalar_and_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = _synthetic(
        tmp_path,
        [
            _row("hello world", ["a", "b"]),
            _row("hello world", ["a", "b"]),
            _row("world", ["a"]),
            _row("hello", ["other"]),
            _row("hello", ["a"], drop="duplicate"),
            _row("hello", ["b"], split="validation"),
        ],
    )
    tokenizer = _tokenizer(tmp_path / "tokenizer")
    policy = CorpusReadinessPolicy(min_unique_train_tokens_by_domain={"a": 1, "b": 1})
    original_read_text = Path.read_text

    def no_jsonl_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == "documents.jsonl":
            raise AssertionError("documents inventory must be streamed as bytes")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_jsonl_read_text)
    first = _measure_source_domains(
        release, tokenizer, policy, scratch=tmp_path / "scratch", batch_documents=1
    )
    second = _measure_source_domains(
        release, tokenizer, policy, scratch=tmp_path / "scratch", batch_documents=4
    )
    assert first["rows_scanned"] == 6
    assert first["documents_size"] == (release / "documents.jsonl").stat().st_size
    assert (
        first["domains"]
        == second["domains"]
        == {
            "a": {
                "eligible_documents": 3,
                "distinct_documents": 2,
                "source_bytes": 16,
                "source_tokens": 3,
            },
            "b": {
                "eligible_documents": 2,
                "distinct_documents": 1,
                "source_bytes": 11,
                "source_tokens": 2,
            },
        }
    )
    assert first["batch_count"] != second["batch_count"]
    model = Tokenizer.from_file(str(tokenizer))
    assert len(model.encode("hello world").ids) == 3
    assert first["domains"]["a"]["source_tokens"] == sum(
        len(model.encode(text, add_special_tokens=False).ids)
        for text in ("hello world", "world")
    )
    assert first["documents_sha256"] == sha256_file(release / "documents.jsonl")


@pytest.mark.parametrize(
    "padding,truncation", [(True, False), (False, True), (True, True)]
)
def test_raw_source_tokens_ignore_artifact_padding_and_truncation(
    tmp_path: Path, padding: bool, truncation: bool
) -> None:
    release = _synthetic(
        tmp_path, [_row("hello world world hello", ["a"]), _row("hello", ["a"])]
    )
    tokenizer = _tokenizer(tmp_path / "tokenizer")
    model = Tokenizer.from_file(str(tokenizer))
    if padding:
        model.enable_padding(length=128, pad_id=0, pad_token=model.id_to_token(0))
    if truncation:
        model.enable_truncation(max_length=2)
    model.save(str(tokenizer))
    result = _measure_source_domains(
        release,
        tokenizer,
        CorpusReadinessPolicy(min_unique_train_tokens_by_domain={"a": 1}),
        scratch=tmp_path / "scratch",
    )
    assert result["domains"]["a"]["source_tokens"] == 5


def test_streamed_source_must_match_manifest_even_with_blank_lines(
    tmp_path: Path,
) -> None:
    release = _synthetic(tmp_path, [_row("hello", ["a"])])
    policy = CorpusReadinessPolicy(min_unique_train_bytes_by_domain={"a": 1})
    documents = release / "documents.jsonl"
    with documents.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="streamed documents differ"):
        _measure_source_domains(
            release,
            None,
            policy,
            scratch=tmp_path / "scratch",
        )


def test_mixed_readiness_all_domain_bytes_and_large_scalar(tmp_path: Path) -> None:
    text = "a" * (1_048_576 + 1)
    release = _synthetic(tmp_path, [_row(text, ["a", "b"])])
    tokenizer = _tokenizer(tmp_path / "tokenizer")
    policy = CorpusReadinessPolicy(min_heldout_families=1)
    result = _measure_source_domains(
        release,
        tokenizer,
        policy,
        scratch=tmp_path / "scratch",
        all_domains=True,
        max_document_source_bytes=None,
    )
    assert set(result["domains"]) == {"a", "b"}
    for domain in ("a", "b"):
        assert result["domains"][domain] == {
            "eligible_documents": 1,
            "distinct_documents": 1,
            "source_bytes": len(text),
            "source_tokens": 1,
        }
    assert result["batch_count"] == 1


def test_strict_single_document_and_batch_limits(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path / "tokenizer")
    policy = CorpusReadinessPolicy(min_unique_train_tokens_by_domain={"a": 1})
    release = _synthetic(tmp_path, [_row("a" * (1_048_576 + 1), ["a"])])
    with pytest.raises(ValueError, match="exceeds 1 MiB"):
        _measure_source_domains(
            release,
            tokenizer,
            policy,
            scratch=tmp_path / "scratch",
            batch_source_bytes=4_194_304,
        )
    small = _synthetic(tmp_path / "small", [_row("hello world " * 6, ["a"])])
    with pytest.raises(ValueError, match="batch byte limit"):
        _measure_source_domains(
            small,
            tokenizer,
            policy,
            scratch=tmp_path / "scratch",
            batch_source_bytes=32,
        )
    with pytest.raises(ValueError, match="tokenizer_batch_documents"):
        _measure_source_domains(release, tokenizer, policy, batch_documents=0)


def test_explicit_four_mib_document_bound_counts_one_large_document(
    tmp_path: Path,
) -> None:
    tokenizer = _tokenizer(tmp_path / "tokenizer")
    policy = CorpusReadinessPolicy(min_unique_train_tokens_by_domain={"a": 1})
    release = _synthetic(tmp_path, [_row("hello " * 200_000, ["a"])])
    result = _measure_source_domains(
        release,
        tokenizer,
        policy,
        scratch=tmp_path / "scratch",
        batch_source_bytes=4_194_304,
        max_document_source_bytes=4_194_304,
    )
    assert result["domains"]["a"]["source_tokens"] == 200_000


def test_public_api_requires_token_domain_policy(
    frozen: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    release, tokenizer, _ = frozen
    policy = tmp_path / "bytes-only.yaml"
    policy.write_text("min_unique_train_bytes_by_domain:\n  developer: 1\n")
    output = tmp_path / "no-token-policy.json"
    with pytest.raises(ValueError, match="token-domain"):
        measure_source_tokens(release, tokenizer, policy, output)
    assert not output.exists()


def test_checked_in_measurement_policy_authenticates_below_100m_supply(
    frozen: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    release, tokenizer, _ = frozen
    policy = (
        Path(__file__).resolve().parents[1]
        / "experiments/research/kernel-memory-lab/card05-base-50m/current/measurement-domains.yaml"
    )
    output = tmp_path / "measured-supply.json"
    receipt = measure_source_tokens(
        release, tokenizer, policy, output, batch_documents=1
    )
    assert receipt["status"] == "COMPLETE"
    assert receipt["policy_sha256"] == sha256_file(policy)
    assert set(receipt["domains"]) == {
        "general_prose",
        "explanatory_prose",
        "incident_response_docs",
    }
    comparison_floors = {
        "general_prose": 40_500_000,
        "explanatory_prose": 9_000_000,
        "incident_response_docs": 500_000,
    }
    assert all(
        receipt["domains"][domain]["source_tokens"] < floor
        for domain, floor in comparison_floors.items()
    )
    assert read_source_token_receipt(output, release, tokenizer, policy) == receipt


def test_complete_receipt_reuse_and_mutation_refusal(
    frozen: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, tokenizer, policy = frozen
    output = tmp_path / "denominator.json"
    receipt = measure_source_tokens(
        release, tokenizer, policy, output, batch_documents=1
    )
    assert receipt["status"] == "COMPLETE"
    assert receipt["evidence"] is None
    assert "tokenizer_origin_release_id" not in receipt
    assert "family_inventory_sha256" not in receipt
    assert receipt["release_id"] == release.name
    assert receipt["documents_sha256"] == sha256_file(release / "documents.jsonl")
    assert receipt["tokenizer_sha256"] == sha256_file(tokenizer)
    assert receipt["policy_sha256"] == sha256_file(policy)
    assert receipt["domains"]["absent"] == {
        "eligible_documents": 0,
        "distinct_documents": 0,
        "source_bytes": 0,
        "source_tokens": 0,
    }
    assert read_source_token_receipt(output, release, tokenizer, policy) == receipt
    import sparselab.corpus.token_denominator as denominator

    def no_encoding(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("read-only reuse must not load or encode tokenizer")

    monkeypatch.setattr(denominator, "load_tokenizer", no_encoding)
    assert measure_source_tokens(release, tokenizer, policy, output) == receipt
    for name in ("policy", "tokenizer", "release"):
        if name == "policy":
            other = tmp_path / "changed-policy.yaml"
            other.write_text(policy.read_text() + "\n")
            args = (release, tokenizer, other)
        elif name == "tokenizer":
            other = _tokenizer(tmp_path / "changed-tokenizer")
            other.write_bytes(other.read_bytes() + b" ")
            manifest_path = other.with_name("tokenizer_manifest.json")
            metadata = json.loads(manifest_path.read_bytes())
            metadata["sha256"] = sha256_file(other)
            manifest_path.write_text(json.dumps(metadata))
            args = (release, other, policy)
        else:
            other = tmp_path / "changed-release" / release.name
            shutil.copytree(release, other)
            with (other / "documents.jsonl").open("ab") as stream:
                stream.write(b"\n")
            args = (other, tokenizer, policy)
        with pytest.raises((ValueError, OSError)):
            read_source_token_receipt(output, *args)
    altered = json.loads(output.read_bytes())
    altered["implementation_sha256"] = "0" * 64
    changed = tmp_path / "changed-receipt.json"
    changed.write_bytes(canonical_json(altered) + b"\n")
    with pytest.raises(ValueError, match="implementation binding"):
        read_source_token_receipt(changed, release, tokenizer, policy)
    altered = json.loads(output.read_bytes())
    altered["domains"]["developer"]["source_tokens"] += 1
    changed.write_bytes(canonical_json(altered) + b"\n")
    with pytest.raises(ValueError, match="scientific identity"):
        read_source_token_receipt(changed, release, tokenizer, policy)
    output.write_bytes(b'{"status":"COMPLETE"')
    with pytest.raises(ValueError):
        measure_source_tokens(release, tokenizer, policy, output)
    assert output.read_bytes() == b'{"status":"COMPLETE"'


def test_cross_release_origin_inventory_and_cli_reuse(
    cross_release: tuple[Path, Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, origin, tokenizer, policy, inventory = cross_release
    output = tmp_path / "cross.json"
    command = [
        sys.executable,
        "-m",
        "sparselab",
        "corpus",
        "measure-tokens",
        str(target),
        "--tokenizer",
        str(tokenizer),
        "--tokenizer-origin-release",
        str(origin),
        "--family-inventory",
        str(inventory),
        "--policy",
        str(policy),
        "--output",
        str(output),
        "--json",
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=120
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt == read_source_token_receipt(
        output,
        target,
        tokenizer,
        policy,
        tokenizer_origin_release=origin,
        family_inventory=inventory,
    )
    assert receipt["release_id"] == target.name
    assert receipt["tokenizer_origin_release_id"] == origin.name
    assert receipt["family_inventory_sha256"] == sha256_file(inventory)
    assert receipt["tokenizer_sha256"] == sha256_file(tokenizer)
    assert receipt["tokenizer_config_sha256"] == sha256_file(
        tokenizer.parent.parent / "tokenizer.yaml"
    )
    assert receipt["documents_sha256"] == sha256_file(target / "documents.jsonl")
    assert receipt["family_inventory_rows"] > 0
    assert receipt["evidence"] is None
    import sparselab.corpus.token_denominator as denominator

    def no_scan(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("verified cached receipt must not encode content")

    monkeypatch.setattr(denominator, "_measure_source_domains", no_scan)
    assert (
        measure_source_tokens(
            target,
            tokenizer,
            policy,
            output,
            tokenizer_origin_release=origin,
            family_inventory=inventory,
        )
        == receipt
    )
    with pytest.raises(ValueError, match="requires its tokenizer origin"):
        read_source_token_receipt(output, target, tokenizer, policy)


def test_cross_release_rejects_wrong_origin_tokenizer_and_inventory(
    cross_release: tuple[Path, Path, Path, Path, Path], tmp_path: Path
) -> None:
    target, origin, tokenizer, policy, inventory = cross_release
    output = tmp_path / "cross.json"
    receipt = measure_source_tokens(
        target,
        tokenizer,
        policy,
        output,
        tokenizer_origin_release=origin,
        family_inventory=inventory,
    )
    assert receipt["status"] == "COMPLETE"
    with pytest.raises(ValueError, match="origin"):
        read_source_token_receipt(
            output,
            target,
            tokenizer,
            policy,
            tokenizer_origin_release=target,
            family_inventory=inventory,
        )
    with pytest.raises(ValueError, match="declared together"):
        read_source_token_receipt(
            output,
            target,
            tokenizer,
            policy,
            tokenizer_origin_release=origin,
        )
    with pytest.raises(TypeError, match="evidence_commit"):
        measure_source_tokens(
            target,
            tokenizer,
            policy,
            tmp_path / "untrusted-reuse.json",
            tokenizer_origin_release=origin,
            family_inventory=inventory,
            evidence_commit="0" * 40,
            release_evidence=tmp_path / "release-evidence.json",
            selection_evidence=tmp_path / "selection-evidence.json",
        )
    wrong = _tokenizer(tmp_path / "unbound-tokenizer")
    with pytest.raises(ValueError, match="origin|tokenizer"):
        read_source_token_receipt(
            output,
            target,
            wrong,
            policy,
            tokenizer_origin_release=origin,
            family_inventory=inventory,
        )
    changed = tmp_path / "changed-families.jsonl"
    changed.write_bytes(
        inventory.read_bytes().replace(b"fixture-family:", b"substituted-family:")
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        read_source_token_receipt(
            output,
            target,
            tokenizer,
            policy,
            tokenizer_origin_release=origin,
            family_inventory=changed,
        )
    omitted = tmp_path / "omitted-families.jsonl"
    omitted.write_bytes(b"".join(inventory.read_bytes().splitlines(keepends=True)[1:]))
    with pytest.raises(ValueError, match="omits"):
        read_source_token_receipt(
            output,
            target,
            tokenizer,
            policy,
            tokenizer_origin_release=origin,
            family_inventory=omitted,
        )
    rows = [json.loads(line) for line in inventory.read_bytes().splitlines()]
    train_family = next(row["family_id"] for row in rows if row["split"] == "train")
    heldout = next(row for row in rows if row["split"] != "train")
    heldout["family_id"] = train_family
    leaked = tmp_path / "leaked-families.jsonl"
    leaked.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match="leaks across splits"):
        measure_source_tokens(
            target,
            tokenizer,
            policy,
            tmp_path / "leaked-output.json",
            tokenizer_origin_release=origin,
            family_inventory=leaked,
        )
    assert not (tmp_path / "leaked-output.json").exists()


@pytest.mark.parametrize("changed", ["target", "origin", "tokenizer"])
def test_cross_release_cold_read_rejects_altered_release(
    cross_release: tuple[Path, Path, Path, Path, Path],
    tmp_path: Path,
    changed: str,
) -> None:
    target, origin, tokenizer, policy, inventory = cross_release
    output = tmp_path / "cross.json"
    measure_source_tokens(
        target,
        tokenizer,
        policy,
        output,
        tokenizer_origin_release=origin,
        family_inventory=inventory,
    )
    paths = {
        "target": target / "documents.jsonl",
        "origin": origin / "documents.jsonl",
        "tokenizer": tokenizer,
    }
    with paths[changed].open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError):
        read_source_token_receipt(
            output,
            target,
            tokenizer,
            policy,
            tokenizer_origin_release=origin,
            family_inventory=inventory,
        )


@pytest.mark.parametrize("changed", ["target", "origin", "inventory", "config"])
def test_cross_release_scan_stability(
    cross_release: tuple[Path, Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    import sparselab.corpus.token_denominator as denominator

    target, origin, tokenizer, policy, inventory = cross_release
    original = denominator._measure_source_domains
    paths = {
        "target": target / "documents.jsonl",
        "origin": origin / "documents.jsonl",
        "inventory": inventory,
        "config": tokenizer.parent.parent / "tokenizer.yaml",
    }

    def change_input(*args: object, **kwargs: object) -> dict:
        measured = original(*args, **kwargs)
        with paths[changed].open("ab") as stream:
            stream.write(b"\n")
        return measured

    monkeypatch.setattr(denominator, "_measure_source_domains", change_input)
    output = tmp_path / "unstable.json"
    with pytest.raises(ValueError, match="changed during source scan"):
        measure_source_tokens(
            target,
            tokenizer,
            policy,
            output,
            tokenizer_origin_release=origin,
            family_inventory=inventory,
        )
    assert not output.exists()
    assert len(list(tmp_path.glob(".unstable.json.*.partial"))) == 1


def test_batch_independent_scientific_identity_and_partial(
    frozen: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    release, tokenizer, policy = frozen
    first = measure_source_tokens(
        release, tokenizer, policy, tmp_path / "first.json", batch_documents=1
    )
    second = measure_source_tokens(
        release, tokenizer, policy, tmp_path / "second.json", batch_documents=16
    )
    from sparselab.corpus.release import _iter_rows

    largest_source = max(
        len(row["text"].encode("utf-8"))
        for row in _iter_rows(release / "documents.jsonl")
        if row["split"] == "train" and row["drop_reason"] is None
    )
    byte_bounded = measure_source_tokens(
        release,
        tokenizer,
        policy,
        tmp_path / "byte-bounded.json",
        batch_documents=16,
        batch_source_bytes=largest_source,
    )
    assert byte_bounded["scientific_sha256"] == second["scientific_sha256"]
    assert (
        byte_bounded["operational"]["batch_count"]
        > second["operational"]["batch_count"]
    )
    assert first["scientific_sha256"] == second["scientific_sha256"]
    assert first["operational"]["batch_count"] != second["operational"]["batch_count"]
    output = tmp_path / "interrupted.json"
    output.write_text('{"measurement_version":1,"status":"INTERRUPTED"}')
    with pytest.raises(ValueError, match="not complete"):
        read_source_token_receipt(output, release, tokenizer, policy)


def test_policy_preserves_exact_authored_weights_and_rejects_duplicate_keys(
    frozen: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    release, tokenizer, policy = frozen
    policy.write_text(
        "passes:\n  basis: tokens\n  requested_total: 2\n  max_required: 1\n"
        "  mixture:\n    developer: 0.50000000000000001\n"
        "    technical_docs: 0.49999999999999999\n"
    )
    receipt = measure_source_tokens(release, tokenizer, policy, tmp_path / "exact.json")
    assert receipt["policy"]["passes"]["mixture"] == {
        "developer": "0.50000000000000001",
        "technical_docs": "0.49999999999999999",
    }
    policy.write_text(
        "min_unique_train_tokens_by_domain:\n  developer: 1\n  developer: 999\n"
    )
    output = tmp_path / "duplicate.json"
    with pytest.raises(ValueError, match="duplicate YAML key"):
        measure_source_tokens(release, tokenizer, policy, output)
    assert not output.exists()


def test_changed_policy_during_scan_cannot_publish_complete(
    frozen: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sparselab.corpus.token_denominator as denominator

    release, tokenizer, policy = frozen
    original = denominator._measure_source_domains

    def change_policy(*args: object, **kwargs: object) -> dict:
        measured = original(*args, **kwargs)
        with policy.open("ab") as stream:
            stream.write(b"\n")
        return measured

    monkeypatch.setattr(denominator, "_measure_source_domains", change_policy)
    output = tmp_path / "changed-during-scan.json"
    with pytest.raises(ValueError, match="input changed during source scan"):
        measure_source_tokens(release, tokenizer, policy, output)
    assert not output.exists()
    partial = next(tmp_path.glob(".changed-during-scan.json.*.partial"))
    assert json.loads(partial.read_bytes())["status"] == "INTERRUPTED"


def test_cli_json_over_authenticated_frozen_release(
    frozen: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    release, tokenizer, policy = frozen
    output = tmp_path / "cli-receipt.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "sparselab",
            "--work-dir",
            str(tmp_path / "state"),
            "corpus",
            "measure-tokens",
            str(release),
            "--tokenizer",
            str(tokenizer),
            "--policy",
            str(policy),
            "--output",
            str(output),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    printed = json.loads(process.stdout)
    assert printed == read_source_token_receipt(output, release, tokenizer, policy)
    assert printed["status"] == "COMPLETE"
    assert "STARTED" in process.stderr and "COMPLETE" in process.stderr


@pytest.mark.parametrize("name", ["release-link", "release@link"])
def test_cli_rejects_symlinked_literal_release_paths(
    frozen: tuple[Path, Path, Path], tmp_path: Path, name: str
) -> None:
    release, tokenizer, policy = frozen
    linked = tmp_path / name
    linked.symlink_to(release, target_is_directory=True)
    output = tmp_path / "rejected.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "sparselab",
            "corpus",
            "measure-tokens",
            str(linked),
            "--tokenizer",
            str(tokenizer),
            "--policy",
            str(policy),
            "--output",
            str(output),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert process.returncode != 0
    assert not output.exists()


def test_cli_reference_rejects_shadowing_dangling_symlink(
    frozen: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    release, tokenizer, policy = frozen
    reference = f"tiny-campaign@{release.name}"
    command = [
        sys.executable,
        "-m",
        "sparselab",
        "--work-dir",
        str(release.parents[3]),
        "corpus",
        "measure-tokens",
        reference,
        "--tokenizer",
        str(tokenizer),
        "--policy",
        str(policy),
        "--json",
    ]
    accepted = tmp_path / "reference.json"
    process = subprocess.run(
        [*command, "--output", str(accepted)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    receipt = json.loads(process.stdout)
    assert receipt["status"] == "COMPLETE"
    assert receipt["release_id"] == release.name

    (tmp_path / reference).symlink_to(
        tmp_path / "missing-release", target_is_directory=True
    )
    rejected = tmp_path / "rejected-reference.json"
    process = subprocess.run(
        [*command, "--output", str(rejected)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert process.returncode != 0
    assert not rejected.exists()


def test_interrupted_scan_retains_marker_and_progress(
    frozen: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sparselab.corpus.token_denominator as denominator

    release, tokenizer, policy = frozen
    output = tmp_path / "failed.json"

    def interrupted(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("tokenizer interrupted")

    monkeypatch.setattr(denominator, "_measure_source_domains", interrupted)
    with pytest.raises(RuntimeError, match="tokenizer interrupted"):
        measure_source_tokens(release, tokenizer, policy, output)
    assert not output.exists()
    markers = list(tmp_path.glob(".failed.json.*.partial"))
    assert len(markers) == 1
    assert json.loads(markers[0].read_bytes())["status"] == "INTERRUPTED"
    log = next(tmp_path.glob(".failed.json.*.progress.jsonl"))
    events = [json.loads(line) for line in log.read_bytes().splitlines()]
    assert [event["status"] for event in events] == ["STARTED", "INTERRUPTED"]
    assert events[0]["launch"] == events[1]["launch"]


@pytest.mark.parametrize("failure_site", ["directory", "completion_progress"])
def test_publication_failure_never_leaves_complete_result(
    frozen: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    release, tokenizer, policy = frozen
    output = tmp_path / "publication-failed.json"
    real_fsync = os.fsync
    directory_synced = False
    injected = False

    def fail_publication(fd: int) -> None:
        nonlocal directory_synced, injected
        is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        if not injected and (
            (failure_site == "directory" and is_directory)
            or (
                failure_site == "completion_progress"
                and directory_synced
                and not is_directory
            )
        ):
            injected = True
            raise OSError("injected publication failure")
        real_fsync(fd)
        if is_directory:
            directory_synced = True

    monkeypatch.setattr(os, "fsync", fail_publication)
    with pytest.raises(OSError, match="injected publication failure"):
        measure_source_tokens(release, tokenizer, policy, output)
    assert not output.exists()
    markers = list(tmp_path.glob(".publication-failed.json.*.partial"))
    assert len(markers) == 1
    assert json.loads(markers[0].read_bytes())["status"] == "INTERRUPTED"
    log = next(tmp_path.glob(".publication-failed.json.*.progress.jsonl"))
    events = [json.loads(line) for line in log.read_bytes().splitlines()]
    assert events[-1]["status"] == "INTERRUPTED"
    assert events[-1]["rows_scanned"] == 12


def test_interrupt_after_link_rolls_back_only_its_own_publication(
    frozen: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, tokenizer, policy = frozen
    output = tmp_path / "post-link-interrupted.json"
    real_link = os.link

    def interrupted_link(source: Path, destination: Path) -> None:
        real_link(source, destination)
        raise KeyboardInterrupt("injected post-link interruption")

    monkeypatch.setattr(os, "link", interrupted_link)
    with pytest.raises(KeyboardInterrupt):
        measure_source_tokens(release, tokenizer, policy, output)
    assert not output.exists()
    markers = list(tmp_path.glob(".post-link-interrupted.json.*.partial"))
    assert len(markers) == 1
    assert json.loads(markers[0].read_bytes())["status"] == "INTERRUPTED"


def test_competing_publisher_is_never_removed(
    frozen: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, tokenizer, policy = frozen
    output = tmp_path / "competing.json"
    competing_bytes = b"owned by another publisher\n"

    def competing_link(_source: Path, destination: Path) -> None:
        destination.write_bytes(competing_bytes)
        raise FileExistsError(destination)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(FileExistsError):
        measure_source_tokens(release, tokenizer, policy, output)
    assert output.read_bytes() == competing_bytes


def test_source_output_conflict_and_symlink(
    frozen: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    release, tokenizer, policy = frozen
    with pytest.raises(ValueError, match="conflict"):
        measure_source_tokens(release, tokenizer, policy, policy)
    linked = tmp_path / "linked.json"
    linked.symlink_to(tmp_path / "real.json")
    with pytest.raises(ValueError, match="symlink"):
        measure_source_tokens(release, tokenizer, policy, linked)


def test_unrelated_evidence_sha_never_bypasses_full_verification(
    frozen: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    release, tokenizer, policy = frozen
    output = tmp_path / "untrusted.json"
    with pytest.raises(TypeError, match="evidence_commit"):
        measure_source_tokens(
            release,
            tokenizer,
            policy,
            output,
            evidence_commit="0" * 40,
            release_evidence=tmp_path / "claimed-primary.json",
            selection_evidence=tmp_path / "claimed-selection.json",
        )
    assert not output.exists()


def test_large_stream_stays_below_512_mib(tmp_path: Path) -> None:
    # Isolated subprocess: build 256 MiB without accumulating rows, then measure
    # with exactly the same streaming primitive used by the authenticated API.
    script = r"""
import hashlib, json, resource, sys
from pathlib import Path
from sparselab.campaign.policy import CorpusReadinessPolicy
from sparselab.corpus.token_denominator import _measure_source_domains
root = Path(sys.argv[1]); root.mkdir()
documents = root / "documents.jsonl"
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
tokenizer = Tokenizer(WordLevel({"[UNK]":0}, unk_token="[UNK]"))
tokenizer_path = root / "tokenizer.json"; tokenizer.save(str(tokenizer_path))
hash_value = hashlib.sha256(); size = 0; rows = 0
with documents.open("wb") as stream:
    while size < 256 * 1024 * 1024:
        text = "x" * 3900 + str(rows)
        row = json.dumps({"split":"train", "drop_reason":None, "domains":["developer"],
            "content_sha256":hashlib.sha256(text.encode()).hexdigest(),
            "text":text}).encode() + b"\n"
        stream.write(row); hash_value.update(row); size += len(row); rows += 1
(root / "manifest.json").write_text(json.dumps({"files":{"documents.jsonl":{
    "sha256":hash_value.hexdigest(), "size":size}}}))
result = _measure_source_domains(root, tokenizer_path,
    CorpusReadinessPolicy(min_unique_train_tokens_by_domain={"developer":1}),
    scratch=root.parent / "scratch")
assert result["rows_scanned"] == rows
assert result["documents_sha256"] == hash_value.hexdigest()
assert result["domains"]["developer"]["distinct_documents"] == rows
assert result["domains"]["developer"]["eligible_documents"] == rows
assert result["domains"]["developer"]["source_tokens"] == rows
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_bytes = peak if sys.platform == "darwin" else peak * 1024
print(json.dumps({"peak_rss_bytes":peak_bytes, "source_size":size, "rows":rows}))
"""
    # Linux ru_maxrss retains the pre-exec pytest image's high-water mark.
    # A small fresh interpreter forks the measured image, not the test runner.
    launcher = (
        "import subprocess,sys; "
        "raise SystemExit(subprocess.run("
        "[sys.executable,'-c',sys.argv[1],sys.argv[2]],check=False).returncode)"
    )
    env = dict(os.environ, SPARSELAB_WORK_DIR=str(tmp_path / "state"))
    result = subprocess.run(
        [sys.executable, "-c", launcher, script, str(tmp_path / "large")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr
    measurements = json.loads(result.stdout)
    assert measurements["peak_rss_bytes"] < 512 * 1024 * 1024, measurements
    print(json.dumps(measurements, sort_keys=True))


def test_receipt_cannot_reactivate_committed_evidence_trust(
    frozen: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    release, tokenizer, policy = frozen
    output = tmp_path / "receipt.json"
    receipt = measure_source_tokens(release, tokenizer, policy, output)
    receipt["evidence"] = {"commit": "0" * 40}
    output.write_bytes(canonical_json(receipt) + b"\n")
    with pytest.raises(ValueError, match="cold input authentication"):
        read_source_token_receipt(output, release, tokenizer, policy)
