"""Campaign readiness over real, independently frozen local corpus releases."""

from __future__ import annotations

import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from sparselab.campaign.plan import load_campaign
from sparselab.campaign.policy import CorpusReadinessPolicy, measure_readiness
from sparselab.campaign.state import CampaignStore
from sparselab.corpus.acquisition import acquire
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import load_project
from sparselab.corpus.release import _verification_operation, freeze, verify_release
from sparselab.corpus.token_denominator import measure_source_tokens
from sparselab.training.manifest import sha256_file


@pytest.fixture
def local_recipe(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "examples/tiny-campaign"
    destination = tmp_path / "tiny-campaign"
    shutil.copytree(source, destination)
    return destination


def _release(recipe: Path, work: Path) -> Path:
    project = load_project(recipe / "corpus.yaml")
    acquire(project, work, offline=False)
    released = freeze(build(project, work, offline=True), work)
    verify_release(released)
    return released


def _declaration(
    recipe: Path, weights: tuple[str, str], total: int, suffix: str
) -> Path:
    payload = {
        "campaign_version": 1,
        "id": "exact-source-passes",
        "stages": [
            {
                "id": "corpus",
                "kind": "corpus_release",
                "scope": "corpus",
                "project": "tiny-campaign/corpus.yaml",
            },
            {
                "id": "ready",
                "kind": "corpus_readiness",
                "scope": "corpus",
                "requires": ["corpus"],
                "corpus": "corpus",
                "policy": {
                    "passes": {
                        "basis": "bytes",
                        "requested_total": total,
                        "mixture": {"developer": 0.5, "technical_docs": 0.5},
                        "max_required": 1,
                    }
                },
            },
        ],
    }
    path = recipe.parent / f"campaign.{suffix}"
    if suffix == "json":
        text = json.dumps(payload, separators=(",", ":"))
        text = text.replace(
            '"developer":0.5,"technical_docs":0.5',
            f'"developer":{weights[0]},"technical_docs":{weights[1]}',
        )
    else:
        text = yaml.safe_dump(payload)
        text = text.replace("developer: 0.5", f"developer: {weights[0]}")
        text = text.replace("technical_docs: 0.5", f"technical_docs: {weights[1]}")
    path.write_text(text)
    return path


@pytest.mark.parametrize("suffix", ["yaml", "json"])
def test_exact_authored_weights_change_verified_release_decision(
    local_recipe: Path, tmp_path: Path, suffix: str
) -> None:
    release = _release(local_recipe, tmp_path / "work")
    amounts = measure_readiness(release, CorpusReadinessPolicy(min_heldout_families=1))[
        "measurements"
    ]["unique_train_bytes_by_domain"]
    # Both 0.5 weights fit a single pass at this threshold. The extra 1e-17
    # of developer weight needs a second pass exactly at the integer boundary.
    assert amounts["developer"] > 0 and amounts["technical_docs"] > 0
    total = 2 * amounts["developer"]
    if amounts["developer"] > amounts["technical_docs"]:
        # Make the smaller domain the weighted boundary regardless of recipe size.
        total = 2 * amounts["technical_docs"]
        upper_domain = "technical_docs"
        weights = ("0.49999999999999999", "0.50000000000000001")
    else:
        upper_domain = "developer"
        weights = ("0.50000000000000001", "0.49999999999999999")
    baseline = load_campaign(_declaration(local_recipe, ("0.5", "0.5"), total, suffix))
    baseline_policy = baseline.stages[1].policy
    assert measure_readiness(release, baseline_policy)["state"] == "COMPLETE"
    baseline_sha = CampaignStore(baseline, tmp_path / "work").declaration_sha

    plan = load_campaign(_declaration(local_recipe, weights, total, suffix))
    mixture = plan.stages[1].policy.passes.mixture
    assert mixture[upper_domain] == Decimal("0.50000000000000001")
    assert CampaignStore(plan, tmp_path / "work").declaration_sha != baseline_sha
    decision = measure_readiness(release, plan.stages[1].policy)
    assert decision["state"] == "BLOCKED"
    assert decision["outcome"] == "EXPAND_MORE"
    assert (
        decision["measurements"]["projected_source_passes_by_domain"][upper_domain] == 2
    )
    assert {item["domain"] for item in decision["deficits"]} == {upper_domain}


@pytest.mark.parametrize("suffix", ["yaml", "json"])
def test_reject_authored_string_and_nonfinite_weights(
    tmp_path: Path, suffix: str
) -> None:
    for bad in ('"0.5"', '"NaN"', "1e9999"):
        path = _declaration(tmp_path, (bad, "0.5"), 2, suffix)
        with pytest.raises((TypeError, ValueError)):
            load_campaign(path)


def test_validation_only_selected_view_does_not_satisfy_train_shape(
    local_recipe: Path, tmp_path: Path
) -> None:
    selection_path = local_recipe / "release.yaml"
    selection = yaml.safe_load(selection_path.read_text())
    selection["lm"]["training_splits"] = ["validation"]
    selection_path.write_text(yaml.safe_dump(selection))
    split_path = local_recipe / "splits.yaml"
    splits = yaml.safe_load(split_path.read_text())
    splits["assignments"]["tiny_test_family"] = "validation"
    split_path.write_text(yaml.safe_dump(splits))
    release = _release(local_recipe, tmp_path / "work")
    assert verify_release(release)["build_identity"]["release"]["lm"]["selected"]
    assert (release / "lm/train.lineage.jsonl").read_text().strip()
    decision = measure_readiness(
        release, CorpusReadinessPolicy(required_nonzero_shapes=("raw_document",))
    )
    assert decision["state"] == "BLOCKED"
    assert decision["outcome"] == "EXPAND_MORE"
    assert decision["measurements"]["selected_train_shapes"] == {}
    assert decision["deficits"] == [
        {
            "dimension": "train_shape",
            "observed": 0,
            "required": 1,
            "domain": "raw_document",
        }
    ]


def _fixture_tokenizer(folder: Path) -> Path:
    folder.mkdir()
    path = folder / "tokenizer.json"
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))
    (folder / "tokenizer_manifest.json").write_text(
        json.dumps(
            {
                "source": "local_text",
                "revision": "readiness-fixture",
                "vocab_size": 2,
                "sha256": sha256_file(path),
            }
        )
    )
    return path


def test_token_only_receipt_uses_authenticated_counts_without_encoding(
    local_recipe: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "state"))
    release = _release(local_recipe, tmp_path / "work")
    tokenizer = _fixture_tokenizer(tmp_path / "tokenizer")
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(
        yaml.safe_dump(
            {"min_unique_train_tokens_by_domain": {"developer": 1, "absent": 0}}
        )
    )
    policy = CorpusReadinessPolicy.model_validate(
        yaml.safe_load(policy_file.read_bytes())
    )
    output = tmp_path / "tokens.json"
    receipt = measure_source_tokens(release, tokenizer, policy_file, output)
    import sparselab.corpus.token_denominator as denominator

    def no_encoding(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("receipt readiness must not tokenize")

    monkeypatch.setattr(denominator, "load_tokenizer", no_encoding)
    original_read_text = Path.read_text

    def no_corpus_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.name in {"documents.jsonl", "lineage.jsonl"}:
            raise AssertionError("corpus ledgers must be streamed")
        return original_read_text(path, *args, **kwargs)

    result = measure_readiness(
        release,
        policy,
        tokenizer,
        measurement_receipt=output,
        measurement_sha256=sha256_file(output),
    )
    facts = result["measurements"]
    assert facts["unique_train_tokens_by_domain"] == {
        domain: counts["source_tokens"] for domain, counts in receipt["domains"].items()
    }
    assert facts["train_languages"] is None
    assert facts["selected_train_shapes"] is None
    assert facts["heldout_families"] is None
    assert facts["unique_train_bytes_by_domain"]["absent"] == 0
    assert result["state"] == "COMPLETE"
    import sparselab.campaign.policy as readiness_module

    with _verification_operation():
        verify_release(release)
        monkeypatch.setattr(Path, "read_text", no_corpus_read_text)

        def no_lineage(path: Path):
            if "lineage" in path.name:
                raise AssertionError("token-only readiness must not scan lineage")
            return iter(())

        monkeypatch.setattr(readiness_module, "_iter_rows", no_lineage)
        assert (
            measure_readiness(
                release,
                policy,
                tokenizer,
                measurement_receipt=output,
                measurement_sha256=sha256_file(output),
            )["measurements"]
            == facts
        )
        absent_tokenizer = measure_readiness(release, policy)
        assert absent_tokenizer["state"] == "BLOCKED"
        assert absent_tokenizer["measurements"]["unique_train_tokens_by_domain"] is None
        assert absent_tokenizer["measurements"]["train_languages"] is None
    monkeypatch.setattr(Path, "read_text", original_read_text)

    with pytest.raises(ValueError, match="together"):
        measure_readiness(release, policy, tokenizer, measurement_receipt=output)
    with pytest.raises(ValueError, match="SHA-256"):
        measure_readiness(
            release,
            policy,
            tokenizer,
            measurement_receipt=output,
            measurement_sha256="0" * 64,
        )
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_bytes(b'{"status":"COMPLETE"')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        measure_readiness(
            release,
            policy,
            tokenizer,
            measurement_receipt=incomplete,
            measurement_sha256=sha256_file(incomplete),
        )
    altered = policy.model_copy(
        update={"min_unique_train_tokens_by_domain": {"developer": 100000}}
    )
    with pytest.raises(ValueError, match="inline policy"):
        measure_readiness(
            release,
            altered,
            tokenizer,
            measurement_receipt=output,
            measurement_sha256=sha256_file(output),
        )
    altered_tokenizer = _fixture_tokenizer(tmp_path / "different-tokenizer")
    altered_tokenizer.write_bytes(altered_tokenizer.read_bytes() + b" ")
    with pytest.raises(ValueError):
        measure_readiness(
            release,
            policy,
            altered_tokenizer,
            measurement_receipt=output,
            measurement_sha256=sha256_file(output),
        )
    altered_release = tmp_path / "different-release" / release.name
    shutil.copytree(release, altered_release)
    with (altered_release / "documents.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError):
        measure_readiness(
            altered_release,
            policy,
            tokenizer,
            measurement_receipt=output,
            measurement_sha256=sha256_file(output),
        )
    policy_file.write_bytes(policy_file.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="binding mismatch"):
        measure_readiness(
            release,
            policy,
            tokenizer,
            measurement_receipt=output,
            measurement_sha256=sha256_file(output),
        )


def test_mixed_readiness_preserves_all_domains_and_missing_tokenizer(
    local_recipe: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "state"))
    release = _release(local_recipe, tmp_path / "work")
    tokenizer = _fixture_tokenizer(tmp_path / "tokenizer")
    policy = CorpusReadinessPolicy(
        min_unique_train_bytes_by_domain={"developer": 1},
        min_unique_train_tokens_by_domain={"developer": 1},
        required_nonzero_languages=("en",),
        required_nonzero_shapes=("raw_document",),
        min_heldout_families=1,
    )
    result = measure_readiness(release, policy, tokenizer)
    facts = result["measurements"]
    assert facts["unique_train_bytes_by_domain"]["developer"] > 0
    assert facts["unique_train_bytes_by_domain"]["technical_docs"] > 0
    assert facts["unique_train_tokens_by_domain"]["developer"] > 0
    assert facts["train_languages"] is not None
    assert facts["selected_train_shapes"] is not None
    assert facts["heldout_families"] >= 1
    without = measure_readiness(release, policy)
    assert without["state"] == "BLOCKED"
    assert without["measurements"]["unique_train_tokens_by_domain"] is None
    assert (
        without["measurements"]["unique_train_bytes_by_domain"]
        == facts["unique_train_bytes_by_domain"]
    )
    assert without["measurements"]["train_languages"] == facts["train_languages"]
    assert (
        without["measurements"]["selected_train_shapes"]
        == facts["selected_train_shapes"]
    )
    assert without["measurements"]["heldout_families"] == facts["heldout_families"]
    assert any(
        item["dimension"] == "unique_train_tokens" and item["observed"] is None
        for item in without["deficits"]
    )
