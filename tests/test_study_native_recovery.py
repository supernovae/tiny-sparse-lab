"""Tiny native preparation/serialization only; never launch or train a worker."""

import json
from pathlib import Path

import pytest
import yaml
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from sparselab.config.loading import load_config
from sparselab.experiments import study as subject
from sparselab.workers.controller import Controller
from sparselab.workers.models import ExperimentSpec
from sparselab.workers.store import ControllerStore


@pytest.fixture
def native_study(tmp_path, monkeypatch):
    # Same bounds as test_three_seed_cli_submission_uses_one_workspace:
    # synthetic CPU, 100 documents per split, 129/81 packed tokens, 16-wide
    # single-layer declaration. Only freeze inputs; no optimizer or worker.
    tokenizer_path = tmp_path / "tokenizer.json"
    vocabulary = {
        token: index
        for index, token in enumerate(
            ["<unk>", "<pad>", "<bos>", "<eos>", *sorted(ByteLevel.alphabet())]
        )
    }
    tokenizer = Tokenizer(BPE(vocab=vocabulary, merges=[], unk_token="<unk>"))
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/runtime_smoke_cpu.yaml"
    )
    config = config.model_copy(
        update={
            "model": config.model.model_copy(update={"vocab_size": len(vocabulary)}),
            "tokenizer": config.tokenizer.model_copy(update={"path": tokenizer_path}),
            "dataset": config.dataset.model_copy(
                update={"cache_dir": tmp_path / "cache"}
            ),
        }
    )
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")))
    (tmp_path / "matrix.yaml").write_text(
        yaml.safe_dump(
            {
                "matrix_version": 1,
                "base_config": "base.yaml",
                "axes": {
                    "seed": [
                        {"label": f"s{seed}", "set": {"seed": seed}}
                        for seed in (17, 42, 73)
                    ]
                },
            }
        )
    )
    path = tmp_path / "study.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "study_version": 1,
                "name": "native-recovery",
                "matrix": "matrix.yaml",
                "cards": ["chat-alias-recall-v1"],
                "comparisons": [],
            }
        )
    )

    def forbidden(*args, **kwargs):
        pytest.fail("native serialization test tried to launch or train")

    for name in ("tick", "run", "_launch", "_rpc_result"):
        monkeypatch.setattr(Controller, name, forbidden)
    return subject.plan_study(path), tokenizer, tokenizer_path


def test_real_preparation_failure_then_explicit_fresh_receipt_retry(
    native_study, tmp_path
):
    study, tokenizer, tokenizer_path = native_study
    receipt = tmp_path / "receipt.json"
    store, _ = subject.study_execution_paths(study, receipt=receipt)
    with pytest.raises(FileNotFoundError, match="tokenizer") as failure:
        subject.submit_study(study, receipt, store=store)
    assert any("fresh --receipt" in note for note in failure.value.__notes__)
    queue = ControllerStore(store, read_only=True)
    assert queue.attempts() == []
    intent_path = subject.study_intent_path(receipt)
    original = intent_path.read_bytes()
    with pytest.raises(FileExistsError, match="intent already exists.*fresh --receipt"):
        subject.submit_study(study, receipt, store=store)
    with pytest.raises(ValueError, match="no native submissions are bound"):
        subject.recover_study_receipt(
            study, intent_path, tmp_path / "recovered.json", store=store
        )
    assert queue.attempts() == []
    assert intent_path.read_bytes() == original

    # Fix the actual missing local asset, preserving the study/config identity.
    tokenizer.save(str(tokenizer_path))
    fresh = tmp_path / "receipt-retry-1.json"
    retry_store, _ = subject.study_execution_paths(study, receipt=fresh)
    assert retry_store == store
    result = subject.submit_study(study, fresh, store=retry_store)
    rows = queue.attempts()
    assert len(rows) == len(result["runs"]) == 3
    assert {row["run_id"] for row in rows} == {row["run_id"] for row in result["runs"]}
    assert len({row["attempt_id"] for row in rows}) == 3
    assert intent_path.read_bytes() == original
    assert not receipt.exists()
    assert (
        json.loads(subject.study_intent_path(fresh).read_bytes())["submission_id"]
        != json.loads(original)["submission_id"]
    )
    with pytest.raises(FileExistsError):
        subject.submit_study(study, fresh, store=store)
    assert queue.attempts() == rows
    # Later jobs under another intent do not upgrade the failed original intent.
    with pytest.raises(ValueError, match="no native submissions are bound"):
        subject.recover_study_receipt(
            study, intent_path, tmp_path / "recovered-old.json", store=store
        )


def test_real_controller_spec_round_trips_through_read_only_recovery(
    native_study, tmp_path, monkeypatch
):
    study, tokenizer, tokenizer_path = native_study
    tokenizer.save(str(tokenizer_path))
    receipt = tmp_path / "receipt.json"
    store, _ = subject.study_execution_paths(study, receipt=receipt)
    submitted = subject.submit_study(study, receipt, store=store)
    queue = ControllerStore(store, read_only=True)
    rows = queue.attempts()
    for row in rows:
        spec = ExperimentSpec.model_validate(row["spec"])
        assert spec.declaration_provenance["study_submission"]
    original_receipt = receipt.read_bytes()
    original_intent = subject.study_intent_path(receipt).read_bytes()
    database = queue.path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("recovery constructed a controller or submitted")

    for name in ("__init__", "submit", "submit_many", "resume"):
        monkeypatch.setattr(Controller, name, forbidden)
    output = tmp_path / "recovered.json"
    recovered = subject.recover_study_receipt(
        study, subject.study_intent_path(receipt), output, store=store
    )
    assert recovered["runs"] == submitted["runs"]
    assert recovered["cards"] == submitted["cards"]
    assert queue.path.read_bytes() == database
    assert receipt.read_bytes() == original_receipt
    assert subject.study_intent_path(receipt).read_bytes() == original_intent
    assert (
        recovered["recovery"]["intent_sha256"]
        == rows[0]["spec"]["declaration_provenance"]["study_submission"][
            "intent_sha256"
        ]
    )
    assert (
        recovered["recovery"]["intent_file_sha256"]
        == subject.hashlib.sha256(original_intent).hexdigest()
    )
