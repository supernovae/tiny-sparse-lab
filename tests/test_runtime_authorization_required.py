"""Real authorization denial reaches suite evidence without loading models."""

import json
from pathlib import Path

import pytest

from sparselab.config.loading import load_config
from sparselab.evaluation import inference, suite
from sparselab.runtime_profile import (
    RuntimeAuthorizationRequired,
    require_authorization,
)


def _metadata_suite(tmp_path, monkeypatch, backend):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke_cpu.yaml")
    config = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"backend": backend})}
    )
    source = tmp_path / "suite.json"
    source.write_text(
        json.dumps(
            {
                "evaluation_suite_version": 1,
                "id": "authorization",
                "evaluations": [{"id": "loss", "role": "gate", "kind": "heldout_lm"}],
            }
        )
    )
    runs = tmp_path / "runs"
    selected = runs / "run/checkpoints/step_2"
    selected.mkdir(parents=True)
    digest = "a" * 64
    (selected / "manifest.json").write_text(json.dumps({"sha256": digest}))
    selections = []
    monkeypatch.setattr(
        suite,
        "_selected_checkpoint",
        lambda run, generation, sha, **_verification: selections.append(
            (run, generation, sha)
        ),
    )
    monkeypatch.setattr(inference, "evaluation_config", lambda *_a, **_k: config)
    return config, source, runs, digest, selections


@pytest.mark.parametrize("override,stored", [("cuda", "cpu"), (None, "rocm")])
def test_real_authorization_required_retains_unavailable_null_evidence(
    tmp_path, monkeypatch, override, stored
):
    config, source, runs, digest, selections = _metadata_suite(
        tmp_path, monkeypatch, override or stored
    )
    calls = []

    def denied(*args, **options):
        calls.append(args)
        require_authorization(config, options["authorization"])
        pytest.fail("unauthorized runtime allowed")

    monkeypatch.setattr(inference, "load_run", denied)
    with pytest.raises(RuntimeAuthorizationRequired) as failure:
        require_authorization(config, None)
    assert isinstance(failure.value, ValueError)
    path = suite.run_suite(source, "run", "step_2", runs, backend=override)
    record = suite.verify_evaluation_index(path)
    (row,) = record["evaluations"]
    assert row["status"] == "UNAVAILABLE"
    assert row["reason"] == str(failure.value)
    assert row["result"] is row["path"] is row["sha256"] is None
    assert record["evaluation_runtime"]["observed"] is None
    assert record["evaluation_runtime"]["backend"] == (override or stored)
    assert record["checkpoint"] == "checkpoints/step_2"
    assert record["checkpoint_sha256"] == digest
    assert len(calls) == 1
    assert selections == [(runs / "run", "checkpoints/step_2", digest)] * 2


@pytest.mark.parametrize(
    "message",
    [
        "cuda requires --runtime-profile or a registered compatible worker",
        "cuda requires --runtime ID or an explicit profile",
        "runtime selection mismatch",
    ],
)
def test_unrelated_value_errors_are_not_runtime_unavailability(
    tmp_path, monkeypatch, message
):
    _, source, runs, _, selections = _metadata_suite(tmp_path, monkeypatch, "cuda")

    def malformed(*_a, **_k):
        raise ValueError(message)

    monkeypatch.setattr(inference, "load_run", malformed)
    with pytest.raises(ValueError) as failure:
        suite.run_suite(source, "run", "step_2", runs, backend="cuda")
    assert type(failure.value) is ValueError and str(failure.value) == message
    assert selections == []
    assert not (runs / "run/evaluations").exists()
