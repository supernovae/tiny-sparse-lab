"""Small CPU fixtures for the existing engine, without acquisition or GPU work."""

import hashlib
import json
import math
import shutil

import numpy as np
import pytest
from test_training import config as tiny_config

from sparselab.config.loading import load_config
from sparselab.config.models import EvaluationConfig, FixedValidationPanel, RunConfig
from sparselab.data.packing import _tokenizer_sha256
from sparselab.data.tokenizer import load_tokenizer
from sparselab.engines.pytorch import PyTorchEngine
from sparselab.evaluation.evidence import experiment_evidence
from sparselab.evaluation.panel_data import load_fixed_panels
from sparselab.staging import materialize_prepared_inputs, verify_prepared_inputs
from sparselab.training.manifest import canonical_json, config_sha256, sha256_file
from sparselab.training.trainer import train


def panel(config, name, *, masked=False):
    root = config.tokenizer.path.parent.parent / name
    root.mkdir()
    identity = {"packing": {"memory": "none"}}
    manifest = {
        "cache_identity": identity,
        "settings_sha256": hashlib.sha256(canonical_json(identity)).hexdigest(),
        "packing_version": "contiguous-eos-v6",
        "tokenizer_sha256": _tokenizer_sha256(load_tokenizer(config.tokenizer.path)),
        "source_identity_sha256": "b" * 64,
        "supervision": {"kind": "token-loss-mask-v1" if masked else "all_tokens"},
    }
    for split in ("train", "validation"):
        path = root / f"{split}.npy"
        np.save(
            path, np.arange(33, dtype=np.int32) + (40 if name == "technical" else 0)
        )
        manifest[split] = {
            "dtype": "int32",
            "shape": [33],
            "tokens": 33,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if masked:
            path = root / f"{split}_supervision.npy"
            np.save(path, np.arange(33) % 2 == 0)
            manifest["supervision"][split] = {
                "dtype": "bool",
                "shape": [33],
                "tokens": 33,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    return FixedValidationPanel.model_validate(
        {
            "name": name,
            "prepared_path": root,
            "prepared_manifest_sha256": manifest["manifest_sha256"],
            "tokenizer_file_sha256": sha256_file(config.tokenizer.path),
            "tokenizer_canonical_sha256": manifest["tokenizer_sha256"],
            "split": "validation",
            "sampling": "first_eligible_blocks",
            "sequence_length": 16,
            "blocks": 2,
            "supervised_targets": 16 if masked else 32,
            "batch_size": 1 if masked else 2,
            "max_forward_positions": 32,
        }
    )


@pytest.fixture
def measured(tmp_path):
    config = tiny_config(tmp_path)
    payload = config.model_dump(mode="python")
    payload["evaluation"] = {
        "every_steps": 2,
        "max_batches": 1,
        "panels": [panel(config, "english"), panel(config, "technical", masked=True)],
    }
    payload["training"].update(max_steps=4, max_tokens=128)
    payload["optimizer"].update(warmup_steps=1)
    payload["checkpoint"].update(every_steps=4)
    return RunConfig.model_validate(payload)


def test_pair_schema_identity_and_relative_paths(measured, tmp_path):
    assert EvaluationConfig().model_dump() == {"every_steps": 25, "max_batches": 16}
    for panels in [
        (measured.evaluation.panels[0],),
        (measured.evaluation.panels[0],) * 2,
        measured.evaluation.panels * 2,
    ]:
        with pytest.raises(ValueError, match="exactly two distinct"):
            EvaluationConfig(panels=panels)
    raw = measured.model_dump(mode="json")
    original = config_sha256(raw)
    for item in raw["evaluation"]["panels"]:
        item["prepared_path"] = item["name"]
    source = tmp_path / "config.json"
    source.write_text(json.dumps(raw))
    loaded = load_config(source)
    assert loaded.evaluation.panels[0].prepared_path == tmp_path / "english"
    assert config_sha256(loaded.model_dump(mode="json")) == original
    raw["evaluation"]["panels"][0]["supervised_targets"] -= 1
    assert config_sha256(raw) != original


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("prepared_manifest_sha256", "c" * 64, "manifest identity"),
        ("tokenizer_file_sha256", "c" * 64, "training tokenizer"),
        ("tokenizer_canonical_sha256", "c" * 64, "training tokenizer"),
        ("supervised_targets", 31, "target coverage"),
        ("blocks", 3, "insufficient eligible"),
    ],
)
def test_authentication_refuses_before_engine(
    measured, monkeypatch, field, value, match
):
    first, second = measured.evaluation.panels
    changed = first.model_copy(update={field: value})
    config = measured.model_copy(
        update={
            "evaluation": measured.evaluation.model_copy(
                update={"panels": (changed, second)}
            )
        }
    )

    def unexpected(*args, **kwargs):
        pytest.fail("engine initialized before panel authentication")

    monkeypatch.setattr(PyTorchEngine, "initialize", unexpected)
    with pytest.raises(ValueError, match=match):
        train(config, run_id="refused")


def test_active_pair_cadence_weighting_owned_evidence(measured, monkeypatch):
    calls = []
    initializations = []
    evaluate = PyTorchEngine.evaluate
    initialize = PyTorchEngine.initialize

    def measured_evaluate(self, batches):
        result = evaluate(self, batches)
        calls.append(result)
        return result

    def measured_initialize(self, *args, **kwargs):
        initializations.append(id(self))
        return initialize(self, *args, **kwargs)

    monkeypatch.setattr(PyTorchEngine, "evaluate", measured_evaluate)
    monkeypatch.setattr(PyTorchEngine, "initialize", measured_initialize)
    train(measured, run_id="paired")
    run = measured.logging.root_dir / "paired"
    assert len(initializations) == 1
    assert len(calls) == 6  # Existing baseline at 0, then 2 and 4; two calls each.
    reports = [json.loads(p.read_text()) for p in (run / "evaluations").glob("*.json")]
    assert {p["step"] for p in reports} == {0, 4}
    assert {
        p.name.split("_gen_")[0] for p in (run / "checkpoints").glob("step_*_gen_*")
    } <= {"step_00000000", "step_00000002", "step_00000004"}
    for report in reports:
        assert set(report["panels"]) == {"english", "technical"}
        assert report["valid_targets"] == 48 and report["batches"] == 3
        assert report["batch_size"] is None and report["max_batches"] is None
        expected = math.fsum(p["nll_sum"] for p in report["panels"].values()) / 48
        assert report["loss"] == expected
        assert report["panels"]["english"]["valid_targets"] == 32
        assert report["panels"]["technical"]["valid_targets"] == 16
    evidence = experiment_evidence(run)
    assert evidence["evidence_level"] == "checkpointed_held_out"
    assert evidence["quality_observations"]
    # Re-sealing a report cannot turn an unweighted mean into valid evidence.
    target = next((run / "evaluations").glob("*.json"))
    payload = json.loads(target.read_text())
    payload["panels"]["english"]["nll_sum"] += 1
    payload.pop("sha256")
    payload["sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    target.write_text(json.dumps(payload))
    rejected = experiment_evidence(run)
    assert rejected["rejected_reports"]


def test_resume_owns_panels_and_refuses_changed_selection(measured):
    train(measured, run_id="parent", stop_after_step=2)
    parent = measured.logging.root_dir / "parent"
    for item in measured.evaluation.panels:
        shutil.rmtree(item.prepared_path)
    measured.tokenizer.path.unlink()
    train(measured, run_id="resumed", resume=parent / "checkpoints/latest.json")
    child = measured.logging.root_dir / "resumed"
    assert load_fixed_panels(
        measured, inputs_root=child, tokenizer_path=child / "tokenizer.json"
    )
    assert experiment_evidence(child)["evidence_level"] == "checkpointed_held_out"
    first, second = measured.evaluation.panels
    changed = measured.model_copy(
        update={
            "evaluation": measured.evaluation.model_copy(
                update={
                    "panels": (
                        first.model_copy(update={"supervised_targets": 31}),
                        second,
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="resume configuration differs"):
        train(changed, run_id="changed", resume=parent / "checkpoints/latest.json")


def test_owned_panel_mutation_refuses_resume(measured):
    train(measured, run_id="parent", stop_after_step=2)
    parent = measured.logging.root_dir / "parent"
    path = parent / "panels/english/validation.npy"
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\xff")
    with pytest.raises(ValueError, match="artifact integrity"):
        train(measured, run_id="refused", resume=parent / "checkpoints/latest.json")


def test_prepared_stage_owns_panel_inputs(measured, tmp_path):
    staged = materialize_prepared_inputs(measured, tmp_path / "staged")
    for item in measured.evaluation.panels:
        shutil.rmtree(item.prepared_path)
    measured.tokenizer.path.unlink()
    verified = verify_prepared_inputs(staged, measured)
    assets = staged / "assets"
    assert verified["assets_root"] == str(assets)
    assert set(
        load_fixed_panels(
            measured, inputs_root=assets, tokenizer_path=assets / "tokenizer.json"
        )
    ) == {"english", "technical"}
    copied = materialize_prepared_inputs(
        measured, tmp_path / "copied", prepared_inputs=staged
    )
    assert verify_prepared_inputs(copied, measured)


def test_same_bundle_with_two_names_refuses(measured, monkeypatch):
    first = measured.evaluation.panels[0]
    duplicate = first.model_copy(update={"name": "technical"})
    config = measured.model_copy(
        update={
            "evaluation": measured.evaluation.model_copy(
                update={"panels": (first, duplicate)}
            )
        }
    )

    def unexpected(*args, **kwargs):
        pytest.fail("duplicate panel inputs reached model initialization")

    monkeypatch.setattr(PyTorchEngine, "initialize", unexpected)
    with pytest.raises(ValueError, match="distinct prepared validation inputs"):
        train(config, run_id="duplicate")


def test_exact_training_array_alias_refuses(measured):
    root = measured.evaluation.panels[0].prepared_path
    manifest = json.loads((root / "manifest.json").read_text())
    # The already authenticated training manifest supplies the array identity;
    # no token-overlap scan or extra full training-array read is introduced.
    with pytest.raises(ValueError, match="aliases the training array"):
        load_fixed_panels(measured, training_manifest={"train": manifest["validation"]})


def test_explicit_workflow_budget_counts_both_panels(measured, tmp_path):
    from test_workflow_execution import admitted, execution_fixture

    assert EvaluationConfig(max_batches=4).batch_limit == 4
    assert measured.evaluation.max_batches == 1
    assert measured.evaluation.batch_limit == 3
    raw = measured.model_dump(mode="json")
    for item in raw["evaluation"]["panels"]:
        item.update(
            prepared_path=f"panels/{item['name']}",
            blocks=5,
            supervised_targets=80,
            batch_size=1,
            max_forward_positions=80,
        )
    source, bindings = execution_fixture(tmp_path / "recipe", raw)
    bound = json.loads(bindings.read_text())
    bound["config_paths"] = {
        f"evaluation.panels.{index}.prepared_path": f"panels/{item['name']}"
        for index, item in enumerate(raw["evaluation"]["panels"])
    }
    bindings.write_text(json.dumps(bound))
    with pytest.raises(ValueError, match="declared evaluation limit"):
        admitted(source, bindings, tmp_path / "work")
    assert not (tmp_path / "work").exists()
