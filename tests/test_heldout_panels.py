"""Tiny arrays and a fake uniform scorer; no trained model or dataset acquisition."""

import hashlib
import json
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from sparselab.data.packing import _tokenizer_sha256
from sparselab.evaluation import panels
from sparselab.training.manifest import canonical_json, sha256_file


class Uniform(torch.nn.Module):
    def __init__(self, failure=None):
        super().__init__()
        self.calls = 0
        self.failure = failure
        self.register_buffer("unchanged", torch.tensor([3.0]))

    def forward(self, inputs):
        self.calls += 1
        torch.rand(1)
        random.random()
        np.random.random()
        if self.calls == 2 and self.failure:
            raise self.failure
        return torch.zeros((*inputs.shape, 3))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    tokenizer = Tokenizer(WordLevel({"a": 0, "b": 1, "c": 2}, unk_token="a"))
    run = tmp_path / "runs" / "fixture"
    run.mkdir(parents=True)
    tokenizer.save(str(run / "tokenizer.json"))
    (run / "checkpoint.bin").write_bytes(b"immutable fake checkpoint and cursor")
    model = Uniform()
    loaded = SimpleNamespace(
        engine=None,
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        config=SimpleNamespace(
            model=SimpleNamespace(
                memory="none", ffn="dense", semantic_memory_dim=None, max_seq_len=4
            ),
            dataset=SimpleNamespace(allocation_manifest_path=None),
        ),
        identity={
            "checkpoint_sha256": "a" * 64,
            "tokens_seen": 123,
            "tokenizer_sha256": sha256_file(run / "tokenizer.json"),
            "runtime": {"engine": "pytorch", "backend": "cpu", "precision": "fp32"},
        },
    )

    def load(*args, expected_checkpoint_sha256, **kwargs):
        if expected_checkpoint_sha256 != loaded.identity["checkpoint_sha256"]:
            raise ValueError("checkpoint digest mismatch")
        return loaded

    monkeypatch.setattr(panels, "load_run", load)

    def declare(name="english", blocks=2, mask=None, canonical=None, allocation=False):
        root = tmp_path / name
        root.mkdir()
        identity = {"packing": {"memory": "none"}}
        manifest = {
            "cache_identity": identity,
            "settings_sha256": hashlib.sha256(canonical_json(identity)).hexdigest(),
            "packing_version": "contiguous-eos-v6",
            "tokenizer_sha256": canonical or _tokenizer_sha256(tokenizer),
            "source_identity_sha256": "b" * 64,
            "supervision": {"kind": "all_tokens"},
        }
        for split in ("train", "validation"):
            values = np.array(
                [2] * 7 if split == "train" else [0, 1, 2, 0, 1, 2, 0],
                dtype=np.int32,
            )
            path = root / f"{split}.npy"
            np.save(path, values)
            manifest[split] = {
                "dtype": "int32",
                "shape": [7],
                "tokens": 7,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if mask is not None:
                path = root / f"{split}_supervision.npy"
                np.save(
                    path, np.array([True] * 7 if split == "train" else mask, dtype=bool)
                )
                manifest["supervision"].update(kind="token-loss-mask-v1")
                manifest["supervision"][split] = {
                    "dtype": "bool",
                    "shape": [7],
                    "tokens": 7,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        if allocation:
            manifest["allocation"] = {"format": "sparselab-prepared-allocation-v1"}
            for split in ("train", "validation"):
                owner = root / f"{split}_owner_ids.npy"
                np.save(owner, np.zeros(7, dtype=np.uint8))
                manifest["allocation"][split] = {
                    "owner": {
                        "dtype": "uint8",
                        "shape": [7],
                        "tokens": 7,
                        "size_bytes": owner.stat().st_size,
                        "sha256": sha256_file(owner),
                    }
                }
        manifest["manifest_sha256"] = hashlib.sha256(
            canonical_json(manifest)
        ).hexdigest()
        (root / "manifest.json").write_text(json.dumps(manifest))
        declaration = {
            "schema_version": 1,
            "name": name,
            "purpose": "development",
            "prepared_path": name,
            "prepared_manifest_sha256": manifest["manifest_sha256"],
            "tokenizer_file_sha256": loaded.identity["tokenizer_sha256"],
            "tokenizer_canonical_sha256": _tokenizer_sha256(tokenizer),
            "checkpoint_sha256": "a" * 64,
            "split": "validation",
            "sampling": "first_eligible_blocks",
            "sequence_length": 2,
            "blocks": blocks,
            "supervised_targets": blocks * 2 if mask is None else 2,
            "batch_size": 1,
            "max_forward_positions": blocks * 2,
        }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(declaration))
        return path, declaration, manifest

    def execute(path, output="result.json"):
        return panels.run_panel(
            "fixture", tmp_path / "runs", "step_0001", path, tmp_path / output
        )

    return SimpleNamespace(
        declare=declare, execute=execute, loaded=loaded, run=run, root=tmp_path
    )


def test_unequal_panels_masks_exact_identities_and_replay(fixture):
    f = fixture
    english, _, manifest = f.declare()
    technical, _, _ = f.declare(
        "technical", mask=[False, True, False, False, False, True, False]
    )
    before = {p.name: p.read_bytes() for p in f.run.iterdir()}
    a = f.execute(english, "english-result.json")
    b = f.execute(technical, "technical-result.json")
    again = f.execute(english, "replay.json")
    for row, count in [(a, 4), (b, 2), (again, 4)]:
        assert row["evidence_complete"] and row["status"] == "complete"
        assert row["valid_targets"] == count
        assert row["nll_sum"] == pytest.approx(count * math.log(3))
        assert row["loss"] == pytest.approx(math.log(3))
        assert row["identity"]["tokens_seen"] == 123
        assert row["identity"]["checkpoint_sha256"] == "a" * 64
        assert row["attempted_forward_positions"] == 4
    assert a["data_identity"] == {
        "prepared_manifest_sha256": manifest["manifest_sha256"],
        "validation_sha256": manifest["validation"]["sha256"],
        "supervision": {"kind": "all_tokens"},
        "packing_version": "contiguous-eos-v6",
        "source_identity_sha256": "b" * 64,
    }
    assert a["data_identity"]["validation_sha256"] != manifest["train"]["sha256"]
    assert b["data_identity"]["supervision"]["kind"] == "token-loss-mask-v1"
    assert a["protocol_sha256"] == again["protocol_sha256"] != b["protocol_sha256"]
    assert {p.name: p.read_bytes() for p in f.run.iterdir()} == before


@pytest.mark.parametrize(
    "error,status",
    [(RuntimeError("fake failure"), "failed"), (KeyboardInterrupt(), "interrupted")],
)
def test_partial_failure_and_rng_mode_restore(fixture, error, status):
    f = fixture
    path, _, _ = f.declare()
    f.loaded.model = Uniform(error)
    model = f.loaded.model
    model.train()
    cpu = torch.get_rng_state().clone()
    py = random.getstate()
    np_state = np.random.get_state()
    result = f.execute(path)
    assert result["status"] == status and not result["evidence_complete"]
    assert result["valid_targets"] == 2 and result["batches"] == 1
    assert result["nll_sum"] == pytest.approx(2 * math.log(3))
    assert result["accounting_complete"]
    assert result["evaluation_seconds"] >= 0
    assert result["loss"] is None
    assert result["attempted_forward_positions"] == 4
    assert model.training and torch.equal(torch.get_rng_state(), cpu)
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert model.unchanged.tolist() == [3.0]
    assert json.loads((f.root / "result.json").read_text()) == result


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("prepared_manifest_sha256", "c" * 64, "prepared manifest identity"),
        ("tokenizer_file_sha256", "c" * 64, "checkpoint tokenizer identity"),
        ("tokenizer_canonical_sha256", "c" * 64, "prepared tokenizer identity"),
        ("checkpoint_sha256", "c" * 64, "checkpoint digest"),
        ("blocks", 4, "insufficient eligible blocks"),
        ("supervised_targets", 3, "supervised-target coverage"),
        ("sequence_length", 5, "insufficient eligible blocks"),
    ],
)
def test_mismatch_refuses_before_forward(fixture, field, value, reason):
    f = fixture
    path, declaration, _ = f.declare()
    declaration[field] = value
    declaration["max_forward_positions"] = 100
    path.write_text(json.dumps(declaration))
    result = f.execute(path)
    assert not result["evidence_complete"] and result["status"] == "failed"
    assert reason in result["reason"]
    assert f.loaded.model.calls == 0
    assert result["valid_targets"] == result["attempted_forward_positions"] == 0


@pytest.mark.parametrize("failure", ["missing", "tampered", "empty", "symlink"])
def test_prepared_input_refusals(fixture, failure):
    f = fixture
    path, _, _ = f.declare(mask=[False] * 7 if failure == "empty" else None)
    target = f.root / "english" / "validation.npy"
    if failure == "missing":
        target.unlink()
    if failure == "tampered":
        target.write_bytes(b"bad array")
    if failure == "symlink":
        target.unlink()
        target.symlink_to(f.root / "english" / "train.npy")
    result = f.execute(path)
    assert result["status"] == "failed" and not result["evidence_complete"]
    assert f.loaded.model.calls == 0


def test_no_clobber_and_no_writes_in_input_root(fixture):
    f = fixture
    path, _, _ = f.declare()
    result = f.execute(path)
    content = (f.root / "result.json").read_bytes()
    calls = f.loaded.model.calls
    with pytest.raises(FileExistsError):
        f.execute(path)
    assert (f.root / "result.json").read_bytes() == content
    assert f.loaded.model.calls == calls
    for output in ("english/result.json", "runs/fixture/result.json"):
        with pytest.raises(ValueError, match="input roots"):
            f.execute(path, output)
    assert result["evidence_complete"]


@pytest.mark.parametrize(
    "change",
    [
        {"split": "test"},
        {"purpose": "final"},
        {"blocks": 0},
        {"batch_size": True},
        {"max_forward_positions": 3},
        {"unexpected": 1},
    ],
)
def test_strict_development_declaration(fixture, change):
    path, declaration, _ = fixture.declare()
    path.write_text(json.dumps(declaration | change))
    with pytest.raises(ValueError):
        fixture.execute(path)
    assert fixture.loaded.model.calls == 0
    assert not (fixture.root / "result.json").exists()


def test_incomplete_attempt_binding_refuses_before_load(fixture, monkeypatch):
    path, _, _ = fixture.declare()
    monkeypatch.setenv("SPARSELAB_ATTEMPT_ACTIVITY", "evaluate")

    def forbidden(*a, **k):
        pytest.fail("model loaded despite missing attempt budget")

    monkeypatch.setattr(panels, "load_run", forbidden)
    result = fixture.execute(path)
    assert result["status"] == "failed"
    assert "attempt binding" in result["reason"]


def test_cli_public_route_and_failed_exit(fixture, monkeypatch, capsys):
    import sys
    from pathlib import Path

    from sparselab.cli.main import main
    from sparselab.config.loading import load_config

    f = fixture
    config = load_config(Path("configs/smoke_cpu.yaml"))
    (f.run / "resolved_config.yaml").write_text(config.model_dump_json())
    path, declaration, _ = f.declare()
    argv = [
        "sparselab",
        "eval",
        "fixture",
        "--runs-dir",
        str(f.root / "runs"),
        "--checkpoint",
        "step_0001",
        "--panel",
        str(path),
        "--output",
        str(f.root / "cli.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    assert json.loads(capsys.readouterr().out)["evidence_complete"]
    declaration["checkpoint_sha256"] = "c" * 64
    path.write_text(json.dumps(declaration))
    argv[-1] = str(f.root / "failed.json")
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert not json.loads(capsys.readouterr().out)["evidence_complete"]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("memory", "ngram", "dense PyTorch"),
        ("ffn", "moe", "dense PyTorch"),
        ("max_seq_len", 1, "context"),
    ],
)
def test_unsupported_model_or_context_refuses(fixture, field, value, reason):
    path, _, _ = fixture.declare()
    setattr(fixture.loaded.config.model, field, value)
    result = fixture.execute(path)
    assert result["status"] == "failed" and reason in result["reason"]
    assert fixture.loaded.model.calls == 0


def test_native_loader_checkpoint_pin_fails_before_model_construction(monkeypatch):
    from pathlib import Path

    from sparselab.evaluation import inference

    monkeypatch.setattr(
        inference,
        "_verified_checkpoint",
        lambda *a, **k: (None, None, None, {"sha256": "a" * 64}, None),
    )
    with pytest.raises(ValueError, match="checkpoint digest"):
        inference.load_run(
            "fixture", Path("/unused"), expected_checkpoint_sha256="b" * 64
        )


def test_forward_budget_reservation_precedes_model_and_retains_partial(
    fixture, monkeypatch
):
    path, _, _ = fixture.declare()
    calls = []
    monkeypatch.setattr(
        panels.AttemptBudget, "forward_allocation_active_from_environment", lambda: True
    )

    def reserve(label, *, kind, positions):
        calls.append((label, kind, positions))
        if len(calls) == 2:
            raise ValueError("fixture forward allocation exhausted")

    monkeypatch.setattr(
        panels.AttemptBudget, "reserve_forward_from_environment", reserve
    )
    result = fixture.execute(path)
    assert result["status"] == "failed"
    assert fixture.loaded.model.calls == 1
    assert result["attempted_forward_positions"] == 2
    assert result["valid_targets"] == 2
    assert len(calls) == 2 and calls[0][0] != calls[1][0]
    assert all(
        kind == "fixed_profile" and positions == 2 for _, kind, positions in calls
    )


def test_comparison_group_binds_protocol_runtime_not_checkpoint(fixture):
    f = fixture
    path, declaration, _ = f.declare()
    first = f.execute(path, "first.json")
    f.loaded.identity["checkpoint_sha256"] = "d" * 64
    declaration["checkpoint_sha256"] = "d" * 64
    path.write_text(json.dumps(declaration))
    second = f.execute(path, "second.json")
    assert first["comparison_group"] == second["comparison_group"]
    f.loaded.identity["runtime"] = {
        "engine": "pytorch",
        "backend": "cpu",
        "precision": "other",
    }
    third = f.execute(path, "third.json")
    assert third["comparison_group"] != second["comparison_group"]
    declaration["batch_size"] = 2
    path.write_text(json.dumps(declaration))
    fourth = f.execute(path, "fourth.json")
    assert fourth["protocol_sha256"] != third["protocol_sha256"]


def test_forward_cap_refuses_before_oversized_native_call(fixture, monkeypatch):
    path, _, _ = fixture.declare()

    def excessive(*args, before_batch, **kwargs):
        before_batch(5)
        pytest.fail("oversized forward admitted")

    monkeypatch.setattr(panels, "evaluate", excessive)
    result = fixture.execute(path)
    assert result["status"] == "failed" and "forward budget" in result["reason"]
    assert result["attempted_forward_positions"] == 0


def test_validation_tokens_determine_loss_and_initial_evidence(fixture):
    class Unequal(torch.nn.Module):
        def forward(self, inputs):
            initial = json.loads((fixture.root / "result.json").read_text())
            assert initial["accounting_complete"] is False
            assert initial["status"] == "incomplete"
            return torch.tensor([0.0, 1.0, 2.0]).expand(*inputs.shape, 3)

    path, _, _ = fixture.declare()
    fixture.loaded.model = Unequal()
    result = fixture.execute(path)
    # The first two validation blocks predict [1, 2, 0, 1]; train predicts [2]*4.
    expected = 4 * math.log(1 + math.e + math.exp(2)) - 4
    assert result["status"] == "complete"
    assert result["nll_sum"] == pytest.approx(expected)
    assert result["accounting_complete"] is True


@pytest.mark.parametrize("bind_panel", [False, True])
def test_independent_tokenizer_cross_bindings(fixture, bind_panel):
    second = Tokenizer(WordLevel({"a": 1, "b": 0, "c": 2}, unk_token="a"))
    digest = _tokenizer_sha256(second)
    path, declaration, _ = fixture.declare(canonical=digest)
    if bind_panel:
        declaration["tokenizer_canonical_sha256"] = digest
        path.write_text(json.dumps(declaration))
    result = fixture.execute(path)
    expected = "checkpoint" if bind_panel else "prepared"
    assert f"{expected} tokenizer identity mismatch" in result["reason"]
    assert fixture.loaded.model.calls == 0


@pytest.mark.parametrize("prepared", [True, False])
def test_allocation_refusals(fixture, prepared):
    path, _, _ = fixture.declare(allocation=prepared)
    if not prepared:
        fixture.loaded.config.dataset.allocation_manifest_path = "fixture-allocation"
    result = fixture.execute(path)
    reason = "allocation sidecars" if prepared else "dense PyTorch"
    assert result["status"] == "failed" and reason in result["reason"]
    assert fixture.loaded.model.calls == 0


def test_small_real_forward_allocation_refuses_before_load(fixture, monkeypatch):
    from sparselab.training.attempt_budget import AttemptBudget

    contract = {
        "contract_version": 1,
        "max_optimizer_updates": 0,
        "max_actual_target_positions": 0,
        "max_generation_calls": 0,
        "max_generated_tokens": 0,
        "max_wall_seconds": 120.0,
        "content_identity_sha256": "a" * 64,
        "monitor_policy_sha256": "b" * 64,
        "workspace_baseline_sha256": "c" * 64,
        "max_fixed_profile_forward_positions": 3,
        "max_nontraining_forward_positions": 4,
        "max_operational_validation_batches": 0,
        "max_operational_validation_forward_positions": 0,
    }
    raw = json.dumps(contract).encode()
    contract_path = fixture.root / "contract.json"
    contract_path.write_bytes(raw)
    budget = AttemptBudget.create_contract(
        fixture.root / "attempt.sqlite",
        contract_path=contract_path,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )
    budget.reserve_vector(
        "panel",
        updates=0,
        target_positions=0,
        generation_calls=0,
        generated_tokens=0,
        content_identity_sha256="a" * 64,
    )
    for key, value in {
        "SPARSELAB_ATTEMPT_BUDGET_LEDGER": str(budget.path),
        "SPARSELAB_ATTEMPT_CONTENT_IDENTITY_SHA256": "a" * 64,
        "SPARSELAB_ATTEMPT_PHASE_LABEL": "panel",
        "SPARSELAB_ATTEMPT_ACTIVITY": "evaluate",
    }.items():
        monkeypatch.setenv(key, value)

    def forbidden(*a, **k):
        pytest.fail("model loaded despite insufficient fixed-profile allowance")

    monkeypatch.setattr(panels, "load_run", forbidden)
    path, _, _ = fixture.declare()
    result = fixture.execute(path)
    assert result["status"] == "failed"
    assert "declared model allocation" in result["reason"]
    assert result["attempted_forward_positions"] == 0


def test_protocol_does_not_depend_on_prepared_location(fixture):
    path, declaration, _ = fixture.declare()
    first = fixture.execute(path, "first.json")
    source = fixture.root / "english"
    moved = fixture.root / "moved"
    source.rename(moved)
    declaration["prepared_path"] = str(moved)
    path.write_text(json.dumps(declaration))
    second = fixture.execute(path, "second.json")
    assert first["status"] == second["status"] == "complete"
    assert first["protocol_sha256"] == second["protocol_sha256"]
    assert first["comparison_group"] == second["comparison_group"]


def test_engine_refuses_before_forward(fixture):
    path, _, _ = fixture.declare()
    fixture.loaded.engine = object()
    fixture.loaded.device = "metal"
    result = fixture.execute(path)
    assert "dense PyTorch" in result["reason"]
    assert fixture.loaded.model.calls == 0
