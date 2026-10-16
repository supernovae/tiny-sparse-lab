"""Stage A uses tiny arrays and fake engine updates; no model forward passes."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sparselab.data.packing import BatchCursor, TokenBlockDataset
from sparselab.training.exposure import checkpoint_exposure, exposure_context
from sparselab.training.manifest import ArtifactIdentity


def context(dataset, **kwargs):
    artifacts = tuple(
        ArtifactIdentity(name, str(i) * 64, 1)
        for i, name in enumerate(
            [
                "data/manifest.json",
                "data/train.npy",
                "tokenizer.json",
                "data/train_supervision.npy",
            ],
            1,
        )
    )
    return exposure_context(
        dataset,
        artifacts,
        seed=7,
        run_id="fixture",
        continuation="FRESH",
        prior=None,
        **kwargs,
    )


@pytest.mark.parametrize(
    "mask, expected",
    [
        (None, (3, 0, 12)),
        (
            [
                True,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
            ],
            (2, 1, 5),
        ),
        ([False] * 15, (0, 3, 0)),
    ],
)
def test_hand_counted_supply(mask, expected):
    dataset = TokenBlockDataset(
        np.arange(15, dtype=np.int32),
        4,
        supervision=None if mask is None else np.array(mask),
    )
    supply = dataset.exposure_supply()
    assert supply == {
        "raw_packed_positions": 15,
        "full_blocks": 3,
        "eligible_blocks": expected[0],
        "zero_target_blocks_skipped": expected[1],
        "trailing_target_positions": 2,
        "usable_supervised_targets_per_pass": expected[2],
    }
    assert (
        sum(
            np.count_nonzero(dataset.numpy_block(i)[1] != -100)
            for i in range(len(dataset))
        )
        == expected[2]
    )


@pytest.mark.parametrize(
    "cursor, passes, block", [(BatchCursor(0, 3), 1, 0), (BatchCursor(1, 2), 1, 2)]
)
def test_cursor_normalization_does_not_claim_fine_consumption(cursor, passes, block):
    row = checkpoint_exposure(
        context(TokenBlockDataset(np.arange(13, dtype=np.int32), 4)),
        tokens=7,
        cursor=cursor,
        wall_seconds=4,
        invocation_seconds=5,
        update_seconds=2,
        wall_complete=True,
        updates_complete=True,
    )
    assert row["committed_supervised_targets"] == 7
    assert row["effective_passes"] == 7 / 12
    assert row["cursor"] == {
        "epoch": cursor.epoch,
        "next_block": cursor.next_block,
        "completed_loader_passes": passes,
        "next_eligible_block": block,
    }
    assert row["finer_consumed_positions"] is None
    assert row["timing"]["optimizer_update_seconds"] == 2
    assert row["timing"]["training_loop_elapsed_seconds"] == 4
    assert row["timing"]["other_training_overhead_seconds"] is None


def test_checkpoint_roundtrip_and_historical_absence(tmp_path):
    from test_checkpoints import _snapshot

    from sparselab.training.checkpoints import CheckpointManager

    snapshot = _snapshot(
        0
    )  # Creates fixture tensors; its training loop has zero iterations.
    manager = CheckpointManager(tmp_path)
    old = manager.save(snapshot)
    old_path = manager.root / old.relative_path / "manifest.json"
    old_bytes = old_path.read_bytes()
    assert "exposure" not in json.loads(old_bytes)
    assert manager.load(old_path.parent).exposure is None
    snapshot.exposure = checkpoint_exposure(
        context(TokenBlockDataset(np.arange(13, dtype=np.int32), 4)),
        tokens=0,
        cursor=BatchCursor(),
        wall_seconds=0,
        invocation_seconds=0,
        update_seconds=0,
        wall_complete=True,
        updates_complete=True,
    )
    new = manager.save(snapshot)
    loaded = manager.load(manager.root / new.relative_path)
    assert loaded.exposure == snapshot.exposure
    from sparselab.training.state_digest import canonical_training_state

    observation = {
        "step": 0,
        "tokens_seen": 0,
        "learning_rate": 0.0,
        "loss": None,
        "gradient_norm": None,
        "overflow": False,
        "retry_state": {},
    }
    original_state = canonical_training_state(loaded, observation=observation)
    loaded.exposure = None
    assert canonical_training_state(loaded, observation=observation) == original_state
    assert old_path.read_bytes() == old_bytes
    snapshot.exposure["committed_supervised_targets"] = 1
    with pytest.raises(ValueError, match="committed checkpoint targets"):
        manager.save(snapshot)


@pytest.fixture
def fake_lifecycle(tmp_path, monkeypatch):
    from pathlib import Path

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from sparselab.config.loading import load_config
    from sparselab.config.models import RunConfig
    from sparselab.engines.base import EngineState, EvaluationResult, UpdateResult
    from sparselab.training import trainer
    from sparselab.training.checkpoints import CheckpointRecord

    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    settings = load_config(Path("configs/smoke_cpu.yaml")).model_dump(mode="json")
    settings["model"].update(
        vocab_size=512, hidden_dim=16, num_layers=1, ffn_dim=32, max_seq_len=16
    )
    settings["tokenizer"]["path"] = str(tokenizer_path)
    settings["dataset"].update(
        cache_dir=str(tmp_path / "cache"),
        train_max_documents=2,
        validation_max_documents=2,
        train_max_tokens=13,
        validation_max_tokens=13,
    )
    settings["training"].update(
        seq_len=4, micro_batch_size=2, max_steps=8, max_tokens=23
    )
    settings["evaluation"].update(every_steps=8, max_batches=1)
    settings["logging"]["root_dir"] = str(tmp_path / "runs")
    settings["checkpoint"]["every_steps"] = 1
    config = RunConfig.model_validate(settings)
    snapshots, calls, outcomes = [], [], []
    monkeypatch.setattr(trainer.PyTorchEngine, "initialize", lambda *a, **k: None)
    monkeypatch.setattr(trainer.PyTorchEngine, "close", lambda *a: None)
    monkeypatch.setattr(
        trainer.PyTorchEngine, "restore_training_state", lambda *a: None
    )
    monkeypatch.setattr(trainer.PyTorchEngine, "export_weights", lambda *a: None)
    monkeypatch.setattr(
        trainer.PyTorchEngine, "export_training_state", lambda *a: EngineState({}, {})
    )
    monkeypatch.setattr(trainer, "triage_completed_run", lambda *a, **k: None)

    def evaluate(self, batches):
        rows = list(batches)
        count = sum(int(np.count_nonzero(b.targets != -100)) for b in rows)
        return EvaluationResult(float(count), count, len(rows))

    def update(self, batches, step, valid_targets):
        assert valid_targets == sum(
            int(np.count_nonzero(b.targets != -100)) for b in batches
        )
        calls.append((step, valid_targets))
        outcome = outcomes.pop(0) if outcomes else "APPLIED"
        return UpdateResult(
            outcome,
            float(valid_targets) if outcome == "APPLIED" else None,
            valid_targets,
            valid_targets if outcome == "APPLIED" else 0,
            0.25,
        )

    def save(self, snapshot, validation_loss=None):
        snapshots.append(copy.deepcopy(snapshot))
        return CheckpointRecord(
            len(snapshots),
            f"step_{snapshot.step}",
            "a" * 64,
            snapshot.step,
            snapshot.tokens_seen,
            "2026-01-01T00:00:00+00:00",
            0,
            validation_loss,
            "pytorch",
            "cpu",
        )

    monkeypatch.setattr(trainer.PyTorchEngine, "evaluate", evaluate)
    monkeypatch.setattr(trainer.PyTorchEngine, "train_update", update)
    monkeypatch.setattr(trainer.CheckpointManager, "save", save)
    return SimpleNamespace(
        config=config,
        trainer=trainer,
        snapshots=snapshots,
        calls=calls,
        outcomes=outcomes,
    )


def test_native_fake_updates_cross_pass_partial_budget_and_retry(fake_lifecycle):
    f = fake_lifecycle
    f.outcomes.extend(["OVERFLOW", "APPLIED"])
    f.trainer.train(f.config, run_id="fresh")
    last = f.snapshots[-1]
    assert f.calls == [(1, 8), (1, 8), (2, 8), (3, 7)]
    assert last.tokens_seen == 23 and last.cursor == (1, 3)
    assert last.exposure["supply"]["usable_supervised_targets_per_pass"] == 12
    assert last.exposure["effective_passes"] == 23 / 12
    assert last.exposure["cursor"]["completed_loader_passes"] == 2
    assert last.exposure["timing"]["optimizer_update_seconds"] == 1.0
    assert last.exposure["protocol"]["identities"]["tokenizer_sha256"]
    assert last.exposure["protocol"]["identities"]["prepared_manifest_sha256"]


def test_wholly_failed_update_never_seals_uncommitted_targets(fake_lifecycle):
    f = fake_lifecycle
    f.outcomes.extend(["APPLIED", "OVERFLOW", "OVERFLOW", "OVERFLOW"])
    with pytest.raises(FloatingPointError, match="three attempts"):
        f.trainer.train(f.config, run_id="failure")
    assert f.snapshots[-1].tokens_seen == 8
    assert f.snapshots[-1].exposure["committed_supervised_targets"] == 8
    assert f.snapshots[-1].cursor == (0, 2)
    assert f.calls == [(1, 8), (2, 8), (2, 8), (2, 8)]
    assert f.snapshots[-1].exposure["timing"]["optimizer_update_seconds"] == 0.25


@pytest.mark.parametrize("kind", ["RESUMED", "PROMOTED", "historical"])
def test_resume_and_warm_start_scope(fake_lifecycle, monkeypatch, kind):
    from sparselab.training.continuation import Continuation

    f = fake_lifecycle
    f.trainer.train(f.config, run_id="first", stop_after_step=1)
    prior = copy.deepcopy(f.snapshots[-1])
    if kind == "historical":
        prior.exposure = None
        prior.cumulative_update_seconds = prior.cumulative_wall_seconds = None
    continuation = "RESUMED" if kind == "historical" else kind
    monkeypatch.setattr(
        f.trainer,
        "load_continuation",
        lambda *a, **k: Continuation(
            kind=continuation,
            snapshot=prior,
            source_run=f.config.logging.root_dir / "first",
            parent_run_id="first",
            parent_checkpoint_sha256="a" * 64,
        ),
    )
    before_calls = len(f.calls)
    f.trainer.train(f.config, run_id="second", stop_after_step=2)
    assert f.calls[before_calls:] == (
        [(1, 8), (2, 8)] if kind == "PROMOTED" else [(2, 8)]
    )
    row = f.snapshots[-1].exposure
    assert (
        row["committed_supervised_targets"] == 16
    )  # Never previous + already cumulative.
    assert row["scope"]["origin_run_id"] == (
        None if kind == "historical" else "second" if kind == "PROMOTED" else "first"
    )
    assert row["scope"]["counter_basis"] == (
        "new run" if kind == "PROMOTED" else "resumed cumulative"
    )
    assert row["timing"]["optimizer_update_seconds"] == (
        None if kind == "historical" else 0.5
    )
    assert (
        row["timing"]["training_loop_elapsed_seconds"] is None
        if kind == "historical"
        else row["timing"]["training_loop_elapsed_seconds"] >= 0
    )

    if kind == "historical":
        second = copy.deepcopy(f.snapshots[-1])
        monkeypatch.setattr(
            f.trainer,
            "load_continuation",
            lambda *a, **k: Continuation(
                kind="RESUMED",
                snapshot=second,
                source_run=f.config.logging.root_dir / "second",
                parent_run_id="second",
                parent_checkpoint_sha256="b" * 64,
            ),
        )
        f.trainer.train(f.config, run_id="third", stop_after_step=3)
        latest = f.snapshots[-1]
        assert latest.tokens_seen == 23
        assert latest.cumulative_update_seconds == 0.5
        assert latest.exposure["timing"]["optimizer_update_seconds"] is None
        assert latest.exposure["timing"]["training_loop_elapsed_seconds"] is None


def test_public_report_retains_exposure_without_work_or_rewrites(
    fake_lifecycle, monkeypatch, tmp_path, capsys
):
    from test_report_front_door import _report, _snapshot

    from sparselab.training.checkpoints import CheckpointManager

    f = fake_lifecycle
    f.trainer.train(f.config, run_id="reported", stop_after_step=1)
    run = f.config.logging.root_dir / "reported"
    expected = f.snapshots[-1].exposure
    # Fixture verification stands in for weights; native save/load authentication
    # is exercised separately without a model forward pass.
    for generation, exposure in enumerate((None, expected)):
        path = run / "checkpoints" / f"step_00000001_gen_{generation:08d}"
        path.mkdir()
        payload = {
            "step": 1,
            "tokens_seen": 8,
            "validation_loss": None,
            "sha256": "b" * 64,
        }
        if exposure is not None:
            payload["exposure"] = exposure
        (path / "manifest.json").write_text(json.dumps(payload))
    monkeypatch.setattr(
        CheckpointManager,
        "verify",
        lambda *a, **k: SimpleNamespace(valid=True, errors=[], resume_level="weights"),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("report attempted model or training work")

    monkeypatch.setattr(f.trainer, "train", forbidden)
    monkeypatch.setattr(f.trainer.PyTorchEngine, "__init__", forbidden)
    monkeypatch.setattr(CheckpointManager, "load", forbidden)
    capsys.readouterr()
    before = _snapshot(tmp_path)
    _report(
        monkeypatch,
        tmp_path / "absent-work",
        "reported",
        "--kind",
        "run",
        "--runs-dir",
        str(f.config.logging.root_dir),
        "--json",
    )
    rows = json.loads(capsys.readouterr().out)["evidence"]["checkpoints"]
    assert [row["exposure"] for row in rows] == [None, expected]
    assert _snapshot(tmp_path) == before


def test_artifact_locations_do_not_change_exposure_protocol():
    dataset = TokenBlockDataset(np.arange(13, dtype=np.int32), 4)
    names = ["data/manifest.json", "data/train.npy", "tokenizer.json"]
    artifacts = tuple(ArtifactIdentity(name, "a" * 64, 1) for name in names)
    first = exposure_context(
        dataset, artifacts, seed=7, run_id="one", continuation="FRESH", prior=None
    )
    windows = tuple(
        ArtifactIdentity(a.relative_path.replace("/", "\\"), a.sha256, a.size_bytes)
        for a in artifacts
    )
    other = exposure_context(
        dataset, windows, seed=7, run_id="two", continuation="FRESH", prior=None
    )
    assert first["protocol_sha256"] == other["protocol_sha256"]
    with pytest.raises(KeyError):
        exposure_context(
            dataset,
            artifacts[:-1],
            seed=7,
            run_id="missing",
            continuation="FRESH",
            prior=None,
        )
