"""Model-free cancellation propagation through the lab arm boundary."""

from pathlib import Path

import pytest

from sparselab import lab_mode
from sparselab.config.loading import load_config
from sparselab.lab_context import LabCancelled, LabContext


@pytest.mark.parametrize("cancelled", [True, False])
def test_training_startup_interrupt_needs_actual_cancel_marker(
    tmp_path, monkeypatch, cancelled
):
    config = load_config(
        Path(__file__).resolve().parents[1] / "examples/lab-job-try/baseline.yaml"
    )
    config = config.model_copy(
        update={
            "logging": config.logging.model_copy(update={"root_dir": tmp_path / "runs"})
        }
    )
    context = LabContext(tmp_path / "CANCEL")

    def stop(*args, **kwargs):
        if cancelled:
            context.cancel_path.write_text("retain stop")
        raise InterruptedError("startup interrupted")

    def forbidden(*args, **kwargs):
        pytest.fail("startup interruption tried to read progress or score a model")

    monkeypatch.setattr(lab_mode, "_progress", forbidden)
    expected = LabCancelled if cancelled else InterruptedError
    with pytest.raises(expected) as caught:
        lab_mode._train_and_score(
            arm="baseline",
            config=config,
            run_id="never-created",
            reuse=False,
            key=None,
            protocol={},
            context=context,
            max_wall_seconds=None,
            authorization=None,
            tokenizer_batch_documents=1,
            tokenizer_batch_source_bytes=100,
            train_fn=stop,
        )
    assert not config.logging.root_dir.exists()
    if cancelled:
        assert caught.value.arm == "baseline"
        assert caught.value.phase == "training"
        assert caught.value.reason == "cancelled"
        assert caught.value.row is None
        assert context.cancel_path.read_text() == "retain stop"
    else:
        assert str(caught.value) == "startup interrupted"


def test_public_try_seals_startup_cancel_without_run_or_checkpoint(
    tmp_path, monkeypatch
):
    import json

    from test_training import config as tiny_config

    from sparselab.training import trainer

    config = tiny_config(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(config.model_dump_json())
    delta = tmp_path / "delta.json"
    delta.write_text(
        json.dumps({"question": "fixture only", "set": {"model.ffn_dim": 64}})
    )
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("cancelled try reached model initialization")

    def cancel_then_train(config, **kwargs):
        calls.append(kwargs["run_id"])
        kwargs["cancel_path"].write_text("retain try cancellation")
        return trainer.train(config, **kwargs)

    monkeypatch.setattr(trainer.PyTorchEngine, "initialize", forbidden)
    record, path = lab_mode.run_try(
        delta, baseline, work_dir=tmp_path / "work", train_fn=cancel_then_train
    )
    assert record["status"] == "interrupted"
    assert record["exit_status"] == lab_mode.EXIT_INTERRUPTED
    assert record["arms"] == {}
    assert record["interruption"] == {
        "arm": "baseline",
        "phase": "training",
        "reason": "cancelled",
    }
    assert len(calls) == 1
    assert not (tmp_path / "work/lab/runs" / calls[0]).exists()
    assert lab_mode.read_record(path) == record
    assert path.with_name("CANCEL").read_text() == "retain try cancellation"
