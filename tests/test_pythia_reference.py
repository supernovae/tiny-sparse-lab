from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparselab.config.loading import load_config
from sparselab.research.pythia import (
    DEFAULT_STEPS,
    PYTHIA_70M_DEDUPED_CHECKPOINTS,
    PYTHIA_LICENSE,
    PYTHIA_MODEL_ID,
    REPORT_FORMAT,
    REPORT_VERSION,
    _evaluate_checkpoint,
    _validate_model_metadata,
    _validate_snapshot,
    _write_exclusive_json,
    checkpoint_for_step,
    validate_reference_config,
)


def _fineweb_config():
    config = load_config(Path("configs/smoke_cpu.yaml"))
    dataset = config.dataset.model_copy(
        update={
            "source": "fineweb_edu",
            "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
            "dataset_config": "sample-10BT",
        }
    )
    return config.model_copy(update={"dataset": dataset})


def test_registered_pythia_steps_are_fixed_official_commits() -> None:
    assert PYTHIA_MODEL_ID == "EleutherAI/pythia-70m-deduped"
    assert PYTHIA_LICENSE == "Apache-2.0"
    assert DEFAULT_STEPS == ("step0", "step10000", "step143000")
    assert {
        step: checkpoint.commit
        for step, checkpoint in PYTHIA_70M_DEDUPED_CHECKPOINTS.items()
    } == {
        "step0": "c913ae980de9355947d0bf73f9f10d580eb79301",
        "step10000": "c890c8f6d8f86c36b2af66c3012a14ef1d35d3f3",
        "step143000": "9a7c847e93250c8f24d4b7e7134dbf369e8fc9cb",
    }
    with pytest.raises(ValueError, match="unsupported Pythia step"):
        checkpoint_for_step("step1")


def test_reference_requires_pinned_fineweb_edu() -> None:
    config = _fineweb_config()
    validate_reference_config(config)

    missing_revision = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"revision": None})}
    )
    with pytest.raises(ValueError, match="FineWeb-Edu"):
        validate_reference_config(missing_revision)


def test_result_writer_preserves_machine_readable_identity(tmp_path: Path) -> None:
    output = tmp_path / "reference.json"
    payload = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "model": {"id": PYTHIA_MODEL_ID, "license": PYTHIA_LICENSE},
        "dataset": {
            "source": "fineweb_edu",
            "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
            "dataset_config": "sample-10BT",
            "content_sha256": "a" * 64,
            "validation_max_documents": 2,
            "validation_max_tokens": 64,
        },
        "checkpoints": [
            {
                "step": "step0",
                "commit": checkpoint_for_step("step0").commit,
                "loss": 1.0,
                "loss_sum": 2.0,
                "target_tokens": 2,
            }
        ],
    }

    assert _write_exclusive_json(output, payload) == output
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == payload
    assert written["model"]["license"] == "Apache-2.0"
    assert written["dataset"]["content_sha256"] == "a" * 64
    with pytest.raises(FileExistsError, match="already exists"):
        _write_exclusive_json(output, payload)


def _minimal_snapshot(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gpt_neox",
                "architectures": ["GPTNeoXForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"")
    return path


@pytest.mark.parametrize("filename", ["pytorch_model.bin", "modeling_custom.py"])
def test_snapshot_rejects_pickle_and_python_files(
    tmp_path: Path, filename: str
) -> None:
    snapshot = _minimal_snapshot(tmp_path / "snapshot")
    (snapshot / filename).write_bytes(b"unsafe")

    with pytest.raises(ValueError, match="disallowed file"):
        _validate_snapshot(snapshot)


@pytest.mark.parametrize("field", ["auto_map", "quantization_config"])
def test_snapshot_metadata_rejects_remote_code_and_quantization(
    tmp_path: Path, field: str
) -> None:
    snapshot = _minimal_snapshot(tmp_path / "snapshot")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    config[field] = {"AutoModel": "custom.module.Model"}
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="forbids remote code or quantization"):
        _validate_model_metadata(snapshot)


def test_checkpoint_windows_respect_context_and_count_every_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_lengths: list[int] = []

    class _Config:
        max_position_embeddings = 2

        @classmethod
        def from_pretrained(cls, *_: object, **__: object) -> _Config:
            return cls()

    class _Model:
        def to(self, _: torch.device) -> _Model:
            return self

        def eval(self) -> None:
            return None

        def __call__(self, *, input_ids: torch.Tensor) -> SimpleNamespace:
            input_lengths.append(input_ids.shape[1])
            return SimpleNamespace(
                logits=torch.zeros((*input_ids.shape, 4), device=input_ids.device)
            )

    class _ModelLoader:
        @classmethod
        def from_pretrained(cls, *_: object, **__: object) -> _Model:
            return _Model()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(GPTNeoXConfig=_Config, GPTNeoXForCausalLM=_ModelLoader),
    )
    result = _evaluate_checkpoint(
        checkpoint_for_step("step0"),
        tmp_path,
        torch.device("cpu"),
        ((0, 1, 2, 3),),
        2,
    )

    assert result["target_tokens"] == 3
    assert input_lengths and max(input_lengths) <= 2
