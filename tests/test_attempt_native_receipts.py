"""Native receipt reconciliation fixtures; no model or generator is invoked."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab.training import attempt_receipts


def test_ledger_bound_train_suppresses_unaccounted_diagnostic_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sparselab.training.trainer import _attempt_contract_suppresses_triage

    monkeypatch.delenv("SPARSELAB_ATTEMPT_BUDGET_LEDGER", raising=False)
    monkeypatch.delenv("SPARSELAB_ATTEMPT_ACTIVITY", raising=False)
    assert _attempt_contract_suppresses_triage() is False
    monkeypatch.setenv("SPARSELAB_ATTEMPT_BUDGET_LEDGER", "/isolated/ledger.sqlite")
    monkeypatch.setenv("SPARSELAB_ATTEMPT_ACTIVITY", "train")
    assert _attempt_contract_suppresses_triage() is True
    monkeypatch.setenv("SPARSELAB_ATTEMPT_ACTIVITY", "inspect")
    assert _attempt_contract_suppresses_triage() is False


def _fresh_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    run = tmp_path / "run_one"
    run.mkdir()
    (run / "manifest.json").write_text("mock")
    manifest = {
        "run_id": run.name,
        "continuation_kind": "FRESH",
        "parent_run_id": None,
    }
    monkeypatch.setattr(attempt_receipts, "read_manifest", lambda _path: manifest)
    from sparselab.evaluation import evidence

    digest = attempt_receipts.hashlib.sha256(
        attempt_receipts.canonical_json(manifest)
    ).hexdigest()
    (run / "progress.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "manifest_sha256": digest,
                "step": 2,
                "tokens_seen": 7,
                "latest": {
                    "relative_path": "step_2_gen_1",
                    "manifest_sha256": "a" * 64,
                },
            }
        )
    )
    monkeypatch.setattr(
        evidence,
        "experiment_evidence",
        lambda _path, **_kwargs: {
            "run_id": run.name,
            "verified_checkpoints": True,
            "checkpoints": [
                {
                    "path": "step_2_gen_1",
                    "digest": "a" * 64,
                    "step": 2,
                    "tokens_seen": 7,
                }
            ],
        },
    )
    return run


def test_training_counters_require_verified_terminal_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _fresh_run(tmp_path, monkeypatch)
    assert attempt_receipts.verify_native_phase_counters(
        "train", run, expected_content_sha256="b" * 64
    ) == (2, 7, 0, 0)
    progress = json.loads((run / "progress.json").read_text())
    progress["tokens_seen"] = 8
    (run / "progress.json").write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="terminal counters"):
        attempt_receipts.verify_native_phase_counters(
            "train", run, expected_content_sha256="b" * 64
        )


def test_resumed_frontier_subtracts_only_verified_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = _fresh_run(tmp_path, monkeypatch)
    parent = tmp_path / "parent"
    checkpoint = parent / "checkpoints" / "step_1_gen_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text(
        json.dumps({"sha256": "c" * 64, "step": 1, "tokens_seen": 3})
    )
    manifest = {
        "run_id": child.name,
        "continuation_kind": "RESUMED",
        "parent_run_id": parent.name,
        "checkpoint_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        attempt_receipts,
        "read_manifest",
        lambda path: manifest if path.parent == child else {"run_id": parent.name},
    )
    progress = json.loads((child / "progress.json").read_text())
    progress["manifest_sha256"] = attempt_receipts.hashlib.sha256(
        attempt_receipts.canonical_json(manifest)
    ).hexdigest()
    (child / "progress.json").write_text(json.dumps(progress))

    class FakeManager:
        def __init__(self, _run: Path, *, manifest_sha256: str):
            self.manifest_sha256 = manifest_sha256

        def verify(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(valid=True)

        def _resolve(self, path: Path) -> Path:
            return checkpoint if path.name == "latest.json" else path

    from sparselab.training import checkpoints

    monkeypatch.setattr(checkpoints, "CheckpointManager", FakeManager)
    assert attempt_receipts.verify_native_phase_counters(
        "train",
        child,
        parent_checkpoint_path=checkpoint,
        expected_content_sha256="b" * 64,
    ) == (1, 4, 0, 0)
    pointer = checkpoint.parent / "latest.json"
    pointer.write_text(json.dumps({"relative_path": checkpoint.name}))
    assert attempt_receipts.verify_native_phase_counters(
        "train",
        child,
        parent_checkpoint_path=pointer,
        expected_content_sha256="b" * 64,
    ) == (1, 4, 0, 0)
    manifest["checkpoint_sha256"] = "d" * 64
    progress["manifest_sha256"] = attempt_receipts.hashlib.sha256(
        attempt_receipts.canonical_json(manifest)
    ).hexdigest()
    (child / "progress.json").write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="parent identity"):
        attempt_receipts.verify_native_phase_counters(
            "train",
            child,
            parent_checkpoint_path=checkpoint,
            expected_content_sha256="b" * 64,
        )


def test_panel_counts_only_verified_complete_token_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.evaluation import panel

    result = tmp_path / "panel.json"
    result.write_text("mock")
    rows = [
        {"status": "COMPLETED", "token_ids": [1, 2]},
        {"status": "COMPLETED", "token_ids": []},
    ]
    monkeypatch.setattr(panel, "verify_panel_result", lambda _p, **_kw: {"rows": rows})
    assert attempt_receipts.verify_native_phase_counters(
        "panel", result, expected_content_sha256="b" * 64
    ) == (0, 0, 2, 2)
    rows[1]["status"] = "FAILED"
    with pytest.raises(ValueError, match="incomplete or failed"):
        attempt_receipts.verify_native_phase_counters(
            "panel", result, expected_content_sha256="b" * 64
        )


def test_campaign_receipt_rejects_wrong_scientific_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.campaign import engine as campaign_engine

    source = tmp_path / "campaign.yaml"
    source.write_text("mock")

    class FakeEngine:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.stages = {"run": SimpleNamespace(kind="experiment_run", plan="plan")}

        def inspect(self, _command: str) -> dict:
            return {
                "stages": [
                    {
                        "id": "run",
                        "state": "COMPLETE",
                        "availability": {"path": str(tmp_path / "run")},
                    }
                ]
            }

        def _lock(self, _rows: dict, _plan: str) -> SimpleNamespace:
            return SimpleNamespace(scientific_sha256="c" * 64)

    monkeypatch.setattr(campaign_engine, "CampaignEngine", FakeEngine)
    with pytest.raises(ValueError, match="scientific lock"):
        attempt_receipts.verify_native_phase_counters(
            "campaign_run",
            source,
            campaign_stage="run",
            campaign_work_dir=tmp_path,
            expected_content_sha256="b" * 64,
        )
