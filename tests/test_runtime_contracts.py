"""Golden reader contracts for externally stored runtime/staging records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparselab.training.manifest import canonical_json, read_manifest
from sparselab.training.metrics import ExperimentStore, _envelope_digest
from sparselab.training.stages import ExperimentStage, StageHistory

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


def _json(name: str) -> dict[str, object]:
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_hash_verified_manifest_fixture_is_readable_and_rejects_newer_version(
    tmp_path: Path,
) -> None:
    fixture = FIXTURES / "run_manifest_v1.json"
    path = tmp_path / "manifest.json"
    path.write_bytes(fixture.read_bytes())
    manifest = read_manifest(path)

    corrupted = _json("run_manifest_v1.json")
    corrupted["continuation_kind"] = "FRESH"
    path.write_bytes(canonical_json(corrupted) + b"\n")
    with pytest.raises(ValueError):
        read_manifest(path)
    incompatible = dict(manifest)
    incompatible["manifest_version"] = 2
    incompatible["sha256"] = (
        __import__("hashlib").sha256(canonical_json(incompatible)).hexdigest()
    )
    path.write_bytes(canonical_json(incompatible) + b"\n")
    with pytest.raises(ValueError, match="unsupported manifest version"):
        read_manifest(path)


def test_stage_fixture_replays_legal_history_and_rejects_post_complete_transition() -> (
    None
):
    records = json.loads(
        (FIXTURES / "stage_history_v1.json").read_text(encoding="utf-8")
    )
    assert isinstance(records, list)
    history = StageHistory()
    for value in records:
        assert isinstance(value, dict)
        stage = ExperimentStage(value["stage"])
        history.start(
            stage,
            step=value["step"],
            tokens_seen=value["tokens_seen"],
            payload=value["payload"],
        )
        history.finish(
            value["status"],
            reason=value["reason"],
            step=value["step"],
            tokens_seen=value["tokens_seen"],
            payload=value["payload"],
        )
    assert [record.stage.value for record in history.records] == [
        value["stage"] for value in records
    ]
    with pytest.raises(ValueError, match="illegal stage transition"):
        history.start(ExperimentStage.TRAINING, step=20, tokens_seen=640)


def test_outbox_fixture_import_is_idempotent_and_rejects_unknown_version(
    tmp_path: Path,
) -> None:
    envelope = _json("outbox_envelope_v1.json")
    store = ExperimentStore(tmp_path)
    store.import_records([envelope])
    store.import_records([envelope])
    assert store.export_records()["records"] == []

    incompatible = dict(envelope)
    incompatible["schema_version"] = 2
    incompatible["sha256"] = _envelope_digest(incompatible)
    with pytest.raises(ValueError, match="unsupported"):
        store.import_records([incompatible])
