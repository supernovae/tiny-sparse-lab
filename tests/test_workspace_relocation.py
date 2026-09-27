import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from sparselab.training.metrics import ExperimentStore
from sparselab.workspace_relocation import consolidate_standalone_stores


def _source(root: Path, seed: int, status: str = "completed") -> Path:
    store = ExperimentStore(root)
    parent = None
    for suffix in ("parent", "child"):
        run_id = f"seed{seed}-{suffix}"
        run = root / run_id
        run.mkdir()
        config = {"seed": seed, "logging": {"root_dir": str(root)}}
        store.create_run(run_id, config, {}, parent_run_id=parent)
        manifest = {"run_id": run_id, "parent_run_id": parent, "config": config}
        payload = json.dumps(manifest).encode()
        (run / "manifest.json").write_bytes(payload)
        store.register_manifest(run_id, hashlib.sha256(payload).hexdigest(), manifest)
        store.log_metrics(run_id, 1, 16, 0.5, {"loss": 1.25})
        store.log_event(run_id, 1, 16, 0.5, "historical_location", {"path": str(run)})
        store.finish_run(run_id, status, str(run / "checkpoints/latest.json"))
        parent = run_id
    return root


def test_three_historical_stores_consolidate_without_rewriting_identity(tmp_path):
    sources = [_source(tmp_path / f"old{seed}", seed) for seed in (17, 42, 73)]
    # Key by run ID because every leaf manifest uses the same filename.
    original = {
        path.parent.name: (path.read_bytes(), path.stat().st_ino)
        for source in sources
        for path in source.glob("*/manifest.json")
    }
    destination = tmp_path / "external-disk/experiments/study/runs"
    consolidate_standalone_stores(sources, destination)
    for source in sources:
        for run in source.iterdir():
            if run.is_dir():
                run.rename(destination / run.name)
    with sqlite3.connect(destination / "experiments.sqlite3") as connection:
        rows = connection.execute(
            "SELECT run_id,parent_run_id,config_json,latest_checkpoint FROM runs"
        ).fetchall()
        assert len(rows) == 6
        for run_id, parent, config, pointer in rows:
            path = destination / run_id / "manifest.json"
            assert (path.read_bytes(), path.stat().st_ino) == original[run_id]
            assert json.loads(config)["logging"]["root_dir"] != str(destination)
            assert Path(pointer).is_relative_to(destination)
            if run_id.endswith("child"):
                assert parent == run_id.replace("child", "parent")
        histories = connection.execute("SELECT payload_json FROM events").fetchall()
        assert len(histories) == 6
        assert all("old" in json.loads(row[0])["path"] for row in histories)
        assert connection.execute(
            "SELECT COUNT(DISTINCT origin_id) FROM ingested_records"
        ).fetchone() == (3,)
    assert all((source / "experiments.sqlite3").is_file() for source in sources)


def test_active_and_duplicate_runs_rejected_before_creating_destination(tmp_path):
    active = _source(tmp_path / "active", 17, "running")
    destination = tmp_path / "new/runs"
    with pytest.raises(ValueError, match="not terminal"):
        consolidate_standalone_stores([active], destination)
    assert not destination.exists()
    sources = [_source(tmp_path / f"old{i}", 42) for i in range(2)]
    with pytest.raises(ValueError, match="duplicate run ID"):
        consolidate_standalone_stores(sources, destination)
    assert not destination.exists()
