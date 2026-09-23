from __future__ import annotations

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml

from sparselab.workers import cli

ROOT = Path(__file__).resolve().parents[1]


def test_cli_matrix_dry_run_needs_no_assets_or_store(tmp_path: Path) -> None:
    base = yaml.safe_load((ROOT / "configs/runtime_smoke_cpu.yaml").read_text())
    base["tokenizer"]["path"] = str(tmp_path / "missing-tokenizer.json")
    base["dataset"]["cache_dir"] = str(tmp_path / "unprepared-cache")
    base["logging"]["root_dir"] = str(tmp_path / "uncreated-runs")
    config = tmp_path / "base.yaml"
    config.write_text(yaml.safe_dump(base))
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "matrix_version": 1,
                "base_config": str(config),
                "axes": {
                    "seed": [
                        {"label": f"seed-{seed}", "set": {"seed": seed}}
                        for seed in (7, 17, 41)
                    ]
                },
            },
            sort_keys=False,
        )
    )
    store = tmp_path / "controller"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sparselab.cli.main import main; main()",
            "experiment",
            "submit",
            "--matrix",
            str(matrix),
            "--dry-run",
            "--store",
            str(store),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    records = json.loads(result.stdout)
    assert [record["coordinate"] for record in records] == [
        {"seed": "seed-7"},
        {"seed": "seed-17"},
        {"seed": "seed-41"},
    ]
    assert len({record["config_sha256"] for record in records}) == 3
    assert not store.exists()
    assert not (tmp_path / "unprepared-cache").exists()
    assert not (tmp_path / "uncreated-runs").exists()


def test_composed_run_observes_reconnection_and_artifact_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter(
        [
            ("QUEUED", "PENDING"),
            ("RUNNING", "PENDING"),
            ("UNKNOWN", "PENDING"),
            ("RUNNING", "PENDING"),
            ("CANCELLED", "PENDING"),
            ("CANCELLED", "ERROR"),
            ("CANCELLED", "COMPLETE"),
        ]
    )

    class ExistingController:
        poll_seconds = 0.001

        def _process_lock(self, *, optional: bool):
            assert optional
            return nullcontext(False)

        def tick(self) -> None:
            raise AssertionError("another controller owns scheduling")

        def list_experiments(self) -> list[dict[str, str]]:
            status, ingestion = next(states)
            return [
                {"run_id": "waiting", "status": status, "ingestion_status": ingestion}
            ]

    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    result = cli._wait_for_run(ExistingController(), "waiting")
    assert result["status"] == "CANCELLED"
    assert result["ingestion_status"] == "COMPLETE"
