"""Independent CLI processes authenticate the same scientific input identities."""

from __future__ import annotations

import json
import subprocess
import sys

from test_training import config as training_config

from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer
from sparselab.experiments.plan import ExperimentPlan
from sparselab.training.manifest import sha256_file


def test_lock_explain_warm_and_cold_identity_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "private-config"))
    config = training_config(tmp_path)
    prepared = prepare_data(config, load_tokenizer(config.tokenizer.path))
    plan = ExperimentPlan.model_validate(
        {
            "plan_version": 1,
            "id": "cli-reuse",
            "base_run": config,
            "artifacts": {
                "tokenizer": {
                    "kind": "tokenizer",
                    "version": 1,
                    "producer": "fixture",
                    "identifier": config.tokenizer.path.parent.name,
                    "sha256": sha256_file(config.tokenizer.path),
                    "path": str(config.tokenizer.path),
                },
                "packed": {
                    "kind": "prepared_data",
                    "version": 1,
                    "producer": "fixture",
                    "identifier": prepared.manifest["settings_sha256"],
                    "sha256": prepared.manifest["manifest_sha256"],
                    "path": str(prepared.root),
                },
            },
            "inputs": {"tokenizer": "tokenizer", "training": "packed"},
        }
    )
    source = tmp_path / "plan.yaml"
    source.write_text(plan.model_dump_json())

    def command(*args, check=True):
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "from sparselab.cli.entry import main; main()",
                "--work-dir",
                str(tmp_path),
                "experiment",
                *map(str, args),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=check,
        )
        return process, json.loads(process.stdout)

    _, locked = command("lock", source)
    path = tmp_path / "experiments/cli-reuse/locks" / (locked["plan_sha256"] + ".json")
    _, warm = command("explain", path)
    _, cold = command("explain", path, "--cold-verify")
    for key in ("scientific_sha256", "plan_sha256"):
        assert locked[key] == warm[key] == cold[key]
    assert warm["verification"]["proof_hits"] == 2
    assert cold["verification"] == {
        "mode": "cold",
        "proof_hits": 0,
        "proof_misses": 0,
        "proof_records": 0,
    }
    array = prepared.root / "train.npy"
    raw = array.read_bytes()
    array.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    for flags in ((), ("--cold-verify",)):
        process, result = command("explain", path, *flags, check=False)
        assert process.returncode == 2
        assert result["status"] == "error"
        assert "digest mismatch" in result["error"]
