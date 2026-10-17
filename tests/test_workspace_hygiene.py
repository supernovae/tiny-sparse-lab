"""Workspace routing checks; execution is stubbed and no model is trained."""

import subprocess
from pathlib import Path

import pytest

from sparselab.cli import main as cli

REPO = Path(__file__).resolve().parents[1]


def test_reference_exercise_defaults_inside_selected_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.evaluation import reference_exercise

    work = tmp_path / "sparselab-work" / "experiments" / "dense-lm-v1"
    persistent_root = tmp_path / "persistent-state"
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(persistent_root))
    observation = {"identity": "fixed", "historical_location": "runs-old/parent"}
    monkeypatch.setattr(
        reference_exercise, "exercise_checkpoint", lambda *a, **kw: observation
    )
    command = [
        "model",
        "exercise",
        "seed42-child",
        "--checkpoint",
        str(work / "runs" / "seed42-child" / "checkpoints" / "generation"),
        "--runs-dir",
        str(work / "runs"),
    ]
    args = cli.build_parser().parse_args(command)
    args.handler(args)
    outputs = list((work / "exercises").glob("*.json"))
    assert len(outputs) == 1
    import json

    assert json.loads(outputs[0].read_text()) == observation
    explicit = tmp_path / "custom-output.json"
    args = cli.build_parser().parse_args([*command, "--output", str(explicit)])
    args.handler(args)
    assert explicit.read_bytes() == outputs[0].read_bytes()
    assert not list(tmp_path.glob("runs*"))
    # Explicit run and output destinations must not be reinterpreted as paths
    # under the selected persistent root.
    assert not persistent_root.exists()


def test_workspace_is_git_ignored() -> None:
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "sparselab-work/experiments/example/runs/run/manifest.json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_resource_retirement_records_originals_and_destination() -> None:
    import json

    record = json.loads((REPO / "docs/research-resource-retirement.json").read_text())
    assert len(record["files"]) == 18
    assert len({row["source"] for row in record["files"]}) == 18
    for row in record["files"]:
        assert row["destination"] == "provenance/lab/" + row["source"]
        assert len(row["sha256"]) == 64
        assert row["size_bytes"] > 0
        if not row["source"].endswith("/lifecycle.json") and "disposition" not in row:
            assert not (REPO / row["source"]).exists()
    assert record["lifecycle_supplement"] == "provenance/lab-lifecycle-supplement.json"
