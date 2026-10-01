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


def test_registered_historical_documents_keep_their_byte_identities() -> None:
    """Old command examples can be frozen evidence; supersede, never rewrite."""
    import hashlib
    import json

    registry = json.loads(
        (REPO / "src/sparselab/research/resources/lifecycle.json").read_text()
    )
    checked = []
    for evidence in registry["evidence"]:
        artifact = evidence.get("artifact", {})
        relative = artifact.get("relative_path", "")
        if evidence["kind"] != "document" or not relative.startswith("docs/"):
            continue
        payload = (REPO / relative).read_bytes()
        assert len(payload) == artifact["size_bytes"], relative
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"], relative
        checked.append(relative)
    assert checked
