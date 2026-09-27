"""Workspace routing checks; execution is stubbed and no model is trained."""

import subprocess
from pathlib import Path

import pytest

from sparselab.cli import main as cli
from sparselab.config import load_config
from sparselab.training.manifest import config_sha256

REPO = Path(__file__).resolve().parents[1]


def test_default_read_and_controller_store_is_ignored_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.chdir(tmp_path)
    parser = cli.build_parser()
    expected = tmp_path / "sparselab-work" / "runs"
    assert Path(parser.parse_args(["eval", "example"]).runs_dir) == expected
    assert Path(parser.parse_args(["controller", "run"]).store) == expected
    custom = tmp_path / "external-disk" / "experiment" / "runs"
    assert (
        Path(parser.parse_args(["eval", "example", "--runs-dir", str(custom)]).runs_dir)
        == custom
    )
    assert not expected.exists()  # Parsing never launches work.


@pytest.mark.parametrize("resume", [False, True])
def test_train_explicit_store_preserves_config_identity_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    config_path = REPO / "configs" / "dense_lm_v1.yaml"
    original = load_config(config_path)
    work = tmp_path / "external-disk" / "experiments" / "dense-lm-v1"
    store = work / "runs"
    checkpoint = store / "parent" / "checkpoints" / "generation" / "manifest.json"
    captured = []

    def record(config, **kwargs):
        captured.append((config, kwargs))
        return config.logging.root_dir / kwargs["run_id"]

    monkeypatch.setattr(cli, "train", record)
    command = ["train", str(config_path), "--run-id", "child", "--runs-dir", str(store)]
    if resume:
        command.extend(["--resume", str(checkpoint)])
    args = cli.build_parser().parse_args(command)
    args.handler(args)
    config, kwargs = captured.pop()
    assert config.logging.root_dir == store
    assert config_sha256(config.model_dump(mode="json")) == config_sha256(
        original.model_dump(mode="json")
    )
    assert kwargs["resume"] == (checkpoint if resume else None)
    # Explicit historical config roots retain their meaning without an override.
    args = cli.build_parser().parse_args(
        ["train", str(config_path), "--run-id", "legacy"]
    )
    args.handler(args)
    assert captured[0][0].logging.root_dir == original.logging.root_dir
    assert not list(tmp_path.glob("runs*"))


def test_reference_exercise_defaults_inside_selected_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.evaluation import reference_exercise

    work = tmp_path / "sparselab-work" / "experiments" / "dense-lm-v1"
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
