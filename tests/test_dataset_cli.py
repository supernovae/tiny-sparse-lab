"""Public authoring rejects retired loaders before creating execution state."""

import sys
from pathlib import Path

import pytest
import yaml

from sparselab.cli.main import build_parser, main


@pytest.mark.parametrize("command", ["train", "stage", "run", "data", "tokenizer"])
def test_retired_direct_source_rejected_before_side_effects(
    tmp_path, monkeypatch, command
):
    sample = Path(__file__).parents[1] / "experiments/samples/tinystories-microlab"
    filename = "tokenizer.yaml" if command == "tokenizer" else "run.yaml"
    config = yaml.safe_load((sample / filename).read_text())
    config["dataset"]["source"] = "tinystories"
    for key in ("train_path", "validation_path", "source_manifest_path", "license"):
        config["dataset"].pop(key)
    path = tmp_path / filename
    path.write_text(yaml.safe_dump(config))
    work = tmp_path / "work"
    argv = ["sparselab", "--work-dir", str(work), command]
    if command == "data":
        argv.append("prepare")
    elif command == "tokenizer":
        argv.append("train")
    argv.append(str(path))
    if command == "stage":
        argv.extend(["--output", str(tmp_path / "stage")])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="retired for new execution"):
        main()
    assert not work.exists()
    assert not (tmp_path / "stage").exists()


def test_explicit_snapshot_cli_has_no_dataset_specific_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "data",
            "snapshot",
            "source.lock.json",
            "--output",
            "snapshot",
            "--resume",
            "--json",
        ]
    )
    assert args.lock == Path("source.lock.json")
    assert args.resume and args.json
    with pytest.raises(SystemExit):
        parser.parse_args(["data", "snapshot", "snapshot", "--train-count", "10"])
