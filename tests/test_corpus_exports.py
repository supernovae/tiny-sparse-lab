"""Frozen release exports exercise real ingestion, fitting and packing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.corpus.acquisition import acquire
from sparselab.corpus.export import export_release, verify_release_export
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import load_project
from sparselab.corpus.release import freeze
from sparselab.data.conversations import iter_rendered_conversations
from sparselab.data.datasets import iter_documents
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer, train_tokenizer


@pytest.fixture(scope="module")
def frozen_sample(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("forge-export")
    project = load_project(Path("corpora/devmind-sample-v0/corpus.yaml"))
    acquire(project, root)
    build_dir = build(project, root, offline=True)
    assert build(project, root, offline=True) == build_dir
    release_dir = freeze(build_dir, root)
    assert freeze(build_dir, root) == release_dir
    return release_dir, root


def test_lm_export_fits_every_train_document(frozen_sample: tuple[Path, Path]) -> None:
    release, root = frozen_sample
    exported = export_release(
        release, "lm", Path("configs/runtime_smoke_cpu.yaml"), 300, root
    )
    assert exported == export_release(
        release, "lm", Path("configs/runtime_smoke_cpu.yaml"), 300, root
    )
    tokenizer_config = load_tokenizer_config(exported / "tokenizer.yaml")
    run_config = load_config(exported / "run.yaml")
    dataset = run_config.dataset
    assert dataset.source == "local_text"
    binding = verify_release_export(dataset)
    assert binding["release_id"] == release.name
    assert list(iter_documents(dataset, "train")) == [
        json.loads(row)["text"]
        for row in (release / "lm/train.jsonl").read_text().splitlines()
    ]
    artifact = train_tokenizer(tokenizer_config)
    manifest = json.loads((artifact.parent / "tokenizer_manifest.json").read_text())
    assert manifest["corpus_export"] == binding
    assert manifest["selected_documents"] == binding_split_count(exported, "train")
    prepared = prepare_data(run_config, load_tokenizer(artifact))
    assert prepared.manifest["corpus_export"] == binding
    assert prepared.manifest["train"]["retained_documents"] == binding_split_count(
        exported, "train"
    )
    assert prepared.manifest["train"]["truncated_documents"] == 0
    assert prepare_data(run_config, load_tokenizer(artifact)).root == prepared.root


def test_changed_base_config_uses_distinct_export(
    frozen_sample: tuple[Path, Path], tmp_path: Path
) -> None:
    release, root = frozen_sample
    original = Path("configs/runtime_smoke_cpu.yaml")
    changed = tmp_path / "base.yaml"
    changed.write_text(
        original.read_text(encoding="utf-8").replace("seed: 7", "seed: 11"),
        encoding="utf-8",
    )
    first = export_release(release, "lm", original, 300, root)
    second = export_release(release, "lm", changed, 300, root)
    assert first != second
    assert (
        verify_release_export(load_config(second / "run.yaml").dataset)["release_id"]
        == release.name
    )


def binding_split_count(exported: Path, split: str) -> int:
    return json.loads((exported / "export.json").read_text())["splits"][split][
        "records"
    ]


def test_chat_export_preserves_v2_records(frozen_sample: tuple[Path, Path]) -> None:
    release, root = frozen_sample
    exported = export_release(
        release, "chat", Path("configs/runtime_smoke_cpu.yaml"), 300, root
    )
    dataset = load_config(exported / "run.yaml").dataset
    assert dataset.source == "local_chat"
    assert verify_release_export(dataset)["view"] == "chat"
    for split in ("train", "validation"):
        path = dataset.train_path if split == "train" else dataset.validation_path
        assert path is not None
        assert list(iter_documents(dataset, split)) == [
            row.text for row in iter_rendered_conversations(path)
        ]
        assert all(
            set(json.loads(row)) == {"format_version", "loss_mode", "messages"}
            for row in path.read_text().splitlines()
        )


def test_local_text_rejects_malformed_rows(tmp_path: Path) -> None:
    config = load_config(Path("configs/runtime_smoke_cpu.yaml")).dataset.model_copy(
        update={
            "source": "local_text",
            "train_path": tmp_path / "train.jsonl",
            "validation_path": tmp_path / "validation.jsonl",
            "license": "MIT",
        }
    )
    for bad in (
        '{"text": ""}\n',
        '{"text": "x", "extra": 1}\n',
        '{"text": 2}\n',
        "not-json\n",
    ):
        config.train_path.write_text(bad)
        with pytest.raises(ValueError, match="local_text"):
            list(iter_documents(config, "train"))


def test_cli_export_authenticates_release_once(
    frozen_sample: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    import argparse

    from sparselab.corpus import release as publication
    from sparselab.corpus.cli import _handle

    release, _ = frozen_sample
    real_files = publication._files
    scans = 0

    def observe_files(root: Path, inventory: dict) -> None:
        nonlocal scans
        if root == release:
            scans += 1
        real_files(root, inventory)

    monkeypatch.setattr(publication, "_files", observe_files)
    _handle(
        argparse.Namespace(
            work_dir=tmp_path / "work",
            corpus_command="export",
            release=str(release),
            view="lm",
            base_run_config="configs/runtime_smoke_cpu.yaml",
            vocab_size=300,
        )
    )
    exported = Path(capsys.readouterr().out.strip())
    run = load_config(exported / "run.yaml")
    assert run.dataset.corpus_release_path == release
    assert (
        json.loads((exported / "export.json").read_bytes())["release_id"]
        == release.name
    )
    assert scans == 1
