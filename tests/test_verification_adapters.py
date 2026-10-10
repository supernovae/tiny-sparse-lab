"""Zero-model CLI adapters: hand-written snapshots and vocabulary, no fitting."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from test_kml_card03_items import _fixture, _write
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from sparselab.cli import main as cli
from sparselab.config.loading import load_config
from sparselab.data.local_stories import DATASET, LICENSE, REVISION
from sparselab.data.tokenizer import SPECIAL_TOKENS
from sparselab.data.tokenizer_cli import verify_configured_tokenizer
from sparselab.evaluation import kml_card03_items as items
from sparselab.training.manifest import sha256_file


def _tokenizer_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    splits = {}
    for split in ("train", "validation"):
        text = f"tiny {split} text"
        path = source / f"{split}.jsonl"
        _write(path, {"ordinal": 0, "text": text})
        encoded = text.encode()
        splits[split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "count": 1,
            "text_bytes": len(encoded),
            "source_records_consumed": 1,
            "duplicates": 0,
            "overlap": 0,
            "empty": 0,
            "content_sha256": hashlib.sha256(
                len(encoded).to_bytes(8, "big") + encoded
            ).hexdigest(),
        }
    excluded = source / "excluded.jsonl"
    excluded.write_text("")
    origin = source / "manifest.json"
    _write(
        origin,
        {
            "schema_version": 1,
            "source": DATASET,
            "revision": REVISION,
            "license": LICENSE,
            "splits": splits,
            "excluded": {
                "path": excluded.name,
                "sha256": sha256_file(excluded),
                "count": 0,
            },
        },
    )
    tokenizer = tmp_path / "tokenizer.json"
    vocab = dict(zip(SPECIAL_TOKENS, range(4), strict=True))
    vocab.update({f"token-{i}": i for i in range(4, 512)})
    Tokenizer(WordLevel(vocab, unk_token="<unk>")).save(str(tokenizer))
    _write(
        tmp_path / "tokenizer_manifest.json",
        {
            "sha256": sha256_file(tokenizer),
            "vocab_size": 512,
            "special_ids": dict(zip(SPECIAL_TOKENS, range(4), strict=True)),
            "source": "local_stories",
            "revision": REVISION,
            "source_manifest_sha256": sha256_file(origin),
        },
    )
    config = load_config(Path("configs/runtime_smoke_cpu.yaml")).model_dump(mode="json")
    config["tokenizer"]["path"] = str(tokenizer)
    config["dataset"].update(
        source="local_stories",
        revision=REVISION,
        license=LICENSE,
        train_path=str(source / "train.jsonl"),
        validation_path=str(source / "validation.jsonl"),
        source_manifest_path=str(origin),
        cache_dir=str(tmp_path / "unused-cache"),
    )
    path = tmp_path / "run.yaml"
    _write(path, config)
    return path


def test_tokenizer_verify_main_is_read_only(tmp_path, monkeypatch, capsys):
    config = _tokenizer_fixture(tmp_path)
    from sparselab.data import tokenizer

    monkeypatch.setattr(
        tokenizer, "train_tokenizer", lambda *a, **k: pytest.fail("fit")
    )
    monkeypatch.setattr(
        tokenizer, "iter_documents", lambda *a, **k: pytest.fail("source replay")
    )
    work = tmp_path / "absent-work"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "--work-dir",
            str(work),
            "tokenizer",
            "verify",
            str(config),
            "--json",
        ],
    )
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["vocab_size"] == 512
    assert result["special_ids"] == dict(zip(SPECIAL_TOKENS, range(4), strict=True))
    assert result["origin"]["source_manifest_sha256"] == sha256_file(
        tmp_path / "source/manifest.json"
    )
    assert not work.exists()
    assert before == {
        str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "change",
    ["missing", "tamper", "underfilled", "special", "config", "source", "unbound"],
)
def test_tokenizer_verify_fails_closed(tmp_path, change):
    config = _tokenizer_fixture(tmp_path)
    path = tmp_path / "tokenizer.json"
    manifest_path = tmp_path / "tokenizer_manifest.json"
    if change == "missing":
        manifest_path.unlink()
    elif change == "tamper":
        path.write_bytes(path.read_bytes() + b" ")
    elif change in {"underfilled", "special"}:
        raw = json.loads(path.read_text())
        if change == "underfilled":
            del raw["model"]["vocab"]["token-511"]
        else:
            raw["model"]["vocab"]["<bos>"], raw["model"]["vocab"]["<eos>"] = 3, 2
        _write(path, raw)
        manifest = json.loads(manifest_path.read_text())
        manifest["sha256"] = sha256_file(path)
        _write(manifest_path, manifest)
    elif change == "source":
        (tmp_path / "source/train.jsonl").write_text('{"ordinal":0,"text":"changed"}\n')
    else:
        raw = json.loads(config.read_text())
        if change == "config":
            raw["model"]["vocab_size"] = 513
        else:
            raw["dataset"] = {
                "source": "synthetic",
                "cache_dir": str(tmp_path),
                "train_max_documents": 1,
                "validation_max_documents": 1,
                "train_max_tokens": 4,
                "validation_max_tokens": 4,
            }
        _write(config, raw)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_configured_tokenizer(config)


def _complete_draft(data):
    template = deepcopy(data["items"][0])
    data["items"] = []
    for suite, categories, count in (
        ("closed_book", items.LANGUAGE_CATEGORIES, 20),
        ("open_book", items.EVIDENCE_CATEGORIES, 40),
    ):
        for category in categories:
            for index in range(count):
                item = deepcopy(template)
                item.update(
                    id=f"{suite}/{category}/{index}", suite=suite, category=category
                )
                if suite == "closed_book" or category == "missing_ambiguous_evidence":
                    item["support_chunk_ids"] = []
                    item["controls"] = dict.fromkeys(item["controls"])
                if category == "missing_ambiguous_evidence":
                    item.update(
                        answerability="clarify" if index % 2 else "unanswerable",
                        clarification_target="Which alarm?",
                    )
                item["content_sha256"] = items._sha(
                    {k: v for k, v in item.items() if k != "content_sha256"}
                )
                data["items"].append(item)
    return data


def _freeze_args(release, families, draft, output):
    return cli.build_parser().parse_args(
        [
            "evaluation",
            "freeze-items",
            "--release",
            str(release),
            "--family-inventory",
            str(families),
            "--draft",
            str(draft),
            "--output",
            str(output),
            "--json",
        ]
    )


def test_freeze_cli_complete_digests_and_collision(tmp_path, monkeypatch, capsys):
    release, families, data = _fixture(tmp_path, monkeypatch)
    draft = tmp_path / "draft.json"
    _write(draft, _complete_draft(data))
    output = tmp_path / "frozen"
    args = _freeze_args(release, families, draft, output)
    args.handler(args)
    result = json.loads(capsys.readouterr().out)
    frozen = json.loads((output / "items.json").read_text())
    assert len(frozen["items"]) == 600
    assert result["complete_denominators"] is True
    assert result["items_sha256"] == items._sha(frozen["items"])
    assert result["chunks_sha256"] == items._sha(frozen["chunks"])
    assert result["denominators_sha256"] == items._sha(
        {
            k: frozen[k]
            for k in ("category_denominators", "missing_evidence_denominators")
        }
    )
    before = (output / "items.json").read_bytes()
    with pytest.raises(FileExistsError):
        args.handler(args)
    assert (output / "items.json").read_bytes() == before


@pytest.mark.parametrize(
    "change",
    ["incomplete", "unreviewed", "missing", "controls", "release", "offset", "train"],
)
def test_freeze_cli_rejections_leave_no_output(tmp_path, monkeypatch, change):
    release, families, data = _fixture(tmp_path, monkeypatch)
    data = _complete_draft(data)
    item = data["items"][0]
    if change in {"incomplete", "missing"}:
        data["items"] = data["items"][:-1] if change == "incomplete" else []
    elif change == "unreviewed":
        item["review_status"] = "draft"
    elif change == "controls":
        item["controls"] = {}
    elif change == "release":
        data["release_id"] = "0" * 64
    elif change == "offset":
        data["chunks"][0]["start"] = 1
    else:
        item["parent_document_ids"] = ["train-1"]
    item["content_sha256"] = items._sha(
        {k: v for k, v in item.items() if k != "content_sha256"}
    )
    draft = tmp_path / "draft.json"
    _write(draft, data)
    output = tmp_path / "frozen"
    args = _freeze_args(release, families, draft, output)
    with pytest.raises(ValueError):
        args.handler(args)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["empty", "symlink"])
def test_freeze_cli_rejects_existing_directory_before_verification(tmp_path, kind):
    output = tmp_path / "output"
    if kind == "empty":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "absent", target_is_directory=True)
    args = _freeze_args(tmp_path, tmp_path / "missing", tmp_path / "missing", output)
    with pytest.raises(FileExistsError):
        args.handler(args)


def test_freeze_cli_cold_verifier_rejects_unverifiable_release(tmp_path, monkeypatch):
    from sparselab.corpus.release import verify_release

    release, families, data = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(items, "verify_release", verify_release)
    draft = tmp_path / "draft.json"
    _write(draft, _complete_draft(data))
    output = tmp_path / "frozen"
    args = _freeze_args(release, families, draft, output)
    with pytest.raises((ValueError, FileNotFoundError)):
        args.handler(args)
    assert not output.exists()


def test_freeze_cli_rejects_train_family_leakage(tmp_path, monkeypatch):
    release, families, data = _fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in families.read_text().splitlines()]
    rows[2]["family_id"] = rows[0]["family_id"]
    families.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data["family_inventory_sha256"] = sha256_file(families)
    draft = tmp_path / "draft.json"
    _write(draft, _complete_draft(data))
    output = tmp_path / "frozen"
    args = _freeze_args(release, families, draft, output)
    with pytest.raises(ValueError, match="family leaks"):
        args.handler(args)
    assert not output.exists()
