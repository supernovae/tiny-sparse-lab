"""Native continuation/source overlap preserves supplied and panel provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from sparselab.corpus import memorization_cli
from sparselab.evaluation import panel

pytest_plugins = ("test_generation_panel",)


def _write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _file_input(
    path: Path, continuation_path: Path, digest: str, **extra: object
) -> Path:
    return _write(
        path,
        {
            "format": "sparselab-memorization-input-v1",
            "continuation": {
                "kind": "file",
                "path": continuation_path.name,
                "file_sha256": digest,
            },
            "sources": [
                {"id": "a", "text": "the cat sat."},
                {"id": "b", "text": "unrelated"},
            ],
            "ngram_size": 2,
            "max_edit_chars": 4,
        }
        | extra,
    )


def test_file_continuation_retains_raw_bindings_and_zero_edit_bound(
    tmp_path: Path,
) -> None:
    continuation = tmp_path / "continuation.txt"
    raw = b"The cat sat."
    continuation.write_bytes(raw)
    source = _file_input(
        tmp_path / "input.json", continuation, hashlib.sha256(raw).hexdigest()
    )

    report = memorization_cli.analyze_memorization(source)

    assert report["format"] == "sparselab-memorization-report-v1"
    assert report["role"] == "descriptive_not_quality_gate"
    assert report["origin"] == {
        "kind": "file",
        "path": str(continuation.absolute()),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert report["continuation_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["normalization"] == "NFC+casefold+whitespace-collapse-v1"
    assert report["best_source"]["source_id"] == "a"
    match = report["sources"][0]
    assert match["source_id"] == "a"
    assert match["passage_sha256"] == hashlib.sha256(b"the cat sat.").hexdigest()
    assert match["longest_exact_match"] == "the cat sat."
    assert (
        match["character_overlap"]
        == match["word_overlap"]
        == match["ngram_overlap"]
        == 1.0
    )
    assert match["edit_similarity"] is None


def test_empty_continuation_uses_core_zero_denominator_semantics(
    tmp_path: Path,
) -> None:
    continuation = tmp_path / "empty.txt"
    continuation.write_bytes(b"")
    source = _file_input(
        tmp_path / "input.json", continuation, hashlib.sha256(b"").hexdigest()
    )

    report = memorization_cli.analyze_memorization(source)

    assert report["sources"][0]["character_overlap"] == 0.0
    assert report["sources"][0]["word_overlap"] == 0.0
    assert report["sources"][0]["ngram_overlap"] == 0.0


@pytest.mark.parametrize(
    "change",
    [
        {"sources": []},
        {"sources": [{"id": "a", "text": "one"}, {"id": "a", "text": "two"}]},
        {"sources": [{"id": "a"}]},
        {"ngram_size": True},
        {"ngram_size": 11},
        {"max_edit_chars": False},
        {"max_edit_chars": 2049},
        {"continuation": {"kind": "file", "path": "x"}},
        {"continuation": {"kind": "generation_panel", "result": "x", "row": True}},
        {
            "continuation": {
                "kind": "file",
                "path": "x",
                "file_sha256": "A" * 64,
                "result": "ambiguous",
            }
        },
    ],
)
def test_strict_declaration_rejects_invalid_sources_bounds_and_origins(
    tmp_path: Path, change: dict[str, object]
) -> None:
    continuation = tmp_path / "continuation.txt"
    continuation.write_text("text", encoding="utf-8")
    source = _file_input(
        tmp_path / "input.json",
        continuation,
        hashlib.sha256(b"text").hexdigest(),
        **change,
    )

    with pytest.raises(ValueError):
        memorization_cli.analyze_memorization(source)


@pytest.mark.parametrize(
    "change",
    [
        {"continuation": None},
        {"sources": [{"text": "missing identifier"}]},
    ],
)
def test_strict_declaration_rejects_absent_origin_and_missing_source_id(
    tmp_path: Path, change: dict[str, object]
) -> None:
    continuation = tmp_path / "continuation.txt"
    raw = b"text"
    continuation.write_bytes(raw)
    source = _file_input(
        tmp_path / "input.json", continuation, hashlib.sha256(raw).hexdigest(), **change
    )

    with pytest.raises(ValueError):
        memorization_cli.analyze_memorization(source)


@pytest.mark.parametrize("unsafe", ["../continuation.txt", "link.txt"])
def test_file_binding_rejects_bad_digest_utf8_and_unsafe_paths(
    tmp_path: Path, unsafe: str
) -> None:
    continuation = tmp_path / "continuation.txt"
    continuation.write_bytes(b"\xff")
    if unsafe == "link.txt":
        (tmp_path / unsafe).symlink_to(continuation)
    source = _file_input(
        tmp_path / "input.json", continuation, hashlib.sha256(b"wrong").hexdigest()
    )
    value = json.loads(source.read_text())
    value["continuation"]["path"] = unsafe
    source.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError):
        memorization_cli.analyze_memorization(source)


def test_file_binding_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    continuation = real / "continuation.txt"
    raw = b"text"
    continuation.write_bytes(raw)
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    source = _file_input(
        tmp_path / "input.json", continuation, hashlib.sha256(raw).hexdigest()
    )
    value = json.loads(source.read_text(encoding="utf-8"))
    value["continuation"]["path"] = "linked/continuation.txt"
    source.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="symlink"):
        memorization_cli.analyze_memorization(source)


def test_file_and_declaration_bindings_are_rechecked_before_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    continuation = tmp_path / "continuation.txt"
    raw = b"text"
    continuation.write_bytes(raw)
    source = _file_input(
        tmp_path / "input.json", continuation, hashlib.sha256(raw).hexdigest()
    )
    original = memorization_cli.diagnose_memorization

    def mutate_continuation(*args, **kwargs):
        result = original(*args, **kwargs)
        continuation.write_bytes(b"changed")
        return result

    monkeypatch.setattr(memorization_cli, "diagnose_memorization", mutate_continuation)
    with pytest.raises(ValueError, match="SHA-256"):
        memorization_cli.analyze_memorization(source)

    continuation.write_bytes(raw)

    def mutate_declaration(*args, **kwargs):
        result = original(*args, **kwargs)
        source.write_text("{}\n", encoding="utf-8")
        return result

    monkeypatch.setattr(memorization_cli, "diagnose_memorization", mutate_declaration)
    with pytest.raises(ValueError, match="declaration changed"):
        memorization_cli.analyze_memorization(source)


def test_file_binding_rejects_invalid_utf8_after_digest_validation(
    tmp_path: Path,
) -> None:
    continuation = tmp_path / "continuation.txt"
    raw = b"\xff"
    continuation.write_bytes(raw)
    source = _file_input(
        tmp_path / "input.json", continuation, hashlib.sha256(raw).hexdigest()
    )

    with pytest.raises(ValueError, match="UTF-8"):
        memorization_cli.analyze_memorization(source)


def test_generation_panel_origin_is_authenticated_and_rechecked(panel_run) -> None:
    root, index = panel_run
    declaration = root / "panel-input.json"
    panel_declaration = root / "panel.json"
    panel_declaration.write_text(
        json.dumps(
            {
                "generation_panel_version": 1,
                "id": "overlap",
                "role": "descriptive_not_quality_gate",
                "prompts": ["hello"],
                "decoder": {
                    "temperature": 0,
                    "top_k": 0,
                    "max_new_tokens": 3,
                    "seed": 42,
                },
                "checkpoint_selection": "identical to heldout evaluation binding",
            }
        ),
        encoding="utf-8",
    )
    result = panel.run_panel(panel_declaration, index, backend="cpu")
    _write(
        declaration,
        {
            "format": "sparselab-memorization-input-v1",
            "continuation": {
                "kind": "generation_panel",
                "result": result.relative_to(root).as_posix(),
                "row": 0,
            },
            "sources": [{"id": "source", "text": "hello"}],
        },
    )

    report = memorization_cli.analyze_memorization(declaration)

    assert report["origin"]["kind"] == "generation_panel"
    assert report["origin"]["result"] == str(result.absolute())
    assert report["origin"]["run_id"] == "panel-run"
    assert report["origin"]["checkpoint_sha256"]
    assert report["origin"]["evaluation_index_sha256"]
    assert report["origin"]["row"] == 0

    record = json.loads(result.read_text(encoding="utf-8"))
    assert report["origin"]["result_sha256"] == record["record_sha256"]
    assert report["origin"]["panel_sha256"] == record["panel_sha256"]
    assert report["origin"]["checkpoint_sha256"] == record["checkpoint_sha256"]
    assert (
        report["origin"]["evaluation_index_sha256"] == record["evaluation_index_sha256"]
    )
    record["rows"][0]["completion"] = "tampered"
    result.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        memorization_cli.analyze_memorization(declaration)


def test_panel_requires_exact_completed_row(panel_run) -> None:
    root, index = panel_run
    panel_declaration = root / "failed-panel.json"
    panel_declaration.write_text(
        json.dumps(
            {
                "generation_panel_version": 1,
                "id": "failed-overlap",
                "role": "descriptive_not_quality_gate",
                "prompts": ["hello", "failure"],
                "decoder": {
                    "temperature": 0,
                    "top_k": 0,
                    "max_new_tokens": 3,
                    "seed": 42,
                },
                "checkpoint_selection": "identical to heldout evaluation binding",
            }
        ),
        encoding="utf-8",
    )
    original = panel.generate_with_token_ids

    def fail_second(*args, **kwargs):
        if args[2] == "failure":
            raise RuntimeError("retained failure")
        return original(*args, **kwargs)

    # A native panel contains the durable failed attempt; the adapter must not use it.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(panel, "generate_with_token_ids", fail_second)
    try:
        result = panel.run_panel(panel_declaration, index, backend="cpu")
    finally:
        monkeypatch.undo()
    input_path = _write(
        root / "failed-input.json",
        {
            "format": "sparselab-memorization-input-v1",
            "continuation": {
                "kind": "generation_panel",
                "result": result.relative_to(root).as_posix(),
                "row": 1,
            },
            "sources": [{"id": "source", "text": "hello"}],
        },
    )
    with pytest.raises(ValueError, match="not completed"):
        memorization_cli.analyze_memorization(input_path)

    out_of_range = _write(
        root / "missing-row-input.json",
        {
            "format": "sparselab-memorization-input-v1",
            "continuation": {
                "kind": "generation_panel",
                "result": result.relative_to(root).as_posix(),
                "row": 2,
            },
            "sources": [{"id": "source", "text": "hello"}],
        },
    )
    with pytest.raises(ValueError, match="outside retained"):
        memorization_cli.analyze_memorization(out_of_range)


def test_panel_rejects_interrupted_row(
    panel_run, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, index = panel_run
    panel_declaration = root / "interrupted-overlap-panel.json"
    panel_declaration.write_text(
        json.dumps(
            {
                "generation_panel_version": 1,
                "id": "interrupted-overlap",
                "role": "descriptive_not_quality_gate",
                "prompts": ["interrupt"],
                "decoder": {
                    "temperature": 0,
                    "top_k": 0,
                    "max_new_tokens": 3,
                    "seed": 42,
                },
                "checkpoint_selection": "identical to heldout evaluation binding",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        panel,
        "generate_with_token_ids",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt("interrupted")),
    )
    with pytest.raises(KeyboardInterrupt):
        panel.run_panel(panel_declaration, index, backend="cpu")
    monkeypatch.undo()
    result = panel.run_panel(panel_declaration, index, backend="cpu")
    input_path = _write(
        root / "interrupted-input.json",
        {
            "format": "sparselab-memorization-input-v1",
            "continuation": {
                "kind": "generation_panel",
                "result": result.relative_to(root).as_posix(),
                "row": 0,
            },
            "sources": [{"id": "source", "text": "interrupt"}],
        },
    )

    with pytest.raises(ValueError, match="not completed"):
        memorization_cli.analyze_memorization(input_path)


def test_panel_dependency_and_durable_attempt_tampering_are_rejected(panel_run) -> None:
    root, index = panel_run
    panel_declaration = root / "tamper-overlap-panel.json"
    panel_declaration.write_text(
        json.dumps(
            {
                "generation_panel_version": 1,
                "id": "tamper-overlap",
                "role": "descriptive_not_quality_gate",
                "prompts": ["hello"],
                "decoder": {
                    "temperature": 0,
                    "top_k": 0,
                    "max_new_tokens": 3,
                    "seed": 42,
                },
                "checkpoint_selection": "identical to heldout evaluation binding",
            }
        ),
        encoding="utf-8",
    )
    result = panel.run_panel(panel_declaration, index, backend="cpu")
    input_path = _write(
        root / "tamper-input.json",
        {
            "format": "sparselab-memorization-input-v1",
            "continuation": {
                "kind": "generation_panel",
                "result": result.relative_to(root).as_posix(),
                "row": 0,
            },
            "sources": [{"id": "source", "text": "hello"}],
        },
    )
    assert memorization_cli.analyze_memorization(input_path)["origin"]["result"] == str(
        result.absolute()
    )
    record = json.loads(result.read_text(encoding="utf-8"))
    dependencies = [
        Path(record["panel"]),
        Path(record["evaluation_index"]),
        Path(record["run"]) / record["checkpoint"] / "manifest.json",
        result.with_suffix(".attempts") / "0.result.json",
    ]
    for dependency in dependencies:
        original = dependency.read_bytes()
        dependency.write_bytes(original + b"tampered")
        try:
            with pytest.raises(ValueError):
                memorization_cli.analyze_memorization(input_path)
        finally:
            dependency.write_bytes(original)

    original_result = result.read_bytes()
    rehashed = json.loads(original_result)
    rehashed["rows"][0]["completion"] = "substituted"
    rehashed["record_sha256"] = panel._digest(
        {key: value for key, value in rehashed.items() if key != "record_sha256"}
    )
    result.write_bytes(panel.canonical_json(rehashed) + b"\n")
    try:
        with pytest.raises(ValueError, match="durable attempt"):
            memorization_cli.analyze_memorization(input_path)
    finally:
        result.write_bytes(original_result)


def test_json_error_envelope_and_parser_registration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    memorization_cli.register_parser(commands)
    args = parser.parse_args(
        ["memorization", "analyze", str(tmp_path / "missing.json"), "--json"]
    )
    with pytest.raises(SystemExit, match="2"):
        args.handler(args)
    assert json.loads(capsys.readouterr().out)["status"] == "error"
