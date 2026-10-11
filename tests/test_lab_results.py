"""Everyday result lookup and dispatch: metadata only, no models or downloads."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from sparselab.lab_records import KINDS, seal
from sparselab.lab_results import (
    ResultLookupError,
    resolve_checkpoint,
    resolve_record,
    resolve_result,
)

ROOT = Path(__file__).resolve().parents[1]


def _run(lab: Path, name: str = "candidate", backend: str = "cpu") -> Path:
    from sparselab.config.models import RunConfig

    run = lab / "runs" / name
    run.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")
    raw = yaml.safe_load((ROOT / "configs/smoke_moe_cpu.yaml").read_text())
    raw["runtime"]["backend"] = backend
    (run / "resolved_config.yaml").write_text(
        RunConfig.model_validate(raw).model_dump_json()
    )
    (run / "checkpoints" / "step-1").mkdir(parents=True)
    return run


def _record(lab: Path, run: Path, kind: str = "try", **updates: Any) -> Path:
    fmt, folder, filename = KINDS[kind]
    row = {
        "run_id": run.name,
        "run_dir": str(run),
        "checkpoint": "checkpoints/step-1",
        "checkpoint_sha256": "a" * 64,
    }
    record = {
        "format": fmt,
        "try_id" if kind == "try" else "probe_id": f"{kind}-example",
        "arms" if kind == "try" else "target": {"candidate": row}
        if kind == "try"
        else row,
        **updates,
    }
    path = lab / folder / f"{kind}-example" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seal(record)))
    return path


@pytest.mark.parametrize("kind", ["try", "probe"])
def test_one_record_handle_selects_exact_checkpoint_and_retains_record(
    tmp_path: Path, kind: str
) -> None:
    run = _run(tmp_path)
    path = _record(tmp_path, run, kind)
    # The latest pointer has advanced; lookup must never use it for a record.
    (run / "checkpoints" / "latest.json").write_text('{"relative_path":"step-2"}')
    for spec in (path.parent.name, str(path.parent), str(path)):
        assert resolve_record(spec, tmp_path) == path
        target = resolve_checkpoint(spec, lab_dir=tmp_path)
        assert target.run == run
        assert target.checkpoint == str(run / "checkpoints/step-1")
        assert target.expected_sha256 == "a" * 64


def test_direct_run_checkpoint_and_custom_runs_directory(tmp_path: Path) -> None:
    run = _run(tmp_path / "external")
    for spec in (run.name, str(run)):
        target = resolve_checkpoint(spec, lab_dir=tmp_path, runs_dir=run.parent)
        assert target.run == run and target.checkpoint is None
    checkpoint = run / "checkpoints/step-1"
    assert resolve_checkpoint(str(checkpoint), lab_dir=tmp_path).checkpoint == str(
        checkpoint
    )
    # Runs do not imply a protocol, record, final split or candidate role.
    with pytest.raises(ResultLookupError, match="unique sealed result"):
        resolve_record(str(run), tmp_path)


def test_missing_recorded_checkpoint_never_falls_back_to_latest(tmp_path: Path) -> None:
    run = _run(tmp_path)
    path = _record(tmp_path, run)
    (run / "checkpoints/step-1").rmdir()
    (run / "checkpoints/step-2").mkdir()
    with pytest.raises(ResultLookupError, match="recorded checkpoint is unavailable"):
        resolve_checkpoint(path.parent.name, lab_dir=tmp_path)
    assert resolve_record(path.parent.name, tmp_path) == path


@pytest.mark.parametrize(
    "checkpoint", ["../other", "/other", "checkpoints/latest.json"]
)
def test_record_cannot_redirect_to_a_pointer_or_another_run(
    tmp_path: Path, checkpoint: str
) -> None:
    run = _run(tmp_path)
    path = _record(tmp_path, run)
    record = json.loads(path.read_text())
    record["arms"]["candidate"]["checkpoint"] = checkpoint
    path.write_text(json.dumps(seal(record)))
    with pytest.raises(
        ResultLookupError, match="does not name a checkpoint generation"
    ):
        resolve_checkpoint(str(path), lab_dir=tmp_path)


def test_record_cannot_follow_checkpoint_symlink_outside_run(tmp_path: Path) -> None:
    run = _run(tmp_path)
    path = _record(tmp_path, run)
    selected = run / "checkpoints/step-1"
    selected.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    selected.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ResultLookupError, match="escapes its run"):
        resolve_checkpoint(str(path), lab_dir=tmp_path)


def test_stopped_record_remains_readable_without_a_scored_candidate(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)
    path = _record(
        tmp_path, run, arms={}, status="resources", stop={"kind": "resources"}
    )
    assert resolve_result(str(path), lab_dir=tmp_path).record["stop"] == {
        "kind": "resources"
    }
    with pytest.raises(ResultLookupError, match="retained stop/missing evidence"):
        resolve_checkpoint(str(path), lab_dir=tmp_path)


def test_colliding_ids_require_an_explicit_path(tmp_path: Path) -> None:
    run = _run(tmp_path, "try-example")
    record = _record(tmp_path, run)
    for resolver in (
        lambda: resolve_record("try-example", tmp_path),
        lambda: resolve_checkpoint("try-example", lab_dir=tmp_path),
    ):
        with pytest.raises(ResultLookupError, match="ambiguous_result"):
            resolver()
    assert resolve_record(str(record), tmp_path) == record
    assert resolve_checkpoint(str(run), lab_dir=tmp_path).run == run


def test_duplicate_run_ids_across_roots_are_not_silently_shadowed(
    tmp_path: Path,
) -> None:
    _run(tmp_path / "lab")
    external = _run(tmp_path / "external")
    with pytest.raises(ResultLookupError, match="ambiguous_result"):
        resolve_checkpoint(
            "candidate", lab_dir=tmp_path / "lab", runs_dir=external.parent
        )


def test_record_digest_is_checked_on_every_surface(tmp_path: Path) -> None:
    run = _run(tmp_path)
    path = _record(tmp_path, run)
    record = json.loads(path.read_text())
    record["arms"]["candidate"]["checkpoint_sha256"] = "b" * 64
    path.write_text(json.dumps(record))
    for resolver in (
        lambda: resolve_record(str(path), tmp_path),
        lambda: resolve_checkpoint(str(path), lab_dir=tmp_path),
    ):
        with pytest.raises(ResultLookupError, match="digest mismatch"):
            resolver()


def test_folder_with_two_records_is_ambiguous(tmp_path: Path) -> None:
    run = _run(tmp_path)
    first = _record(tmp_path, run)
    second = _record(tmp_path, run, "probe")
    (first.parent / "probe.json").write_bytes(second.read_bytes())
    with pytest.raises(ResultLookupError, match="multiple records"):
        resolve_record(str(first.parent), tmp_path)


def _cli(monkeypatch: pytest.MonkeyPatch, work: Path, *args: str) -> None:
    from sparselab.cli.main import main

    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(work))
    monkeypatch.setattr(sys, "argv", ["sparselab", "--work-dir", str(work), *args])
    main()


@pytest.mark.parametrize("command", ["report", "compare", "probe", "explore"])
def test_unknown_id_has_the_same_text_and_json_error_on_all_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    with pytest.raises(SystemExit) as text_error:
        _cli(monkeypatch, tmp_path, command, "missing")
    with pytest.raises(SystemExit) as json_error:
        _cli(monkeypatch, tmp_path, command, "missing", "--json")
    assert json_error.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "missing_evidence"
    assert payload["result"] == "missing"
    assert payload["reason"] in str(text_error.value)
    assert payload["hint"] in str(text_error.value)


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
def test_probe_saved_backend_error_is_explicit_in_text_and_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    backend: str,
) -> None:
    run = _run(tmp_path / "lab", backend=backend)
    for flag in ([], ["--json"]):
        with pytest.raises(SystemExit) as error:
            _cli(monkeypatch, tmp_path, "probe", run.name, *flag)
        if flag:
            payload = json.loads(capsys.readouterr().out)
            assert payload["status"] == "runtime_required"
            message = payload["reason"] + payload["hint"]
        else:
            message = str(error.value)
        assert f"saved backend {backend}" in message
        assert "--backend cpu" in message and "no automatic fallback" in message
    assert not (tmp_path / "lab/probes").exists()


def test_explicit_cpu_override_reaches_native_authorization_without_changing_saved_config(
    tmp_path: Path,
) -> None:
    from sparselab.cli.main import _prepare_runtime_command, build_parser

    run = _run(tmp_path / "lab", backend="cuda")
    original = (run / "resolved_config.yaml").read_bytes()
    for command in ("probe", "explore"):
        args = build_parser(tmp_path).parse_args(
            ["--work-dir", str(tmp_path), command, run.name, "--backend", "cpu"]
        )
        _prepare_runtime_command(args)
        assert args.backend == "cpu" and args.runtime_authorization is None
    assert (run / "resolved_config.yaml").read_bytes() == original


def test_record_digest_is_forwarded_to_native_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sparselab.evaluation import inference
    from sparselab.probes.cli import load_target

    run = _run(tmp_path)
    path = _record(tmp_path, run)
    seen = []
    monkeypatch.setattr(inference, "load_run", lambda *a, **kw: seen.append((a, kw)))
    load_target(
        path.parent.name,
        lab_dir=tmp_path,
        runs_dir=None,
        backend="cpu",
        authorization=None,
    )
    assert seen == [
        (
            (run.name, run.parent, str(run / "checkpoints/step-1"), "cpu"),
            {"authorization": None, "expected_checkpoint_sha256": "a" * 64},
        )
    ]


def test_native_loader_rejects_record_digest_mismatch_before_device_or_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sparselab.evaluation import inference

    monkeypatch.setattr(
        inference,
        "_verified_checkpoint",
        lambda *a, **kw: (None, None, None, {"sha256": "b" * 64}, None),
    )
    monkeypatch.setattr(
        inference,
        "_ephemeral_config",
        lambda *a: pytest.fail("must reject before runtime selection"),
    )
    with pytest.raises(ValueError, match="digest differs from the sealed lab record"):
        inference.load_run("candidate", tmp_path, expected_checkpoint_sha256="a" * 64)


def test_report_retains_resource_stop_and_compare_explains_missing_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lab = tmp_path / "lab"
    path = _record(
        lab,
        _run(lab),
        arms={},
        status="interrupted",
        interruption={"arm": "baseline", "phase": "training", "reason": "resources"},
    )
    original = path.read_bytes()
    _cli(monkeypatch, tmp_path, "report", path.parent.name)
    text = capsys.readouterr().out
    assert "stopped safely" in text and "resources" in text
    _cli(monkeypatch, tmp_path, "report", path.parent.name, "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload == json.loads(original)
    for command in ("probe", "explore", "compare"):
        with pytest.raises(SystemExit):
            _cli(monkeypatch, tmp_path, command, path.parent.name, "--json")
        error = json.loads(capsys.readouterr().out)
        assert error["status"] == "missing_evidence"
        assert "retained stop/missing evidence" in error["hint"]
    assert path.read_bytes() == original


def test_record_cli_dispatch_preserves_explicit_backend_and_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sparselab import explorer

    lab = tmp_path / "lab"
    record = _record(lab, _run(lab, backend="cuda"))
    envelope = tmp_path / "envelope.yaml"
    envelope.write_text(
        "resource_envelope_version: 1\nmax_rss_bytes: 1125899906842624\n"
    )
    seen = []

    def inspect(target: str, **kwargs: Any) -> tuple[dict, None]:
        seen.append((target, kwargs))
        return {"unavailable": "fixture: no model loaded"}, None

    monkeypatch.setattr(explorer, "explore", inspect)
    _cli(
        monkeypatch,
        tmp_path,
        "explore",
        record.parent.name,
        "--backend",
        "cpu",
        "--resource-envelope",
        str(envelope),
        "--json",
    )
    assert (
        json.loads(capsys.readouterr().out)["unavailable"] == "fixture: no model loaded"
    )
    assert seen[0][0] == record.parent.name
    assert seen[0][1]["backend"] == "cpu"
    assert seen[0][1]["resource_envelope"].max_rss_bytes == 1 << 50
