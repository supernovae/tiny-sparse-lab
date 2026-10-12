"""Selector evidence is sealed at execution, independent of a reader's cwd."""

from pathlib import Path

import pytest
from test_lab_results import _record, _run

from sparselab.lab_records import read_lab_record
from sparselab.lab_results import ResultLookupError, baseline_text
from sparselab.probes.cli import run_probe


@pytest.mark.parametrize(
    "selector_kind",
    ["run-id", "record-id", "probe-id", "relative", "absolute", "checkpoint"],
)
def test_probe_seals_native_baseline_selection(tmp_path, monkeypatch, selector_kind):
    from sparselab.probes import cli

    lab = tmp_path / "lab"
    run = _run(lab)
    record_path = _record(lab, run, "probe" if selector_kind == "probe-id" else "try")
    monkeypatch.chdir(tmp_path)
    selectors = {
        "run-id": run.name,
        "record-id": "try-example",
        "probe-id": "probe-example",
        "relative": "lab/runs/candidate",
        "absolute": str(run),
        "checkpoint": str(run / "checkpoints/step-1"),
    }
    spec = selectors[selector_kind]
    loads = []

    def load(*args, **kwargs):
        loads.append(kwargs["selected"])
        return object()

    def battery(target, baseline, **options):
        # Changing cwd between resolution and load cannot reinterpret a selector.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        baseline.load()
        return {
            "format": cli.PROBE_FORMAT,
            "stop": {"kind": "done"},
            "target": None,
            "baseline": None,
        }

    monkeypatch.setattr(cli, "load_target", load)
    monkeypatch.setattr(cli, "run_battery", battery)
    result, path = run_probe(str(run), spec, lab_dir=lab)
    assert result["request"]["baseline"] == spec
    provenance = result["baseline_provenance"]
    assert provenance["selector"] == spec
    assert provenance["working_directory"] == str(tmp_path)
    assert provenance["status"] == "resolved"
    assert provenance["resolved"]["run_dir"] == str(run)
    assert loads[0].run == run
    if selector_kind in {"record-id", "probe-id"}:
        assert provenance["resolved"]["path"] == str(record_path)
        assert loads[0].expected_sha256 == "a" * 64
        assert loads[0].checkpoint == str(run / "checkpoints/step-1")
    before = path.read_bytes()
    assert read_lab_record(path)[1] == result
    assert str(tmp_path) in baseline_text(result)
    assert path.read_bytes() == before


@pytest.mark.parametrize("failure", ["missing", "ambiguous", "invalid", "interrupted"])
def test_probe_setup_failure_keeps_selector_without_inventing_path(
    tmp_path, monkeypatch, failure
):
    from sparselab.probes import cli
    from sparselab.probes.render import render

    lab = tmp_path / "lab"
    run = _run(lab)
    monkeypatch.chdir(tmp_path)
    spec = "missing"
    error = ResultLookupError
    if failure == "ambiguous":
        spec = run.name
        (tmp_path / spec).mkdir()
    elif failure == "invalid":
        spec = "bad.json"
        (tmp_path / spec).write_text("{}")
    elif failure == "interrupted":
        spec = run.name
        error = KeyboardInterrupt
        monkeypatch.setattr(
            cli,
            "run_battery",
            lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    with pytest.raises(error):
        run_probe(str(run), spec, lab_dir=lab)
    (path,) = (lab / "probes").glob("*/probe.json")
    record = read_lab_record(path)[1]
    assert record["request"]["baseline"] == spec
    assert record["status"] == ("interrupted" if failure == "interrupted" else "failed")
    if failure != "interrupted":
        assert "resolved" not in record["baseline_provenance"]
    assert spec in render(record)
    with pytest.raises(ResultLookupError, match="no local scored candidate"):
        cli.resolve_checkpoint(str(path), lab_dir=lab)


@pytest.mark.parametrize("absolute", [False, True])
def test_failed_try_retains_original_selector_and_execution_path(
    tmp_path, monkeypatch, absolute
):
    from sparselab.lab_mode import run_try, summarize

    monkeypatch.chdir(tmp_path)
    baseline = str(tmp_path / "missing.yaml") if absolute else "missing.yaml"
    (tmp_path / "delta.yaml").write_text("set:\n  model.ffn_dim: 128\n")
    with pytest.raises(ValueError):
        run_try(Path("delta.yaml"), Path(baseline), work_dir=tmp_path / "work")
    (path,) = (tmp_path / "work/lab/tries").glob("*/try.json")
    record = read_lab_record(path)[1]
    assert record["request"]["baseline"] == baseline
    assert record["baseline_provenance"]["resolved"] == {
        "kind": "config",
        "path": str(tmp_path / "missing.yaml"),
    }
    monkeypatch.chdir(tmp_path / "work")
    assert baseline in summarize(record)
    assert record == read_lab_record(path)[1]


def test_historical_reader_does_not_infer_cwd_or_reseal(tmp_path, monkeypatch):
    from sparselab.lab_mode import summarize
    from sparselab.lab_records import TRY_FORMAT, write_sealed

    path = tmp_path / "try.json"
    record = write_sealed(
        path,
        {
            "format": TRY_FORMAT,
            "try_id": "old",
            "status": "failed",
            "request": {"baseline": "relative.yaml"},
        },
    )
    before = path.read_bytes()
    destination = tmp_path / "reader"
    destination.mkdir()
    monkeypatch.chdir(destination)
    assert "original working directory unknown" in summarize(record)
    assert "baseline_provenance" not in read_lab_record(path)[1]
    assert path.read_bytes() == before
    assert list(destination.iterdir()) == []


def test_interrupted_try_has_sealed_provenance_before_parsing(tmp_path, monkeypatch):
    from sparselab import lab_mode

    monkeypatch.chdir(tmp_path)

    def interrupted(path):
        (record_path,) = (tmp_path / "work/lab/tries").glob("*/try.json")
        started = read_lab_record(record_path)[1]
        assert started["status"] == "started"
        assert started["request"]["baseline"] == "baseline.yaml"
        raise KeyboardInterrupt

    monkeypatch.setattr(lab_mode, "read_candidate", interrupted)
    record, path = lab_mode.run_try(
        Path("delta.yaml"), Path("baseline.yaml"), work_dir=tmp_path / "work"
    )
    assert record["status"] == "interrupted"
    assert record["baseline_provenance"]["working_directory"] == str(tmp_path)
    assert read_lab_record(path)[1] == record


@pytest.mark.parametrize("command", ["report", "compare"])
@pytest.mark.parametrize("json_output", [False, True])
def test_relocated_read_only_cli_preserves_history_without_creating_store(
    tmp_path, monkeypatch, capsys, command, json_output
):
    import json

    from test_lab_results import _cli

    from sparselab.lab_records import TRY_FORMAT, write_sealed

    path = tmp_path / "export.json"
    record = write_sealed(
        path,
        {
            "format": TRY_FORMAT,
            "try_id": "old",
            "status": "failed",
            "request": {"baseline": "relative.yaml"},
        },
    )
    original = path.read_bytes()
    reader = tmp_path / "reader"
    reader.mkdir()
    monkeypatch.chdir(reader)
    work = reader / "absent-work"
    if command == "compare":
        with pytest.raises(SystemExit):
            _cli(
                monkeypatch,
                work,
                command,
                str(path),
                *(["--json"] if json_output else []),
            )
    else:
        _cli(
            monkeypatch, work, command, str(path), *(["--json"] if json_output else [])
        )
    output = capsys.readouterr().out
    if command == "report":
        if json_output:
            assert json.loads(output) == record
        else:
            assert "original working directory unknown" in output
    assert path.read_bytes() == original
    assert list(reader.iterdir()) == []


@pytest.mark.parametrize("kind", ["try", "probe"])
def test_historical_relative_run_location_cannot_use_reader_cwd(
    tmp_path, monkeypatch, kind
):
    import json

    from sparselab.lab_records import seal
    from sparselab.lab_results import resolve_checkpoint

    run = _run(tmp_path)
    path = _record(tmp_path, run, kind)
    record = read_lab_record(path)[1]
    row = record["arms"]["candidate"] if kind == "try" else record["target"]
    row["run_dir"] = "runs/candidate"
    path.write_text(json.dumps(seal(record)))
    before = path.read_bytes()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        ResultLookupError, match="original working directory is unknown"
    ):
        resolve_checkpoint(str(path), lab_dir=tmp_path)
    assert read_lab_record(path)[1] == seal(record)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "record",
    [
        {"baseline_provenance": {"selector": None, "status": "not_requested"}},
        {"request": {"baseline": None}},
    ],
)
def test_explicit_no_baseline_display(record):
    assert baseline_text(record) == "  baseline: none requested"


def test_missing_historical_request_retains_uncertainty():
    assert "original working directory unknown" in baseline_text({"baseline": None})
