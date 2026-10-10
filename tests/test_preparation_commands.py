"""Canonical argv/template/path expansion only; no production execution."""

import json
from pathlib import Path

import pytest

from sparselab.cli.main import build_parser
from sparselab.corpus.declaration_render import render_declaration
from sparselab.training.preparation import (
    compile_phase,
    load_preparation,
    supervised_phase_command,
)

PLAN = (
    Path(__file__).resolve().parents[1]
    / "experiments/research/kernel-memory-lab/card05-base-50m/current/preparation.json"
)


def test_all_current_phase_paths_and_commands_are_disjoint(tmp_path):
    plan = load_preparation(PLAN)
    outputs = set()
    assert len(plan["phases"]) == 32
    assert plan["defaults"]["BATCH_SOURCE_BYTES"] == "8388608"
    for label, phase in plan["phases"].items():
        needed = {word[2:-1] for word in phase["args"] if word.startswith("${")}
        values = {
            key: str(tmp_path / key)
            for key in needed - plan["defaults"].keys() - {"TEMPLATE", "VALUES_JSON"}
        }
        compiled = compile_phase(PLAN, tmp_path, label, values)
        for key in ("leaf", "completion", "inner_monitor"):
            if key not in compiled:
                continue
            path = Path(compiled[key])
            assert all(
                path != old
                and not path.is_relative_to(old)
                and not old.is_relative_to(path)
                for old in outputs
            )
            outputs.add(path)
        command = supervised_phase_command(
            compiled,
            work_root=tmp_path,
            ledger=tmp_path / "ledger",
            label=label,
            content_identity="a" * 64,
            whole_policy=tmp_path / "whole",
            preparation_policy=tmp_path / "prep-policy",
            baseline=tmp_path / "baseline",
        )
        parsed = build_parser().parse_args(command[1:])
        assert parsed.attempt_command == "run"
        assert (
            parsed.updates
            == parsed.target_positions
            == parsed.generation_calls
            == parsed.generated_tokens
            == 0
        )
        assert parsed.completion == Path(compiled["completion"])


def test_production_templates_get_paths_only_from_plan(tmp_path):
    for label in ("pre-freeze-declaration", "final-build-declaration"):
        compiled = compile_phase(PLAN, tmp_path, label, {})
        args = compiled["args"]
        rendered = render_declaration(
            Path(args[args.index("--template") + 1]),
            args[args.index("--values-json") + 1],
            Path(compiled["leaf"]),
            tmp_path,
        )
        assert Path(rendered["path"]).is_file()
    with pytest.raises(ValueError, match="cannot override"):
        compile_phase(
            PLAN,
            tmp_path,
            "pre-freeze-declaration",
            {"VALUES_JSON": json.dumps({"SPLITS_FILE": "splits-acquire.yaml"})},
        )
    with pytest.raises(ValueError, match="exactly cover"):
        compiled = compile_phase(
            PLAN,
            tmp_path,
            "final-build-declaration",
            {"VALUES_JSON": json.dumps({"OLD_ALIAS": "value"})},
        )
        args = compiled["args"]
        render_declaration(
            Path(args[args.index("--template") + 1]),
            args[args.index("--values-json") + 1],
            tmp_path / "other.yaml",
            tmp_path,
        )


def test_binding_errors_and_no_executable_legacy_paths(tmp_path):
    with pytest.raises(ValueError, match="unknown preparation phase"):
        compile_phase(PLAN, tmp_path, "train", {})
    with pytest.raises(ValueError, match="missing preparation bindings"):
        compile_phase(PLAN, tmp_path, "freeze-release", {})
    with pytest.raises(ValueError, match="unused preparation bindings"):
        compile_phase(PLAN, tmp_path, "verify-snapshots", {"TYPO": "value"})
    with pytest.raises(SystemExit):
        build_parser().parse_args(["attempt", "phase-paths"])
    data = load_preparation(PLAN)
    data["phases"]["verify-snapshots"]["args"] = [
        "train",
        "config.yaml",
        "--run-id",
        "test",
        "--runs-dir",
        "/tmp/runs",
    ]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="cannot dispatch model work"):
        compile_phase(bad, tmp_path, "verify-snapshots", {})
