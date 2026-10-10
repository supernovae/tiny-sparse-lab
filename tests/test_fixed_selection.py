"""Receipt-only fixed validation selection; no model or tokenizer fixture."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab.cli.main import build_parser
from sparselab.evaluation import fixed_selection as selection
from sparselab.training.manifest import canonical_json, sha256_file

STEPS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
STRATA = ("general_prose", "explanatory_prose", "incident_response_docs")


def _score(run: Path, step: int, config: dict, profile_sha: str, loss: float) -> Path:
    items = [
        {
            "id": f"v{index}",
            "stratum": STRATA[index // 4],
            "targets": 256,
            "nll_nats": loss * 256,
        }
        for index in range(12)
    ]
    payload = {
        "mode": "validation",
        "profile_sha256": profile_sha,
        "forward_input_positions": 3084,
        "scored_targets": 3072,
        "items": items,
        "loss": loss,
        "identity": {
            "run_id": "fresh",
            "step": step,
            "tokens_seen": step * 12,
            "checkpoint_relative_path": f"checkpoints/step_{step:08d}_gen_000001",
            "checkpoint_sha256": f"{step + 1:064x}",
            "config": config,
            "source_identity_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
        },
    }
    content = canonical_json(payload) + b"\n"
    digest = hashlib.sha256(content).hexdigest()
    path = run / "evaluations" / f"fixed-slices-{digest}.json"
    path.write_bytes(content)
    return path


def _replace_score(path: Path, change) -> Path:
    payload = json.loads(path.read_text())
    change(payload)
    content = canonical_json(payload) + b"\n"
    replacement = path.with_name(
        f"fixed-slices-{hashlib.sha256(content).hexdigest()}.json"
    )
    path.unlink()
    replacement.write_bytes(content)
    return replacement


@pytest.fixture
def receipt_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run = tmp_path / "runs" / "fresh"
    (run / "evaluations").mkdir(parents=True)
    (run / "manifest.json").write_text("run manifest")
    (run / "progress.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "step": 100,
                "tokens_seen": 1234,
                "parent_run_id": None,
            }
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fixture config")
    (run / "resolved_config.yaml").write_text("fixture effective config")
    requested_config = {"seed": 17, "tokenizer": {"path": "origin"}}
    config = {"seed": 17, "tokenizer": {"path": "stage"}}
    monkeypatch.setattr(
        selection,
        "load_config",
        lambda path: SimpleNamespace(
            model_dump=lambda **__: requested_config if path == config_path else config
        ),
    )
    manifest = {
        "run_id": "fresh",
        "requested_config": requested_config,
        "effective_config": config,
        "source_identity": {"sha256": "a" * 64},
    }
    monkeypatch.setattr(selection, "read_manifest", lambda _: manifest)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "tokenizer_sha256": "b" * 64,
                "loss_slices": [
                    {
                        "id": f"v{index}",
                        "stratum": STRATA[index // 4],
                        "split": "validation",
                    }
                    for index in range(12)
                ],
            }
        )
    )
    profile_sha = sha256_file(profile_path)
    paths = {
        step: _score(run, step, config, profile_sha, 1 + abs(step - 20) / 1000)
        for step in STEPS
    }
    checkpoints = [
        {
            "step": step,
            "path": f"step_{step:08d}_gen_000001",
            "tokens_seen": step * 12,
            "digest": f"{step + 1:064x}",
            "verified": True,
            "errors": [],
        }
        for step in STEPS
    ]
    observations = [
        {
            "step": step,
            "checkpoint": row["path"],
            "checkpoint_sha256": row["digest"],
            "batches": 1,
            "max_batches": 1,
            "loss": 1.0,
        }
        for step, row in zip(STEPS, checkpoints, strict=True)
    ]
    evidence = {
        "run_id": "fresh",
        "verified_checkpoints": True,
        "missing_reports": [],
        "rejected_reports": [],
        "checkpoints": checkpoints,
        "quality_observations": observations,
    }
    monkeypatch.setattr(selection, "experiment_evidence", lambda _: evidence)

    class Manager:
        def __init__(self, *args, **kwargs):
            pass

        def verify(self, *args, **kwargs):
            return SimpleNamespace(valid=True, resume_level="full")

    monkeypatch.setattr(selection, "CheckpointManager", Manager)
    declaration = tmp_path / "selection.json"
    declaration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "fresh",
                "runs_dir": str(run.parent),
                "config": str(config_path),
                "config_sha256": sha256_file(config_path),
                "profile": str(profile_path),
                "profile_sha256": profile_sha,
                "steps": STEPS,
                "final_step": 100,
                "target_positions": 1234,
            }
        )
    )
    return SimpleNamespace(
        run=run, paths=paths, evidence=evidence, declaration=declaration
    )


def test_select_tie_break_and_cold_reverification(receipt_fixture, tmp_path):
    fixture = receipt_fixture
    _replace_score(
        fixture.paths[10],
        lambda score: score.update(
            loss=1.0, items=[{**row, "nll_nats": 256.0} for row in score["items"]]
        ),
    )
    _replace_score(
        fixture.paths[20],
        lambda score: score.update(
            loss=1.0, items=[{**row, "nll_nats": 256.0} for row in score["items"]]
        ),
    )
    output = tmp_path / "selected.json"
    result = selection.select_fixed_validation(fixture.declaration, output)
    assert result["selected_step"] == 10
    assert len(result["scores"]) == 11
    assert selection.verify_fixed_selection(fixture.declaration, output) == result
    with pytest.raises(FileExistsError):
        selection.select_fixed_validation(fixture.declaration, output)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "duplicate",
        "run",
        "checkpoint",
        "checkpoint_path",
        "profile",
        "window",
        "target_count",
        "nonfinite",
        "test_before",
        "bad_hash",
        "config",
        "operational",
        "progress",
    ],
)
def test_invalid_or_premature_selection_fails(receipt_fixture, tmp_path, damage):
    fixture = receipt_fixture
    first = fixture.paths[0]
    if damage == "missing":
        first.unlink()
    elif damage == "duplicate":
        _replace_score(first, lambda score: score["identity"].update(step=10))
    elif damage == "run":
        _replace_score(first, lambda score: score["identity"].update(run_id="other"))
    elif damage == "checkpoint":
        _replace_score(
            first, lambda score: score["identity"].update(checkpoint_sha256="c" * 64)
        )
    elif damage == "checkpoint_path":
        _replace_score(
            first,
            lambda score: score["identity"].update(
                checkpoint_relative_path="checkpoints/substituted"
            ),
        )
    elif damage == "profile":
        _replace_score(first, lambda score: score.update(profile_sha256="c" * 64))
    elif damage == "window":
        _replace_score(first, lambda score: score["items"][0].update(id="v1"))
    elif damage == "target_count":
        _replace_score(first, lambda score: score["items"][0].update(targets=255))
    elif damage == "nonfinite":
        payload = json.loads(first.read_text())
        payload["loss"] = float("nan")
        content = json.dumps(payload, sort_keys=True).encode() + b"\n"
        first.unlink()
        (
            fixture.run
            / "evaluations"
            / f"fixed-slices-{hashlib.sha256(content).hexdigest()}.json"
        ).write_bytes(content)
    elif damage == "test_before":
        _replace_score(first, lambda score: score.update(mode="test"))
    elif damage == "bad_hash":
        first.write_bytes(first.read_bytes() + b" ")
    elif damage == "config":
        _replace_score(
            first, lambda score: score["identity"].update(config={"seed": 19})
        )
    elif damage == "operational":
        fixture.evidence["quality_observations"][0]["loss"] = float("inf")
    else:
        (fixture.run / "progress.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises((ValueError, TypeError)):
        selection.select_fixed_validation(
            fixture.declaration, tmp_path / "selected.json"
        )
    assert not (tmp_path / "selected.json").exists()


def test_selected_receipt_survives_later_test_output_but_not_score_substitution(
    receipt_fixture, tmp_path
):
    fixture = receipt_fixture
    output = tmp_path / "selected.json"
    selection.select_fixed_validation(fixture.declaration, output)
    saved = json.loads(fixture.paths[0].read_text())
    saved["mode"] = "test"
    content = canonical_json(saved) + b"\n"
    (
        fixture.run
        / "evaluations"
        / f"fixed-slices-{hashlib.sha256(content).hexdigest()}.json"
    ).write_bytes(content)
    assert (
        selection.verify_fixed_selection(fixture.declaration, output)["selected_step"]
        == 20
    )
    _replace_score(fixture.paths[0], lambda score: score.update(mode="test"))
    with pytest.raises(ValueError):
        selection.verify_fixed_selection(fixture.declaration, output)


def test_requested_and_effective_config_bindings_are_distinct(
    receipt_fixture, tmp_path
):
    fixture = receipt_fixture
    output = tmp_path / "selected.json"
    selection.select_fixed_validation(fixture.declaration, output)
    (fixture.run / "resolved_config.yaml").write_text("changed effective bytes")
    with pytest.raises(ValueError, match="receipt differs"):
        selection.verify_fixed_selection(fixture.declaration, output)
    assert output.exists()


def test_requested_config_substitution_fails(receipt_fixture, tmp_path, monkeypatch):
    fixture = receipt_fixture
    original = selection.read_manifest
    monkeypatch.setattr(
        selection,
        "read_manifest",
        lambda path: {**original(path), "requested_config": {"seed": 19}},
    )
    with pytest.raises(ValueError, match="run/config"):
        selection.select_fixed_validation(
            fixture.declaration, tmp_path / "selected.json"
        )


def test_cli_selection_is_zero_runtime_command():
    parser = build_parser()
    select = parser.parse_args(
        ["evaluation", "fixed-slices", "select", "d.json", "--output", "s.json"]
    )
    verify = parser.parse_args(
        ["evaluation", "fixed-slices", "verify-selection", "d.json", "s.json"]
    )
    assert select.fixed_command == "select" and select.output == "s.json"
    assert verify.fixed_command == "verify-selection" and verify.receipt == "s.json"


@pytest.mark.parametrize("bad_steps", [STEPS[::-1], STEPS[:-1], STEPS[:-1] + (90,)])
def test_missing_duplicate_or_unordered_declaration_fails(
    receipt_fixture, tmp_path, bad_steps
):
    payload = json.loads(receipt_fixture.declaration.read_text())
    payload["steps"] = bad_steps
    receipt_fixture.declaration.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        selection.select_fixed_validation(
            receipt_fixture.declaration, tmp_path / "selected.json"
        )
