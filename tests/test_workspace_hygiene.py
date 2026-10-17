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


def test_resource_retirement_records_originals_and_destination() -> None:
    import json

    record = json.loads((REPO / "docs/research-resource-retirement.json").read_text())
    assert len(record["files"]) == 26
    assert len({row["source"] for row in record["files"]}) == 26
    for row in record["files"]:
        assert row["destination"] == "provenance/lab/" + row["source"]
        assert len(row["sha256"]) == 64
        assert row["size_bytes"] > 0
        if not row["source"].endswith("/lifecycle.json") and "disposition" not in row:
            assert not (REPO / row["source"]).exists()
    assert record["lifecycle_supplement"] == "provenance/lab-lifecycle-supplement.json"


def test_retained_historical_evidence_keeps_exact_bytes() -> None:
    import hashlib
    import json

    record = json.loads((REPO / "docs/research-resource-retirement.json").read_text())
    rows = record["retained_in_lab"]
    paths = {row["relative_path"] for row in rows}
    assert len(rows) == len(paths) == 77
    assert {
        "docs/research/sample-report.md",
        "docs/context-engram-study.md",
        "docs/portable-engram.md",
        "docs/research/learned-engram-portability.md",
    } <= paths
    assert "docs/path-domain-corpus.md" not in paths
    for row in rows:
        content = (REPO / row["relative_path"]).read_bytes()
        assert len(content) == row["size_bytes"], row["relative_path"]
        assert hashlib.sha256(content).hexdigest() == row["sha256"], row[
            "relative_path"
        ]


def test_active_research_guides_use_external_resource_authoring() -> None:
    for relative in (
        "docs/research/README.md",
        "papers.md",
        "experiments/samples/README.md",
    ):
        text = (REPO / relative).read_text()
        assert "--resource-root /path/to/resources" in text
        assert "research scaffold engram-" not in text
        assert "research describe engram-" not in text
        assert "research scaffold memory-allocation-curve-v1" not in text
    for name in (
        "engram-ffn-substitution",
        "engram-mla-compression",
        "engram-moe-capacity",
        "engram-placement",
        "engram-sparse-budget",
        "lexical-memory-heavy",
    ):
        text = (REPO / f"docs/research/{name}.md").read_text()
        assert "1aa84218a95fabafc96f2cc7d667c0c969eb8d5e/provenance/lab/" in text
        assert "new experiment identity" in text
        assert "research scaffold " not in text


def test_devmind_v2_retirement_keeps_neutral_corpus_coverage() -> None:
    import json

    record = json.loads((REPO / "docs/devmind-v2-retirement.json").read_text())
    rows = record["files"]
    assert len(rows) == len({row["source"] for row in rows}) == 23
    for row in rows:
        assert row["source"].startswith("corpora/devmind-v2/")
        assert row["destination"] == "provenance/lab/" + row["source"]
        assert len(row["sha256"]) == 64
        assert row["size_bytes"] > 0
        assert not (REPO / row["source"]).exists()
    assert (REPO / "corpora/devmind-sample-v0/corpus.yaml").is_file()
    for name in ("large_build", "pipeline", "rights_header", "structure_cleaning"):
        text = (REPO / f"tests/test_corpus_{name}.py").read_text()
        assert "corpora/devmind-v2" not in text
        assert "neutral_corpus(" in text
