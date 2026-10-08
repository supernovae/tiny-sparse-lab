"""Run the frozen 200-prompt native panel at the preselected Card 05 generation."""

from __future__ import annotations

import json
import os
from pathlib import Path

from sparselab.evaluation.inference import evaluation_config
from sparselab.evaluation.panel import run_panel, verify_panel_result
from sparselab.evaluation.suite import run_suite, verify_evaluation_index
from sparselab.runtime_environments import profile_for_id
from sparselab.runtime_profile import authorize_profile
from sparselab.training.manifest import sha256_file

RUN_ID = "kml-card05-full-tranche-v1"


def evaluate(root: Path, checkout: Path) -> dict[str, object]:
    project = checkout / "experiments/research/kernel-memory-lab"
    selected = json.loads((root / "selected-checkpoint.json").read_text())
    if selected.get("run_id") != RUN_ID or selected.get("selected_step") not in {
        0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4883
    }:
        raise ValueError("selected checkpoint receipt invalid")
    runs = root.parent / "card04-synthetic/runs"
    config = evaluation_config(RUN_ID, runs, str(selected["checkpoint"]), "rocm")
    authorization = authorize_profile(profile_for_id("rocm-7900xtx"), config)
    index_path = run_suite(
        project / "card05-selected-evaluation-suite-v1.json",
        RUN_ID,
        str(selected["checkpoint"]),
        runs,
        "rocm",
        authorization=authorization,
    )
    index = verify_evaluation_index(index_path)
    if (
        index["run_id"] != RUN_ID
        or index["checkpoint_sha256"] != selected["checkpoint_sha256"]
    ):
        raise ValueError("one-batch index differs from selected checkpoint")
    panel_path = run_panel(
        project / "card05-language-panel-v1.json",
        index_path,
        backend="rocm",
        authorization=authorization,
    )
    panel = verify_panel_result(panel_path)
    rows = panel["rows"]
    if len(rows) != 200 or any(row["status"] != "COMPLETED" for row in rows):
        raise ValueError("language generation incomplete; retain native partial panel")
    result: dict[str, object] = {
        "format": "kml-card05-language-generation-v1",
        "run_id": RUN_ID,
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "evaluation_index": str(index_path),
        "evaluation_index_file_sha256": sha256_file(index_path),
        "panel": str(panel_path),
        "panel_file_sha256": sha256_file(panel_path),
        "completed_rows": 200,
    }
    output = root / "evaluation-generation-complete.json"
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


if __name__ == "__main__":
    evaluate(Path(os.environ["KML_PROFILE_ROOT"]), Path(os.environ["KML_CHECKOUT"]))
