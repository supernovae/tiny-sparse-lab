"""Offline public dispatch qualification: no model, network or tokenizer work."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sparselab.cli.main import build_parser
from sparselab.operational_monitor import capture_workspace_baseline
from sparselab.training.attempt_budget import (
    AttemptBudget,
    AttemptBudgetError,
    AttemptContract,
)
from sparselab.training.attempt_commands import classify_attempt_command


def _project(root: Path) -> Path:
    recipe = root / "recipe"
    (recipe / "sources").mkdir(parents=True)
    blob = b"# A harmless offline fixture\n"
    oid = hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest()
    payloads = {
        "corpus.yaml": {
            "schema_version": 1,
            "id": "attempt_fixture",
            "sources": ["sources/one.yaml"],
            "transforms": [],
            "splits": "splits.yaml",
            "release": "release.yaml",
            "transport_budget": {
                "attempt_id": "attempt_fixture",
                "max_source_body_bytes": 128,
                "max_metadata_body_bytes": 128,
                "max_transfers": 1,
                "max_retries_per_shard": 0,
                "max_wall_seconds": 120,
                "max_disk_bytes": 1024,
            },
        },
        "splits.yaml": {
            "schema_version": 1,
            "unit": "document",
            "assignments": {"document": "train"},
        },
        "release.yaml": {
            "schema_version": 1,
            "mixture": {"technical_docs": 1.0},
            "lm": {"selected": False, "training_splits": []},
            "chat": {"selected": False, "training_splits": []},
        },
        "sources/one.yaml": {
            "schema_version": 1,
            "id": "one",
            "kind": "git",
            "canonical_uri": "https://github.com/example/offline-fixture",
            "revision": "a" * 40,
            "license": "MIT",
            "redistribution": "redistributable",
            "domains": ["technical_docs"],
            "document_kinds": ["markdown"],
            "source_family": "docs",
            "acquisition": {
                "max_bytes": len(blob),
                "tree_oid": "b" * 40,
                "bounded_blobs": [
                    {
                        "path": "docs/data.md",
                        "git_blob_oid": oid,
                        "max_bytes": len(blob),
                    }
                ],
            },
        },
    }
    for name, data in payloads.items():
        (recipe / name).write_text(yaml.safe_dump(data))
    return recipe / "corpus.yaml"


def _fixture(
    tmp_path: Path,
    *,
    preparation_bytes: int = 536870912,
    require_config: bool = False,
    require_eval_baseline: bool = False,
    initialize: bool = True,
) -> dict[str, Path | str]:
    root = tmp_path / "owned"
    root.mkdir()
    project = _project(root)
    baseline_path = root / "baseline.json"
    baseline = capture_workspace_baseline(root, baseline_path, seconds=2)
    whole = root / "whole.yaml"
    prep = root / "prep.yaml"
    evaluation = root / "evaluation.yaml"
    for path in (whole, prep, evaluation):
        added = preparation_bytes if path == prep else 536870912
        path.write_text(
            "monitor_policy_version: 1\n"
            "interval_seconds: 0.1\n"
            "termination_grace_seconds: 0.2\n"
            "max_tree_rss_bytes: 4294967296\n"
            f"max_added_workspace_bytes: {added}\n"
            "max_added_workspace_inodes: 1000\n"
            "max_wall_seconds: 120\n"
        )
    contract = root / "contract.json"
    identity = "a" * 64
    contract.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "max_optimizer_updates": 0,
                "max_actual_target_positions": 0,
                "max_generation_calls": 0,
                "max_generated_tokens": 0,
                "max_wall_seconds": 120,
                "content_identity_sha256": identity,
                "acquisition_project_sha256": hashlib.sha256(
                    project.read_bytes()
                ).hexdigest(),
                "fixed_profile_sha256": "a" * 64,
                "fixed_family_inventory_sha256": "b" * 64,
                "monitor_policy_sha256": hashlib.sha256(whole.read_bytes()).hexdigest(),
                "preparation_monitor_policy_sha256": hashlib.sha256(
                    prep.read_bytes()
                ).hexdigest(),
                "evaluation_monitor_policy_sha256": hashlib.sha256(
                    evaluation.read_bytes()
                ).hexdigest(),
                "workspace_baseline_sha256": baseline.sha256,
                "require_resolved_train_config_binding": require_config,
                "require_evaluation_baseline_binding": require_eval_baseline,
            },
            sort_keys=True,
        )
    )
    ledger = root / "ledger.sqlite"
    if initialize:
        AttemptBudget.create_contract(
            ledger,
            contract_path=contract,
            expected_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),
        )
    return {
        "root": root,
        "project": project,
        "baseline": baseline_path,
        "whole": whole,
        "prep": prep,
        "evaluation": evaluation,
        "contract": contract,
        "ledger": ledger,
        "identity": identity,
    }


def _native(root: Path, *words: str) -> list[str]:
    return [
        "uv",
        "run",
        "--locked",
        "--no-sync",
        "sparselab",
        "--work-dir",
        str(root),
        *words,
    ]


def _wrapped(paths: dict[str, Path | str], *words: str) -> list[str]:
    root = paths["root"]
    assert isinstance(root, Path)
    return _native(
        root,
        "monitor",
        "--policy",
        str(paths["prep"]),
        "--log-dir",
        str(root / "preparation-monitor"),
        "--workspace",
        str(root),
        "--baseline",
        str(paths["baseline"]),
        "--",
        *_native(root, *words),
    )


def _dispatch(
    paths: dict[str, Path | str], *, label: str = "fixture-preparation"
) -> list[str]:
    root = paths["root"]
    assert isinstance(root, Path)
    return _native(
        root,
        "attempt",
        "run",
        "--ledger",
        str(paths["ledger"]),
        "--label",
        label,
        "--activity",
        "inspect",
        "--content-identity-sha256",
        str(paths["identity"]),
        "--policy",
        str(paths["whole"]),
        "--baseline",
        str(paths["baseline"]),
        "--workspace",
        str(root),
        "--completion",
        str(root / f"{label}.json"),
        "--updates",
        "0",
        "--target-positions",
        "0",
        "--generation-calls",
        "0",
        "--generated-tokens",
        "0",
        "--receipt-kind",
        "none",
        "--",
        *_wrapped(paths, "corpus", "budget-init", str(paths["project"])),
    )


def _render_preledger_contract(paths: dict[str, Path | str]) -> None:
    root = paths["root"]
    contract = paths["contract"]
    assert isinstance(root, Path) and isinstance(contract, Path)
    declaration = json.loads(contract.read_text())
    declaration["workspace_baseline_sha256"] = (
        "${VERIFIED_WORKSPACE_BASELINE_IDENTITY_SHA256}"
    )
    declaration["require_preledger_monitor_binding"] = True
    template = root / "contract.template.json"
    template.write_text(json.dumps(declaration, sort_keys=True))
    contract.unlink()
    completed = subprocess.run(
        _native(
            root,
            "corpus",
            "render-declaration",
            "--template",
            str(template),
            "--values-json",
            "{}",
            "--workspace-baseline",
            str(paths["baseline"]),
            "--output",
            str(contract),
        ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)


def _init_preledger(
    paths: dict[str, Path | str], *, workspace: Path | None = None
) -> subprocess.CompletedProcess[str]:
    root = paths["root"]
    contract = paths["contract"]
    assert isinstance(root, Path) and isinstance(contract, Path)
    return subprocess.run(
        _native(
            root,
            "attempt",
            "init",
            "--ledger",
            str(paths["ledger"]),
            "--contract",
            str(contract),
            "--contract-sha256",
            hashlib.sha256(contract.read_bytes()).hexdigest(),
            "--policy",
            str(paths["whole"]),
            "--baseline",
            str(paths["baseline"]),
            "--workspace",
            str(workspace or root),
        ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.skipif(os.name != "posix", reason="owned subprocess qualification")
def test_verified_baseline_render_preledger_init_and_real_dispatch(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, initialize=False)
    _render_preledger_contract(paths)
    root = paths["root"]
    assert isinstance(root, Path)
    from sparselab.operational_monitor import load_workspace_baseline

    assert (
        json.loads(paths["contract"].read_text())["workspace_baseline_sha256"]
        == load_workspace_baseline(paths["baseline"], root).sha256
    )
    assert not paths["ledger"].exists()
    initialized = _init_preledger(paths)
    assert initialized.returncode == 0, initialized.stderr
    completed = subprocess.run(
        _dispatch(paths), capture_output=True, text=True, timeout=90, check=False
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert (
        json.loads((root / "fixture-preparation.json").read_text())[
            "living_descendants"
        ]
        == 0
    )
    assert AttemptBudget(paths["ledger"]).status()["charged_updates"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "file-hash",
        "tampered-baseline",
        "wrong-root",
        "changed-policy",
        "changed-binding",
    ],
)
def test_preledger_binding_rejects_bad_inputs_without_ledger(
    tmp_path: Path, fault: str
) -> None:
    paths = _fixture(tmp_path, initialize=False)
    _render_preledger_contract(paths)
    contract = paths["contract"]
    assert isinstance(contract, Path)
    workspace = None
    if fault in {"file-hash", "changed-binding"}:
        declaration = json.loads(contract.read_text())
        declaration["workspace_baseline_sha256"] = (
            hashlib.sha256(paths["baseline"].read_bytes()).hexdigest()
            if fault == "file-hash"
            else "c" * 64
        )
        contract.write_text(json.dumps(declaration, sort_keys=True))
    elif fault == "tampered-baseline":
        baseline = paths["baseline"]
        assert isinstance(baseline, Path)
        receipt = json.loads(baseline.read_text())
        receipt["apparent_bytes"] += 1
        baseline.write_text(json.dumps(receipt))
    elif fault == "wrong-root":
        workspace = tmp_path / "other"
        workspace.mkdir()
    else:
        whole = paths["whole"]
        assert isinstance(whole, Path)
        whole.write_text(whole.read_text() + "# changed\n")
    completed = _init_preledger(paths, workspace=workspace)
    assert completed.returncode != 0
    assert not paths["ledger"].exists()


def test_preledger_binding_cannot_be_omitted(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, initialize=False)
    _render_preledger_contract(paths)
    root = paths["root"]
    contract = paths["contract"]
    assert isinstance(root, Path) and isinstance(contract, Path)
    completed = subprocess.run(
        _native(
            root,
            "attempt",
            "init",
            "--ledger",
            str(paths["ledger"]),
            "--contract",
            str(contract),
            "--contract-sha256",
            hashlib.sha256(contract.read_bytes()).hexdigest(),
        ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode != 0
    assert not paths["ledger"].exists()


def test_renderer_refuses_unverified_baseline_identity_override(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, initialize=False)
    root = paths["root"]
    assert isinstance(root, Path)
    template = root / "baseline-slot.template.json"
    template.write_text('{"identity":"${VERIFIED_WORKSPACE_BASELINE_IDENTITY_SHA256}"}')
    output = root / "bad-render.json"
    completed = subprocess.run(
        _native(
            root,
            "corpus",
            "render-declaration",
            "--template",
            str(template),
            "--values-json",
            json.dumps({"VERIFIED_WORKSPACE_BASELINE_IDENTITY_SHA256": "f" * 64}),
            "--workspace-baseline",
            str(paths["baseline"]),
            "--output",
            str(output),
        ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode != 0
    assert not output.exists()


@pytest.mark.skipif(os.name != "posix", reason="owned subprocess qualification")
def test_public_cli_owned_whole_and_preparation_monitors_dispatch_native_budget_init(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    root = paths["root"]
    assert isinstance(root, Path)
    command = _dispatch(paths)
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=90, check=False
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert (root / "corpora" / "attempt_fixture" / "transport-budget.sqlite").is_file()
    owned = json.loads((root / "fixture-preparation.json").read_text())
    whole = json.loads(
        (root / "fixture-preparation.json.monitor" / "completion.json").read_text()
    )
    preparation = json.loads(
        (root / "preparation-monitor" / "completion.json").read_text()
    )
    assert owned["living_descendants"] == 0
    assert whole["status"] == preparation["status"] == "COMPLETE"
    assert preparation["launch"]["command"] == _native(
        root, "corpus", "budget-init", str(paths["project"])
    )
    status = AttemptBudget(paths["ledger"]).status()
    assert status["charged_updates"] == status["charged_actual_target_positions"] == 0
    assert status["charged_generation_calls"] == status["charged_generated_tokens"] == 0
    assert len(status["reservations"]) == 1
    assert status["reservations"][0]["actual_updates"] == 0
    transport = subprocess.run(
        _native(root, "corpus", "budget-status", str(paths["project"])),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert transport.returncode == 0, transport.stderr
    assert json.loads(transport.stdout)["source_charged"] == 0
    repeated_init = subprocess.run(
        _native(root, "corpus", "budget-init", str(paths["project"])),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert repeated_init.returncode != 0
    assert "already exists" in repeated_init.stderr


@pytest.mark.parametrize(
    "words",
    [
        ["bash", "-c", "sparselab corpus budget-init anything"],
        ["uv", "run", "--locked", "--no-sync", "python", "-c", "pass"],
        ["sparselab", "corpus", "build", "project", "--execute-runs"],
        ["sparselab", "stage", "config", "--through", "warmup", "--output", "out"],
        [
            "sparselab",
            "evaluation",
            "fixed-slices",
            "score",
            "p",
            "r",
            "--mode",
            "test",
            "--max-forward-positions",
            "9999",
        ],
        ["sparselab", "monitor", "--policy", "p", "--", "bash", "-c", "true"],
    ],
)
def test_hidden_commands_and_malformed_options_are_rejected(words: list[str]) -> None:
    with pytest.raises(ValueError):
        classify_attempt_command(words)


@pytest.mark.parametrize(
    "words,effect",
    [
        (["corpus", "budget-init", "project"], "preparation"),
        (
            [
                "corpus",
                "alias-snapshot",
                "project",
                "--source-id",
                "one",
                "--snapshot",
                "/x",
            ],
            "preparation",
        ),
        (["corpus", "acquire", "project"], "preparation"),
        (["corpus", "acquire", "project", "--offline"], "preparation"),
        (
            [
                "corpus",
                "admission-draft",
                "project",
                "--template",
                "/t",
                "--policy-document",
                "/p",
                "--output",
                "/o",
            ],
            "preparation",
        ),
        (
            ["corpus", "split-inventory", "project", "--output", "/o", "--json"],
            "preparation",
        ),
        (
            [
                "corpus",
                "freeze-splits",
                "project",
                "--inventory",
                "/i",
                "--clusters",
                "/c",
                "--output",
                "/o",
                "--json",
            ],
            "preparation",
        ),
        (["corpus", "build", "project", "--offline"], "preparation"),
        (["corpus", "freeze", "/build"], "preparation"),
        (["corpus", "audit", "/release"], "inspection"),
        (["corpus", "near-duplicates", "/release"], "inspection"),
        (
            [
                "corpus",
                "finalize-family-inventory",
                "/release",
                "--splits",
                "/s",
                "--output",
                "/o",
                "--json",
            ],
            "preparation",
        ),
        (
            [
                "corpus",
                "audit-protected-lineage",
                "--prior-release",
                "/p",
                "--candidate-release",
                "/c",
                "--prior-inventory",
                "/pi",
                "--candidate-inventory",
                "/ci",
                "--profile",
                "/q",
                "--suite",
                "/s",
                "--output",
                "/o",
                "--json",
            ],
            "preparation",
        ),
        (
            [
                "corpus",
                "measure-tokens",
                "/r",
                "--tokenizer",
                "/t",
                "--tokenizer-origin-release",
                "/o",
                "--family-inventory",
                "/f",
                "--policy",
                "/p",
                "--output",
                "/m",
                "--batch-source-bytes",
                "8388608",
                "--json",
            ],
            "preparation",
        ),
        (
            ["corpus", "materialize-mixture", "/m", "--output", "/o", "--json"],
            "preparation",
        ),
        (["corpus", "verify-mixture", "/m", "--output", "/o", "--json"], "inspection"),
        (["corpus", "verify-export", "/c"], "inspection"),
        (
            [
                "data",
                "prepared-inputs",
                "publish",
                "/c",
                "--output",
                "/b",
                "--resource-envelope",
                "/e",
                "--tokenizer-batch-source-bytes",
                "8388608",
            ],
            "preparation",
        ),
        (["data", "prepared-inputs", "verify", "/c", "/b"], "inspection"),
        (["inspect", "/c", "--json"], "inspection"),
        (["workspace", "preflight", "/c"], "inspection"),
        (
            [
                "stage",
                "/c",
                "--through",
                "validate",
                "--prepared-inputs",
                "/b",
                "--cold-verify",
                "--output",
                "/o",
                "--runtime",
                "rocm-7900xtx",
                "--resource-envelope",
                "/e",
            ],
            "stage",
        ),
        (
            [
                "train",
                "/c",
                "--stage-bundle",
                "/b",
                "--run-id",
                "run",
                "--runs-dir",
                "/runs",
                "--runtime",
                "rocm-7900xtx",
                "--resource-envelope",
                "/e",
            ],
            "train",
        ),
        (
            [
                "evaluation",
                "fixed-slices",
                "score",
                "/q",
                "run",
                "--release",
                "/h",
                "--tokenizer-config",
                "/t",
                "--family-inventory",
                "/f",
                "--expected-profile-sha256",
                "a" * 64,
                "--expected-family-sha256",
                "b" * 64,
                "--checkpoint",
                "/c",
                "--runs-dir",
                "/runs",
                "--backend",
                "rocm",
                "--runtime",
                "rocm-7900xtx",
                "--mode",
                "validation",
                "--max-forward-positions",
                "3084",
                "--json",
            ],
            "fixed_score",
        ),
        (
            [
                "evaluation",
                "fixed-slices",
                "continuations",
                "/q",
                "run",
                "--release",
                "/h",
                "--tokenizer-config",
                "/t",
                "--family-inventory",
                "/f",
                "--expected-profile-sha256",
                "a" * 64,
                "--expected-family-sha256",
                "b" * 64,
                "--checkpoint",
                "/c",
                "--runs-dir",
                "/runs",
                "--backend",
                "rocm",
                "--runtime",
                "rocm-7900xtx",
                "--json",
            ],
            "continuation",
        ),
        (
            ["evaluation", "fixed-slices", "select", "/d", "--output", "/o", "--json"],
            "preparation",
        ),
        (
            ["evaluation", "fixed-slices", "verify-selection", "/d", "/r", "--json"],
            "inspection",
        ),
        (
            ["monitor-baseline", "/root", "--output", "/o", "--seconds", "4", "--json"],
            "preparation",
        ),
        (["checkpoint", "verify", "/c", "--json"], "inspection"),
    ],
)
def test_complete_packet_leaf_map_parses_without_execution(
    tmp_path: Path, words: list[str], effect: str
) -> None:
    command = _native(tmp_path, *words)
    assert classify_attempt_command(command).effect == effect
    parsed = build_parser(tmp_path).parse_args(["--work-dir", str(tmp_path), *words])
    assert parsed.command == words[0]


def test_legacy_contract_omits_new_optional_bindings(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    raw = json.loads(Path(paths["contract"]).read_text())
    raw.pop("preparation_monitor_policy_sha256")
    raw.pop("evaluation_monitor_policy_sha256")
    raw.pop("acquisition_project_sha256")
    raw.pop("fixed_profile_sha256")
    raw.pop("fixed_family_inventory_sha256")
    raw.pop("require_resolved_train_config_binding")
    raw.pop("require_evaluation_baseline_binding")
    parsed = AttemptContract.model_validate(raw)
    dumped = parsed.model_dump(mode="json")
    assert "preparation_monitor_policy_sha256" not in dumped
    assert "acquisition_project_sha256" not in dumped
    assert "fixed_profile_sha256" not in dumped
    assert "fixed_family_inventory_sha256" not in dumped
    assert "evaluation_monitor_policy_sha256" not in dumped
    assert "require_resolved_train_config_binding" not in dumped
    assert "require_evaluation_baseline_binding" not in dumped


def test_wrong_monitor_identity_rejected_before_ledger_reservation(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    root = paths["root"]
    assert isinstance(root, Path)
    other = root / "other.yaml"
    other.write_text("monitor_policy_version: 1\nmax_wall_seconds: 1\n")
    command = _wrapped(paths, "corpus", "budget-init", str(paths["project"]))
    command[command.index("--policy") + 1] = str(other)
    with pytest.raises(AttemptBudgetError, match="nested monitor policy"):
        AttemptBudget(paths["ledger"]).run_contract(
            command,
            activity="inspect",
            label="wrong-policy",
            content_identity_sha256=str(paths["identity"]),
            monitor_policy_path=paths["whole"],
            workspace_baseline_path=paths["baseline"],
            workspace_root=root,
            completion=root / "wrong.json",
        )
    assert AttemptBudget(paths["ledger"]).status()["reservations"] == []


def test_changed_acquisition_project_rejected_before_dispatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    root = paths["root"]
    assert isinstance(root, Path)
    project = paths["project"]
    assert isinstance(project, Path)
    project.write_bytes(project.read_bytes() + b"\n")
    with pytest.raises(AttemptBudgetError, match="acquisition project"):
        AttemptBudget(paths["ledger"]).run_contract(
            _wrapped(paths, "corpus", "budget-init", str(project)),
            activity="inspect",
            label="project-drift",
            content_identity_sha256=str(paths["identity"]),
            monitor_policy_path=paths["whole"],
            workspace_baseline_path=paths["baseline"],
            workspace_root=root,
            completion=root / "project-drift.json",
        )
    assert AttemptBudget(paths["ledger"]).status()["reservations"] == []


def test_live_acquisition_needs_existing_bound_transport_ledger(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    root = paths["root"]
    assert isinstance(root, Path)
    with pytest.raises(ValueError, match="transport budget ledger missing"):
        AttemptBudget(paths["ledger"]).run_contract(
            _wrapped(paths, "corpus", "acquire", str(paths["project"])),
            activity="inspect",
            label="unfunded-transfer",
            content_identity_sha256=str(paths["identity"]),
            monitor_policy_path=paths["whole"],
            workspace_baseline_path=paths["baseline"],
            workspace_root=root,
            completion=root / "unfunded.json",
        )
    assert AttemptBudget(paths["ledger"]).status()["reservations"] == []


def test_fixed_scoring_rejected_by_attempt_without_forward_allocation(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    root = paths["root"]
    assert isinstance(root, Path)
    leaf = _native(
        root,
        "evaluation",
        "fixed-slices",
        "score",
        "profile.json",
        "run",
        "--release",
        "release",
        "--tokenizer-config",
        "tokenizer.yaml",
        "--family-inventory",
        "families.jsonl",
        "--expected-profile-sha256",
        "a" * 64,
        "--expected-family-sha256",
        "b" * 64,
        "--checkpoint",
        "initial",
        "--runs-dir",
        str(root / "runs"),
        "--backend",
        "cpu",
        "--runtime",
        "cpu",
        "--mode",
        "validation",
        "--max-forward-positions",
        "3084",
        "--json",
    )
    with pytest.raises(AttemptBudgetError, match="model allocation"):
        AttemptBudget(paths["ledger"]).run_contract(
            leaf,
            activity="evaluate",
            label="unmetered-score",
            content_identity_sha256=str(paths["identity"]),
            monitor_policy_path=paths["whole"],
            workspace_baseline_path=paths["baseline"],
            workspace_root=root,
            completion=root / "unmetered.json",
        )
    assert AttemptBudget(paths["ledger"]).status()["reservations"] == []


@pytest.mark.parametrize(
    "flag", ["--expected-profile-sha256", "--expected-family-sha256"]
)
def test_fixed_score_rejects_substituted_frozen_identity_before_dispatch(
    tmp_path: Path, flag: str
) -> None:
    paths = _fixture(tmp_path)
    root = paths["root"]
    assert isinstance(root, Path)
    leaf = _native(
        root,
        "evaluation",
        "fixed-slices",
        "score",
        "profile.json",
        "run",
        "--release",
        "release",
        "--tokenizer-config",
        "tokenizer.yaml",
        "--family-inventory",
        "families.jsonl",
        "--expected-profile-sha256",
        "a" * 64,
        "--expected-family-sha256",
        "b" * 64,
        "--checkpoint",
        "initial",
        "--runs-dir",
        str(root / "runs"),
        "--backend",
        "cpu",
        "--runtime",
        "cpu",
        "--mode",
        "validation",
        "--max-forward-positions",
        "3084",
        "--json",
    )
    leaf[leaf.index(flag) + 1] = "c" * 64
    with pytest.raises(AttemptBudgetError, match="fixed evaluation identity"):
        AttemptBudget(paths["ledger"]).run_contract(
            leaf,
            activity="evaluate",
            label="substituted-fixed-score",
            content_identity_sha256=str(paths["identity"]),
            monitor_policy_path=paths["whole"],
            workspace_baseline_path=paths["baseline"],
            workspace_root=root,
            completion=root / "substituted.json",
        )
    assert AttemptBudget(paths["ledger"]).status()["reservations"] == []


def test_fixed_score_requires_cold_bound_evaluation_monitor_before_allocation(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, require_eval_baseline=True)
    root = paths["root"]
    assert isinstance(root, Path)
    baseline = root / "evaluation-baseline.json"
    capture_workspace_baseline(root, baseline, seconds=1)
    budget = AttemptBudget(paths["ledger"])
    budget.bind_resolved_artifact(
        kind="evaluation_baseline",
        path=baseline,
        expected_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        content_identity_sha256=str(paths["identity"]),
        workspace_root=root,
    )
    leaf = _native(
        root,
        "evaluation",
        "fixed-slices",
        "score",
        "profile.json",
        "fixture-run",
        "--release",
        "release",
        "--tokenizer-config",
        "tokenizer.yaml",
        "--family-inventory",
        "families.jsonl",
        "--expected-profile-sha256",
        "a" * 64,
        "--expected-family-sha256",
        "b" * 64,
        "--checkpoint",
        "initial",
        "--runs-dir",
        str(root / "runs"),
        "--backend",
        "cpu",
        "--runtime",
        "cpu",
        "--mode",
        "validation",
        "--max-forward-positions",
        "3084",
        "--json",
    )

    def wrapped(baseline_path: Path) -> list[str]:
        return _native(
            root,
            "monitor",
            "--policy",
            str(paths["evaluation"]),
            "--log-dir",
            str(root / "evaluation-monitor"),
            "--workspace",
            str(root),
            "--baseline",
            str(baseline_path),
            "--",
            *leaf,
        )

    with pytest.raises(AttemptBudgetError, match="nested monitor policy or baseline"):
        budget.run_contract(
            wrapped(paths["baseline"]),
            activity="evaluate",
            label="wrong-evaluation-baseline",
            content_identity_sha256=str(paths["identity"]),
            monitor_policy_path=paths["whole"],
            workspace_baseline_path=paths["baseline"],
            workspace_root=root,
            completion=root / "wrong-evaluation.json",
        )
    with pytest.raises(AttemptBudgetError, match="model allocation"):
        budget.run_contract(
            wrapped(baseline),
            activity="evaluate",
            label="unfunded-evaluation",
            content_identity_sha256=str(paths["identity"]),
            monitor_policy_path=paths["whole"],
            workspace_baseline_path=paths["baseline"],
            workspace_root=root,
            completion=root / "unfunded-evaluation.json",
        )
    assert budget.status()["reservations"] == []


def test_resolved_config_binding_is_one_time_and_cold(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    root = paths["root"]
    assert isinstance(root, Path)
    config = root / "resolved-run.yaml"
    shutil.copyfile("configs/micro_dense.yaml", config)
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    budget = AttemptBudget(paths["ledger"])
    with pytest.raises(AttemptBudgetError, match="bytes differ"):
        budget.bind_resolved_artifact(
            kind="train_config",
            path=config,
            expected_sha256="b" * 64,
            content_identity_sha256=str(paths["identity"]),
            workspace_root=root,
        )
    result = budget.bind_resolved_artifact(
        kind="train_config",
        path=config,
        expected_sha256=digest,
        content_identity_sha256=str(paths["identity"]),
        workspace_root=root,
    )
    assert result["sha256"] == digest
    assert budget.status()["resolved_artifacts"] == [result]
    train = _native(
        root,
        "train",
        str(config),
        "--run-id",
        "fresh-fixture",
        "--runs-dir",
        str(root / "runs"),
    )
    # Static packet identity and the later config digest differ by design.
    with pytest.raises(AttemptBudgetError, match="config differs"):
        budget._preflight_native_command(
            train,
            native_receipt_kind="train",
            native_receipt_path=root / "runs" / "fresh-fixture",
            campaign_stage=None,
            stage_limits=None,
            reserved=(200, 204800, 0, 0),
            parent_checkpoint_path=None,
            content_identity_sha256=str(paths["identity"]),
        )
    budget._preflight_native_command(
        train,
        native_receipt_kind="train",
        native_receipt_path=root / "runs" / "fresh-fixture",
        campaign_stage=None,
        stage_limits=None,
        reserved=(200, 204800, 0, 0),
        parent_checkpoint_path=None,
        content_identity_sha256=budget._resolved_artifact(
            "train_config", workspace_root=root
        )[1],
    )
    with pytest.raises(AttemptBudgetError, match="already bound"):
        budget.bind_resolved_artifact(
            kind="train_config",
            path=config,
            expected_sha256=digest,
            content_identity_sha256=str(paths["identity"]),
            workspace_root=root,
        )
    config.write_bytes(config.read_bytes() + b"\n")
    with pytest.raises(AttemptBudgetError, match="identity changed"):
        budget._resolved_artifact("train_config", workspace_root=root)


def test_packet_stage_refuses_unbound_resolved_config_before_dispatch(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, require_config=True)
    root = paths["root"]
    assert isinstance(root, Path)
    stage = _native(
        root,
        "stage",
        str(root / "prospective.yaml"),
        "--through",
        "validate",
        "--prepared-inputs",
        str(root / "prepared"),
        "--cold-verify",
        "--output",
        str(root / "stage"),
        "--runtime",
        "rocm-7900xtx",
        "--resource-envelope",
        str(root / "envelope.yaml"),
    )
    with pytest.raises(AttemptBudgetError, match="resolved training config"):
        AttemptBudget(paths["ledger"]).run_contract(
            stage,
            activity="validate",
            label="unbound-stage",
            content_identity_sha256=str(paths["identity"]),
            monitor_policy_path=paths["whole"],
            workspace_baseline_path=paths["baseline"],
            workspace_root=root,
            completion=root / "unbound.json",
        )
    assert AttemptBudget(paths["ledger"]).status()["reservations"] == []


@pytest.mark.parametrize("changed", ["profile_sha256", "config_sha256", "config"])
def test_selector_rejects_substituted_late_and_frozen_identities(
    tmp_path: Path, changed: str
) -> None:
    paths = _fixture(tmp_path, require_config=True)
    root = paths["root"]
    assert isinstance(root, Path)
    config = root / "resolved-run.yaml"
    shutil.copyfile("configs/micro_dense.yaml", config)
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    budget = AttemptBudget(paths["ledger"])
    budget.bind_resolved_artifact(
        kind="train_config",
        path=config,
        expected_sha256=digest,
        content_identity_sha256=str(paths["identity"]),
        workspace_root=root,
    )
    declaration = root / "selection.json"
    values = {
        "profile_sha256": "a" * 64,
        "config_sha256": digest,
        "config": str(config),
    }
    values[changed] = "c" * 64 if changed != "config" else str(root / "other.yaml")
    declaration.write_text(json.dumps(values))
    with pytest.raises(AttemptBudgetError, match="fixed selection declaration"):
        budget.run_contract(
            _native(
                root,
                "evaluation",
                "fixed-slices",
                "select",
                str(declaration),
                "--output",
                str(root / "selected.json"),
                "--json",
            ),
            activity="inspect",
            label="substituted-selection",
            content_identity_sha256=str(paths["identity"]),
            monitor_policy_path=paths["whole"],
            workspace_baseline_path=paths["baseline"],
            workspace_root=root,
            completion=root / "substituted-selection.json",
        )
    assert budget.status()["reservations"] == []


@pytest.mark.skipif(os.name != "posix", reason="owned subprocess qualification")
def test_resource_failure_retains_charge_and_shutdown_receipts(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, preparation_bytes=1)
    root = paths["root"]
    assert isinstance(root, Path)
    completed = subprocess.run(
        _dispatch(paths, label="resource-stop"),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode != 0
    owned = json.loads((root / "resource-stop.json").read_text())
    whole = json.loads(
        (root / "resource-stop.json.monitor" / "completion.json").read_text()
    )
    assert owned["living_descendants"] == 0
    assert whole["status"] == "FAILED"
    assert (
        "added workspace cap exceeded before launch"
        in (root / "resource-stop.json.monitor" / "command.stderr.log").read_text()
    )
    assert not (root / "preparation-monitor" / "completion.json").exists()
    assert len(AttemptBudget(paths["ledger"]).status()["reservations"]) == 1


def test_expired_deadline_rejects_before_native_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.training import attempt_budget as module

    paths = _fixture(tmp_path)
    budget = AttemptBudget(paths["ledger"])
    with budget._connect() as connection:
        deadline = connection.execute(
            "SELECT deadline_ns FROM budget WHERE id=1"
        ).fetchone()[0]
    monkeypatch.setattr(module.time, "time_ns", lambda: deadline + 1)
    root = paths["root"]
    assert isinstance(root, Path)
    with pytest.raises(AttemptBudgetError, match="wall-time limit"):
        budget.run_contract(
            _wrapped(paths, "corpus", "budget-init", str(paths["project"])),
            activity="inspect",
            label="expired",
            content_identity_sha256=str(paths["identity"]),
            monitor_policy_path=paths["whole"],
            workspace_baseline_path=paths["baseline"],
            workspace_root=root,
            completion=root / "expired.json",
        )
    assert not (root / "expired.json").exists()


@pytest.mark.parametrize("operation", ["score", "continuations"])
def test_fixed_model_leaf_rejects_missing_allocation_before_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from sparselab.config import loading
    from sparselab.evaluation import fixed_slices, inference
    from sparselab.evaluation.cli import _fixed_slices

    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        loading,
        "load_tokenizer_config",
        lambda _path: SimpleNamespace(output_dir=tmp_path),
    )
    monkeypatch.setattr(
        fixed_slices, "bind_fixed_slices", lambda *_args, **_kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        inference, "load_run", lambda *_args, **_kwargs: pytest.fail("model loaded")
    )
    monkeypatch.setenv("SPARSELAB_ATTEMPT_BUDGET_LEDGER", str(paths["ledger"]))
    monkeypatch.setenv(
        "SPARSELAB_ATTEMPT_CONTENT_IDENTITY_SHA256", str(paths["identity"])
    )
    monkeypatch.setenv("SPARSELAB_ATTEMPT_PHASE_LABEL", "fake-model-leaf")
    monkeypatch.setenv("SPARSELAB_ATTEMPT_ACTIVITY", "evaluate")
    args = argparse.Namespace(
        fixed_command=operation,
        tokenizer_config=str(tmp_path / "tokenizer.yaml"),
        profile=str(tmp_path / "profile.json"),
        release=str(tmp_path / "release"),
        family_inventory=str(tmp_path / "families.jsonl"),
        expected_profile_sha256="a" * 64,
        expected_family_sha256="b" * 64,
        mode="test",
        max_forward_positions=3084,
    )
    with pytest.raises(AttemptBudgetError, match="model allocation"):
        _fixed_slices(args)
