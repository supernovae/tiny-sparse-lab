"""Tiny, offline public-path preparation sequence; no tokenizer or model work."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from sparselab.cli.main import build_parser
from sparselab.config.loading import load_config, load_tokenizer_config
from sparselab.corpus.acquisition import (
    _adapter,
    _digest,
    _project_sha,
    declaration_sha256,
)
from sparselab.corpus.admission_draft import draft_admission_manifest
from sparselab.corpus.declaration_render import render_declaration
from sparselab.corpus.export import (
    _licenses,
    _split_stats,
    export_release,
    verify_release_export,
)
from sparselab.corpus.pipeline import build
from sparselab.corpus.project import load_project, source_declaration_payload
from sparselab.corpus.protected_lineage import audit_protected_lineage
from sparselab.corpus.release import freeze, verify_release
from sparselab.corpus.release_review import verify_admission_review
from sparselab.corpus.split_freeze import finalize_family_inventory, freeze_splits
from sparselab.corpus.split_inventory import write_split_inventory
from sparselab.operational_monitor import capture_workspace_baseline
from sparselab.training.attempt_budget import AttemptBudget, AttemptContract
from sparselab.training.attempt_commands import classify_attempt_command
from sparselab.training.manifest import canonical_json, sha256_file


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_data_only_render_and_legacy_contract_compatibility(tmp_path: Path) -> None:
    template = tmp_path / "template.json"
    template.write_text('{"identity":"${RELEASE_SHA256}"}\n')
    output = tmp_path / "rendered.json"
    rendered = render_declaration(
        template, json.dumps({"RELEASE_SHA256": "a" * 64}), output, tmp_path
    )
    assert rendered["sha256"] == sha256_file(output)
    assert json.loads(output.read_text()) == {"identity": "a" * 64}
    array_template = tmp_path / "array-template.json"
    array_template.write_text('{"reviewed":"${REVIEWED_JSON}"}\n')
    array_output = tmp_path / "array-rendered.json"
    render_declaration(
        array_template,
        json.dumps({"REVIEWED_JSON": '[{"source_id":"covered"}]'}),
        array_output,
        tmp_path,
    )
    assert json.loads(array_output.read_text()) == {
        "reviewed": [{"source_id": "covered"}]
    }
    with pytest.raises(TypeError, match="must be an array"):
        render_declaration(
            array_template,
            json.dumps({"REVIEWED_JSON": "{}"}),
            tmp_path / "not-an-array.json",
            tmp_path,
        )
    with pytest.raises(ValueError, match="exactly cover"):
        render_declaration(template, "{}", tmp_path / "missing.json", tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        render_declaration(template, "{}", tmp_path.parent / "outside.json", tmp_path)
    symlink_parent = tmp_path / "alias"
    symlink_parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        render_declaration(
            template,
            json.dumps({"RELEASE_SHA256": "a" * 64}),
            symlink_parent / "via-alias.json",
            tmp_path,
        )
    assert (
        classify_attempt_command(
            [
                "sparselab",
                "corpus",
                "render-declaration",
                "--template",
                str(template),
                "--values-json",
                "{}",
                "--output",
                str(tmp_path / "declaration.json"),
            ]
        ).effect
        == "preparation"
    )
    assert (
        build_parser()
        .parse_args(["corpus", "budget-status", "project.yaml"])
        .corpus_command
        == "budget-status"
    )
    old = AttemptContract.model_validate(
        {
            "contract_version": 1,
            "max_optimizer_updates": 0,
            "max_actual_target_positions": 0,
            "max_generation_calls": 0,
            "max_generated_tokens": 0,
            "max_wall_seconds": 30,
            "content_identity_sha256": "a" * 64,
            "monitor_policy_sha256": "b" * 64,
            "workspace_baseline_sha256": "c" * 64,
        }
    )
    payload = old.model_dump(mode="json")
    assert "preparation_normalizer" not in payload
    assert "offline_retained_sources_only" not in payload
    assert "require_release_acceptance_binding" not in payload
    assert "require_preledger_monitor_binding" not in payload


def test_reviewed_admission_cannot_substitute_an_edited_draft(tmp_path: Path) -> None:
    draft = tmp_path / "draft.json"
    draft.write_text('{"decision":"quarantine"}\n', encoding="utf-8")
    admitted = tmp_path / "admission.json"
    admitted.write_text('{"decision":"admit"}\n', encoding="utf-8")
    (tmp_path / "admission.json.review.json").write_bytes(
        canonical_json(
            {
                "format": "sparselab-admission-review-v1",
                "decision": "ACCEPTED",
                "reviewer": "Fixture reviewer",
                "reviewed_on": "2026-10-09",
                "draft_path": str(draft),
                "draft_sha256": sha256_file(draft),
                "admission_sha256": sha256_file(admitted),
                "spot_audits": [
                    {
                        "source_id": "source",
                        "location": "row-1",
                        "outcome": "pass",
                        "note": "Reviewed the row.",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_admission_review(admitted, tmp_path)


def _sources(root: Path, project_id: str) -> tuple[Path, dict[str, str]]:
    recipe = root / project_id
    strata = {
        "books": "general_prose",
        "incident": "incident_response_docs",
        "wiki": "explanatory_prose",
    }
    files_by_source: dict[str, list[tuple[str, bytes]]] = {}
    for source_id, stratum in strata.items():
        files = []
        for index in range(12):
            number = 0 if index == 10 else index
            prose = (
                f"# {source_id} topic {number}\n"
                f"The {source_id} topic {number} explains a concrete practice. "
                "A reader can follow the steps and check the result.\n"
            )
            if index == 11:
                prose = "# Exception\nSPDX-License-Identifier: Proprietary\n"
            files.append((f"docs/{index:02}.md", prose.encode()))
        files_by_source[source_id] = files
        _write_yaml(
            recipe / "sources" / f"{source_id}.yaml",
            {
                "schema_version": 2,
                "id": source_id,
                "kind": "git",
                "canonical_uri": f"https://github.com/example/{source_id}",
                "revision": "a" * 40,
                "license": "Apache-2.0 source claim",
                "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
                "rights": {
                    "training_eligibility": "review_required",
                    "redistribution_mode": "review_required",
                },
                "domains": [stratum],
                "document_kinds": ["markdown"],
                "source_family": source_id,
                "acquisition": {"include": ["docs/*.md"], "max_bytes": 65536},
            },
        )
    _write_yaml(
        recipe / "splits-placeholder.yaml",
        {
            "schema_version": 1,
            "unit": "source_document_family",
            "family_key": "source_family",
            "assignments": {source_id: "train" for source_id in strata},
        },
    )
    _write_yaml(
        recipe / "release-quarantine.yaml",
        {
            "schema_version": 2,
            "publication_mode": "metadata_reconstruction_only",
            "mixture": {
                "general_prose": 0.4,
                "incident_response_docs": 0.2,
                "explanatory_prose": 0.4,
            },
            "lm": {"selected": False, "training_splits": []},
            "chat": {"selected": False, "training_splits": []},
        },
    )
    _write_yaml(
        recipe / "transforms/lm.yaml",
        {
            "id": "fixture_lm",
            "version": "1",
            "kind": "lm_text",
            "parameters": {},
            "inputs": list(strata),
        },
    )
    _write_yaml(
        recipe / "acquire.yaml",
        {
            "schema_version": 1,
            "id": project_id,
            "sources": [f"sources/{source_id}.yaml" for source_id in strata],
            "transforms": [],
            "splits": "splits-placeholder.yaml",
            "release": "release-quarantine.yaml",
        },
    )
    project = load_project(recipe / "acquire.yaml")
    entries = {}
    for source in project.sources:
        files = [
            {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
            for name, raw in files_by_source[source.id]
        ]
        identity = _digest(
            {
                "declaration_sha256": declaration_sha256(source),
                "adapter": _adapter(source),
                "files": files,
            }
        )
        snapshot = root / "corpora" / project_id / "snapshots" / source.id / identity
        for name, raw in files_by_source[source.id]:
            destination = snapshot / "files" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        manifest = {
            "schema_version": 1,
            "source_id": source.id,
            "declaration": source_declaration_payload(source),
            "declaration_sha256": declaration_sha256(source),
            "adapter": _adapter(source),
            "files": files,
            "retrieval": {"fixture": "network-disabled immutable snapshot"},
            "snapshot_sha256": identity,
        }
        (snapshot / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        entries[source.id] = {
            "declaration_sha256": declaration_sha256(source),
            "snapshot_sha256": identity,
            "snapshot_path": str(snapshot),
            "receipt": {"status": "acquired", "retrieval": manifest["retrieval"]},
        }
    lock = {
        "schema_version": 1,
        "project_id": project_id,
        "project_sha256": _project_sha(project),
        "sources": entries,
    }
    (root / "corpora" / project_id / "acquisition.json").write_bytes(
        canonical_json(lock) + b"\n"
    )
    return recipe, strata


def _policy(recipe: Path, strata: dict[str, str]) -> tuple[Path, Path]:
    document = recipe / "policy.md"
    document.write_text("Reviewed fixture source policy, Apache-2.0 local use.\n")
    template = recipe / "application.json"
    template.write_bytes(
        canonical_json(
            {
                "policy_id": "fixture-policy-v1",
                "policy_sha256": sha256_file(document),
                "sources": [
                    {
                        "source_id": source_id,
                        "license_label": "Apache-2.0 local research",
                        "rights": {
                            "training_eligibility": "eligible_with_obligations",
                            "redistribution_mode": "metadata_reconstruction_only",
                            "spdx_expression": "Apache-2.0",
                            "license_references": [
                                "https://www.apache.org/licenses/LICENSE-2.0"
                            ],
                            "notices": ["Keep source attribution"],
                        },
                    }
                    for source_id in strata
                ],
            }
        )
        + b"\n"
    )
    return template, document


def _reviewed_release(recipe: Path, admission: Path) -> None:
    _write_yaml(
        recipe / "release-reviewed.yaml",
        {
            "schema_version": 2,
            "publication_mode": "metadata_reconstruction_only",
            "mixture": {
                "general_prose": 0.4,
                "incident_response_docs": 0.2,
                "explanatory_prose": 0.4,
            },
            "lm": {"selected": True, "training_splits": ["train", "validation"]},
            "chat": {"selected": False, "training_splits": []},
            "record_admission": {
                "path": admission.name,
                "sha256": sha256_file(admission),
            },
            "normalizer": "normalizer-structure-v3",
        },
    )
    _write_yaml(
        recipe / "pre-freeze.yaml",
        {
            **yaml.safe_load((recipe / "acquire.yaml").read_text()),
            "release": "release-reviewed.yaml",
        },
    )


def _clusters(
    path: Path,
    inventory: Path,
    strata: dict[str, str],
    prior: tuple[Path, Path] | None = None,
    *,
    seed: str = "fixture-seed",
) -> None:
    payload = {
        "schema_version": 1,
        "inventory_sha256": sha256_file(inventory),
        "seed": seed,
        "source_strata": strata,
        "merges": [
            {
                "family_id": f"git-file:{source_id}:docs/00.md",
                "member_hints": [
                    f"git-file:{source_id}:docs/00.md",
                    f"git-file:{source_id}:docs/10.md",
                ],
            }
            for source_id in strata
        ],
        "reviewer": "Fixture independent reviewer",
        "reviewed_on": "2026-10-09",
    }
    if prior is not None:
        family, release = prior
        payload["prior_family_inventory"] = {
            "path": str(family),
            "sha256": sha256_file(family),
            "release_path": str(release),
        }
    path.write_bytes(canonical_json(payload) + b"\n")


def _prior(root: Path) -> tuple[Path, Path, dict[str, str]]:
    recipe, strata = _sources(root, "fixture-prior")
    template, policy = _policy(recipe, strata)
    admission = recipe / "admission.json"
    result = draft_admission_manifest(
        load_project(recipe / "acquire.yaml"), root, template, policy, admission
    )
    assert result["counts"] == {"qualify": 33, "exclude": 0, "quarantine": 3}
    _reviewed_release(recipe, admission)
    project = load_project(recipe / "pre-freeze.yaml")
    inventory = recipe / "inventory.jsonl"
    write_split_inventory(project, root, inventory)
    clusters = recipe / "clusters.json"
    _clusters(clusters, inventory, strata)
    frozen = recipe / "splits-reviewed.yaml"
    freeze_splits(project, root, inventory, clusters, frozen)
    _write_yaml(
        recipe / "build.yaml",
        {
            **yaml.safe_load((recipe / "pre-freeze.yaml").read_text()),
            "transforms": ["transforms/lm.yaml"],
            "splits": frozen.name,
        },
    )
    release = freeze(
        build(load_project(recipe / "build.yaml"), root, offline=True), root
    )
    assert verify_release(release)["release_id"] == release.name
    family = recipe / "family.jsonl"
    finalize_family_inventory(release, frozen, family)
    return release, family, strata


def _authenticated_fixture_tokenizer(root: Path, release: Path) -> tuple[Path, Path]:
    """Reuse the existing cross-release test pattern with a prebuilt WordLevel model."""
    exported = export_release(
        release, "lm", Path("configs/runtime_smoke_cpu.yaml"), 300, root
    )
    config_path = exported / "tokenizer.yaml"
    config = load_tokenizer_config(config_path)
    tokenizer_path = config.output_dir / "tokenizer.json"
    tokenizer_path.parent.mkdir(parents=True)
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "[UNK]": 3}
    vocab.update({f"unused-{index}": index for index in range(4, 300)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tokenizer_path))
    binding = verify_release_export(config.dataset)
    (tokenizer_path.parent / "tokenizer_manifest.json").write_bytes(
        canonical_json(
            {
                "source": "local_text",
                "revision": release.name,
                "vocab_size": 300,
                "sha256": sha256_file(tokenizer_path),
                "corpus_export": binding,
                "training_contract": {"corpus_export": binding},
            }
        )
        + b"\n"
    )
    return config_path, tokenizer_path


def _native(root: Path, *words: str) -> list[str]:
    return [sys.executable, "-m", "sparselab", "--work-dir", str(root), *words]


def _phase(
    paths: dict[str, Path | str], label: str, *words: str
) -> subprocess.CompletedProcess[str]:
    root = paths["root"]
    assert isinstance(root, Path)
    leaf = _native(root, *words)
    guarded = _native(
        root,
        "monitor",
        "--policy",
        str(paths["prep"]),
        "--log-dir",
        str(root / "logs" / label),
        "--workspace",
        str(root),
        "--baseline",
        str(paths["baseline"]),
        "--",
        *leaf,
    )
    command = _native(
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
        str(root / "logs" / f"{label}.json"),
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
        *guarded,
    )
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
        env=paths["env"],
    )


def _run(paths: dict[str, Path | str], label: str, *words: str) -> None:
    completed = _phase(paths, label, *words)
    assert completed.returncode == 0, (label, completed.stdout, completed.stderr)
    root = paths["root"]
    assert isinstance(root, Path)
    receipt = json.loads((root / "logs" / f"{label}.json").read_text())
    assert receipt["living_descendants"] == 0


def _render(
    paths: dict[str, Path | str],
    label: str,
    template: Path,
    output: Path,
    values: dict[str, str] | None = None,
) -> None:
    _run(
        paths,
        label,
        "corpus",
        "render-declaration",
        "--template",
        str(template),
        "--values-json",
        json.dumps(values or {}, sort_keys=True),
        "--output",
        str(output),
    )


def _attempt(root: Path, acquisition: Path) -> dict[str, Path | str]:
    (root / "logs").mkdir(exist_ok=True)
    baseline = root / "baseline.json"
    value = capture_workspace_baseline(root, baseline, seconds=2)
    whole = root / "whole.yaml"
    prep = root / "prep.yaml"
    policy = {
        "monitor_policy_version": 1,
        "interval_seconds": 0.1,
        "termination_grace_seconds": 0.2,
        "max_tree_rss_bytes": 4 * 1024**3,
        "max_added_workspace_bytes": 512 * 1024**2,
        "max_added_workspace_inodes": 5000,
        "max_wall_seconds": 300,
    }
    _write_yaml(whole, policy)
    _write_yaml(prep, policy)
    project = load_project(acquisition)
    identity = "a" * 64
    contract = root / "contract.json"
    contract.write_bytes(
        canonical_json(
            {
                "contract_version": 1,
                "max_optimizer_updates": 0,
                "max_actual_target_positions": 0,
                "max_generation_calls": 0,
                "max_generated_tokens": 0,
                "max_wall_seconds": 300,
                "content_identity_sha256": identity,
                "monitor_policy_sha256": sha256_file(whole),
                "workspace_baseline_sha256": value.sha256,
                "preparation_monitor_policy_sha256": sha256_file(prep),
                "acquisition_project_sha256": sha256_file(acquisition),
                "preparation_acquisition_identity_sha256": _project_sha(project),
                "preparation_normalizer": "normalizer-structure-v3",
                "require_release_acceptance_binding": True,
                "offline_retained_sources_only": True,
            }
        )
        + b"\n"
    )
    ledger = root / "fixture-ledger.sqlite"
    AttemptBudget.create_contract(
        ledger, contract_path=contract, expected_sha256=sha256_file(contract)
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_OFFLINE"] = "1"
    return {
        "root": root,
        "ledger": ledger,
        "baseline": baseline,
        "whole": whole,
        "prep": prep,
        "identity": identity,
        "env": env,
    }


def _bind(paths: dict[str, Path | str], kind: str, path: Path) -> None:
    root = paths["root"]
    assert isinstance(root, Path)
    command = _native(
        root,
        "attempt",
        "bind-artifact",
        "--ledger",
        str(paths["ledger"]),
        "--kind",
        kind,
        "--path",
        str(path),
        "--sha256",
        sha256_file(path),
        "--content-identity-sha256",
        str(paths["identity"]),
        "--workspace",
        str(root),
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        env=paths["env"],
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)


@pytest.mark.skipif(os.name != "posix", reason="owned-process fixture requires POSIX")
def test_offline_preparation_sequence_through_native_supervision(
    tmp_path: Path,
) -> None:
    root = tmp_path / "work"
    root.mkdir()
    prior_release, prior_family, strata = _prior(root)
    tokenizer_config, tokenizer_path = _authenticated_fixture_tokenizer(
        root, prior_release
    )
    recipe, _ = _sources(root, "fixture-candidate")
    template, policy = _policy(recipe, strata)
    (root / "sitecustomize.py").write_text(
        "import socket\n"
        "def blocked(*args, **kwargs):\n"
        "    raise RuntimeError('network-disabled fixture')\n"
        "socket.socket.connect = blocked\n"
        "socket.create_connection = blocked\n"
    )
    release_template = recipe / "release-reviewed.template.yaml"
    release_spec = yaml.safe_load(
        (root / "fixture-prior/release-reviewed.yaml").read_text()
    )
    release_spec["record_admission"] = {
        "path": "admission.json",
        "sha256": "${ADMISSION_SHA256}",
    }
    _write_yaml(release_template, release_spec)
    admission_review_template = recipe / "admission-review.template.json"
    admission_review_template.write_bytes(
        canonical_json(
            {
                "format": "sparselab-admission-review-v1",
                "decision": "ACCEPTED",
                "reviewer": "Fixture independent reviewer",
                "reviewed_on": "2026-10-09",
                "draft_path": "${DRAFT_PATH}",
                "draft_sha256": "${DRAFT_SHA256}",
                "admission_sha256": "${ADMISSION_SHA256}",
                "spot_audits": [
                    {
                        "source_id": source_id,
                        "location": "docs/00.md",
                        "outcome": "pass",
                        "note": "Covered fixture prose; exception remains quarantined.",
                    }
                    for source_id in strata
                ],
            }
        )
        + b"\n"
    )
    prefreeze_template = recipe / "pre-freeze.template.yaml"
    _write_yaml(
        prefreeze_template,
        {
            **yaml.safe_load((recipe / "acquire.yaml").read_text()),
            "release": "release-reviewed.yaml",
        },
    )
    cluster_template = recipe / "clusters.template.json"
    _clusters(
        cluster_template,
        root / "fixture-prior/inventory.jsonl",
        strata,
        (prior_family, prior_release),
    )
    cluster_spec = json.loads(cluster_template.read_text())
    cluster_spec["inventory_sha256"] = "${INVENTORY_SHA256}"
    cluster_template.write_bytes(canonical_json(cluster_spec) + b"\n")
    build_template = recipe / "build.template.yaml"
    _write_yaml(
        build_template,
        {
            **yaml.safe_load(prefreeze_template.read_text()),
            "transforms": ["transforms/lm.yaml"],
            "splits": "splits-reviewed.yaml",
        },
    )
    prior_rows = [json.loads(line) for line in prior_family.read_text().splitlines()]
    protected = next(row for row in prior_rows if row["split"] == "test")
    profile = recipe / "profile.json"
    profile.write_bytes(
        canonical_json(
            {
                "release_id": prior_release.name,
                "release_manifest_sha256": sha256_file(prior_release / "manifest.json"),
                "documents_sha256": sha256_file(prior_release / "documents.jsonl"),
                "loss_slices": [protected],
            }
        )
    )
    suite = recipe / "suite.json"
    suite.write_bytes(
        canonical_json(
            {
                "release_id": prior_release.name,
                "family_inventory_sha256": sha256_file(prior_family),
                "content_sha256": "b" * 64,
                "items": [
                    {
                        "split": "test",
                        "parent_document_ids": [protected["document_id"]],
                        "parent_family": protected["family_id"],
                    }
                ],
                "chunks": [{"document_id": protected["document_id"]}],
            }
        )
    )
    acceptance_template = recipe / "release-acceptance.template.json"
    acceptance_template.write_bytes(
        canonical_json(
            {
                "format": "sparselab-release-review-v1",
                "decision": "ACCEPTED",
                "reviewer": "Fixture independent reviewer",
                "reviewed_on": "2026-10-09",
                "release_path": "${RELEASE_PATH}",
                "release_id": "${RELEASE_ID}",
                "release_manifest_sha256": "${RELEASE_MANIFEST_SHA256}",
                "family_inventory_path": "${FAMILY_INVENTORY_PATH}",
                "family_inventory_sha256": "${FAMILY_INVENTORY_SHA256}",
                "protected_lineage_path": "${PROTECTED_LINEAGE_PATH}",
                "protected_lineage_sha256": "${PROTECTED_LINEAGE_SHA256}",
            }
        )
        + b"\n"
    )
    floors = recipe / "token-floors.yaml"
    _write_yaml(
        floors,
        {"min_unique_train_tokens_by_domain": {name: 1 for name in strata.values()}},
    )
    mixture_template = recipe / "mixture.template.yaml"
    _write_yaml(
        mixture_template,
        {
            "schema_version": 1,
            "release_path": "${RELEASE_PATH}",
            "tokenizer_config": str(tokenizer_config),
            "family_inventory": "${FAMILY_INVENTORY_PATH}",
            "source_strata": strata,
            "target_quotas": {
                "general_prose": 40,
                "incident_response_docs": 20,
                "explanatory_prose": 40,
            },
            "seed": 17,
            "max_exposures": 2,
            "tokenizer_origin_release_id": prior_release.name,
        },
    )
    base_config = load_config(Path("configs/runtime_smoke_cpu.yaml")).model_dump(
        mode="json"
    )
    base_config["name"] = "fixture-prepared-inputs-only"
    base_config["model"]["vocab_size"] = 300
    base_config["tokenizer"]["path"] = str(tokenizer_path)
    base_config["dataset"] = {
        "source": "local_token_mixture",
        "revision": "${RELEASE_ID}",
        "cache_dir": "${CACHE_DIR}",
        "train_path": "${TRAIN_PATH}",
        "validation_path": "${VALIDATION_PATH}",
        "train_max_documents": "${TRAIN_DOCUMENTS}",
        "validation_max_documents": "${VALIDATION_DOCUMENTS}",
        "train_max_tokens": 100,
        "validation_max_tokens": "${VALIDATION_TOKENS}",
        "license": "${LICENSE}",
        "mixture_declaration_path": "${MIXTURE_DECLARATION_PATH}",
        "mixture_output_path": "${MIXTURE_OUTPUT_PATH}",
    }
    base_config["training"].update(seq_len=16, max_steps=7, max_tokens=100)
    base_config["checkpoint"]["every_steps"] = 1
    base_config["evaluation"]["every_steps"] = 1
    base_config["logging"]["root_dir"] = "${RUNS_DIR}"
    run_template = recipe / "prepared-run.template.yaml"
    _write_yaml(run_template, base_config)
    envelope = recipe / "resource-envelope.yaml"
    _write_yaml(
        envelope,
        {
            "resource_envelope_version": 1,
            "max_rss_bytes": 4 * 1024**3,
            "min_disk_bytes": 0,
            "min_inodes": 0,
            "max_workers": 2,
            "spill_to_disk": True,
        },
    )
    paths = _attempt(root, recipe / "acquire.yaml")
    denied_live = _phase(
        paths,
        "forbidden-live-acquisition",
        "corpus",
        "acquire",
        str(recipe / "acquire.yaml"),
    )
    assert denied_live.returncode != 0
    assert "verified offline reuse only" in denied_live.stderr
    _run(
        paths,
        "verify-snapshots",
        "corpus",
        "acquire",
        str(recipe / "acquire.yaml"),
        "--offline",
    )
    _run(
        paths,
        "draft-admission",
        "corpus",
        "admission-draft",
        str(recipe / "acquire.yaml"),
        "--template",
        str(template),
        "--policy-document",
        str(policy),
        "--output",
        str(recipe / "draft.json"),
    )
    draft = json.loads((recipe / "draft.json").read_text())
    assert all(
        sum(row["decision"] == "quarantine" for row in item["files"]) == 1
        for item in draft["sources"]
    )
    _render(
        paths, "reviewed-admission", recipe / "draft.json", recipe / "admission.json"
    )
    admission = recipe / "admission.json"
    _render(
        paths,
        "admission-review",
        admission_review_template,
        admission.with_name(admission.name + ".review.json"),
        {
            "DRAFT_PATH": str(recipe / "draft.json"),
            "DRAFT_SHA256": sha256_file(recipe / "draft.json"),
            "ADMISSION_SHA256": sha256_file(admission),
        },
    )
    _render(
        paths,
        "reviewed-release-declaration",
        release_template,
        recipe / "release-reviewed.yaml",
        {"ADMISSION_SHA256": sha256_file(admission)},
    )
    prefreeze = recipe / "pre-freeze.yaml"
    _render(paths, "pre-freeze-declaration", prefreeze_template, prefreeze)
    _bind(paths, "pre_freeze_project", prefreeze)
    inventory = recipe / "inventory.jsonl"
    _run(
        paths,
        "split-inventory",
        "corpus",
        "split-inventory",
        str(prefreeze),
        "--output",
        str(inventory),
        "--json",
    )
    clusters = recipe / "clusters.json"
    _render(
        paths,
        "reviewed-family-decisions",
        cluster_template,
        clusters,
        {"INVENTORY_SHA256": sha256_file(inventory)},
    )
    frozen = recipe / "splits-reviewed.yaml"
    _run(
        paths,
        "family-freeze",
        "corpus",
        "freeze-splits",
        str(prefreeze),
        "--inventory",
        str(inventory),
        "--clusters",
        str(clusters),
        "--output",
        str(frozen),
        "--json",
    )
    _render(paths, "final-build-declaration", build_template, recipe / "build.yaml")
    _bind(paths, "build_project", recipe / "build.yaml")
    _run(
        paths,
        "offline-build",
        "corpus",
        "build",
        str(recipe / "build.yaml"),
        "--offline",
    )
    builds = list((root / "corpora" / "fixture-candidate" / "builds").iterdir())
    assert len(builds) == 1
    _run(paths, "freeze-release", "corpus", "freeze", str(builds[0]))
    releases = list((root / "corpora" / "fixture-candidate" / "releases").iterdir())
    assert len(releases) == 1
    release = releases[0]
    assert verify_release(release)["release_id"] == release.name
    _run(paths, "release-audit", "corpus", "audit", str(release))
    _run(paths, "near-duplicate-audit", "corpus", "near-duplicates", str(release))
    family = recipe / "family.jsonl"
    _run(
        paths,
        "final-family",
        "corpus",
        "finalize-family-inventory",
        str(release),
        "--splits",
        str(frozen),
        "--output",
        str(family),
        "--json",
    )
    lineage = recipe / "lineage.json"
    _run(
        paths,
        "protected-lineage",
        "corpus",
        "audit-protected-lineage",
        "--prior-release",
        str(prior_release),
        "--candidate-release",
        str(release),
        "--prior-inventory",
        str(prior_family),
        "--candidate-inventory",
        str(family),
        "--profile",
        str(profile),
        "--suite",
        str(suite),
        "--output",
        str(lineage),
        "--json",
    )
    assert json.loads(lineage.read_text())["status"] == "PASS"
    denied = _phase(
        paths,
        "supply-before-acceptance",
        "corpus",
        "measure-tokens",
        str(release),
        "--tokenizer",
        str(recipe / "missing-tokenizer.json"),
        "--tokenizer-origin-release",
        str(prior_release),
        "--family-inventory",
        str(family),
        "--policy",
        str(recipe / "missing-policy.yaml"),
        "--output",
        str(recipe / "forbidden-supply.json"),
        "--batch-source-bytes",
        "1024",
        "--json",
    )
    assert denied.returncode != 0 and "not been accepted" in denied.stderr
    review = recipe / "release-acceptance.json"
    _render(
        paths,
        "release-acceptance",
        acceptance_template,
        review,
        {
            "RELEASE_PATH": str(release),
            "RELEASE_ID": release.name,
            "RELEASE_MANIFEST_SHA256": sha256_file(release / "manifest.json"),
            "FAMILY_INVENTORY_PATH": str(family),
            "FAMILY_INVENTORY_SHA256": sha256_file(family),
            "PROTECTED_LINEAGE_PATH": str(lineage),
            "PROTECTED_LINEAGE_SHA256": sha256_file(lineage),
        },
    )
    _bind(paths, "release_acceptance", review)
    measured = recipe / "measured-supply.json"
    _run(
        paths,
        "measure-accepted-supply",
        "corpus",
        "measure-tokens",
        str(release),
        "--tokenizer",
        str(tokenizer_path),
        "--tokenizer-origin-release",
        str(prior_release),
        "--family-inventory",
        str(family),
        "--policy",
        str(floors),
        "--output",
        str(measured),
        "--batch-source-bytes",
        "1024",
        "--json",
    )
    mixture = recipe / "mixture.yaml"
    _render(
        paths,
        "mixture-declaration",
        mixture_template,
        mixture,
        {"RELEASE_PATH": str(release), "FAMILY_INVENTORY_PATH": str(family)},
    )
    mixture_output = recipe / "mixture"
    _run(
        paths,
        "materialize-mixture",
        "corpus",
        "materialize-mixture",
        str(mixture),
        "--output",
        str(mixture_output),
        "--json",
    )
    _run(
        paths,
        "cold-verify-mixture",
        "corpus",
        "verify-mixture",
        str(mixture),
        "--output",
        str(mixture_output),
        "--json",
    )
    mixture_receipt = json.loads((mixture_output / "receipt.json").read_text())
    assert sum(mixture_receipt["actual_target_tokens"].values()) == 100
    validation_documents, validation_bytes = _split_stats(
        release / "lm/validation.jsonl", "lm"
    )
    run_config = recipe / "prepared-run.yaml"
    _render(
        paths,
        "prepared-run-declaration",
        run_template,
        run_config,
        {
            "RELEASE_ID": release.name,
            "CACHE_DIR": str(recipe / "cache"),
            "TRAIN_PATH": str(mixture_output / "train.tokens.jsonl"),
            "VALIDATION_PATH": str(release / "lm/validation.jsonl"),
            "TRAIN_DOCUMENTS": str(mixture_receipt["records"]),
            "VALIDATION_DOCUMENTS": str(validation_documents),
            "VALIDATION_TOKENS": str(validation_bytes + validation_documents + 1),
            "LICENSE": _licenses(release),
            "MIXTURE_DECLARATION_PATH": str(mixture),
            "MIXTURE_OUTPUT_PATH": str(mixture_output),
            "RUNS_DIR": str(recipe / "runs"),
        },
    )
    substituted_config = recipe / "substituted-prepared-run.yaml"
    substituted = yaml.safe_load(run_config.read_text())
    substituted["dataset"]["revision"] = prior_release.name
    _write_yaml(substituted_config, substituted)
    rejected_bundle = _phase(
        paths,
        "reject-substituted-prepared-release",
        "data",
        "prepared-inputs",
        "publish",
        str(substituted_config),
        "--output",
        str(recipe / "forbidden-prepared-bundle"),
        "--resource-envelope",
        str(envelope),
        "--tokenizer-batch-source-bytes",
        "1024",
    )
    assert rejected_bundle.returncode != 0
    assert "prepared config differs from accepted release" in rejected_bundle.stderr
    assert not (recipe / "forbidden-prepared-bundle").exists()
    bundle = recipe / "prepared-bundle"
    _run(
        paths,
        "publish-prepared-bundle",
        "data",
        "prepared-inputs",
        "publish",
        str(run_config),
        "--output",
        str(bundle),
        "--resource-envelope",
        str(envelope),
        "--tokenizer-batch-source-bytes",
        "1024",
    )
    _run(
        paths,
        "cold-verify-prepared-bundle",
        "data",
        "prepared-inputs",
        "verify",
        str(run_config),
        str(bundle),
    )
    status = AttemptBudget(paths["ledger"]).status()
    assert status["charged_updates"] == 0
    assert all(row["actual_updates"] == 0 for row in status["reservations"])
    assert status["resolved_artifacts"][-1]["kind"] == "release_acceptance"


def test_freeze_rejects_missing_admission_and_mismatched_normalizer_or_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "work"
    root.mkdir()
    recipe, strata = _sources(root, "fixture-rejections")
    acquisition = load_project(recipe / "acquire.yaml")
    wrong_inventory = recipe / "acquisition-inventory.jsonl"
    write_split_inventory(acquisition, root, wrong_inventory)
    wrong_clusters = recipe / "acquisition-clusters.json"
    _clusters(wrong_clusters, wrong_inventory, strata)
    with pytest.raises(ValueError, match="reviewed record admission"):
        freeze_splits(
            acquisition,
            root,
            wrong_inventory,
            wrong_clusters,
            recipe / "forbidden.yaml",
        )
    template, policy = _policy(recipe, strata)
    admission = recipe / "admission.json"
    draft_admission_manifest(acquisition, root, template, policy, admission)
    _reviewed_release(recipe, admission)
    prefreeze = load_project(recipe / "pre-freeze.yaml")
    inventory = recipe / "inventory.jsonl"
    write_split_inventory(prefreeze, root, inventory)
    clusters = recipe / "clusters.json"
    _clusters(clusters, inventory, strata)
    release_file = recipe / "release-reviewed.yaml"
    release_data = yaml.safe_load(release_file.read_text())
    release_data["normalizer"] = "normalizer-structure-v2"
    _write_yaml(release_file, release_data)
    with pytest.raises(ValueError, match="inventory differs"):
        freeze_splits(
            load_project(recipe / "pre-freeze.yaml"),
            root,
            inventory,
            clusters,
            recipe / "wrong-normalizer.yaml",
        )
    release_data["normalizer"] = "normalizer-structure-v3"
    _write_yaml(release_file, release_data)
    rows = inventory.read_bytes().splitlines()
    rows[0] = rows[0].replace(b'"books"', b'"other"', 1)
    inventory.write_bytes(b"\n".join(rows) + b"\n")
    with pytest.raises(ValueError, match="invalid reviewed family-cluster"):
        freeze_splits(
            load_project(recipe / "pre-freeze.yaml"),
            root,
            inventory,
            clusters,
            recipe / "wrong-inventory.yaml",
        )


def test_real_lineage_audit_blocks_protected_family_moved_to_train(
    tmp_path: Path,
) -> None:
    root = tmp_path / "work"
    root.mkdir()
    prior_release, prior_family, strata = _prior(root)
    prior_rows = [json.loads(line) for line in prior_family.read_text().splitlines()]
    protected = next(row for row in prior_rows if row["split"] == "test")
    recipe, _ = _sources(root, "fixture-leakage")
    template, policy = _policy(recipe, strata)
    admission = recipe / "admission.json"
    draft_admission_manifest(
        load_project(recipe / "acquire.yaml"), root, template, policy, admission
    )
    _reviewed_release(recipe, admission)
    project = load_project(recipe / "pre-freeze.yaml")
    inventory = recipe / "inventory.jsonl"
    write_split_inventory(project, root, inventory)
    selected: Path | None = None
    for index in range(10):
        clusters = recipe / f"clusters-{index}.json"
        _clusters(clusters, inventory, strata, seed=f"different-{index}")
        frozen = recipe / f"splits-{index}.yaml"
        freeze_splits(project, root, inventory, clusters, frozen)
        assignments = yaml.safe_load(frozen.read_text())["assignments"]
        if assignments[protected["document_id"]] == "train":
            selected = frozen
            break
    assert selected is not None
    _write_yaml(
        recipe / "build.yaml",
        {
            **yaml.safe_load((recipe / "pre-freeze.yaml").read_text()),
            "transforms": ["transforms/lm.yaml"],
            "splits": selected.name,
        },
    )
    release = freeze(
        build(load_project(recipe / "build.yaml"), root, offline=True), root
    )
    family = recipe / "family.jsonl"
    finalize_family_inventory(release, selected, family)
    profile = recipe / "profile.json"
    profile.write_bytes(
        canonical_json(
            {
                "release_id": prior_release.name,
                "release_manifest_sha256": sha256_file(prior_release / "manifest.json"),
                "documents_sha256": sha256_file(prior_release / "documents.jsonl"),
                "loss_slices": [protected],
            }
        )
    )
    suite = recipe / "suite.json"
    suite.write_bytes(
        canonical_json(
            {
                "release_id": prior_release.name,
                "family_inventory_sha256": sha256_file(prior_family),
                "content_sha256": "c" * 64,
                "items": [
                    {
                        "split": "test",
                        "parent_document_ids": [protected["document_id"]],
                        "parent_family": protected["family_id"],
                    }
                ],
                "chunks": [{"document_id": protected["document_id"]}],
            }
        )
    )
    output = recipe / "blocked-lineage.json"
    result = audit_protected_lineage(
        prior_release, release, prior_family, family, profile, suite, output
    )
    assert result["status"] == "BLOCKED"
    assert result["new_train_family_collisions"]
    assert output.exists()
