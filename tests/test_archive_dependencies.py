"""Model-free archive closures; synthetic checkpoint metadata is not model evidence."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

from sparselab import archive
from sparselab.evaluation import suite
from sparselab.training.manifest import canonical_json, sha256_file


def _digest(value: dict) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")
    return path


@pytest.fixture
def graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Real suite/index verification and archive I/O, without checkpoint execution.

    Only upstream checkpoint selection and family availability are substituted.
    No model, tokenizer, tensor files, training, or corpus is constructed.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("Model-free archive fixture.\n")
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/generation"
    run_record = {
        "architecture_sha256": "a" * 64,
        "effective_config": {
            "training": {"max_steps": 1, "max_tokens": 1},
            "dataset": {},
        },
        "artifacts": [],
        "resource_decisions": [],
    }
    _write(run / "manifest.json", run_record)
    checkpoint_record = {
        "manifest_sha256": _digest(run_record),
        "architecture_sha256": "a" * 64,
        "files": [],
    }
    checkpoint_sha = _digest(checkpoint_record)
    _write(
        checkpoint / "manifest.json", {**checkpoint_record, "sha256": checkpoint_sha}
    )
    observed = {
        "engine": "torch",
        "backend": "cpu",
        "precision": "fp32",
        "device_index": 0,
    }
    runtime = {**observed, "observed": observed}
    result = _write(
        run / "evaluations/result.json",
        {
            "identity": {
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_relative_path": "checkpoints/generation",
                "runtime": observed,
            },
            "metrics": {"loss": 1.0},
        },
    )
    monkeypatch.setattr(suite, "_selected_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "sparselab.family.cli.show",
        lambda *args, **kwargs: {
            "nodes": [
                {
                    "availability": {
                        "checkpoint": "PRESENT",
                        "evaluation_index": "PRESENT",
                    }
                }
            ]
        },
    )

    def index(name: str, children: tuple[Path, ...] = ()) -> Path:
        declarations, rows = [], []
        for number, child in enumerate(children):
            supplied = suite.verify_evaluation_index(child)
            reference = _write(
                repo / f"{name}-{number}.json",
                {
                    "format": "scientific-evidence-reference-v1",
                    "kind": "evaluation_index",
                    "external_location": str(child),
                    "sha256": supplied["index_sha256"],
                },
            )
            item = {
                "id": f"branch-{number}",
                "kind": "evidence_reference",
                "role": "diagnostic",
                "source": reference.name,
            }
            declarations.append(item)
            rows.append(
                {
                    **item,
                    "status": "COMPLETED",
                    "path": str(child),
                    "sha256": supplied["index_sha256"],
                    "source_sha256": sha256_file(reference),
                    "result": {"index_sha256": supplied["index_sha256"]},
                }
            )
        if not children:
            item = {"id": "loss", "kind": "heldout_lm", "role": "gate"}
            declarations.append(item)
            rows.append(
                {
                    **item,
                    "status": "COMPLETED",
                    "path": str(result),
                    "sha256": sha256_file(result),
                    "source_sha256": None,
                    "result": {"loss": 1.0},
                }
            )
        source = repo / f"{name}.yaml"
        source.write_text(
            yaml.safe_dump(
                {"evaluation_suite_version": 1, "id": name, "evaluations": declarations}
            )
        )
        record = {
            "format": "evaluation-index-v1",
            "run": str(run),
            "run_id": run.name,
            "checkpoint": "checkpoints/generation",
            "checkpoint_sha256": checkpoint_sha,
            "suite": str(source),
            "suite_sha256": sha256_file(source),
            "evaluations": rows,
            "evaluation_runtime": runtime,
        }
        record["index_sha256"] = _digest(
            suite._index_binding(record["suite_sha256"], checkpoint_sha, rows, runtime)
        )
        record["record_sha256"] = _digest(record)
        path = _write(
            run / "evaluations" / f"suite-{record['index_sha256']}.json", record
        )
        assert suite.verify_evaluation_index(path) == record
        return path

    def recipe(top: Path) -> Path:
        identity = {"id": "fixture", "sha256": "b" * 64}
        (repo / "family.yaml").write_text(
            yaml.safe_dump(
                {
                    "family_version": 1,
                    "id": "fixture",
                    "nodes": [
                        {
                            "id": "model",
                            "parent": None,
                            "parent_checkpoint_sha256": None,
                            "corpus": identity,
                            "tokenizer": identity,
                            "plan": identity,
                            "architecture_sha256": "a" * 64,
                            "objective": "next_token",
                            "budget": {"max_steps": 1, "max_tokens": 1},
                            "checkpoint": {
                                "path": str(checkpoint),
                                "sha256": checkpoint_sha,
                            },
                            "evaluation_index": {
                                "path": str(top),
                                "sha256": suite.verify_evaluation_index(top)[
                                    "index_sha256"
                                ],
                            },
                            "readiness_result": None,
                        }
                    ],
                }
            )
        )
        _write(repo / "readiness.json", {"fixture": "No readiness result declared."})

        def commit():
            for args in (
                ("init", "-q"),
                ("add", "."),
                (
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-qm",
                    "Fixture declarations",
                ),
            ):
                subprocess.run(["git", "-C", str(repo), *args], check=True)

        commit()
        pinned = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        source = repo / "recovery.yaml"
        source.write_text(
            yaml.safe_dump(
                {
                    "recovery_version": 1,
                    "id": "fixture",
                    "source_commit": pinned,
                    "evaluation_suite": "top.yaml",
                    "readiness_policy": "readiness.json",
                    "runtime_requirement": {
                        "engine": "torch",
                        "backend": "cpu",
                        "device_index": 0,
                        "requirements": {},
                    },
                    "family": "family.yaml",
                    "steps": [
                        {
                            "id": "cache",
                            "kind": "optional_cache",
                            "role": "fixture",
                            "reason": "No runtime work.",
                        }
                    ],
                }
            )
        )
        commit()
        return source

    return index, recipe, checkpoint, result


def _repack(source: Path, destination: Path, index: dict, omitted: str) -> None:
    changed = {
        **index,
        "entries": [entry for entry in index["entries"] if entry["path"] != omitted],
    }
    with tarfile.open(source) as original, tarfile.open(destination, "w") as output:
        payloads = {"archive-index.json": canonical_json(changed) + b"\n"}
        for entry in changed["entries"]:
            member = original.extractfile(entry["path"])
            assert member is not None
            payloads[entry["path"]] = member.read()
        for name, payload in payloads.items():
            header = tarfile.TarInfo(name)
            header.size = len(payload)
            output.addfile(header, io.BytesIO(payload))


@pytest.mark.parametrize("shape", ["single", "repeated", "diamond"])
def test_portable_shared_index_branches(graph, tmp_path: Path, shape: str) -> None:
    make_index, recipe, _, _ = graph
    leaf = make_index("leaf")
    children = (leaf,) if shape == "single" else (leaf, leaf)
    if shape == "diamond":
        children = (make_index("left", (leaf,)), make_index("right", (leaf,)))
    top = make_index("top", children)
    output = tmp_path / "portable.tar"
    created = archive.create_archive(
        recipe(top), "portable", output, work_root=tmp_path / "work"
    )
    prefixes = [
        f"evidence/model/evaluation_index/evaluations/branch-{n}/supplied"
        for n in range(len(children))
    ]
    if shape == "diamond":
        prefixes = [prefix + "/evaluations/branch-0/supplied" for prefix in prefixes]
    entries = {entry["path"]: entry for entry in created["entries"]}
    suffixes = (
        "run-manifest.json",
        "suite/leaf.yaml",
        "evaluations/loss/result/result.json",
    )
    for suffix in suffixes:
        copies = [entries[f"{prefix}/{suffix}"] for prefix in prefixes]
        assert len({entry["sha256"] for entry in copies}) == 1
    assert archive.verify_archive(output)["verified_members"] == len(entries)
    # Rewriting the outer inventory cannot legitimize missing branch-local bytes.
    for number, suffix in enumerate(suffixes):
        damaged = tmp_path / f"missing-{number}.tar"
        _repack(output, damaged, created, f"{prefixes[-1]}/{suffix}")
        with pytest.raises(ValueError, match="missing portable evaluation dependency"):
            archive.verify_archive(damaged)


@pytest.mark.parametrize(
    "missing", ["result", "suite", "run-manifest", "index", "source"]
)
def test_shared_index_missing_dependency_refuses(
    graph, tmp_path: Path, missing: str
) -> None:
    make_index, recipe, checkpoint, result = graph
    leaf = make_index("leaf")
    top = make_index("top", (leaf, leaf))
    source = recipe(top)
    top_record = suite.verify_evaluation_index(top)
    record = json.loads(leaf.read_text())
    paths = {
        "result": result,
        "suite": Path(record["suite"]),
        "run-manifest": checkpoint.parent.parent / "manifest.json",
        "index": leaf,
        "source": source.parent / "top-1.json",
    }
    paths[missing].unlink()
    # Exercise the collector even when the public command rejects a deleted
    # declaration earlier at its provenance gate.
    with pytest.raises((ValueError, FileNotFoundError)):
        archive._index_dependencies(
            {}, "evidence/model/evaluation_index", top_record, checkpoint, set()
        )
    output = tmp_path / "missing.tar"
    with pytest.raises((ValueError, FileNotFoundError)):
        archive.create_archive(source, "portable", output, work_root=tmp_path / "work")
    assert not output.exists()


def test_index_dependency_checkpoint_mismatch_refuses(graph, tmp_path: Path) -> None:
    make_index, _, _, _ = graph
    leaf = make_index("leaf")
    with pytest.raises(
        ValueError, match="checkpoint differs from declared family node"
    ):
        archive._index_dependencies(
            {},
            "evidence/model/evaluation_index",
            suite.verify_evaluation_index(leaf),
            tmp_path / "another-checkpoint",
            set(),
        )


def test_index_dependency_cycle_refuses(graph, monkeypatch: pytest.MonkeyPatch) -> None:
    make_index, _, checkpoint, _ = graph
    leaf = make_index("leaf")
    top = make_index("top", (leaf,))
    record = suite.verify_evaluation_index(top)
    # Isolate this collector's cycle guard; upstream recursive cycle handling is
    # separate scope. A verifier returning an ancestor must not imply completion.
    monkeypatch.setattr(suite, "verify_evaluation_index", lambda path: record)
    with pytest.raises(ValueError, match="cyclic portable evaluation index"):
        archive._index_dependencies(
            {}, "evidence/model/evaluation_index", record, checkpoint, set()
        )
