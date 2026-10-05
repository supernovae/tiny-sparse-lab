"""Typed Campaign orchestration over verified Corpus Forge and Experiment APIs."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any

from sparselab.campaign.plan import load_campaign, operational_path, safe_path
from sparselab.campaign.state import CampaignStore, digest
from sparselab.runtime_profile import RuntimeProfile
from sparselab.workdir import storage_checks, warn_storage_checks


def _verified_operation(function: Callable[..., Any]) -> Callable[..., Any]:
    """Retain only full-verifier proofs within this one Campaign command."""

    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        from sparselab.corpus.release import _verification_operation

        engine = args[0]
        previous = getattr(engine, "_verification_options", None)
        if previous is None:
            from sparselab.verification_proofs import verification_options

            engine._verification_options = verification_options(engine.work_dir)
        try:
            with _verification_operation():
                return function(*args, **kwargs)
        finally:
            if previous is None:
                del engine._verification_options

    return guarded


class CampaignEngine:
    def __init__(
        self,
        source: Path,
        work_dir: Path,
        after_commit: Callable[[str, dict], None] | None = None,
        *,
        runtime_profile: RuntimeProfile | None = None,
        observer: Any | None = None,
    ) -> None:
        self.source = Path(source).resolve()
        self.plan = load_campaign(self.source)
        self.store = CampaignStore(self.plan, Path(work_dir))
        self.work_dir = Path(work_dir).resolve()
        self.after_commit = after_commit
        self.runtime_profile = runtime_profile
        self.observer = observer
        self.stages = {stage.id: stage for stage in self.plan.stages}

    def _verification(self) -> dict[str, Any]:
        from sparselab.verification_proofs import verification_options

        return getattr(self, "_verification_options", None) or verification_options(
            self.work_dir
        )

    def _phase(self, name: str, *, host_kind: str | None = None) -> Any:
        return (
            self.observer.phase(name, host_kind=host_kind)
            if self.observer is not None
            else nullcontext()
        )

    def _runtime_binding(
        self,
        rows: dict[str, dict],
        runtime_stage: str,
        lock: Any,
        cell: Any,
        controller: Any,
    ) -> dict | None:
        """Revalidate the accepted cell and source only before new execution."""
        from sparselab.experiments.binding import open_runtime_binding

        accepted = self._upstream(rows, runtime_stage)
        entry = (
            accepted.get("availability", {}).get("runtime_bindings", {}).get(cell.id)
        )
        declaration = self.stages[runtime_stage]
        if entry is None:
            if (
                declaration.profile_id is not None
                or declaration.worker is not None
                or cell.config.runtime.engine != "pytorch"
                or cell.config.runtime.backend != "cpu"
            ):
                raise ValueError("RUNTIME_REQUIRED: accepted cell binding absent")
            return None
        binding = open_runtime_binding(
            Path(entry["path"]), lock, cell, controller=controller
        )
        if binding["binding_sha256"] != entry["binding_sha256"]:
            raise ValueError("accepted runtime binding digest changed")
        if binding["source_kind"] == "profile":
            if (
                self.runtime_profile is None
                or self.runtime_profile.model_dump(mode="json") != binding["descriptor"]
            ):
                raise ValueError(
                    "RUNTIME_REQUIRED: accepted profile must be supplied unchanged"
                )
        elif declaration.worker != binding["descriptor"]["name"]:
            raise ValueError("accepted worker differs from runtime declaration")
        return binding

    def _path(self, reference: str) -> Path:
        return safe_path(self.source.parent, reference)

    def _operational_path(self, reference: str) -> Path:
        return operational_path(self.source.parent, reference)

    @staticmethod
    def _identity(kind: str, identifier: str, sha256: str) -> list[dict[str, str]]:
        return [{"kind": kind, "identifier": identifier, "sha256": sha256}]

    @staticmethod
    def _result(
        state: str = "COMPLETE", outcome: str = "READY_FOR_NEXT_STAGE", **kwargs: Any
    ) -> dict:
        return {
            "state": state,
            "outcome": outcome,
            "reason": kwargs.pop("reason", "verified"),
            "deficits": [],
            "outputs": [],
            "availability": {},
            **kwargs,
        }

    def _provenance_closure(self, provenance: dict) -> list[dict[str, str | None]]:
        """Compare scientific closure bytes, not branch, HEAD or availability paths."""
        entries = provenance.get("declarations") or []
        if entries and provenance.get("status") != "UNKNOWN":
            return sorted(
                (
                    {"path": entry["path"], "sha256": entry["sha256"]}
                    for entry in entries
                ),
                key=lambda entry: entry["path"],
            )
        # An explicit UNKNOWN override still needs a byte-bound closure when
        # declarations live outside a Git repository.
        from sparselab.recovery.provenance import declaration_paths, repository_root
        from sparselab.training.manifest import sha256_file

        root = repository_root(self.source) or self.source.parent

        return sorted(
            (
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
                for path in declaration_paths(self.source, "campaign")
            ),
            key=lambda entry: entry["path"],
        )

    @staticmethod
    def _require_declaration_files(provenance: dict) -> None:
        closure = provenance["closure"]
        missing = [item["path"] for item in closure if item.get("sha256") is None]
        if not closure or missing:
            raise ValueError(
                "MISSING_DECLARATION_INPUT: "
                + (
                    ", ".join(sorted(missing))
                    if missing
                    else "empty scientific closure"
                )
            )

    def _allow_recovery_publication(self, prior: dict, current: dict) -> bool:
        """Accept only committed, verified output inventory additions after a stage."""
        import hashlib
        import subprocess

        import yaml

        from sparselab.campaign.state import read_canonical
        from sparselab.experiments.plan import (
            _reject_constant,
            _unique_json_pairs,
            _UniqueLoader,
        )
        from sparselab.recovery.engine import inspect_manifest
        from sparselab.recovery.evidence import export_evidence
        from sparselab.recovery.manifest import RecoveryManifest, load_manifest
        from sparselab.recovery.provenance import _git, repository_root

        if self.plan.recovery is None or current.get("status") != "CLEAN_AND_COMMITTED":
            return False
        root = repository_root(self.source)
        commit = prior.get("source_commit")
        if root is None or not isinstance(commit, str):
            return False
        manifest_path = self._path(self.plan.recovery)
        name = manifest_path.relative_to(root).as_posix()
        old = {item["path"]: item["sha256"] for item in prior["closure"]}
        new = {item["path"]: item["sha256"] for item in current["closure"]}
        if name not in old or name not in new or old[name] == new[name]:
            return False
        try:
            old_bytes = _git(root, "show", f"{commit}:{name}")
            if hashlib.sha256(old_bytes).hexdigest() != old[name]:
                return False  # An uncommitted prior recipe has no trusted old intent.
            text = old_bytes.decode("utf-8")
            payload = (
                json.loads(
                    text,
                    object_pairs_hook=_unique_json_pairs,
                    parse_constant=_reject_constant,
                )
                if manifest_path.suffix == ".json"
                else yaml.load(text, Loader=_UniqueLoader)
            )
            previous = RecoveryManifest.model_validate(payload)
            updated = load_manifest(manifest_path)
            if previous.model_dump(exclude={"steps", "evidence"}) != updated.model_dump(
                exclude={"steps", "evidence"}
            ):
                return False
            if updated.evidence[: len(previous.evidence)] != previous.evidence:
                return False
            expected_updates: dict[str, str] = {}
            if len(previous.steps) != len(updated.steps):
                return False
            for before, after in zip(previous.steps, updated.steps, strict=True):
                original = before.model_dump()
                revised = after.model_dump()
                for field in tuple(original):
                    if not field.startswith("expected_"):
                        continue
                    old_value, new_value = original[field], revised[field]
                    if old_value is None and isinstance(new_value, str):
                        expected_updates[before.id] = new_value
                        original[field] = new_value
                if original != revised:
                    return False
            appended = updated.evidence[len(previous.evidence) :]
            if not appended and not expected_updates:
                return False
            additions = {
                self._path(self.plan.recovery)
                .parent.joinpath(reference)
                .relative_to(root)
                .as_posix()
                for reference in appended
            }
            if additions & old.keys() or set(new) != set(old) | additions:
                return False
            if any(old[path] != new[path] for path in old if path != name):
                return False
            for reference in appended:
                from sparselab.recovery.provenance import declaration_reference

                evidence_path = declaration_reference(manifest_path, reference)
                record = read_canonical(evidence_path)
                if record.get("format") != "scientific-evidence-reference-v1":
                    return False
                export_evidence(
                    record["kind"],
                    Path(record["external_location"]),
                    evidence_path,
                    source_commit=record["source_commit"],
                    declaration_hashes=record["declaration_hashes"],
                )
            if expected_updates:
                rows = {
                    item["id"]: item
                    for item in inspect_manifest(manifest_path, self.work_dir)["steps"]
                }
                if any(
                    rows[step]["classification"] != "PRESENT"
                    or rows[step]["actual_sha256"] != expected
                    for step, expected in expected_updates.items()
                ):
                    return False
            return True
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            UnicodeError,
            yaml.YAMLError,
            subprocess.CalledProcessError,
        ):
            return False

    def _assert_declaration_identity(self, state: dict, provenance: dict) -> None:
        current = provenance["closure"]
        for row in state["stages"]:
            if "stage_input_sha256" not in row and not row.get("receipt"):
                continue
            prior = row.get("declaration_provenance")
            if not isinstance(prior, dict) or "closure" not in prior:
                raise ValueError(
                    f"DECLARATION_IDENTITY_UNKNOWN: stage {row['id']} "
                    "has no committed closure binding"
                )
            if prior["closure"] != current and not self._allow_recovery_publication(
                prior, provenance
            ):
                raise ValueError(
                    f"DECLARATION_IDENTITY_CHANGED: stage {row['id']} "
                    "was committed against different scientific declaration bytes"
                )

    def _rows(self, state: dict) -> dict[str, dict]:
        return {row["id"]: row for row in state["stages"]}

    def _upstream(self, rows: dict[str, dict], name: str) -> dict:
        row = rows[name]
        if row["state"] != "COMPLETE" or not row.get("outputs"):
            raise ValueError(f"upstream {name} has no completed immutable output")
        self._verify(self.stages[name], row, rows)
        return row

    def _release(self, rows: dict[str, dict], name: str) -> tuple[Path, str]:
        row = self._upstream(rows, name)
        return Path(row["availability"]["path"]), row["outputs"][0]["sha256"]

    def _lock(self, rows: dict[str, dict], name: str) -> Any:
        from sparselab.experiments.lock import open_lock

        row = self._upstream(rows, name)
        lock = open_lock(Path(row["availability"]["path"]), **self._verification())
        if lock.plan_sha256 != row["outputs"][0]["sha256"]:
            raise ValueError("experiment lock identity changed")
        return lock

    def _binding(self, stage: Any, rows: dict[str, dict]) -> tuple[str, list[dict]]:
        identities = [
            {
                "id": name,
                "stage_input_sha256": rows[name]["stage_input_sha256"],
                "outputs": rows[name]["outputs"],
            }
            for name in sorted(stage.bind)
        ]
        binding = digest(
            "campaign-approval-binding-v1",
            {
                "declaration_sha256": self.store.declaration_sha,
                "gate": stage.id,
                "gate_declaration": stage.model_dump(mode="json"),
                "identities": identities,
            },
        )
        return binding, identities

    def _verify(self, stage: Any, row: dict, rows: dict[str, dict]) -> None:
        """Reopen domain artifacts; no completed-invalid stage is ever dispatched again."""
        if row.get("stage_input_sha256") != self.store.input_sha(stage, rows):
            raise ValueError(f"stage input changed: {stage.id}")
        kind = stage.kind
        output = row["outputs"][0] if row.get("outputs") else None
        path = Path(row.get("availability", {}).get("path", ""))
        if kind in {"artifact_reference", "tokenizer_reference"}:
            from sparselab.experiments.artifacts import verify_artifact

            verified = verify_artifact(
                stage.artifact, self.source, **self._verification()
            )
            if output != {
                key: verified[key] for key in ("kind", "identifier", "sha256")
            }:
                raise ValueError(f"external artifact result changed: {stage.id}")
        elif kind == "corpus_release":
            from sparselab.corpus.release import verify_release

            manifest = verify_release(path, expected_id=output["sha256"])
            if manifest["release_id"] != output["identifier"]:
                raise ValueError("corpus release identity changed")
        elif kind == "experiment_plan":
            from sparselab.experiments.lock import open_lock

            lock = open_lock(path, **self._verification())
            if lock.plan_sha256 != output["sha256"] or lock.id != output["identifier"]:
                raise ValueError("experiment plan identity changed")
            self._check_lock(stage, rows, lock)
        elif kind == "experiment_collect":
            from sparselab.experiments.evidence import read_evidence

            lock = self._lock(rows, stage.plan)
            evidence = read_evidence(
                lock,
                Path(row["availability"]["workspace"]),
                path,
                **self._verification(),
            )
            if evidence["index_sha256"] != output["sha256"]:
                raise ValueError("evidence index changed")
            cell = self._check_collection(stage, evidence, rows)
            if row.get("measurements") != self._selected_collection_checkpoint(cell):
                raise ValueError("collected checkpoint binding changed")
        elif kind == "experiment_run":
            from sparselab.evaluation.evidence import experiment_evidence
            from sparselab.workers.controller import Controller

            lock = self._lock(rows, stage.plan)
            controller = Controller(
                Path(row["availability"]["workspace"]) / "controller", read_only=True
            )
            matches = self._attempts(controller, lock, stage, rows)
            if len(matches) != 1 or matches[0]["run_id"] != output["identifier"]:
                raise ValueError("completed run is missing or ambiguous")
            if (
                matches[0]["status"] != "COMPLETE"
                or matches[0]["ingestion_status"] != "COMPLETE"
            ):
                raise ValueError("completed run lost ingestion")
            evidence = experiment_evidence(path, **self._verification())
            if (
                evidence["run_id"] != output["identifier"]
                or digest("campaign-run-v1", self._run_identity(matches[0]))
                != output["sha256"]
            ):
                raise ValueError("completed run receipt identity changed")
        elif kind == "approval":
            binding, identities = self._binding(stage, rows)
            receipt = self.store.read_approval(stage, binding)
            if (
                receipt is None
                or receipt["decision"] != "approve"
                or receipt["identities"] != identities
            ):
                raise ValueError("approval receipt missing or changed")
            if output["sha256"] != binding:
                raise ValueError("approval binding changed")
        elif kind == "evaluation":
            from sparselab.evaluation.suite import verify_evaluation_index
            from sparselab.training.manifest import sha256_file

            index = verify_evaluation_index(path, **self._verification())
            collect = self.stages[stage.collect]
            collected = self._upstream(rows, stage.collect)["measurements"]
            run = self._upstream(rows, collect.run)
            if (
                index["index_sha256"] != output["sha256"]
                or index["run_id"] != run["outputs"][0]["identifier"]
                or index["suite_sha256"] != sha256_file(self._path(stage.suite))
                or index["checkpoint_sha256"] != collected["sha256"]
                or index["checkpoint"] != f"checkpoints/{collected['generation']}"
            ):
                raise ValueError(
                    "evaluation suite or collected checkpoint binding changed"
                )
        elif kind == "model_readiness":
            from sparselab.evaluation.readiness import verify_readiness_result
            from sparselab.training.manifest import sha256_file

            result = verify_readiness_result(path)
            evaluation = self._upstream(rows, stage.evaluation)
            expected_review = (
                str(self._operational_path(stage.review))
                if stage.review is not None
                else None
            )
            if (
                result["result_sha256"] != output["sha256"]
                or result["index_sha256"] != evaluation["outputs"][0]["sha256"]
                or result["policy_sha256"] != sha256_file(self._path(stage.policy))
                or result["review"] != expected_review
            ):
                raise ValueError("model readiness result binding changed")
        elif kind == "corpus_readiness" and stage.measurement_receipt is not None:
            from sparselab.campaign.policy import measure_readiness

            release, _ = self._release(rows, stage.corpus)
            tokenizer = Path(
                self._upstream(rows, stage.tokenizer)["availability"]["path"]
            )
            measured = measure_readiness(
                release,
                stage.policy,
                tokenizer=tokenizer,
                measurement_receipt=self._operational_path(stage.measurement_receipt),
                measurement_sha256=stage.measurement_sha256,
                _scratch=self.store.root / "scratch" / "corpus-readiness",
            )
            if measured["state"] != "COMPLETE" or measured["measurements"] != row.get(
                "measurements"
            ):
                raise ValueError("completed corpus readiness measurement changed")
            expected = digest(f"campaign-{kind}-v1", self._science(row))
            if output is None or output["sha256"] != expected:
                raise ValueError("corpus readiness result identity changed")
        elif kind in {
            "corpus_readiness",
            "token_measurement",
            "runtime_acceptance",
        }:
            expected = digest(f"campaign-{kind}-v1", self._science(row))
            if output is None or output["sha256"] != expected:
                raise ValueError(f"{kind} result identity changed")

    @staticmethod
    def _science(row: dict) -> dict:
        return {
            key: row[key]
            for key in ("stage_input_sha256", "measurements")
            if key in row
        }

    def _check_lock(self, stage: Any, rows: dict[str, dict], lock: Any) -> None:
        tokenizer = self._upstream(rows, stage.tokenizer)["outputs"][0]
        prepared = self._upstream(rows, stage.prepared)["outputs"][0]
        from sparselab.evaluation.suite import load_suite
        from sparselab.training.manifest import sha256_file

        suites = {
            self._path(item.suite)
            for item in self.plan.stages
            if item.kind == "evaluation" and self.stages[item.collect].plan == stage.id
        }
        if suites:
            if len(suites) != 1:
                raise ValueError(
                    "one locked plan cannot bind conflicting evaluation suites"
                )
            suite_path = next(iter(suites))
            suite = load_suite(suite_path)
            if lock.evaluation_suite != {
                "id": suite.id,
                "sha256": sha256_file(suite_path),
            }:
                raise ValueError("Campaign suite differs from locked evaluation suite")
        declared = {
            item.cell
            for item in self.plan.stages
            if item.kind == "experiment_run" and item.plan == stage.id
        }
        if not declared:
            if len(lock.cells) != 1:
                raise ValueError(
                    "ambiguous locked plan without an explicitly selected run cell"
                )
            declared = {lock.cells[0].id}
        for selected in sorted(declared):
            cells = [item for item in lock.cells if item.id == selected]
            if len(cells) != 1:
                raise ValueError(f"declared run cell {selected} is absent or ambiguous")
            cell = cells[0]
            for kind, expected in (
                ("tokenizer", tokenizer),
                ("prepared_data", prepared),
            ):
                actual = [
                    item for item in cell.artifacts.values() if item.get("kind") == kind
                ]
                if len(actual) != 1 or any(
                    actual[0].get(key) != expected[key]
                    for key in ("identifier", "sha256")
                ):
                    raise ValueError(
                        f"locked cell {cell.id} differs from declared {kind}"
                    )
            synthetic = cell.config.dataset.source == "synthetic"
            if synthetic and stage.corpus is not None:
                raise ValueError(
                    "synthetic cell cannot claim the campaign corpus as model input"
                )
            if not synthetic:
                if stage.corpus is None:
                    raise ValueError(
                        "non-synthetic locked cell requires a corpus release"
                    )
                release = self._upstream(rows, stage.corpus)["outputs"][0]
                actual = [
                    item
                    for item in cell.artifacts.values()
                    if item.get("kind") == "corpus_release"
                ]
                if len(actual) != 1 or actual[0]["sha256"] != release["sha256"]:
                    raise ValueError(
                        "locked corpus variant differs from upstream campaign release"
                    )

    def _check_collection(
        self, stage: Any, evidence: dict, rows: dict[str, dict]
    ) -> dict:
        run_stage = self.stages[stage.run]
        cell = evidence["cells"][run_stage.cell]
        if (
            cell["status"] != "complete"
            or cell["selected_run_id"] != rows[stage.run]["outputs"][0]["identifier"]
        ):
            raise ValueError("selected locked cell has no complete ingested evidence")
        return cell

    @staticmethod
    def _selected_collection_checkpoint(cell: dict) -> dict:
        """Pin the unique latest verified generation in the immutable collection."""
        complete = [
            attempt
            for attempt in cell["attempts"]
            if attempt["status"] == "complete"
            and attempt["run_id"] == cell["selected_run_id"]
        ]
        if len(complete) != 1 or not complete[0].get("checkpoints"):
            raise ValueError("collection has no unique verified checkpoint generation")
        checkpoints = complete[0]["checkpoints"]
        latest_step = max(checkpoint["step"] for checkpoint in checkpoints)
        latest = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint["step"] == latest_step
        ]
        if len(latest) != 1:
            raise ValueError("ambiguous latest generation in collected evidence")
        selected = latest[0]
        return {
            "generation": selected["path"],
            "sha256": selected["digest"],
            "step": selected["step"],
        }

    def _availability(self, stage: Any) -> str | None:
        if stage.kind in {"artifact_reference", "tokenizer_reference"}:
            path = self._operational_path(stage.artifact.path)
        elif stage.kind == "corpus_release":
            path = self._path(stage.project)
        elif stage.kind == "experiment_plan":
            path = self._path(stage.source)
            if stage.mode == "reference":
                lock_path = self._operational_path(stage.lock)
                if not lock_path.exists():
                    return f"missing declared local lock: {lock_path}"
        elif stage.kind == "evaluation":
            path = self._path(stage.suite)
        elif stage.kind == "model_readiness":
            path = self._path(stage.policy)
        else:
            return None
        if not path.exists():
            return f"missing declared local input: {path}"
        if stage.kind in {"artifact_reference", "tokenizer_reference"}:
            from sparselab.experiments.artifacts import verify_artifact

            verify_artifact(stage.artifact, self.source, **self._verification())
        elif stage.kind == "corpus_release":
            from sparselab.corpus.project import ProjectConfig, load_project
            from sparselab.experiments.plan import read_document

            config = ProjectConfig.model_validate(read_document(path))
            for reference in (
                *config.sources,
                *config.transforms,
                config.splits,
                config.release,
            ):
                member = safe_path(path.parent, reference)
                if not member.is_file():
                    return f"missing declared corpus recipe: {member}"
            project = load_project(path)
            for source in project.sources:
                if source.kind not in {"local", "deterministic_generator"}:
                    raise ValueError(
                        "campaign corpus sources must be local or deterministic_generator"
                    )
                if source.kind == "local":
                    for entry in source.acquisition.files:
                        member = safe_path(project.root, entry.path)
                        if not member.is_file():
                            return f"missing declared corpus source: {member}"
        elif stage.kind == "experiment_plan" and stage.mode == "reference":
            from sparselab.experiments.lock import open_lock

            open_lock(self._operational_path(stage.lock), **self._verification())
        elif stage.kind == "evaluation":
            from sparselab.evaluation.suite import load_suite

            load_suite(path)
        elif stage.kind == "model_readiness":
            from sparselab.evaluation.readiness import load_policy

            load_policy(path)
        return None

    def _project(self, state: dict) -> dict:
        rows = self._rows(state)
        projected: list[dict] = []
        next_action: dict = {
            "action": "none",
            "reason": "all stages complete",
            "identities": {},
        }
        for stage in self.plan.ordered_stages():
            row = dict(rows[stage.id])
            prior = [rows[name] for name in stage.requires]
            terminal = row["state"] in {
                "COMPLETE",
                "FAILED",
                "INCONCLUSIVE",
                "BLOCKED",
                "AWAITING_APPROVAL",
                "RUNNING",
                "INTERRUPTED",
            }
            if not terminal or (row["state"] == "BLOCKED" and not row.get("receipt")):
                blocked = [item["id"] for item in prior if item["state"] != "COMPLETE"]
                if blocked:
                    row.update(
                        state="BLOCKED",
                        outcome="DO_NOT_ADVANCE",
                        reason="prerequisite did not complete",
                        blocked_by=blocked,
                    )
                elif all(item["state"] == "COMPLETE" for item in prior):
                    row.pop("blocked_by", None)
                    try:
                        missing = self._availability(stage)
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        row.update(
                            state="FAILED", outcome="DO_NOT_ADVANCE", reason=str(error)
                        )
                    else:
                        if missing:
                            row.update(
                                state="BLOCKED",
                                outcome="DO_NOT_ADVANCE",
                                reason=missing,
                            )
                        else:
                            row.update(
                                state="READY",
                                outcome="READY_FOR_NEXT_STAGE",
                                reason="prerequisites complete",
                            )
            if stage.kind == "approval" and row["state"] == "AWAITING_APPROVAL":
                binding, identities = self._binding(stage, rows)
                receipt = self.store.read_approval(stage, binding)
                if receipt is not None:
                    if receipt["identities"] != identities:
                        raise ValueError("committed approval input identities changed")
                    row.update(
                        state="READY",
                        reason="committed approval awaits gate reconciliation",
                    )
            if row["state"] == "COMPLETE":
                self._verify(stage, row, rows)
            if next_action["action"] == "none" and row["state"] in {
                "READY",
                "RUNNING",
                "INTERRUPTED",
                "AWAITING_APPROVAL",
                "BLOCKED",
                "FAILED",
                "INCONCLUSIVE",
            }:
                action = (
                    "approve"
                    if row["state"] == "AWAITING_APPROVAL"
                    else "resume"
                    if row["state"] in {"RUNNING", "INTERRUPTED"}
                    else "apply"
                    if row["state"] == "READY"
                    else "wait"
                )
                if row["state"] == "INTERRUPTED" and stage.kind == "experiment_run":
                    action = "wait"
                next_action = {
                    "action": action,
                    "stage": stage.id,
                    "reason": row.get("reason", row["state"]),
                    "identities": {"outputs": row.get("outputs", [])},
                }
            projected.append(row)
            rows[stage.id] = row
        return {
            "id": self.plan.id,
            "declaration_sha256": self.store.declaration_sha,
            "stages": projected,
            "next_action": next_action,
        }

    def _observed_checkpoints(
        self, state: dict, manifest_path: Path, recipe_rows: list[dict]
    ) -> list[dict]:
        """Separate verified Campaign generations from logical recovery run names."""
        from sparselab.recovery.manifest import load_manifest
        from sparselab.training.checkpoints import CheckpointManager
        from sparselab.training.mlx_checkpoints import strict_json

        recipe = load_manifest(manifest_path)
        persisted = self._rows(state)
        observed: list[dict] = []
        for collect in self.plan.stages:
            if collect.kind != "experiment_collect":
                continue
            row = persisted[collect.id]
            selected = row.get("measurements")
            if row["state"] != "COMPLETE" or not isinstance(selected, dict):
                continue
            if not {"generation", "sha256"} <= selected.keys():
                continue
            run = self.stages[collect.run]
            run_row = persisted[run.id]
            run_path = Path(run_row.get("availability", {}).get("path", ""))
            generation = selected["generation"]
            checkpoint = run_path / "checkpoints" / generation
            expected = selected["sha256"]
            plan_stage = self.stages[collect.plan]
            plan_row = persisted[collect.plan]
            plan_sha = (
                plan_row["outputs"][0]["sha256"] if plan_row.get("outputs") else None
            )
            entry = {
                "id": f"observed_checkpoint:{collect.id}",
                "kind": "checkpoint",
                "expected_sha256": expected,
                "actual_sha256": None,
                "path": str(checkpoint),
                "run_id": (run_row.get("outputs") or [{}])[0].get("identifier"),
                "generation": generation,
                "plan_sha256": plan_sha,
            }
            try:
                self._verify(run, run_row, persisted)
                self._verify(collect, row, persisted)
                report = CheckpointManager(run_path).verify(checkpoint)
                if not report.valid:
                    raise ValueError(f"collected checkpoint invalid: {report.errors}")
                actual = strict_json(checkpoint / "manifest.json")["sha256"]
                if actual != expected:
                    raise ValueError("collected checkpoint digest differs from receipt")
            except (OSError, ValueError, KeyError, TypeError) as error:
                entry.update(
                    classification=(
                        "ERROR" if checkpoint.exists() else "MISSING_NONRECONSTRUCTABLE"
                    ),
                    reason=f"historical Campaign collection cannot be reverified: {error}",
                )
            else:
                entry.update(
                    classification="PRESENT",
                    actual_sha256=actual,
                    reason="freshly verified Campaign collection and exact generation",
                )
                compatible = [
                    step
                    for step in recipe.steps
                    if step.kind == "experiment_lock"
                    and step.expected_plan_sha256 == plan_sha
                    and self._path(plan_stage.source)
                    == safe_path(manifest_path.parent, step.plan)
                ]
                mapped = [
                    step
                    for step in recipe.steps
                    if step.kind == "checkpoint"
                    and step.expected_sha256 == actual
                    and step.cell
                    in {
                        generation,
                        generation.split("_gen_", 1)[0],
                    }
                ]
                if len(compatible) == len(mapped) == 1:
                    for declared in recipe_rows:
                        if declared["id"] == mapped[0].id:
                            declared.update(
                                classification="PRESENT",
                                actual_sha256=actual,
                                action=None,
                                reason="pinned digest and plan match verified Campaign generation",
                            )
                            break
            observed.append(entry)
        return observed

    def _recoverability(self, state: dict) -> list[dict]:
        """Fresh availability, independent of historical stage outcomes."""
        if self.plan.recovery is not None:
            from sparselab.recovery.engine import inspect_manifest

            manifest = self._path(self.plan.recovery)
            if not manifest.is_file():
                return [
                    {
                        "id": "declaration",
                        "kind": "recovery",
                        "classification": "MISSING_EXTERNAL",
                        "reason": f"missing recovery manifest: {manifest}",
                    }
                ]
            try:
                from sparselab.training.manifest import sha256_file

                manifest_sha = sha256_file(manifest)
                rows = [
                    {
                        "id": "declaration",
                        "kind": "recovery",
                        "classification": "PRESENT",
                        "expected_sha256": manifest_sha,
                        "actual_sha256": manifest_sha,
                        "reason": "recovery declaration available",
                    },
                    *inspect_manifest(manifest, self.work_dir)["steps"],
                ]
                rows.extend(self._observed_checkpoints(state, manifest, rows))
                return rows
            except (OSError, ValueError, KeyError, TypeError) as error:
                return [
                    {
                        "id": "declaration",
                        "kind": "recovery",
                        "classification": "ERROR",
                        "reason_code": "RECOVERY_INSPECTION_FAILED",
                        "reason": str(error),
                    }
                ]
        rows = self._rows(state)
        result = []
        for stage in self.plan.ordered_stages():
            row = rows[stage.id]
            output = row.get("outputs", [])
            if not output:
                classification = (
                    "NOT_CREATED"
                    if row["state"] in {"NOT_STARTED", "READY", "BLOCKED"}
                    and not row.get("stage_input_sha256")
                    else "MISSING_EXTERNAL"
                )
                reason = (
                    "no committed output"
                    if classification == "NOT_CREATED"
                    else "output availability unknown"
                )
            else:
                try:
                    self._verify(stage, row, rows)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    path = Path(row.get("availability", {}).get("path", ""))
                    if stage.kind == "experiment_run":
                        classification = "MISSING_NONRECONSTRUCTABLE"
                    elif path.exists() and row.get("availability", {}).get("path"):
                        classification = "ERROR"
                    else:
                        classification = "MISSING_EXTERNAL"
                    reason = str(error)
                else:
                    classification, reason = "PRESENT", "verified"
            result.append(
                {
                    "id": stage.id,
                    "kind": stage.kind,
                    "classification": classification,
                    "expected_sha256": output[0]["sha256"] if output else None,
                    "reason": reason,
                }
            )
        return result

    def reconstruct(
        self,
        *,
        allow_network: bool = False,
        allow_uncommitted_declaration: bool = False,
        evidence_output: Path | None = None,
    ) -> dict:
        if self.plan.recovery is None:
            raise ValueError("campaign reconstruct requires a linked recovery manifest")
        from sparselab.recovery.engine import reconstruct_manifest
        from sparselab.recovery.provenance import declaration_preflight

        provenance = declaration_preflight(
            self.source, "campaign", allow_uncommitted=allow_uncommitted_declaration
        )
        checks = warn_storage_checks(storage_checks(self.work_dir))
        result = reconstruct_manifest(
            self._path(self.plan.recovery),
            self.work_dir,
            allow_network=allow_network,
            allow_uncommitted_declaration=allow_uncommitted_declaration,
            evidence_output=evidence_output,
        )
        return {
            **self.inspect("status"),
            "reconstruction": result,
            "declaration_provenance": provenance,
            "storage_checks": checks,
        }

    def _submitted_run(self, stage: Any, rows: dict[str, dict]) -> bool:
        from sparselab.workers.controller import Controller

        lock = self._lock(rows, stage.plan)
        prior = rows[stage.id].get("measurements", {})
        submitted = (
            isinstance(prior, dict)
            and {"experiment_id", "attempt_id", "run_id"} <= prior.keys()
        )
        workspace = Path(rows[stage.plan]["availability"]["workspace"])
        controller_dir = workspace / "controller"
        if not controller_dir.exists():
            if submitted:
                raise ValueError(
                    "LOST_SUBMISSION: controller state vanished after run submission"
                )
            return False
        controller = Controller(controller_dir, read_only=True)
        matches = self._attempts(controller, lock, stage, rows)
        if len(matches) > 1:
            raise ValueError("ambiguous controller attempts for locked cell")
        if submitted and (
            not matches
            or self._run_identity(matches[0])
            != {key: prior[key] for key in ("experiment_id", "attempt_id", "run_id")}
        ):
            raise ValueError(
                "LOST_SUBMISSION: previously submitted attempt is missing or replaced"
            )
        return bool(matches)

    @_verified_operation
    def inspect(self, command: str = "plan") -> dict:
        if command not in {"plan", "next", "status", "explain"}:
            raise ValueError(f"unknown inspection command {command}")
        persisted = self.store.read()
        if command == "status" and persisted is not None:
            rows = self._rows(persisted)
            for stage in self.plan.stages:
                if (
                    stage.kind == "corpus_readiness"
                    and stage.measurement_receipt is not None
                ):
                    row = rows[stage.id]
                    if row["state"] == "COMPLETE":
                        self._verify(stage, row, rows)
            return {
                **{
                    key: persisted[key]
                    for key in ("id", "declaration_sha256", "stages", "next_action")
                },
                "recoverability": self._recoverability(persisted),
            }
        state = persisted or self.store.initial()
        if command != "status":
            state = self.store.reconcile(state)
        projected = self._project(state)
        if command == "status":
            projected["recoverability"] = self._recoverability(state)
        return projected

    def _record(self, state: dict, row: dict) -> None:
        state["stages"] = [
            row if item["id"] == row["id"] else item for item in state["stages"]
        ]
        projection = self._project(state)
        projected = self._rows(projection)
        state["stages"] = [projected[stage.id] for stage in self.plan.stages]
        state["next_action"] = projection["next_action"]
        self.store.save(state)

    @_verified_operation
    def apply(
        self,
        resume: bool = False,
        max_wait_seconds: float = 120,
        *,
        execute_runs: bool = False,
        allow_uncommitted_declaration: bool = False,
    ) -> dict:
        from sparselab.recovery.provenance import declaration_preflight

        if max_wait_seconds < 0 or not math.isfinite(max_wait_seconds):
            raise ValueError("max wait seconds must be finite and nonnegative")
        provenance = declaration_preflight(
            self.source, "campaign", allow_uncommitted=allow_uncommitted_declaration
        )
        provenance = {
            **provenance,
            "allow_uncommitted_declaration": allow_uncommitted_declaration,
            "closure": self._provenance_closure(provenance),
        }
        self._require_declaration_files(provenance)
        checks = warn_storage_checks(storage_checks(self.work_dir))
        with self.store.locked():
            previous = self.store.read()
            if resume and previous is None:
                raise ValueError("cannot resume a campaign without durable state")
            if previous is not None:
                self._assert_declaration_identity(previous, provenance)
            state = self.store.reconcile(previous or self.store.initial())
            self._assert_declaration_identity(state, provenance)
            if previous is None or state != previous:
                self._record(state, state["stages"][0])
            while True:
                projection = self._project(state)
                action = projection["next_action"]
                if action["action"] not in {"apply", "resume"}:
                    return {
                        **projection,
                        "declaration_provenance": provenance,
                        "storage_checks": checks,
                    }
                stage = self.stages[action["stage"]]
                rows = self._rows(state)
                row = rows[stage.id]
                submitted = False
                dispatch_error: OSError | ValueError | KeyError | TypeError | None = (
                    None
                )
                if stage.kind == "experiment_run":
                    try:
                        submitted = self._submitted_run(stage, rows)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        dispatch_error = error
                if (
                    stage.kind == "experiment_run"
                    and not execute_runs
                    and not submitted
                    and dispatch_error is None
                ):
                    return {
                        **projection,
                        "next_action": {
                            "action": "execute_run",
                            "stage": stage.id,
                            "reason": "new model execution requires --execute-runs",
                            "identities": {"outputs": row.get("outputs", [])},
                        },
                        "declaration_provenance": provenance,
                        "storage_checks": checks,
                    }
                if row["state"] == "RUNNING" and stage.kind != "experiment_run":
                    reason = "uncommitted interrupted attempt"
                    attempts = list(row["attempts"])
                    if attempts:
                        attempts[-1] = {
                            **attempts[-1],
                            "state": "INTERRUPTED",
                            "reason": reason,
                        }
                    row = {
                        **row,
                        "state": "INTERRUPTED",
                        "reason": reason,
                        "attempts": attempts,
                        "observations": [
                            *row["observations"],
                            {"state": "INTERRUPTED", "reason": reason},
                        ],
                    }
                    self._record(state, row)
                row = {
                    **row,
                    "state": "RUNNING",
                    "stage_input_sha256": self.store.input_sha(stage, rows),
                    "attempts": [
                        *row.get("attempts", []),
                        {"state": "RUNNING", "reason": "dispatch"},
                    ],
                    "declaration_provenance": provenance,
                    "storage_checks": checks,
                }
                self._record(state, row)
                rows = self._rows(state)
                try:
                    if dispatch_error is not None:
                        raise dispatch_error
                    result = self.dispatch(
                        stage, rows, max_wait_seconds, execute_runs=execute_runs
                    )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    result = self._result("FAILED", "DO_NOT_ADVANCE", reason=str(error))
                from sparselab.recovery.provenance import (
                    declaration_paths,
                    git_provenance,
                )

                current = git_provenance(declaration_paths(self.source, "campaign"))
                current["closure"] = self._provenance_closure(current)
                self._assert_declaration_identity(state, current)
                row.update(result)
                row["attempts"][-1] = {"state": row["state"], "reason": row["reason"]}
                if row["state"] in {
                    "COMPLETE",
                    "BLOCKED",
                    "FAILED",
                    "INCONCLUSIVE",
                    "INTERRUPTED",
                }:
                    row = self.store.commit(stage, row)
                self._record(state, row)
                if self.after_commit is not None and row["state"] == "COMPLETE":
                    self.after_commit(stage.id, state)
                if row["state"] != "COMPLETE":
                    return {
                        **self._project(state),
                        "declaration_provenance": provenance,
                        "storage_checks": checks,
                    }

    @_verified_operation
    def approve(
        self, gate: str, decision: str = "approve", note: str | None = None
    ) -> dict:
        if decision not in {"approve", "reject"}:
            raise ValueError("approval decision must be approve or reject")
        with self.store.locked():
            state = self.store.read()
            if state is None:
                raise ValueError("campaign has no pending gate")
            from sparselab.recovery.provenance import declaration_paths, git_provenance

            provenance = git_provenance(declaration_paths(self.source, "campaign"))
            provenance["closure"] = self._provenance_closure(provenance)
            self._require_declaration_files(provenance)
            self._assert_declaration_identity(state, provenance)
            state = self.store.reconcile(state)
            self._assert_declaration_identity(state, provenance)
            self._project(state)
            stage = self.stages.get(gate)
            if stage is None or stage.kind != "approval":
                raise ValueError(f"unknown approval gate {gate}")
            rows = self._rows(state)
            row = rows[gate]
            if row["state"] not in {"AWAITING_APPROVAL", "COMPLETE", "BLOCKED"}:
                raise ValueError(f"gate {gate} is not awaiting approval")
            for name in stage.bind:
                self._upstream(rows, name)
            binding, identities = self._binding(stage, rows)
            receipt = self.store.approval(stage, binding, decision, note, identities)
            outcome = (
                "READY_FOR_NEXT_STAGE" if decision == "approve" else "DO_NOT_ADVANCE"
            )
            result = self._result(
                "COMPLETE" if decision == "approve" else "BLOCKED",
                outcome,
                reason=f"approval {decision}",
                outputs=self._identity("approval", gate, binding),
                bindings=identities,
                approval=receipt,
            )
            row.update(result)
            row["stage_input_sha256"] = self.store.input_sha(stage, rows)
            row = self.store.commit(stage, row)
            self._record(state, row)
            return self._project(state)

    def _attempts(
        self, controller: Any, lock: Any, stage: Any, rows: dict
    ) -> list[dict]:
        from sparselab.experiments.binding import inspect_runtime_binding
        from sparselab.experiments.cli import locked_cell_request
        from sparselab.training.manifest import config_sha256
        from sparselab.workers.models import ExperimentSpec

        cell = stage.cell

        matches = [
            row
            for row in controller.list_experiments()
            if (row["spec"].get("plan") or {}).get("plan_sha256") == lock.plan_sha256
            and (row["spec"].get("plan") or {}).get("cell_id") == cell
        ]
        selected = next(item for item in lock.cells if item.id == cell)
        entry = (
            self._upstream(rows, stage.runtime)
            .get("availability", {})
            .get("runtime_bindings", {})
            .get(cell)
        )
        runtime_digest = None
        if entry is not None:
            binding = inspect_runtime_binding(Path(entry["path"]), lock, selected)
            runtime_digest = binding["binding_sha256"]
            if runtime_digest != entry["binding_sha256"]:
                raise ValueError("accepted runtime digest differs from receipt")
        elif (
            selected.config.runtime.backend != "cpu"
            or selected.config.runtime.engine != "pytorch"
        ):
            raise ValueError(
                "RUNTIME_REQUIRED: attempted accelerator cell has no accepted binding"
            )
        for row in matches:
            spec = ExperimentSpec.model_validate(row["spec"])
            expected = locked_cell_request(
                lock,
                selected,
                spec.bound_worker,
                controller.root.parent,
                controller,
                read_only=True,
                runtime_binding_sha256=runtime_digest,
            )
            if (
                spec.config.model_dump(mode="json")
                != selected.config.model_dump(mode="json")
                or spec.config_sha256 != selected.config_sha256
                or config_sha256(spec.config.model_dump(mode="json"))
                != selected.config_sha256
                or spec.plan.model_dump(mode="json") != expected["plan"]
            ):
                raise ValueError(f"controller attempt does not bind locked cell {cell}")
            continuation = spec.continuation
            parent = (
                expected.get("resume")
                or expected.get("extend_budget")
                or expected.get("promote")
            )
            expected_kind = (
                "PROMOTED"
                if "promote" in expected
                else "RESUMED"
                if parent
                else "FRESH"
            )
            if (
                continuation.kind != expected_kind
                or continuation.allow_runtime_drift
                or continuation.budget_extension != ("extend_budget" in expected)
                or continuation.checkpoint_sha256
                != expected["plan"]["parent_checkpoint_sha256"]
            ):
                raise ValueError(
                    f"controller continuation does not bind locked cell {cell}"
                )
            if parent:
                manifest = json.loads((parent / "manifest.json").read_text())
                if (
                    continuation.parent_run_id != manifest["run_id"]
                    or continuation.artifact_identity.sha256
                    != continuation.checkpoint_sha256
                ):
                    raise ValueError(
                        f"controller parent does not bind locked cell {cell}"
                    )
        return matches

    @staticmethod
    def _run_identity(row: dict) -> dict:
        return {key: row[key] for key in ("experiment_id", "attempt_id", "run_id")}

    @_verified_operation
    def dispatch(
        self,
        stage: Any,
        rows: dict[str, dict],
        max_wait_seconds: float,
        *,
        execute_runs: bool = False,
    ) -> dict:
        """Run one declared adapter; return science identities separately from availability."""
        kind = stage.kind
        input_sha = rows[stage.id]["stage_input_sha256"]
        if kind in {"artifact_reference", "tokenizer_reference"}:
            from sparselab.experiments.artifacts import verify_artifact

            path = self._operational_path(stage.artifact.path)
            if not path.exists():
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason=f"missing declared artifact: {path}",
                )
            verified = verify_artifact(
                stage.artifact, self.source, **self._verification()
            )
            return self._result(
                outputs=[
                    {key: verified[key] for key in ("kind", "identifier", "sha256")}
                ],
                availability={"path": verified["path"]},
            )
        if kind == "corpus_release":
            from sparselab.corpus.acquisition import acquire
            from sparselab.corpus.pipeline import build
            from sparselab.corpus.project import load_project
            from sparselab.corpus.release import freeze, verify_release

            path = self._path(stage.project)
            if not path.exists():
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason=f"missing corpus project: {path}",
                )
            project = load_project(path)
            if any(
                source.kind not in {"local", "deterministic_generator"}
                for source in project.sources
            ):
                raise ValueError(
                    "campaign corpus sources must be local or deterministic_generator"
                )
            acquire(project, self.store.root, offline=False)
            release = freeze(
                build(project, self.store.root, offline=True), self.store.root
            )
            identity = verify_release(release)["release_id"]
            return self._result(
                outputs=self._identity("corpus_release", identity, identity),
                availability={"path": str(release)},
            )
        if kind == "corpus_readiness":
            from sparselab.campaign.policy import measure_readiness

            release, _ = self._release(rows, stage.corpus)
            tokenizer = (
                Path(self._upstream(rows, stage.tokenizer)["availability"]["path"])
                if stage.tokenizer
                else None
            )
            measured = measure_readiness(
                release,
                stage.policy,
                tokenizer=tokenizer,
                measurement_receipt=(
                    self._operational_path(stage.measurement_receipt)
                    if stage.measurement_receipt is not None
                    else None
                ),
                measurement_sha256=stage.measurement_sha256,
                _scratch=self.store.root / "scratch" / "corpus-readiness",
            )
            result = self._result(
                measured["state"],
                measured["outcome"],
                reason="corpus policy measured",
                deficits=measured["deficits"],
                measurements=measured["measurements"],
            )
            if result["state"] == "COMPLETE":
                result["outputs"] = self._identity(
                    kind,
                    stage.id,
                    digest(
                        f"campaign-{kind}-v1",
                        {
                            "stage_input_sha256": input_sha,
                            "measurements": result["measurements"],
                        },
                    ),
                )
            return result
        if kind == "token_measurement":
            from sparselab.corpus.release import describe

            release, release_id = self._release(rows, stage.corpus)
            tokenizer = self._upstream(rows, stage.tokenizer)
            measured = describe(
                release, tokenizer=Path(tokenizer["availability"]["path"])
            )
            facts = {
                "release_id": release_id,
                "tokenizer_sha256": tokenizer["outputs"][0]["sha256"],
                "token_totals": measured["token_totals"],
                "views": measured["measurement"]["views"],
            }
            return self._result(
                measurements=facts,
                outputs=self._identity(
                    kind,
                    stage.id,
                    digest(
                        f"campaign-{kind}-v1",
                        {"stage_input_sha256": input_sha, "measurements": facts},
                    ),
                ),
            )
        if kind == "experiment_plan":
            from sparselab.experiments.lock import open_lock, publish_lock, resolve_plan
            from sparselab.experiments.plan import load_plan
            from sparselab.experiments.prepare import prepare_plan
            from sparselab.training.manifest import canonical_json

            source = self._path(stage.source)
            if not source.exists():
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason=f"missing experiment plan: {source}",
                )
            plan = load_plan(source)
            from sparselab.experiments.plan import base_run_config

            config = base_run_config(plan, source)
            configured_checks = [
                check
                for kind, target in (
                    ("cache", config.dataset.cache_dir),
                    ("output", config.logging.root_dir),
                )
                if Path(target).is_absolute()
                for check in storage_checks(target, kind=kind)
            ]
            warn_storage_checks(configured_checks)
            workspace = self.store.root / "experiments" / f"{stage.id}-{plan.id}"
            if stage.mode == "reference":
                path = self._operational_path(stage.lock)
                if not path.exists():
                    return self._result(
                        "BLOCKED",
                        "DO_NOT_ADVANCE",
                        reason=f"missing experiment lock: {path}",
                    )
                lock = open_lock(path, **self._verification())
                self._check_lock(stage, rows, lock)
            else:
                prepared = None
                if plan.corpus_variants:
                    record_path = workspace / "preparation.json"
                    if record_path.exists():
                        raw = record_path.read_bytes()
                        prepared = json.loads(raw)
                        if (
                            raw != canonical_json(prepared) + b"\n"
                            or prepared.get("format") != "experiment-preparation-v1"
                            or prepared.get("id") != plan.id
                        ):
                            raise ValueError(
                                "existing immutable preparation receipt is invalid"
                            )
                    else:
                        prepared = prepare_plan(
                            plan, source, workspace, **self._verification()
                        )
                        with record_path.open("xb") as handle:
                            handle.write(canonical_json(prepared) + b"\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                lock = resolve_plan(
                    plan, source, prepared=prepared, **self._verification()
                )
                self._check_lock(stage, rows, lock)
                path = publish_lock(lock, workspace)
                lock = open_lock(path, **self._verification())
            return self._result(
                outputs=self._identity("experiment_plan", lock.id, lock.plan_sha256),
                availability={"path": str(path), "workspace": str(workspace)},
                storage_checks=[*rows[stage.id]["storage_checks"], *configured_checks],
            )
        if kind == "runtime_acceptance":
            from sparselab.workspace_preflight import training_storage_checks

            lock = self._lock(rows, stage.plan)
            selected = {
                item.cell
                for item in self.plan.stages
                if item.kind == "experiment_run"
                and item.plan == stage.plan
                and item.runtime == stage.id
            }
            if not selected:
                if len(lock.cells) != 1:
                    raise ValueError("runtime has no unambiguous declared cell")
                selected = {lock.cells[0].id}
            cells = [cell for cell in lock.cells if cell.id in selected]
            if len(cells) != len(selected):
                raise ValueError("runtime references an absent locked cell")
            profile = self.runtime_profile if stage.profile_id is not None else None
            if stage.worker is not None and self.runtime_profile is not None:
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason="named worker runtime rejects a simultaneous profile",
                )
            if stage.profile_id is not None and (
                profile is None or profile.id != stage.profile_id
            ):
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason="RUNTIME_REQUIRED: matching profile_id",
                )
            if (
                profile is None
                and stage.worker is None
                and any(
                    cell.config.runtime.backend != "cpu"
                    or cell.config.runtime.engine != "pytorch"
                    for cell in cells
                )
            ):
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason="RUNTIME_REQUIRED: explicit profile or worker",
                )
            workspace = Path(rows[stage.plan]["availability"]["workspace"])
            checks = [
                asdict(check)
                for cell in cells
                for check in training_storage_checks(
                    cell.config, work_dir=workspace, run_dir=workspace / "controller"
                )
            ]
            if any(check["status"] != "adequate" for check in checks):
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason="insufficient training storage headroom",
                    availability={"storage_checks": checks},
                )
            from sparselab.experiments.binding import bind_runtime, open_runtime_binding
            from sparselab.workers.controller import Controller

            bindings = {}
            if profile is not None or stage.worker is not None:
                controller = Controller(workspace / "controller")
                for cell in cells:
                    try:
                        path = bind_runtime(
                            lock,
                            cell,
                            workspace,
                            profile=profile,
                            worker=stage.worker,
                            controller=controller,
                        )
                        binding = open_runtime_binding(
                            path, lock, cell, controller=controller
                        )
                    except (OSError, ValueError, RuntimeError) as error:
                        return self._result(
                            "BLOCKED", "DO_NOT_ADVANCE", reason=str(error)
                        )
                    bindings[cell.id] = {
                        "path": str(path),
                        "binding_sha256": binding["binding_sha256"],
                        "tested_runtime": binding["tested_runtime"],
                    }
            facts = {
                "headroom": "adequate",
                "cells": sorted(selected),
                "runtimes": {
                    cell.id: cell.config.runtime.model_dump(mode="json")
                    for cell in cells
                },
                "runtime_bindings": {
                    key: value["binding_sha256"] for key, value in bindings.items()
                },
            }
            return self._result(
                measurements=facts,
                availability={"storage_checks": checks, "runtime_bindings": bindings},
                outputs=self._identity(
                    kind,
                    stage.id,
                    digest(
                        f"campaign-{kind}-v1",
                        {"stage_input_sha256": input_sha, "measurements": facts},
                    ),
                ),
            )
        if kind == "approval":
            binding, identities = self._binding(stage, rows)
            approval = self.store.read_approval(stage, binding)
            if approval is None:
                return self._result(
                    "AWAITING_APPROVAL",
                    "DO_NOT_ADVANCE",
                    reason="approval required",
                    bindings=identities,
                    binding_sha256=binding,
                )
            if approval["decision"] == "reject":
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason="approval rejected",
                    bindings=identities,
                    binding_sha256=binding,
                )
            return self._result(
                outputs=self._identity("approval", stage.id, binding),
                bindings=identities,
                approval=approval,
            )
        if kind == "experiment_run":
            from sparselab.experiments.cli import locked_cell_request
            from sparselab.workers.controller import Controller
            from sparselab.workers.models import WorkerDefinition

            lock = self._lock(rows, stage.plan)
            cells = [cell for cell in lock.cells if cell.id == stage.cell]
            if len(cells) != 1:
                raise ValueError("experiment_run must select exactly one locked cell")
            cell = cells[0]
            workspace = Path(rows[stage.plan]["availability"]["workspace"])
            if not execute_runs and not self._submitted_run(stage, rows):
                raise ValueError("new model execution requires --execute-runs")
            controller = Controller(
                workspace / "controller",
                observer=self.observer,
                **self._verification(),
            )
            matches = self._attempts(controller, lock, stage, rows)
            if len(matches) > 1:
                raise ValueError("ambiguous controller attempts for locked cell")
            previous = rows[stage.id].get("measurements", {})
            if (
                isinstance(previous, dict)
                and {"experiment_id", "attempt_id", "run_id"} <= previous.keys()
                and (
                    not matches
                    or self._run_identity(matches[0])
                    != {
                        key: previous[key]
                        for key in ("experiment_id", "attempt_id", "run_id")
                    }
                )
            ):
                raise ValueError(
                    "LOST_SUBMISSION: refusing to replace prior run attempt"
                )
            if not matches:
                try:
                    binding = self._runtime_binding(
                        rows, stage.runtime, lock, cell, controller
                    )
                except (OSError, ValueError, RuntimeError) as error:
                    return self._result("BLOCKED", "DO_NOT_ADVANCE", reason=str(error))
                worker = f"campaign-{stage.id}-{lock.id}"
                runtime_digest = None
                if binding is not None:
                    runtime_digest = binding["binding_sha256"]
                    if binding["source_kind"] == "worker":
                        worker = binding["descriptor"]["name"]
                    else:
                        controller.register(
                            WorkerDefinition(
                                worker_id=worker,
                                name=worker,
                                transport="local",
                                python=Path(binding["descriptor"]["python"]),
                                root=workspace / "workers" / worker,
                                engine=cell.config.runtime.engine,
                                backend=cell.config.runtime.backend,
                                device_index=cell.config.runtime.device_index,
                            )
                        )
                else:
                    controller.register(
                        WorkerDefinition(
                            worker_id=worker,
                            name=worker,
                            transport="local",
                            python=Path(sys.executable).absolute(),
                            root=workspace / "workers" / worker,
                            engine="pytorch",
                            backend="cpu",
                            device_index=0,
                        )
                    )
                request = locked_cell_request(
                    lock,
                    cell,
                    worker,
                    workspace,
                    controller,
                    runtime_binding_sha256=runtime_digest,
                )
                request["declaration_provenance"] = rows[stage.id][
                    "declaration_provenance"
                ]
                request["storage_checks"] = [
                    *rows[stage.id]["storage_checks"],
                    *(
                        check
                        for check in rows[stage.plan].get("storage_checks", [])
                        if check not in rows[stage.id]["storage_checks"]
                    ),
                ]
                controller.submit_many([request])
            deadline = time.monotonic() + max_wait_seconds
            while True:
                matches = self._attempts(controller, lock, stage, rows)
                if len(matches) != 1:
                    raise ValueError("controller submission missing or conflicting")
                attempt = matches[0]
                ids = self._run_identity(attempt)
                availability = {
                    "path": str(controller.root / attempt["run_id"]),
                    "workspace": str(workspace),
                }
                if (
                    attempt["status"] == "COMPLETE"
                    and attempt["ingestion_status"] == "COMPLETE"
                ):
                    from sparselab.evaluation.evidence import experiment_evidence

                    with self._phase(
                        "campaign.run_evidence", host_kind="verification_bound"
                    ):
                        experiment_evidence(
                            Path(availability["path"]), **self._verification()
                        )
                    return self._result(
                        outputs=self._identity(
                            "experiment_run",
                            attempt["run_id"],
                            digest("campaign-run-v1", ids),
                        ),
                        availability=availability,
                        measurements=ids,
                    )
                if (
                    attempt["status"] in {"FAILED", "CANCELLED", "INTERRUPTED"}
                    or attempt["ingestion_status"] == "ERROR"
                ):
                    return self._result(
                        "FAILED"
                        if attempt["status"] == "FAILED"
                        or attempt["ingestion_status"] == "ERROR"
                        else "INTERRUPTED",
                        "DO_NOT_ADVANCE",
                        reason=f"controller {attempt['status']}, ingestion {attempt['ingestion_status']}",
                        availability=availability,
                        measurements=ids,
                    )
                if time.monotonic() >= deadline:
                    return self._result(
                        "RUNNING",
                        "READY_FOR_NEXT_STAGE",
                        reason="worker active; resume to reconcile",
                        availability=availability,
                        measurements=ids,
                    )
                with self._phase("campaign.controller_tick"):
                    controller.tick(deadline=deadline)
                refreshed = self._attempts(controller, lock, stage, rows)
                if len(refreshed) != 1:
                    raise ValueError("controller submission missing or conflicting")
                # Re-enter the durable gate immediately; only an unchanged attempt
                # consumes idle time, and never beyond the shared deadline.
                if refreshed[0] != attempt:
                    continue
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    with self._phase("campaign.state_idle"):
                        time.sleep(min(0.1, remaining))
        if kind == "experiment_collect":
            from sparselab.experiments.evidence import collect_evidence, read_evidence

            lock = self._lock(rows, stage.plan)
            workspace = Path(rows[stage.plan]["availability"]["workspace"])
            with self._phase(
                "campaign.checkpoint_collection", host_kind="verification_bound"
            ):
                evidence = collect_evidence(lock, workspace, **self._verification())
                verified = read_evidence(
                    lock,
                    workspace,
                    Path(evidence["index_path"]),
                    **self._verification(),
                )
            cell = self._check_collection(stage, verified, rows)
            selected = self._selected_collection_checkpoint(cell)
            return self._result(
                outputs=self._identity(kind, stage.id, verified["index_sha256"]),
                measurements=selected,
                availability={
                    "path": verified["index_path"],
                    "workspace": str(workspace),
                },
            )
        if kind == "evaluation":
            from sparselab.evaluation.suite import run_suite, verify_evaluation_index

            collect = self.stages[stage.collect]
            lock = self._lock(rows, collect.plan)
            collected = self._upstream(rows, stage.collect)["measurements"]
            run = self._upstream(rows, collect.run)
            run_path = Path(run["availability"]["path"])
            from sparselab.evaluation.inference import evaluation_config
            from sparselab.runtime_profile import rederive_authorization
            from sparselab.workers.controller import Controller

            run_id = run["outputs"][0]["identifier"]
            checkpoint = str(run_path / "checkpoints" / collected["generation"])
            config = evaluation_config(
                run_id,
                run_path.parent,
                checkpoint,
                stage.backend,
                **self._verification(),
            )
            authorization = None
            if stage.runtime is not None:
                run_stage = self.stages[collect.run]
                cell = next(cell for cell in lock.cells if cell.id == run_stage.cell)
                workspace = Path(rows[collect.plan]["availability"]["workspace"])
                try:
                    binding = self._runtime_binding(
                        rows,
                        stage.runtime,
                        lock,
                        cell,
                        Controller(workspace / "controller"),
                    )
                    if binding is None:
                        if config.runtime.backend != "cpu":
                            raise ValueError(
                                "RUNTIME_REQUIRED: evaluation binding absent"
                            )
                    else:
                        if (
                            binding["source_kind"] == "worker"
                            and binding["descriptor"]["transport"] != "local"
                        ):
                            raise ValueError(
                                "remote worker evaluation unsupported; declare backend: cpu"
                            )
                        authorization = rederive_authorization(
                            binding["runtime_authorization"], config
                        )
                except (OSError, ValueError, RuntimeError) as error:
                    return self._result("BLOCKED", "DO_NOT_ADVANCE", reason=str(error))
            elif config.runtime.backend != "cpu" or config.runtime.engine != "pytorch":
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason="RUNTIME_REQUIRED: accelerator checkpoint evaluation needs runtime",
                )
            with self._phase("campaign.evaluation"):
                index_path = run_suite(
                    self._path(stage.suite),
                    run_id,
                    checkpoint,
                    run_path.parent,
                    stage.backend,
                    authorization=authorization,
                    **self._verification(),
                )
            index = verify_evaluation_index(index_path, **self._verification())
            if (
                index["run_id"] != run["outputs"][0]["identifier"]
                or index["checkpoint_sha256"] != collected["sha256"]
                or index["checkpoint"] != f"checkpoints/{collected['generation']}"
            ):
                raise ValueError("suite index differs from collected run/checkpoint")
            facts = {
                "checkpoint_sha256": index["checkpoint_sha256"],
                "index_sha256": index["index_sha256"],
                "evaluation_runtime": index["evaluation_runtime"],
                "evaluations": [
                    {"id": item["id"], "status": item["status"]}
                    for item in index["evaluations"]
                ],
            }
            return self._result(
                measurements=facts,
                outputs=self._identity(kind, stage.id, index["index_sha256"]),
                availability={"path": str(index_path)},
            )
        if kind == "model_readiness":
            from sparselab.evaluation.readiness import (
                assess_readiness,
                verify_readiness_result,
            )

            evaluation = self._upstream(rows, stage.evaluation)
            result_path = assess_readiness(
                self._path(stage.policy),
                Path(evaluation["availability"]["path"]),
                self._operational_path(stage.review) if stage.review else None,
            )
            assessed = verify_readiness_result(result_path)
            state = assessed["state"]
            facts = {
                "checkpoint_sha256": assessed["checkpoint_sha256"],
                "index_sha256": assessed["index_sha256"],
                "completed_evaluations": assessed["completed_evaluations"],
                "missing_gates": assessed["missing_gates"],
            }
            result_state = (
                "COMPLETE"
                if state == "READY_FOR_NEXT_STAGE"
                else "BLOCKED"
                if state == "DO_NOT_ADVANCE"
                else "INCONCLUSIVE"
            )
            outcome = (
                state
                if state in {"READY_FOR_NEXT_STAGE", "DO_NOT_ADVANCE"}
                else "INCONCLUSIVE"
            )
            return self._result(
                result_state,
                outcome,
                reason=f"model policy assessed: {state}",
                measurements=facts,
                outputs=(
                    self._identity(kind, stage.id, assessed["result_sha256"])
                    if state == "READY_FOR_NEXT_STAGE"
                    else []
                ),
                availability={"path": str(result_path)},
            )
        raise ValueError(f"unsupported campaign adapter {kind}")
