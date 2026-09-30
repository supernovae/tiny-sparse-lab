"""Typed Campaign orchestration over verified Corpus Forge and Experiment APIs."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sparselab.campaign.plan import load_campaign, safe_path
from sparselab.campaign.state import CampaignStore, digest


class CampaignEngine:
    def __init__(
        self,
        source: Path,
        work_dir: Path,
        after_commit: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.source = Path(source).resolve()
        self.plan = load_campaign(self.source)
        self.store = CampaignStore(self.plan, Path(work_dir))
        self.after_commit = after_commit
        self.stages = {stage.id: stage for stage in self.plan.stages}

    def _path(self, reference: str) -> Path:
        return safe_path(self.source.parent, reference)

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
        lock = open_lock(Path(row["availability"]["path"]))
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

            verified = verify_artifact(stage.artifact, self.source)
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

            lock = open_lock(path)
            if lock.plan_sha256 != output["sha256"] or lock.id != output["identifier"]:
                raise ValueError("experiment plan identity changed")
            self._check_lock(stage, rows, lock)
        elif kind == "experiment_collect":
            from sparselab.experiments.evidence import read_evidence

            lock = self._lock(rows, stage.plan)
            evidence = read_evidence(lock, Path(row["availability"]["workspace"]), path)
            if evidence["index_sha256"] != output["sha256"]:
                raise ValueError("evidence index changed")
            self._check_collection(stage, evidence, rows)
        elif kind == "experiment_run":
            from sparselab.evaluation.evidence import experiment_evidence
            from sparselab.workers.controller import Controller

            lock = self._lock(rows, stage.plan)
            controller = Controller(
                Path(row["availability"]["workspace"]) / "controller", read_only=True
            )
            matches = self._attempts(controller, lock, stage.cell)
            if len(matches) != 1 or matches[0]["run_id"] != output["identifier"]:
                raise ValueError("completed run is missing or ambiguous")
            if (
                matches[0]["status"] != "COMPLETE"
                or matches[0]["ingestion_status"] != "COMPLETE"
            ):
                raise ValueError("completed run lost ingestion")
            evidence = experiment_evidence(path)
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
        elif kind in {
            "evaluation",
            "model_readiness",
            "corpus_readiness",
            "token_measurement",
            "runtime_acceptance",
        }:
            # Their immutable identity binds every upstream input and measured result.
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

    def _availability(self, stage: Any) -> str | None:
        if stage.kind in {"artifact_reference", "tokenizer_reference"}:
            path = self._path(stage.artifact.path)
        elif stage.kind == "corpus_release":
            path = self._path(stage.project)
        elif stage.kind == "experiment_plan":
            path = self._path(stage.source)
            if stage.mode == "reference" and not self._path(stage.lock).exists():
                return f"missing declared local lock: {self._path(stage.lock)}"
        else:
            return None
        if not path.exists():
            return f"missing declared local input: {path}"
        if stage.kind in {"artifact_reference", "tokenizer_reference"}:
            from sparselab.experiments.artifacts import verify_artifact

            verify_artifact(stage.artifact, self.source)
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
        elif stage.mode == "reference":
            from sparselab.experiments.lock import open_lock

            open_lock(self._path(stage.lock))
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

    def inspect(self, command: str = "plan") -> dict:
        if command not in {"plan", "next", "status", "explain"}:
            raise ValueError(f"unknown inspection command {command}")
        persisted = self.store.read()
        if command == "status" and persisted is not None:
            return {
                key: persisted[key]
                for key in ("id", "declaration_sha256", "stages", "next_action")
            }
        state = persisted or self.store.initial()
        if command != "status":
            state = self.store.reconcile(state)
        return self._project(state)

    def _record(self, state: dict, row: dict) -> None:
        state["stages"] = [
            row if item["id"] == row["id"] else item for item in state["stages"]
        ]
        projection = self._project(state)
        projected = self._rows(projection)
        state["stages"] = [projected[stage.id] for stage in self.plan.stages]
        state["next_action"] = projection["next_action"]
        self.store.save(state)

    def apply(self, resume: bool = False, max_wait_seconds: float = 120) -> dict:
        if max_wait_seconds < 0 or not math.isfinite(max_wait_seconds):
            raise ValueError("max wait seconds must be finite and nonnegative")
        with self.store.locked():
            previous = self.store.read()
            if resume and previous is None:
                raise ValueError("cannot resume a campaign without durable state")
            state = self.store.reconcile(previous or self.store.initial())
            if previous is None or state != previous:
                self._record(state, state["stages"][0])
            while True:
                projection = self._project(state)
                action = projection["next_action"]
                if action["action"] not in {"apply", "resume"}:
                    return projection
                stage = self.stages[action["stage"]]
                rows = self._rows(state)
                row = rows[stage.id]
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
                }
                self._record(state, row)
                rows = self._rows(state)
                try:
                    result = self.dispatch(stage, rows, max_wait_seconds)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    result = self._result("FAILED", "DO_NOT_ADVANCE", reason=str(error))
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
                    return self._project(state)

    def approve(
        self, gate: str, decision: str = "approve", note: str | None = None
    ) -> dict:
        if decision not in {"approve", "reject"}:
            raise ValueError("approval decision must be approve or reject")
        with self.store.locked():
            state = self.store.read()
            if state is None:
                raise ValueError("campaign has no pending gate")
            state = self.store.reconcile(state)
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

    @staticmethod
    def _attempts(controller: Any, lock: Any, cell: str) -> list[dict]:
        from sparselab.experiments.cli import locked_cell_request
        from sparselab.training.manifest import config_sha256
        from sparselab.workers.models import ExperimentSpec

        matches = [
            row
            for row in controller.list_experiments()
            if (row["spec"].get("plan") or {}).get("plan_sha256") == lock.plan_sha256
            and (row["spec"].get("plan") or {}).get("cell_id") == cell
        ]
        selected = next(item for item in lock.cells if item.id == cell)
        for row in matches:
            spec = ExperimentSpec.model_validate(row["spec"])
            expected = locked_cell_request(
                lock,
                selected,
                spec.bound_worker,
                controller.root.parent,
                controller,
                read_only=True,
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

    def dispatch(
        self, stage: Any, rows: dict[str, dict], max_wait_seconds: float
    ) -> dict:
        """Run one declared adapter; return science identities separately from availability."""
        kind = stage.kind
        input_sha = rows[stage.id]["stage_input_sha256"]
        if kind in {"artifact_reference", "tokenizer_reference"}:
            from sparselab.experiments.artifacts import verify_artifact

            path = self._path(stage.artifact.path)
            if not path.exists():
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason=f"missing declared artifact: {path}",
                )
            verified = verify_artifact(stage.artifact, self.source)
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
            measured = measure_readiness(release, stage.policy, tokenizer=tokenizer)
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
            workspace = self.store.root / "experiments" / f"{stage.id}-{plan.id}"
            if stage.mode == "reference":
                path = self._path(stage.lock)
                if not path.exists():
                    return self._result(
                        "BLOCKED",
                        "DO_NOT_ADVANCE",
                        reason=f"missing experiment lock: {path}",
                    )
                lock = open_lock(path)
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
                        prepared = prepare_plan(plan, source, workspace)
                        with record_path.open("xb") as handle:
                            handle.write(canonical_json(prepared) + b"\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                lock = resolve_plan(plan, source, prepared=prepared)
                self._check_lock(stage, rows, lock)
                path = publish_lock(lock, workspace)
                lock = open_lock(path)
            return self._result(
                outputs=self._identity("experiment_plan", lock.id, lock.plan_sha256),
                availability={"path": str(path), "workspace": str(workspace)},
            )
        if kind == "runtime_acceptance":
            from sparselab.workspace_preflight import training_storage_checks

            lock = self._lock(rows, stage.plan)
            selected = {
                item.cell
                for item in self.plan.stages
                if item.kind == "experiment_run" and item.plan == stage.plan
            }
            if not selected:
                if len(lock.cells) != 1:
                    raise ValueError("runtime has no unambiguous declared cell")
                selected = {lock.cells[0].id}
            cells = [cell for cell in lock.cells if cell.id in selected]
            if len(cells) != len(selected):
                raise ValueError("runtime references an absent locked cell")
            if any(
                cell.config.runtime.backend != "cpu"
                or cell.config.runtime.engine != "pytorch"
                for cell in cells
            ):
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason="Campaign v1 executes local CPU/PyTorch cells only",
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
            facts = {
                "headroom": "adequate",
                "engine": "pytorch",
                "backend": "cpu",
                "cells": sorted(selected),
            }
            return self._result(
                measurements=facts,
                availability={"storage_checks": checks},
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
            controller = Controller(workspace / "controller")
            matches = self._attempts(controller, lock, cell.id)
            if len(matches) > 1:
                raise ValueError("ambiguous controller attempts for locked cell")
            if not matches:
                worker = f"campaign-{stage.id}-{lock.id}"
                controller.register(
                    WorkerDefinition(
                        worker_id=worker,
                        name=worker,
                        transport="local",
                        python=Path(sys.executable).absolute(),
                        root=workspace / "workers" / worker,
                        engine="pytorch",
                        backend="cpu",
                        device_index=cell.config.runtime.device_index,
                    )
                )
                controller.submit_many(
                    [locked_cell_request(lock, cell, worker, workspace, controller)]
                )
            deadline = time.monotonic() + max_wait_seconds
            while True:
                matches = self._attempts(controller, lock, cell.id)
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

                    experiment_evidence(Path(availability["path"]))
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
                controller.tick(deadline=deadline)
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        if kind == "experiment_collect":
            from sparselab.experiments.evidence import collect_evidence, read_evidence

            lock = self._lock(rows, stage.plan)
            workspace = Path(rows[stage.plan]["availability"]["workspace"])
            evidence = collect_evidence(lock, workspace)
            verified = read_evidence(lock, workspace, Path(evidence["index_path"]))
            self._check_collection(stage, verified, rows)
            return self._result(
                outputs=self._identity(kind, stage.id, verified["index_sha256"]),
                availability={
                    "path": verified["index_path"],
                    "workspace": str(workspace),
                },
            )
        if kind == "evaluation":
            from sparselab.evaluation.evidence import experiment_evidence

            collect = self.stages[stage.collect]
            self._lock(rows, collect.plan)
            self._upstream(rows, stage.collect)
            run = self._upstream(rows, collect.run)
            observed = experiment_evidence(Path(run["availability"]["path"]))
            if observed["run_id"] != run["outputs"][0]["identifier"]:
                raise ValueError("evaluation run identity changed")
            facts = {
                "run_id": observed["run_id"],
                "evidence_level": observed["evidence_level"],
                "quality_observations": observed["quality_observations"],
                "rejected_reports": observed["rejected_reports"],
                "missing_reports": observed["missing_reports"],
            }
            if observed["evidence_level"] != "checkpointed_held_out":
                return self._result(
                    "INCONCLUSIVE",
                    "INCONCLUSIVE",
                    reason="no complete checkpoint-bound heldout evidence",
                    measurements=facts,
                )
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
        if kind == "model_readiness":
            evaluation = self._upstream(rows, stage.evaluation)["measurements"]
            if (
                stage.max_heldout_loss is None
                or evaluation["evidence_level"] != "checkpointed_held_out"
            ):
                return self._result(
                    "INCONCLUSIVE",
                    "INCONCLUSIVE",
                    reason="no declared threshold or complete heldout evidence",
                )
            observations = evaluation["quality_observations"]
            if not observations:
                return self._result(
                    "INCONCLUSIVE",
                    "INCONCLUSIVE",
                    reason="missing heldout observations",
                )
            latest = max(item["step"] for item in observations)
            matches = [item for item in observations if item["step"] == latest]
            if any(item != matches[0] for item in matches[1:]):
                raise ValueError(
                    "conflicting heldout observations at latest checkpoint step"
                )
            loss = matches[0].get("loss")
            if (
                isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(loss)
            ):
                return self._result(
                    "INCONCLUSIVE",
                    "INCONCLUSIVE",
                    reason="latest heldout loss absent or invalid",
                )
            facts = {
                "step": latest,
                "observed_loss": loss,
                "required_max_heldout_loss": stage.max_heldout_loss,
            }
            if loss > stage.max_heldout_loss:
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason="heldout loss exceeds declared threshold",
                    measurements=facts,
                )
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
        raise ValueError(f"unsupported campaign adapter {kind}")
