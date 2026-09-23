"""Controller-side scheduling for independent local or SSH worker attempts."""

from __future__ import annotations

import logging
import math
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from sparselab.config.models import RunConfig
from sparselab.training.manifest import canonical_json, config_sha256
from sparselab.workers.store import (
    ACTIVE_QUEUE_STATES,
    RECONCILABLE_QUEUE_STATES,
    ControllerStore,
)

_LOGGER = logging.getLogger(__name__)


class Controller:
    """A durable, foreground controller for whole independent experiments.

    Worker optimizer execution is never retried here.  The persisted attempt and
    agent receipt are the only authority after an uncertain RPC delivery.
    """

    def __init__(
        self, root: Path, poll_seconds: float = 2, *, transfer_timeout: float = 1800
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if not math.isfinite(transfer_timeout) or transfer_timeout <= 0:
            raise ValueError("transfer_timeout must be finite and positive")
        self.transfer_timeout = transfer_timeout
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.poll_seconds = poll_seconds
        self.store = ControllerStore(self.root)
        self._lock_path = self.root / ".controller.lock"

    @staticmethod
    def _dump(value: object) -> dict[str, object]:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return value
        raise TypeError("worker protocol object is not JSON-native")

    @staticmethod
    def _model(name: str, value: object) -> Any:
        from sparselab.workers import models

        return getattr(models, name).model_validate(value)

    @staticmethod
    def _reply_result(reply: object) -> dict[str, object]:
        result = getattr(reply, "result", reply)
        if not isinstance(result, dict):
            raise TypeError("worker reply result must be an object")
        return result

    def _rpc_result(
        self,
        worker: Any,
        op: str,
        payload: dict[str, object],
        *,
        attachments: dict[str, Path] | None = None,
    ) -> dict[str, object]:
        """Run metadata-only RPCs without leaking reply directories into CWD."""
        from sparselab.workers.transport import call_worker

        with TemporaryDirectory(prefix=".worker-reply-", dir=self.root) as directory:
            reply = call_worker(
                worker,
                op,
                payload,
                attachments=attachments,
                receive_dir=Path(directory),
            )
            return self._reply_result(reply)

    def _capability_for_definition(self, definition: Any, value: object) -> Any:
        """Bind remote discovery to its registered execution identity.

        SSH invokes the fixed agent remotely, whose local discovery naturally
        reports ``local``/``host=None``.  Those transport fields describe the
        endpoint hop rather than a different worker identity, so only they are
        normalized to the controller registration after every execution-critical
        field has matched exactly.
        """
        capability = self._model("WorkerCapabilities", value)
        for field in (
            "worker_id",
            "name",
            "root",
            "python",
            "engine",
            "backend",
            "device_index",
        ):
            if getattr(capability, field) != getattr(definition, field):
                raise ValueError(f"worker discovery identity mismatch: {field}")
        data = self._dump(capability)
        data["transport"] = definition.transport
        data["host"] = definition.host
        return self._model("WorkerCapabilities", data)

    def register(self, worker: Any) -> Any:
        """Discover a worker through the same protocol used for SSH workers."""
        result = self._rpc_result(worker, "discover", {})
        capabilities = self._capability_for_definition(
            worker, result.get("capabilities", result)
        )
        definition = self._dump(worker)
        record = {"definition": definition, "capabilities": self._dump(capabilities)}
        self.store.save_worker(str(worker.worker_id), record)
        return capabilities

    def workers(self, name: str | None = None, *, refresh: bool = True) -> list[Any]:
        from sparselab.workers.transport import ProtocolError, RemoteProtocolError

        results: list[Any] = []
        for record in self.store.worker_records():
            definition = self._model("WorkerDefinition", record["definition"])
            capabilities_data = record["capabilities"]
            if refresh:
                try:
                    answer = self._rpc_result(definition, "discover", {})
                    capabilities_data = self._dump(
                        self._capability_for_definition(
                            definition, answer.get("capabilities", answer)
                        )
                    )
                except (OSError, TimeoutError, RemoteProtocolError, ProtocolError):
                    # A transport failure only changes availability; it cannot turn an
                    # executing receipt into failure or completion.
                    capabilities_data = {**capabilities_data, "status": "unknown"}
                self.store.save_worker(
                    str(definition.worker_id),
                    {
                        "definition": self._dump(definition),
                        "capabilities": capabilities_data,
                    },
                )
            capability = self._model("WorkerCapabilities", capabilities_data)
            if name is None or capability.name == name:
                results.append(capability)
        return results

    def _prepare_entry(self, request: dict[str, object]) -> dict[str, object]:
        from sparselab.workers.bundles import (
            prepare_dispatch_bundle,
            verify_dispatch_bundle,
        )

        config = request["config"]
        if not isinstance(config, RunConfig):
            raise TypeError("submit requires a RunConfig")
        worker = request.get("worker")
        preferred = request.get("preferred")
        # A command-line worker binding is an explicit scheduling decision.  A
        # matrix preference remains metadata only when it does not conflict.
        if worker is not None:
            preferred = worker
        requirements = request.get("requirements")
        if requirements is None:
            requirements = self._model("SchedulingRequirements", {})
        elif not hasattr(requirements, "model_dump"):
            requirements = self._model("SchedulingRequirements", requirements)
        bundle_dir = self.root / ".dispatch" / uuid.uuid4().hex
        bundle = prepare_dispatch_bundle(
            config,
            bundle_dir,
            stage_bundle=request.get("stage_bundle"),
            promote=request.get("promote"),
            resume=request.get("resume"),
            allow_runtime_drift=bool(request.get("allow_runtime_drift", False)),
        )
        bundle = verify_dispatch_bundle(bundle_dir)
        experiment_id, attempt_id, run_id = (str(uuid.uuid4()) for _ in range(3))
        config_data = config.model_dump(mode="json")
        spec_data = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "config": config_data,
            "config_sha256": config_sha256(config_data),
            "bound_worker": worker,
            "preferred_worker": preferred,
            "requirements": self._dump(requirements),
            "continuation": self._dump(bundle.continuation),
            "required_versions": bundle.required_versions,
            "source_identity_sha256": bundle.source_identity_sha256,
            "dispatch_bundle_digest": bundle.digest(),
            "matrix": request.get("matrix"),
        }
        spec = self._model("ExperimentSpec", spec_data)
        return {
            "experiment_id": experiment_id,
            "attempt_id": attempt_id,
            "run_id": run_id,
            "spec": self._dump(spec),
            "bundle_dir": bundle_dir,
        }

    def submit(
        self,
        config: RunConfig,
        *,
        worker: str | None = None,
        stage_bundle: Path | None = None,
        promote: Path | None = None,
        requirements: Any | None = None,
        preferred: str | None = None,
        matrix: dict | None = None,
    ) -> Any:
        return self.submit_many(
            [
                {
                    "config": config,
                    "worker": worker,
                    "stage_bundle": stage_bundle,
                    "promote": promote,
                    "requirements": requirements,
                    "preferred": preferred,
                    "matrix": matrix,
                }
            ]
        )[0]

    def submit_many(self, requests: list[dict]) -> list[Any]:
        if not requests:
            return []
        # Preparation (including every content hash) completes before the only
        # enqueue transaction, so a bad matrix coordinate leaves no queued rows.
        entries = [self._prepare_entry(dict(request)) for request in requests]
        self.store.enqueue_many(entries)
        return [
            self._model(
                "Submission",
                {
                    "experiment_id": entry["experiment_id"],
                    "attempt_id": entry["attempt_id"],
                    "run_id": entry["run_id"],
                    "status": "QUEUED",
                },
            )
            for entry in entries
        ]

    def list_experiments(self) -> list[dict[str, object]]:
        return self.store.attempts()

    def cancel(self, run_id: str) -> dict[str, object]:
        result = self.store.request_cancel(run_id)
        if result["status"] != "CANCEL_REQUESTED":
            return result
        attempt = self.store.attempt_by_run(run_id)
        assert attempt is not None
        worker = self._worker_for_attempt(attempt)
        if worker is None:
            return result
        from sparselab.workers.transport import (
            ProtocolError,
            RemoteProtocolError,
            call_worker,
        )

        try:
            reply = call_worker(
                worker,
                "cancel",
                {"attempt_id": result["attempt_id"], "reason": "user"},
            )
        except (OSError, TimeoutError, RemoteProtocolError, ProtocolError):
            # The durable request is reconciled on a later poll; no local
            # transport failure can be represented as acknowledgement.
            return result
        receipt = self._reply_result(reply).get("receipt")
        if isinstance(receipt, dict):
            self._reconcile_receipt(attempt, receipt)
        return result

    def _worker_for_attempt(self, attempt: dict[str, Any]) -> Any | None:
        worker_id = attempt.get("worker_id")
        if not worker_id:
            return None
        for record in self.store.worker_records():
            definition = record["definition"]
            if definition.get("worker_id") == worker_id:
                return self._model("WorkerDefinition", definition)
        return None

    @staticmethod
    def _requested_features(spec: Any) -> set[str]:
        memory = spec.config.runtime.memory
        features: set[str] = set()
        if memory.activation_checkpointing.enabled:
            features.add("activation_checkpointing")
        if memory.activation_offload.enabled:
            features.add("activation_offload")
        return features

    @staticmethod
    def _capacity(
        capability: Any, requirements: Any, spec: Any
    ) -> tuple[bool, str | None]:
        runtime = capability.runtime
        if not isinstance(runtime, dict):
            return False, "worker runtime is malformed"
        requested_precision = (
            requirements.precision
            if requirements.precision is not None
            else spec.config.runtime.precision
        )
        if (
            requested_precision != "auto"
            and requested_precision not in capability.supported_precisions
        ):
            return False, f"requested precision {requested_precision} unsupported"
        required_features = Controller._requested_features(spec)
        missing_features = sorted(
            required_features - set(capability.supported_features)
        )
        if missing_features:
            return False, "untested required features: " + ", ".join(missing_features)
        for field, required, label in (
            ("device_total_bytes", requirements.min_device_memory_gb, "device memory"),
            ("system_total_bytes", requirements.min_system_memory_gb, "system memory"),
        ):
            if required is not None:
                available = runtime.get(field)
                if available is None:
                    return False, f"unknown {label} cannot meet minimum"
                if available < required * 1024**3:
                    return False, f"insufficient {label}"
        return True, None

    def _basic_eligibility(self, capability: Any, spec: Any) -> tuple[bool, str | None]:
        from sparselab.workers.models import validate_required_versions

        if spec.bound_worker and capability.name != spec.bound_worker:
            return False, "not hard-bound worker"
        if capability.status != "idle":
            return False, "not idle"
        if capability.max_concurrent_runs < 1:
            return False, "no worker slots"
        if capability.engine != spec.config.runtime.engine:
            return False, "config engine mismatch"
        backend = spec.config.runtime.backend
        if backend != "auto" and backend != capability.backend:
            return False, "config backend mismatch"
        if capability.device_index != spec.config.runtime.device_index:
            return False, "configured device index mismatch"
        drift_allowed = (
            spec.continuation.kind == "RESUMED"
            and spec.continuation.allow_runtime_drift
        )
        if (
            capability.source_identity_sha256 != spec.source_identity_sha256
            and not drift_allowed
        ):
            return False, "source identity mismatch"
        if (
            spec.requirements.backend
            and capability.backend not in spec.requirements.backend
        ):
            return False, "required backend unsupported"
        try:
            validate_required_versions(spec.required_versions, capability)
        except ValueError as error:
            return False, f"schema/codec mismatch: {error}"
        return True, None

    def _validate_candidate(
        self, capability: Any, spec: Any
    ) -> tuple[Any | None, str | None]:
        """Run the concrete probe that converts first-use discovery into evidence."""
        from sparselab.workers.transport import RemoteProtocolError, call_worker

        definition = self._worker_for_attempt({"worker_id": capability.worker_id})
        if definition is None:
            return None, "worker registration is unavailable"
        try:
            answer = self._reply_result(
                call_worker(
                    definition,
                    "validate",
                    {
                        "config": self._dump(spec.config),
                        "bundle_digest": spec.dispatch_bundle_digest,
                    },
                )
            )
        except RemoteProtocolError as error:
            if error.code == "BUSY":
                return None, "validation busy"
            return None, f"validation failed: {error.message}"
        except (OSError, TimeoutError) as error:
            return None, f"validation unavailable: {error}"
        if answer.get("validation_status") != "passed":
            reason = answer.get("reason")
            return None, f"validation failed: {reason or 'probe did not pass'}"
        runtime = answer.get("runtime")
        if not isinstance(runtime, dict):
            return None, "validation returned malformed runtime"
        try:
            discovered = self._rpc_result(definition, "discover", {})
            refreshed = self._capability_for_definition(
                definition, discovered.get("capabilities", discovered)
            )
        except (OSError, TimeoutError, RemoteProtocolError) as error:
            return None, f"validated capability refresh unavailable: {error}"
        if refreshed.validation_status != "passed":
            return None, "validated capability refresh did not persist probe evidence"
        self.store.save_worker(
            str(definition.worker_id),
            {
                "definition": self._dump(definition),
                "capabilities": self._dump(refreshed),
            },
        )
        return refreshed, None

    def _eligible_worker(
        self, attempt: dict[str, Any]
    ) -> tuple[Any | None, str | None]:
        spec = self._model("ExperimentSpec", attempt["spec"])
        choices: list[tuple[int, str, Any]] = []
        reasons: list[str] = []
        active = self.store.attempts(RECONCILABLE_QUEUE_STATES)
        for capability in self.workers(refresh=True):
            if any(
                item.get("worker_id") == capability.worker_id
                and item.get("terminal_receipt") is None
                for item in active
            ):
                reasons.append(f"{capability.name}: controller slot occupied")
                continue
            basic, reason = self._basic_eligibility(capability, spec)
            if not basic:
                if reason != "not hard-bound worker":
                    reasons.append(f"{capability.name}: {reason}")
                continue
            tested = capability
            if capability.validation_status != "passed":
                tested, reason = self._validate_candidate(capability, spec)
                if tested is None:
                    reasons.append(f"{capability.name}: {reason}")
                    continue
            ok, reason = self._capacity(tested, spec.requirements, spec)
            if not ok:
                reasons.append(f"{tested.name}: {reason}")
                continue
            choices.append(
                (
                    0 if spec.preferred_worker == tested.name else 1,
                    str(tested.name),
                    tested,
                )
            )
        if not choices:
            return None, "; ".join(reasons) or "no eligible worker"
        return min(choices, key=lambda choice: choice[:2])[2], None

    def _dispatch_bundle(self, spec: Any) -> tuple[Path, Any]:
        from sparselab.workers.bundles import _read_bundle_manifest

        for path in (self.root / ".dispatch").glob("*/bundle.json"):
            manifest = _read_bundle_manifest(path)
            if manifest.digest() == spec.dispatch_bundle_digest:
                return path.parent, manifest
        raise ValueError("prepared dispatch bundle is unavailable")

    def _launch(self, attempt: dict[str, Any], capability: Any) -> None:
        from sparselab.workers.bundles import verify_dispatch_bundle
        from sparselab.workers.transport import call_worker

        worker = self._worker_for_attempt({"worker_id": capability.worker_id})
        if worker is None:
            raise ValueError("assigned worker registration is unavailable")
        if attempt["status"] == "QUEUED":
            if not self.store.assign(attempt["attempt_id"], capability.worker_id):
                return
        elif (
            attempt["status"] not in {"ASSIGNED", "CANCEL_REQUESTED"}
            or attempt.get("worker_id") != capability.worker_id
        ):
            raise ValueError("only the same assigned attempt may replay launch")
        spec = self._model("ExperimentSpec", attempt["spec"])
        bundle_root, _ = self._dispatch_bundle(spec)
        manifest = verify_dispatch_bundle(bundle_root)
        check = call_worker(
            worker,
            "install_bundle",
            {"manifest_digest": manifest.digest(), "mode": "check"},
            attachments={"bundle.json": bundle_root / "bundle.json"},
            timeout=self.transfer_timeout,
        )
        missing = self._reply_result(check).get("missing_asset_digests")
        if not isinstance(missing, list) or any(
            not isinstance(value, str) for value in missing
        ):
            raise ValueError("worker bundle check returned malformed missing assets")
        identities = {item.sha256: item for item in manifest.files}
        attachments: dict[str, Path] = {"bundle.json": bundle_root / "bundle.json"}
        for digest in missing:
            identity = identities.get(digest)
            if identity is None:
                raise ValueError(
                    "worker requested an asset absent from bundle manifest"
                )
            attachments[f"assets/{digest}"] = bundle_root / identity.relative_path
        install = call_worker(
            worker,
            "install_bundle",
            {"manifest_digest": manifest.digest(), "mode": "install"},
            attachments=attachments,
            timeout=self.transfer_timeout,
        )
        install_result = self._reply_result(install)
        if install_result.get("bundle_digest") != spec.dispatch_bundle_digest:
            raise ValueError("worker installed a mismatched bundle")
        spec_path = self.root / ".dispatch" / f".{attempt['attempt_id']}.spec.json"
        try:
            spec_path.write_bytes(canonical_json(self._dump(spec)))
            reply = call_worker(
                worker,
                "launch",
                {
                    "attempt_id": attempt["attempt_id"],
                    "run_id": attempt["run_id"],
                    "experiment_id": attempt["experiment_id"],
                    "spec_digest": spec.digest(),
                    "bundle_digest": spec.dispatch_bundle_digest,
                },
                attachments={"spec.json": spec_path},
            )
        finally:
            spec_path.unlink(missing_ok=True)
        launch_result = self._reply_result(reply)
        receipt = launch_result.get("receipt", launch_result)
        if not isinstance(receipt, dict):
            raise TypeError("launch response omitted receipt")
        self._reconcile_receipt(attempt, receipt)

    def _reconcile_receipt(
        self, attempt: dict[str, Any], receipt: dict[str, object]
    ) -> None:
        from sparselab.workers.transport import ProtocolError

        durable = self.store.attempt_by_run(attempt["run_id"])
        if durable is None:
            raise ProtocolError("receipt has no durable attempt")
        spec = self._model("ExperimentSpec", durable["spec"])
        expected = {
            "attempt_id": durable["attempt_id"],
            "run_id": durable["run_id"],
            "experiment_id": durable["experiment_id"],
            "worker_id": durable["worker_id"],
            "spec_digest": spec.digest(),
            "bundle_digest": spec.dispatch_bundle_digest,
        }
        if any(receipt.get(name) != value for name, value in expected.items()):
            raise ProtocolError("receipt identity differs from durable assignment")
        state = receipt.get("state")
        mapping = {
            "PREPARED": "ASSIGNED",
            "RUNNING": "RUNNING",
            "COMPLETE": "COMPLETE",
            "FAILED": "FAILED",
            "INTERRUPTED": "INTERRUPTED",
            "UNKNOWN": "UNKNOWN",
        }
        if state not in mapping:
            raise ValueError("unknown attempt receipt state")
        # Completion wins a concurrent cancellation request.  Cancellation only
        # becomes CANCELLED after interruption evidence acknowledges it.
        status = mapping[state]
        if state == "INTERRUPTED" and receipt.get("cancellation_acknowledged"):
            status = "CANCELLED"
        self.store.set_receipt(attempt["attempt_id"], receipt, status)

    def _poll_active(self, attempt: dict[str, Any]) -> None:
        """Reconcile a durable receipt; only transport failures create UNKNOWN."""
        from sparselab.workers.transport import ProtocolError, RemoteProtocolError

        worker = self._worker_for_attempt(attempt)
        if worker is None:
            self.store.mark_unknown_if_nonterminal(
                attempt["attempt_id"], "worker registration is unavailable"
            )
            return
        try:
            if attempt["status"] == "CANCEL_REQUESTED":
                self.cancel(attempt["run_id"])
            result = self._rpc_result(
                worker, "status", {"attempt_id": attempt["attempt_id"]}
            )
            receipt = result.get("receipt")
            if not isinstance(receipt, dict):
                return
            self._reconcile_receipt(attempt, receipt)
        except (OSError, TimeoutError, RemoteProtocolError, ProtocolError) as error:
            self.store.mark_unknown_if_nonterminal(
                attempt["attempt_id"], f"worker status unavailable: {error}"
            )
            return
        origin_id = result.get("origin_id")
        if not isinstance(origin_id, str):
            raise TypeError("worker status omitted outbox origin identity")
        terminal = receipt.get("state") in {
            "COMPLETE",
            "FAILED",
            "INTERRUPTED",
            "UNKNOWN",
        }
        try:
            if terminal:
                self._ingest(attempt, worker, receipt, origin_id)
            else:
                self._ingest_records(worker, origin_id, drain=False)
        except (
            OSError,
            TimeoutError,
            RemoteProtocolError,
            ProtocolError,
            ValueError,
        ) as error:
            if terminal:
                self.store.mark_ingestion_error(attempt["attempt_id"], error)
            else:
                _LOGGER.warning(
                    "Live record import failed for %s: %s", attempt["attempt_id"], error
                )

    @staticmethod
    def _artifact_free_terminal(receipt: dict[str, object]) -> bool:
        """An unsuccessful terminal without run-owned files needs no transfer."""
        return receipt.get("state") in {"FAILED", "INTERRUPTED", "UNKNOWN"} and not any(
            item["relative_path"].startswith("run/")
            for item in receipt.get("artifacts", [])
        )

    def _ingest(
        self,
        attempt: dict[str, Any],
        worker: Any,
        receipt: dict[str, object],
        origin_id: str,
    ) -> None:
        from sparselab.workers.artifacts import ingest_attempt_artifacts

        if self._artifact_free_terminal(receipt):
            self.store.mark_ingestion_not_required(attempt["attempt_id"])
            return
        self._ingest_records(worker, origin_id, drain=True)
        spec = self._model("ExperimentSpec", attempt["spec"])
        _, bundle = self._dispatch_bundle(spec)
        ingest_attempt_artifacts(
            worker,
            self._model("AttemptReceipt", receipt),
            self.root,
            spec=spec,
            bundle=bundle,
            records=self.store.metrics,
            timeout=self.transfer_timeout,
        )
        self.store.mark_ingestion_complete(attempt["attempt_id"])

    def _ingest_records(self, worker: Any, origin_id: str, *, drain: bool) -> None:
        from sparselab.workers.transport import call_worker

        cursor = self.store.imported_sequence(origin_id)
        while True:
            with TemporaryDirectory(prefix=".records-", dir=self.root) as directory:
                reply = call_worker(
                    worker,
                    "records",
                    {
                        "origin_id": origin_id,
                        "after_sequence": cursor,
                        "limit": 1000,
                        "max_bytes": 4_194_304,
                    },
                    receive_dir=Path(directory),
                    timeout=self.transfer_timeout,
                )
                result = self._reply_result(reply)
                if result.get("origin_id") != origin_id:
                    raise ValueError(
                        "record response origin differs from requested worker"
                    )
                attachments = getattr(reply, "attachments", {})
                records_path = (
                    attachments.get("records.json")
                    if isinstance(attachments, dict)
                    else None
                )
                if records_path is None:
                    raise ValueError("records response omitted records.json")
                from sparselab.training.metrics import _strict_json_loads

                records = _strict_json_loads(Path(records_path).read_bytes())
                if not isinstance(records, list):
                    raise TypeError("records attachment is malformed")
                if any(
                    not isinstance(record, dict) or record.get("origin_id") != origin_id
                    for record in records
                ):
                    raise ValueError(
                        "record envelope origin differs from requested worker"
                    )
                next_cursor = records[-1]["sequence"] if records else cursor
                if result["next_sequence"] != next_cursor or (
                    result.get("has_more") and next_cursor <= cursor
                ):
                    raise ValueError(
                        "record stream cursor is inconsistent or not advancing"
                    )
                self.store.import_records(records)
            cursor = int(result["next_sequence"])
            if not drain or not result.get("has_more"):
                break

    def resume(
        self,
        run_id: str,
        *,
        worker: str | None = None,
        allow_runtime_drift: bool = False,
    ) -> Any:
        parent = self.store.attempt_by_run(run_id)
        if parent is None:
            raise KeyError(f"unknown run: {run_id}")
        if parent["status"] in ACTIVE_QUEUE_STATES:
            raise ValueError("cannot resume a confirmed active parent")
        if parent["status"] not in {"INTERRUPTED", "CANCELLED", "FAILED", "UNKNOWN"}:
            raise ValueError("parent has no resumable terminal state")
        if parent["ingestion_status"] != "COMPLETE":
            raise ValueError("parent checkpoint ingestion is not complete")
        with self.store.metrics._connect() as con:
            checkpoint = con.execute(
                "SELECT digest,relative_path,step,tokens_seen FROM checkpoints WHERE run_id=? "
                "AND verification_status='verified' AND resume_level='full' "
                "ORDER BY step DESC,relative_path DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        if checkpoint is None:
            raise ValueError("parent has no locally verified full-resume checkpoint")
        spec = self._model("ExperimentSpec", parent["spec"])
        checkpoint_path = self.root / run_id / "checkpoints" / checkpoint[1]
        if not checkpoint_path.is_dir():
            raise ValueError(
                "parent verified checkpoint files are not locally available"
            )
        from sparselab.training.metrics import _strict_json_loads

        checkpoint_manifest = _strict_json_loads(
            (checkpoint_path / "manifest.json").read_bytes()
        )
        if (
            not isinstance(checkpoint_manifest, dict)
            or checkpoint_manifest.get("sha256") != checkpoint[0]
        ):
            raise ValueError("selected generation differs from its verified projection")
        if (
            checkpoint[2] >= spec.config.training.max_steps
            or checkpoint[3] >= spec.config.training.max_tokens
        ):
            raise ValueError(
                "selected checkpoint already exhausted the experiment budget"
            )
        entries = [
            self._prepare_entry(
                {
                    "config": spec.config,
                    "worker": worker,
                    "requirements": spec.requirements,
                    "preferred": spec.preferred_worker,
                    "matrix": spec.matrix,
                    "resume": checkpoint_path,
                    "allow_runtime_drift": allow_runtime_drift,
                }
            )
        ]
        self.store.enqueue_many(entries)
        return self._model(
            "Submission",
            {
                "experiment_id": entries[0]["experiment_id"],
                "attempt_id": entries[0]["attempt_id"],
                "run_id": entries[0]["run_id"],
                "status": "QUEUED",
            },
        )

    def _replay_assigned(self, attempt: dict[str, Any]) -> bool:
        """Replay delivery only before an executor has claimed the same receipt."""
        from sparselab.workers.transport import ProtocolError, RemoteProtocolError

        worker = self._worker_for_attempt(attempt)
        if worker is None:
            return False
        try:
            result = self._rpc_result(
                worker, "status", {"attempt_id": attempt["attempt_id"]}
            )
        except (OSError, TimeoutError, RemoteProtocolError, ProtocolError):
            # A missing reply is deliberately indistinguishable from a delivery
            # failure.  The durable ASSIGNED row remains eligible for a later
            # same-ID reconciliation, not a new optimizer attempt.
            return False
        receipt = result.get("receipt")
        if isinstance(receipt, dict):
            self._reconcile_receipt(attempt, receipt)
            if (
                receipt.get("state") != "PREPARED"
                or receipt.get("training_started_at") is not None
            ):
                return False
        capability = next(
            (
                item
                for item in self.workers(refresh=False)
                if item.worker_id == attempt["worker_id"]
            ),
            None,
        )
        if capability is None:
            return False
        try:
            if attempt["status"] == "CANCEL_REQUESTED":
                self._rpc_result(
                    worker,
                    "cancel",
                    {"attempt_id": attempt["attempt_id"], "reason": "user"},
                )
            self._launch(attempt, capability)
        except (OSError, TimeoutError, RemoteProtocolError, ProtocolError):
            return False
        return True

    def _retry_ingestion(self, attempt: dict[str, Any]) -> None:
        from sparselab.workers.transport import ProtocolError, RemoteProtocolError

        receipt = attempt["terminal_receipt"]
        if not isinstance(receipt, dict):
            return
        worker = self._worker_for_attempt(attempt)
        if worker is None:
            return
        try:
            result = self._rpc_result(
                worker, "status", {"attempt_id": attempt["attempt_id"]}
            )
            origin_id = result.get("origin_id")
            if not isinstance(origin_id, str):
                raise TypeError("worker status omitted outbox origin identity")
            self._ingest(attempt, worker, receipt, origin_id)
        except (
            OSError,
            TimeoutError,
            RemoteProtocolError,
            ProtocolError,
            ValueError,
        ) as error:
            self.store.mark_ingestion_error(attempt["attempt_id"], error)

    def tick(self) -> dict[str, int]:
        from sparselab.workers.transport import ProtocolError, RemoteProtocolError

        queued = self.store.attempts(frozenset({"QUEUED"}))
        assigned = 0
        for attempt in queued:
            capability, reason = self._eligible_worker(attempt)
            if capability is None:
                self.store.set_queued_reason(attempt["attempt_id"], reason)
                continue
            try:
                self._launch(attempt, capability)
                assigned += 1
            except (OSError, TimeoutError, RemoteProtocolError, ProtocolError):
                # The CAS assignment is retained; the next tick performs the
                # same-ID receipt reconciliation before any replay.
                continue
        for attempt in self.store.attempts(RECONCILABLE_QUEUE_STATES):
            if attempt["terminal_receipt"] is None:
                self._poll_active(attempt)
        for attempt in self.store.attempts(frozenset({"ASSIGNED", "CANCEL_REQUESTED"})):
            if self._replay_assigned(attempt):
                assigned += 1
        for attempt in self.store.pending_ingestion():
            self._retry_ingestion(attempt)
        return {"queued": len(queued), "assigned": assigned}

    @contextmanager
    def _process_lock(self, *, optional: bool = False) -> Any:
        import fcntl

        with self._lock_path.open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                if optional:
                    yield False
                    return
                raise RuntimeError("controller is already running") from error
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def run(self) -> None:
        with self._process_lock() as acquired:
            assert acquired
            _LOGGER.info("Controller ready: %s", self.root)
            while True:
                self.tick()
                time.sleep(self.poll_seconds)
