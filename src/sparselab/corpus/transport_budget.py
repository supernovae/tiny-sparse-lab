"""Durable, conservative response-body reservations for bounded corpus acquisition."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from sparselab.corpus.project import Project
from sparselab.training.manifest import canonical_json


def _binding(project: Project) -> str:
    value = {
        "project": project.config.id,
        "budget": project.config.transport_budget.model_dump(mode="json"),
        "sources": [
            {
                "id": source.id,
                "uri": source.canonical_uri,
                "revision": source.revision,
                "shards": [
                    {
                        "path": shard.path,
                        "sha256": shard.expected_sha256.lower(),
                        "max_bytes": shard.max_shard_bytes,
                    }
                    for shard in source.acquisition.bounded_shards
                ],
            }
            for source in project.sources
            if source.kind == "huggingface_dataset"
            and source.acquisition.bounded_shards is not None
        ]
        + [
            {
                "id": source.id,
                "uri": source.canonical_uri,
                "revision": source.revision,
                "tree_oid": source.acquisition.tree_oid,
                "blobs": [
                    blob.model_dump(mode="json")
                    for blob in source.acquisition.bounded_blobs
                ],
            }
            for source in project.sources
            if source.kind == "git" and source.acquisition.bounded_blobs is not None
        ],
    }
    if project.config.source_effects is not None:
        value["source_effects"] = [
            effect.model_dump(mode="json") for effect in project.config.source_effects
        ]
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _validate_scope(project: Project) -> None:
    effects = (
        {item.source_id: item for item in project.config.source_effects}
        if project.config.source_effects is not None
        else None
    )
    for source in project.sources:
        if effects is not None and effects[source.id].effect == "reuse_only":
            continue
        if source.redistribution == "rejected" or source.kind in {
            "local",
            "deterministic_generator",
            "inference_generator",
        }:
            continue
        if not (
            source.kind == "huggingface_dataset"
            and source.acquisition.bounded_shards is not None
            or source.kind == "git"
            and source.acquisition.bounded_blobs is not None
        ):
            raise ValueError(
                "transport budget cannot cover this unaccounted network source"
            )


class TransportBudget:
    def __init__(self, path: Path, project: Project):
        spec = project.config.transport_budget
        if spec is None:
            raise ValueError("project has no transport budget")
        _validate_scope(project)
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                "transport budget ledger missing; initialize it explicitly"
            )
        self.path = path
        self.spec = spec
        self.project = project
        self.binding = _binding(project)
        self._read()

    def _validate_state(self, state: dict[str, Any]) -> None:
        if state["binding"] != self.binding:
            raise ValueError("transport budget binding changed")
        transfers = state["transfers"]
        if (
            state["deadline"] != state["created_at"] + self.spec.max_wall_seconds
            or not isinstance(transfers, list)
            or len(transfers) > self.spec.max_transfers
            or state["source_charged"] != sum(item["reserved"] for item in transfers)
            or state["source_actual"] != sum(item["actual"] for item in transfers)
            or not 0
            <= state["source_actual"]
            <= state["source_charged"]
            <= self.spec.max_source_body_bytes
            or not 0
            <= state["metadata_actual"]
            <= state["metadata_charged"]
            <= self.spec.max_metadata_body_bytes
            or any(
                item["status"] not in {"started", "interrupted", "failed", "complete"}
                or not 0 <= item["actual"] <= item["reserved"]
                for item in transfers
            )
        ):
            raise ValueError("transport budget ledger counters are invalid")
        for transfer in transfers:
            if "resources" not in transfer:
                continue  # Historical ledgers have no inferred readings.
            readings = transfer["resources"]
            if (
                not isinstance(readings, dict)
                or set(readings)
                != {"expanded_bytes", "peak_staging_bytes", "peak_staging_inodes"}
                or any(
                    (type(value) is not int or value < 0)
                    for key, value in readings.items()
                    if key != "expanded_bytes" or value is not None
                )
            ):
                raise ValueError("transport budget resource counters are invalid")

    def projected_disk_bytes(self) -> int:
        effects = (
            {item.source_id: item for item in self.project.config.source_effects}
            if self.project.config.source_effects is not None
            else None
        )
        transferring = [
            source
            for source in self.project.sources
            if effects is None or effects[source.id].effect == "acquire"
        ]
        bounded_hf = [
            source.acquisition
            for source in transferring
            if source.kind == "huggingface_dataset"
            and source.acquisition.bounded_shards is not None
        ]
        bounded_git = [
            source.acquisition
            for source in transferring
            if source.kind == "git" and source.acquisition.bounded_blobs is not None
        ]
        largest = [
            shard.max_shard_bytes
            for spec in bounded_hf
            for shard in spec.bounded_shards
        ] + [blob.max_bytes for spec in bounded_git for blob in spec.bounded_blobs]
        return (
            sum(spec.max_bytes for spec in (*bounded_hf, *bounded_git))
            + max(largest)
            + 65_536  # ledger, manifests and small staging metadata
        )

    @classmethod
    def initialize(cls, path: Path, project: Project) -> TransportBudget:
        spec = project.config.transport_budget
        if spec is None:
            raise ValueError("project has no transport budget")
        _validate_scope(project)
        effects = (
            {item.source_id: item for item in project.config.source_effects}
            if project.config.source_effects is not None
            else None
        )
        if not any(
            (effects is None or effects[source.id].effect == "acquire")
            and (
                source.kind == "huggingface_dataset"
                and source.acquisition.bounded_shards is not None
                or source.kind == "git"
                and source.acquisition.bounded_blobs is not None
            )
            for source in project.sources
        ):
            raise ValueError("transport budget requires bounded shards or Git blobs")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb"):
                pass
        except FileExistsError as error:
            raise ValueError("transport budget ledger already exists") from error
        with sqlite3.connect(path) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute(
                "CREATE TABLE state (id INTEGER PRIMARY KEY CHECK (id=1), body TEXT NOT NULL)"
            )
            now = time.time()
            state = {
                "binding": _binding(project),
                "created_at": now,
                "deadline": now + spec.max_wall_seconds,
                "source_charged": 0,
                "source_actual": 0,
                "metadata_charged": 0,
                "metadata_actual": 0,
                "transfers": [],
                "failures": [],
            }
            db.execute("INSERT INTO state VALUES (1, ?)", (json.dumps(state),))
        return cls(path, project)

    def _read(self) -> dict[str, Any]:
        try:
            with sqlite3.connect(self.path) as db:
                row = db.execute("SELECT body FROM state WHERE id=1").fetchone()
            if row is None:
                raise ValueError("transport budget ledger is incomplete")
            state = json.loads(row[0])
            self._validate_state(state)
            if time.time() < state["created_at"]:
                raise ValueError("transport budget clock moved backward")
            return state
        except (sqlite3.Error, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("transport budget ledger is invalid") from error

    def _mutate(self, update: Any, *, check_deadline: bool = True) -> Any:
        with sqlite3.connect(self.path, timeout=30, isolation_level=None) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM state WHERE id=1").fetchone()
            if row is None:
                raise ValueError("transport budget ledger is incomplete")
            state = json.loads(row[0])
            self._validate_state(state)
            now = time.time()
            if now < state["created_at"] or (
                check_deadline and now >= state["deadline"]
            ):
                raise ValueError("transport budget deadline exceeded")
            result = update(state)
            db.execute("UPDATE state SET body=? WHERE id=1", (json.dumps(state),))
            db.execute("COMMIT")
            return result

    def remaining_seconds(self) -> float:
        state = self._read()
        remaining = state["deadline"] - time.time()
        if remaining <= 0:
            raise ValueError("transport budget deadline exceeded")
        return remaining

    def reserve_transfer(
        self,
        source_id: str,
        path: str,
        size: int,
        *,
        allow_verified_retry: bool = False,
    ) -> int:
        key = f"{source_id}:{path}"

        def update(state: dict[str, Any]) -> int:
            self._check_transfer(
                state, key, size, allow_verified_retry=allow_verified_retry
            )
            for item in state["transfers"]:
                if item["key"] == key and item["status"] == "started":
                    item["status"] = "interrupted"
            state["source_charged"] += size
            state["transfers"].append(
                {"key": key, "reserved": size, "actual": 0, "status": "started"}
            )
            return len(state["transfers"]) - 1

        return self._mutate(update)

    def _check_transfer(
        self,
        state: dict[str, Any],
        key: str,
        size: int,
        *,
        allow_verified_retry: bool = False,
    ) -> None:
        prior = [item for item in state["transfers"] if item["key"] == key]
        if any(
            item["status"] == "failed"
            or item["status"] == "complete"
            and (not allow_verified_retry or item.get("sha256_verified") is not True)
            for item in prior
        ):
            raise ValueError("transport budget shard is already complete or terminal")
        if len(prior) > self.spec.max_retries_per_shard:
            raise ValueError("transport budget shard retry allowance exhausted")
        if len(state["transfers"]) >= self.spec.max_transfers:
            raise ValueError("transport budget transfer count exhausted")
        if size > self.spec.max_source_body_bytes - state["source_charged"]:
            raise ValueError("transport budget source body allowance exhausted")

    def assert_transfer_available(
        self,
        source_id: str,
        path: str,
        size: int,
        *,
        allow_verified_retry: bool = False,
    ) -> None:
        self.remaining_seconds()
        self._check_transfer(
            self._read(),
            f"{source_id}:{path}",
            size,
            allow_verified_retry=allow_verified_retry,
        )

    def charge_transfer_actual(self, index: int, amount: int) -> None:
        def update(state: dict[str, Any]) -> None:
            transfer = state["transfers"][index]
            if (
                transfer["status"] != "started"
                or amount > transfer["reserved"] - transfer["actual"]
            ):
                raise ValueError("transport budget transfer body cap exceeded")
            transfer["actual"] += amount
            state["source_actual"] += amount

        self._mutate(update, check_deadline=False)

    def finish_transfer(
        self, index: int, *, success: bool, interrupted: bool = False
    ) -> None:
        def update(state: dict[str, Any]) -> None:
            transfer = state["transfers"][index]
            if transfer["status"] != "started":
                raise ValueError("transport budget transfer already ended")
            transfer["status"] = (
                "complete" if success else "interrupted" if interrupted else "failed"
            )
            transfer["sha256_verified"] = success

        self._mutate(update, check_deadline=False)

    def record_failure(self, source_id: str, error: BaseException) -> None:
        def update(state: dict[str, Any]) -> None:
            state["failures"].append(
                {
                    "source_id": source_id,
                    "error_type": type(error).__name__,
                    "message": str(error)[:300],
                }
            )

        self._mutate(update, check_deadline=False)

    def reserve_metadata(self, amount: int) -> None:
        def update(state: dict[str, Any]) -> None:
            if amount > self.spec.max_metadata_body_bytes - state["metadata_charged"]:
                raise ValueError("transport budget metadata body allowance exhausted")
            state["metadata_charged"] += amount

        self._mutate(update)

    def charge_metadata_actual(self, amount: int) -> None:
        def update(state: dict[str, Any]) -> None:
            state["metadata_actual"] += amount
            if state["metadata_actual"] > state["metadata_charged"]:
                raise ValueError("transport budget metadata accounting invalid")

        self._mutate(update, check_deadline=False)

    def receipt(self) -> dict[str, Any]:
        return self._read()

    def record_resources(self, index: int, readings: dict[str, Any]) -> None:
        """Retain measured counters even after a transfer or parser fails."""

        def update(state: dict[str, Any]) -> None:
            state["transfers"][index]["resources"] = readings
            self._validate_state(state)

        self._mutate(update, check_deadline=False)
