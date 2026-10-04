from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from sparselab import host_capacity as host


@pytest.fixture
def capacity(monkeypatch):
    monkeypatch.setattr(
        host.psutil, "cpu_count", lambda *, logical: 24 if logical else 12
    )
    monkeypatch.setattr(
        host.psutil,
        "Process",
        lambda: SimpleNamespace(cpu_affinity=lambda: list(range(6))),
    )
    memory = SimpleNamespace(available=1000, total=2000)
    monkeypatch.setattr(host.psutil, "virtual_memory", lambda: memory)
    return memory


def test_capacity_uses_all_bounds(capacity):
    plan = host.plan_host_workers(
        "sha256", worker_memory_bytes=200, reserve_bytes=200, operator_cap=3
    )
    assert (
        plan.logical_cpus,
        plan.physical_cpus,
        plan.affinity_cpus,
        plan.workers,
    ) == (24, 12, 6, 3)
    assert (
        host.plan_host_workers(
            "sha256", worker_memory_bytes=200, reserve_bytes=200
        ).workers
        == 4
    )
    capacity.available = 200
    with pytest.raises(ValueError, match="inadequate"):
        host.plan_host_workers("sha256", worker_memory_bytes=200, reserve_bytes=200)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"worker_memory_bytes": 0, "reserve_bytes": 1},
        {"worker_memory_bytes": 1, "reserve_bytes": -1},
        {"worker_memory_bytes": True, "reserve_bytes": 1},
        {"worker_memory_bytes": 1, "reserve_bytes": 1, "operator_cap": 0},
    ],
)
def test_nonpositive_bounds_rejected(kwargs):
    with pytest.raises(ValueError):
        host.plan_host_workers("sha256", **kwargs)


def test_unknown_memory_is_sha_only(capacity, monkeypatch):
    def unavailable():
        raise OSError

    monkeypatch.setattr(host.psutil, "virtual_memory", unavailable)
    plan = host.plan_host_workers(
        "sha256", worker_memory_bytes=host.SHA_WORKER_BYTES, reserve_bytes=1
    )
    assert plan.available_bytes is None and plan.workers == 1
    for operation, size in [("encoding", 1), ("sha256", host.SHA_WORKER_BYTES + 1)]:
        with pytest.raises(ValueError, match="unavailable"):
            host.plan_host_workers(operation, worker_memory_bytes=size, reserve_bytes=1)


def test_unknown_cpu_and_affinity_fallback(capacity, monkeypatch):
    monkeypatch.setattr(host.psutil, "cpu_count", lambda **_: None)
    monkeypatch.setattr(host.psutil, "Process", lambda: object())
    assert (
        host.plan_host_workers("sha256", worker_memory_bytes=1, reserve_bytes=1).workers
        == 1
    )


def test_ordered_window_and_cleanup(capacity):
    plan = host.plan_host_workers(
        "sha256", worker_memory_bytes=1, reserve_bytes=1, operator_cap=2
    )
    second = threading.Event()
    consumed = []

    def items():
        for number in range(5):
            consumed.append(number)
            yield number

    def worker(number):
        if number == 0:
            assert second.wait(5)
        elif number == 1:
            second.set()
        return number * 2

    iterator = host.run_ordered(items(), worker, plan=plan)
    assert next(iterator) == 0
    assert consumed == [0, 1]
    assert list(iterator) == [2, 4, 6, 8]
    completed = []

    def failed(number):
        if number == 0:
            raise ValueError("failure")
        completed.append(number)

    with pytest.raises(ValueError, match="failure"):
        list(host.run_ordered(range(10), failed, plan=plan))
    assert set(completed) <= {1}
    assert not [t for t in threading.enumerate() if t.name.startswith("sha256_")]


def test_native_sha_and_canonical_bytes_match(capacity, tmp_path):
    paths = []
    for n in range(8):
        path = tmp_path / f"{n}.bin"
        path.write_bytes(bytes([n]) * 4096)
        paths.append(path)
    plan = host.plan_host_workers("sha256", worker_memory_bytes=1, reserve_bytes=1)

    def worker(path):
        return (path.name, hashlib.sha256(path.read_bytes()).hexdigest())

    from sparselab.training.manifest import canonical_json

    reference = canonical_json(
        list(host.run_ordered(paths, worker, plan=replace(plan, workers=1)))
    )
    for workers in (2, plan.workers):
        assert (
            canonical_json(
                list(
                    host.run_ordered(paths, worker, plan=replace(plan, workers=workers))
                )
            )
            == reference
        )


def test_isa_parsing_and_fallback(monkeypatch):
    assert (
        host._parse_isa("x86_64", "flags: avx2 sha_ni aes\nflags: avx2 aes")["sha_ni"]
        is False
    )
    assert host._parse_isa("x86_64", "flags: sse4_1 sha_ni")["sha_ni"] is True
    assert (
        host._parse_isa("aarch64", "Features: asimd sha2 aes crc32")["asimd/neon"]
        is True
    )
    assert set(host._parse_isa("arm64", None).values()) == {"unknown"}
    assert host._parse_isa("riscv64", None) == {"capabilities": "unknown"}
    monkeypatch.setattr(host.platform, "system", lambda: "unsupported")
    assert host.hardware_observation()["probe"] == "unknown"
