"""Process-wide CPU limits for the test suite.

This module is imported before test collection in both the xdist controller and
its workers.  Child processes inherit the environment established here.
"""

import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

_CPU_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

for _name in _CPU_THREAD_ENVIRONMENT:
    os.environ[_name] = "1"

_torch = sys.modules.get("torch")
if isinstance(_torch, ModuleType):
    _torch.set_num_threads(1)


# Host-RAM planning (``sparselab.host_capacity``) refuses work when measured
# available RAM minus a reserve cannot fit one worker. Under xdist, and on a
# box shared with other jobs, that measurement depends on whatever else is
# running, so unrelated tests failed with "inadequate measured RAM after
# reserve". Every SparseLab RAM reader (reserves from ``total`` as well as
# checks on ``available``) goes through ``host_capacity.measure_memory``, so
# pinning it gives tests one fixed, ample host whatever the physical machine.
# A test that monkeypatches ``psutil.virtual_memory`` itself
# (tests/test_host_capacity.py and friends) still gets its own value, so
# low-memory and reserve logic stays covered.
_GIB = 1024**3


@pytest.fixture(autouse=True)
def _deterministic_host_ram(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    from sparselab import host_capacity

    host_capacity_measure = host_capacity.measure_memory
    original = psutil.virtual_memory
    # Children launched by tests (``python -c``/``-m sparselab``) inherit the
    # same fixed host through the environment.
    monkeypatch.setenv(host_capacity.TEST_HOST_RAM_ENV, f"{64 * _GIB}:{48 * _GIB}")

    def measured() -> object:
        if psutil.virtual_memory is not original:
            return psutil.virtual_memory()
        return host_capacity_measure()

    monkeypatch.setattr(host_capacity, "measure_memory", measured)


# Fast/full split. Tests listed in tests/slow_tests.txt (by base node id, so
# every parametrized case follows) get the `slow` marker; `-m "not slow"` is
# the fast suite. See docs/test-speed.md.
_SLOW_LIST = Path(__file__).with_name("slow_tests.txt")
_SLOW = frozenset(
    line.strip()
    for line in _SLOW_LIST.read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
)


# Known failures with an open issue (tests/known_failures.txt): skipped with
# the precise reason until the issue lands, never silently weakened.
_KNOWN = {
    node.strip(): reason.strip()
    for node, _, reason in (
        line.partition("|")
        for line in Path(__file__)
        .with_name("known_failures.txt")
        .read_text()
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        base = item.nodeid.split("[", 1)[0]
        module = base.split("::", 1)[0]
        if base in _SLOW or module in _SLOW:
            item.add_marker(pytest.mark.slow)
        reason = _KNOWN.get(base) or _KNOWN.get(module)
        if reason:
            item.add_marker(pytest.mark.skip(reason=f"known failure: {reason}"))
