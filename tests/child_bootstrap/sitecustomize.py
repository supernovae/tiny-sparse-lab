"""Test-suite bootstrap for child interpreters launched by tests.

``tests/conftest.py`` puts this directory first on ``PYTHONPATH`` and sets
``SPARSELAB_TEST_HOST_RAM=total:available`` for every test, so a fresh
``python -c``/``-m sparselab`` child plans against the same fixed host as the
parent process. Python imports ``sitecustomize`` at startup only from
``sys.path``; this directory is never there outside pytest, so shipped code
always measures real RAM (``host_capacity.measure_memory`` has no override).
"""

from __future__ import annotations

import os

_PINNED = os.environ.get("SPARSELAB_TEST_HOST_RAM")

if _PINNED:
    try:
        from sparselab import host_capacity
    except ImportError:  # child that does not run SparseLab code
        pass
    else:
        _total, _, _available = _PINNED.partition(":")
        _measure = host_capacity.measure_memory

        def _pinned_measure_memory() -> object:
            return _measure()._replace(total=int(_total), available=int(_available))

        host_capacity.measure_memory = _pinned_measure_memory
