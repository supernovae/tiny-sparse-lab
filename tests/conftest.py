"""Process-wide CPU limits for the test suite.

This module is imported before test collection in both the xdist controller and
its workers.  Child processes inherit the environment established here.
"""

import os
import sys
from types import ModuleType

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
