"""Fail closed at model-update entry points during preparation-only attempts."""

from __future__ import annotations

import os


def require_model_runtime_allowed() -> None:
    if os.environ.get("SPARSELAB_PREPARATION_ONLY") == "1":
        raise RuntimeError(
            "model training and staging are disabled by SPARSELAB_PREPARATION_ONLY"
        )
