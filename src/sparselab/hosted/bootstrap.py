"""Fixed stdlib bootstrap programs for already user-provisioned hosts."""

from __future__ import annotations

import json
from pathlib import Path

from sparselab import runtime_env_subprocess
from sparselab.hosted import bootstrap_worker


def _program(action: str, payload: dict, timeout: float, *, before: str = "") -> str:
    # Both sources are versioned stdlib-only implementations, not user programs.
    bounded = Path(runtime_env_subprocess.__file__).read_text(encoding="utf-8")
    worker = Path(bootstrap_worker.__file__).read_text(encoding="utf-8")
    data = json.dumps({**payload, "timeout": timeout}, sort_keys=True, allow_nan=False)
    return (
        "import json\n_sparselab_bootstrap = {'__name__': 'sparselab_hosted_bootstrap'}\n"
        + f"exec(compile({bounded!r}, '<sparselab-bounded>', 'exec'), _sparselab_bootstrap)\n"
        + f"exec(compile({worker!r}, '<sparselab-bootstrap>', 'exec'), _sparselab_bootstrap)\n"
        + f"_sparselab_bootstrap['_deadline'] = _sparselab_bootstrap['time'].monotonic() + {timeout!r}\n"
        + "try:\n"
        + f" exec(compile({before!r}, '<sparselab-transfer>', 'exec'), {{'__name__': 'sparselab_transfer', '_sparselab_deadline': _sparselab_bootstrap['_deadline']}})\n"
        + f" _sparselab_payload = json.loads({data!r})\n"
        + " _sparselab_payload['timeout'] = _sparselab_bootstrap['remaining']()\n"
        + f" _sparselab_result = _sparselab_bootstrap['dispatch']({action!r}, _sparselab_payload)\n"
        + " print(json.dumps(_sparselab_result, sort_keys=True, allow_nan=False))\n"
        + "except Exception as _sparselab_error:\n"
        + " print(json.dumps({'hosted_error_version': 1, 'error_type': type(_sparselab_error).__name__, 'message': str(_sparselab_error)[:4096]}, sort_keys=True))\n"
        + " raise\n"
    )


def diagnostic_program(root: str, *, timeout: float = 60) -> str:
    return _program("inspect", {"root": root}, timeout)


def preflight_program(
    *,
    root: str,
    runtime_root: str,
    source_bytes: int,
    source_commit: str,
    source_sha256: str,
    recipe: str,
    timeout: float,
    instance_id: str | None = None,
) -> str:
    return _program(
        "preflight",
        {
            "root": root,
            "runtime_root": runtime_root,
            "source_bytes": source_bytes,
            "source_commit": source_commit,
            "source_sha256": source_sha256,
            "recipe": recipe,
            "instance_id": instance_id,
        },
        timeout,
    )


def setup_program(
    *,
    root: str,
    runtime_root: str,
    source_bundle: str,
    source_commit: str,
    source_sha256: str,
    recipe: str,
    instance_id: str,
    timeout: float,
    assembly: str = "",
) -> str:
    return _program(
        "setup",
        {
            "root": root,
            "runtime_root": runtime_root,
            "source_bundle": source_bundle,
            "source_commit": source_commit,
            "source_sha256": source_sha256,
            "recipe": recipe,
            "instance_id": instance_id,
        },
        timeout,
        before=assembly,
    )
