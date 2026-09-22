"""Optional dense MLX engine capability gate."""

from __future__ import annotations

from importlib.util import find_spec

from sparselab.config.models import RunConfig
from sparselab.runtime import RuntimeInfo, discover_runtimes


def validate(config: RunConfig) -> RuntimeInfo:
    if config.runtime.engine != "mlx" or config.runtime.backend != "metal":
        raise ValueError("MLX engine requires runtime.engine=mlx and backend=metal")
    if (
        config.model.ffn != "dense"
        or config.model.memory != "none"
        or config.attention.kind != "dense"
    ):
        raise ValueError("MLX currently supports dense attention and dense FFN only")
    if config.runtime.precision not in {"fp32", "auto"}:
        raise ValueError("MLX precision is currently fp32 only")
    if find_spec("mlx.core") is None:
        raise ValueError("MLX runtime is unavailable")

    return next(info for info in discover_runtimes() if info.engine == "mlx")
