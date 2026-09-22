"""Runtime discovery, device selection, measurements, and reproducibility helpers."""

from __future__ import annotations

import platform
import random
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

import numpy as np
import psutil
import torch


@dataclass(frozen=True)
class RuntimeInfo:
    engine: str
    backend: str
    torch_device: str | None
    device_index: int
    device_name: str | None
    physical_device_id: str | None
    framework_version: str | None
    runtime_version: str | None
    driver_version: str | None
    os: str
    system_total_bytes: int | None
    system_available_bytes: int | None
    device_total_bytes: int | None
    device_free_bytes: int | None
    device_recommended_bytes: int | None
    measurement_source: str | None
    measured_at: str
    precision_capabilities: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    device_driver_allocated_bytes: int | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ram() -> tuple[int, int]:
    memory = psutil.virtual_memory()
    return int(memory.total), int(memory.available)


def _backend_available(backend: str) -> bool:
    if backend == "cpu":
        return True
    if backend == "mps":
        return bool(torch.backends.mps.is_available())
    if backend in {"cuda", "rocm"}:
        return bool(torch.cuda.is_available()) and (
            (backend == "rocm") == bool(torch.version.hip)
        )
    if backend == "xpu":
        xpu = getattr(torch, "xpu", None)
        return bool(xpu and xpu.is_available())
    return False


def _auto_backend() -> str:
    if _backend_available("mps"):
        return "mps"
    if torch.cuda.is_available():
        return "rocm" if torch.version.hip else "cuda"
    if _backend_available("xpu"):
        return "xpu"
    return "cpu"


def torch_device_for(backend: str, device_index: int = 0) -> torch.device:
    if backend in {"cuda", "rocm"}:
        return torch.device("cuda", device_index)
    if backend == "xpu":
        return torch.device("xpu", device_index)
    if backend == "mps":
        return torch.device("mps")
    if backend == "cpu":
        return torch.device("cpu")
    raise ValueError(f"backend has no PyTorch device: {backend}")


def discover_runtimes() -> list[RuntimeInfo]:
    """Passively inventory known runtimes; this never allocates a model."""
    total, available = _ram()
    infos: list[RuntimeInfo] = []
    for backend in ("cpu", "mps", "cuda", "rocm", "xpu"):
        supported = _backend_available(backend)
        if backend in {"cuda", "rocm"} and (
            (backend == "rocm") != bool(torch.version.hip)
        ):
            reason = "PyTorch build targets the other CUDA/HIP API"
        elif not supported:
            reason = "runtime unavailable"
        else:
            reason = ""
        name = None
        device_total = device_free = recommended = None
        driver_allocated = None
        source = None
        if supported and backend in {"cuda", "rocm"}:
            properties = torch.cuda.get_device_properties(0)
            name, device_total = properties.name, int(properties.total_memory)
            free, _ = torch.cuda.mem_get_info(0)
            device_free, source = int(free), "torch.cuda.mem_get_info"
        elif supported and backend == "mps":
            name, source = "Apple Metal", "torch.mps"
            mps = torch.mps
            if hasattr(mps, "recommended_max_memory"):
                recommended = int(mps.recommended_max_memory())
            if hasattr(mps, "driver_allocated_memory"):
                driver_allocated = int(mps.driver_allocated_memory())
        elif supported and backend == "xpu":
            xpu = torch.xpu
            name = (
                xpu.get_device_name(0)
                if hasattr(xpu, "get_device_name")
                else "Intel XPU"
            )
            if hasattr(xpu, "get_device_properties"):
                device_total = int(xpu.get_device_properties(0).total_memory)
        infos.append(
            RuntimeInfo(
                "pytorch",
                backend,
                str(torch_device_for(backend)) if supported else None,
                0,
                name,
                None,
                torch.__version__,
                torch.version.hip or torch.version.cuda,
                None,
                platform.platform(),
                total,
                available,
                device_total,
                device_free,
                recommended,
                source,
                _now(),
                ("fp32",) if supported else (),
                (reason,) if reason else (),
                device_driver_allocated_bytes=driver_allocated,
            )
        )
    try:
        import mlx.core as mx  # type: ignore[import-not-found]

        infos.append(
            RuntimeInfo(
                "mlx",
                "metal",
                None,
                0,
                "Apple Metal",
                None,
                None,
                getattr(mx, "__version__", None),
                None,
                platform.platform(),
                total,
                available,
                None,
                None,
                None,
                "mlx",
                _now(),
                ("fp32",),
                (),
            )
        )
    except ImportError:
        infos.append(
            RuntimeInfo(
                "mlx",
                "metal",
                None,
                0,
                None,
                None,
                None,
                None,
                None,
                platform.platform(),
                total,
                available,
                None,
                None,
                None,
                None,
                _now(),
                (),
                ("MLX package unavailable",),
            )
        )
    return infos


def validate_runtime(config: Any) -> RuntimeInfo:
    """Validate the explicit backend with a disposable small optimizer probe."""
    runtime = config.runtime
    if runtime.engine != "pytorch":
        raise ValueError("MLX validation is not implemented in this build")
    backend = _auto_backend() if runtime.backend == "auto" else runtime.backend
    if not _backend_available(backend):
        raise ValueError(f"requested backend unavailable: {backend}")
    if runtime.precision not in {"fp32", "auto"}:
        raise ValueError(
            "PyTorch training currently executes FP32 only; mixed precision is not implemented"
        )
    if backend in {"cpu", "mps"} and runtime.device_index != 0:
        raise ValueError(f"{backend} supports only device_index=0")
    device = torch_device_for(backend, runtime.device_index)
    # Probe includes a real backward/update and does not silently retry elsewhere.
    parameter = torch.nn.Parameter(torch.ones((2, 2), device=device))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    (parameter @ parameter).float().sum().backward()
    optimizer.step()
    for info in discover_runtimes():
        if info.engine == "pytorch" and info.backend == backend:
            if backend in {"cuda", "rocm"}:
                properties = torch.cuda.get_device_properties(runtime.device_index)
                free, total = torch.cuda.mem_get_info(runtime.device_index)
                return replace(
                    info,
                    device_index=runtime.device_index,
                    torch_device=str(device),
                    device_name=properties.name,
                    device_total_bytes=int(total),
                    device_free_bytes=int(free),
                )
            if backend == "xpu":
                return replace(
                    info,
                    device_index=runtime.device_index,
                    torch_device=str(device),
                    device_name=torch.xpu.get_device_name(runtime.device_index),
                )
            return info
    raise RuntimeError("validated runtime disappeared from inventory")


def select_device(requested: str) -> torch.device:
    backend = _auto_backend() if requested == "auto" else requested
    if backend not in {"cpu", "mps", "cuda", "rocm", "xpu"} or not _backend_available(
        backend
    ):
        raise ValueError(f"requested device unavailable: {requested}")
    return torch_device_for(backend)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)


def allocated_memory_bytes(device: torch.device) -> int | None:
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return int(torch.mps.current_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    if device.type == "xpu" and hasattr(torch.xpu, "memory_allocated"):
        return int(torch.xpu.memory_allocated(device))
    return None


def process_rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def seed_everything(seed: int, *, deterministic_cpu: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cpu:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if (
        "mps" in state
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state["mps"])
