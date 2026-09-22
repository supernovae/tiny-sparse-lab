"""Device selection, timing, memory, and reproducibility primitives."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil
import torch


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    selected: str
    mps_built: bool
    mps_available: bool
    cuda_available: bool


def device_info(requested: str) -> DeviceInfo:
    return DeviceInfo(
        requested=requested,
        selected=str(select_device(requested)),
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
        cuda_available=torch.cuda.is_available(),
    )


def select_device(requested: str) -> torch.device:
    """Select a requested device; explicit unavailable accelerators never fall back."""
    available = {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }
    if requested == "auto":
        for name in ("mps", "cuda", "cpu"):
            if available[name]:
                return torch.device(name)
    if requested not in available:
        raise ValueError(
            f"unknown device {requested!r}; choose auto, mps, cuda, or cpu"
        )
    if not available[requested]:
        if requested == "mps":
            raise RuntimeError(
                "MPS requested but unavailable "
                f"(built={torch.backends.mps.is_built()}, available={torch.backends.mps.is_available()})"
            )
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def allocated_memory_bytes(device: torch.device) -> int | None:
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    return None


def process_rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def seed_everything(seed: int, *, deterministic_cpu: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic_cpu:
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)


def capture_rng_state() -> dict[str, Any]:
    """Capture all locally relevant RNG streams. Accelerator states are optional."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
