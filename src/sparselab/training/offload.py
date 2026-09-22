"""Saved-tensor activation offload hooks for supported discrete accelerators."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OffloadMetrics:
    bytes_to_cpu: int = 0
    bytes_to_device: int = 0
    peak_host_bytes: int = 0
    live_host_bytes: int = 0


class ActivationOffload:
    def __init__(self, device: torch.device) -> None:
        if device.type == "cpu":
            raise ValueError("activation offload is meaningless on CPU")
        self.device = device
        self.metrics = OffloadMetrics()

    def pack(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.dtype, torch.device]:
        if tensor.is_leaf or tensor.device.type == "cpu":
            return tensor, tensor.dtype, tensor.device
        host = tensor.detach().to("cpu").contiguous()
        size = host.numel() * host.element_size()
        self.metrics.bytes_to_cpu += size
        self.metrics.live_host_bytes += size
        self.metrics.peak_host_bytes = max(
            self.metrics.peak_host_bytes, self.metrics.live_host_bytes
        )
        return host, tensor.dtype, tensor.device

    def unpack(
        self, packed: tuple[torch.Tensor, torch.dtype, torch.device]
    ) -> torch.Tensor:
        host, dtype, device = packed
        if device.type == "cpu":
            return host
        size = host.numel() * host.element_size()
        self.metrics.bytes_to_device += size
        self.metrics.live_host_bytes -= size
        return host.to(device=device, dtype=dtype)

    def hooks(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
