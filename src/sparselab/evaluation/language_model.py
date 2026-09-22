"""Held-out next-token evaluation with exact target accounting."""

from __future__ import annotations

import math

import torch
from torch.nn import functional

from sparselab.data.packing import TokenBlockDataset


def _device_rng_state(device: torch.device) -> torch.Tensor | None:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
    ):
        return torch.mps.get_rng_state()
    if device.type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.get_rng_state(device)  # type: ignore[attr-defined]
    return None


def _restore_device_rng_state(device: torch.device, state: torch.Tensor | None) -> None:
    if state is None:
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    elif (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.set_rng_state(state, device)  # type: ignore[attr-defined]


def evaluate(
    model: torch.nn.Module,
    dataset: TokenBlockDataset,
    *,
    batch_size: int,
    max_batches: int,
    device: torch.device,
) -> dict[str, float | int | str | None]:
    """Evaluate no more than ``max_batches`` and report exactly scored labels."""
    if batch_size <= 0 or max_batches <= 0:
        raise ValueError("batch_size and max_batches must be positive")
    was_training = model.training
    cpu_rng = torch.get_rng_state()
    device_rng = _device_rng_state(device)
    total = 0.0
    count = 0
    batches = 0
    try:
        model.eval()
        with torch.inference_mode():
            limit = min(len(dataset), batch_size * max_batches)
            for start in range(0, limit, batch_size):
                batch = [
                    dataset[i] for i in range(start, min(start + batch_size, limit))
                ]
                if not batch:
                    continue
                x = torch.stack([item[0] for item in batch]).to(device)
                y = torch.stack([item[1] for item in batch]).to(device)
                byte_addresses = (
                    torch.stack([item[2] for item in batch]).to(device)
                    if len(batch[0]) == 3
                    else None
                )
                loss = functional.cross_entropy(
                    model(x, byte_addresses=byte_addresses).flatten(0, 1),
                    y.flatten(),
                    ignore_index=-100,
                    reduction="sum",
                )
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite validation loss")
                valid = int((y != -100).sum())
                if valid:
                    total += float(loss)
                    count += valid
                batches += 1
        if count == 0:
            raise ValueError("validation contains no valid labels")
        loss = total / count
        perplexity: float | None
        if loss > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
            perplexity = None
            reason = "loss exceeds finite exp range"
        else:
            perplexity = math.exp(loss)
            reason = None
        return {
            "loss": loss,
            "perplexity": perplexity,
            "perplexity_unavailable_reason": reason,
            "valid_targets": count,
            "batches": batches,
        }
    finally:
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        _restore_device_rng_state(device, device_rng)
