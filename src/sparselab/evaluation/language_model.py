from __future__ import annotations

import math

import torch
from torch.nn import functional

from sparselab.data.packing import TokenBlockDataset


def evaluate(
    model: torch.nn.Module,
    dataset: TokenBlockDataset,
    *,
    batch_size: int,
    max_batches: int,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, min(len(dataset), batch_size * max_batches), batch_size):
            batch = [
                dataset[i] for i in range(start, min(start + batch_size, len(dataset)))
            ]
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
                reduction="sum",
            )
            total += float(loss)
            count += y.numel()
    model.train(was_training)
    loss = total / count
    return {"loss": loss, "perplexity": math.exp(loss)}
