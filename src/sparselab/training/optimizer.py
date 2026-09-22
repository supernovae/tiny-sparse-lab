from __future__ import annotations

import math

import torch
from torch import nn


def learning_rate_for_step(
    step: int, max_steps: int, warmup_steps: int, peak: float, floor: float
) -> float:
    if warmup_steps and step <= warmup_steps:
        return peak * step / warmup_steps
    fraction = (
        ((step - warmup_steps) / (max_steps - warmup_steps))
        if warmup_steps
        else ((step - 1) / max(1, max_steps - 1))
    )
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * fraction))


def make_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> torch.optim.AdamW:
    decay, no_decay, seen = [], [], set()
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=betas,
        eps=eps,
        foreach=False,
    )


def make_adafactor(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    beta2_decay: float,
    eps: tuple[float | None, float],
    d: float,
) -> torch.optim.Adafactor:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.Adafactor(
        parameters,
        lr=learning_rate,
        beta2_decay=beta2_decay,
        eps=eps,
        d=d,
        weight_decay=weight_decay,
        foreach=False,
    )
