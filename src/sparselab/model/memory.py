"""Local causal token n-gram memory adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MemoryDiagnostics:
    lookup_count: Tensor
    unique_addresses: Tensor
    collision_count: Tensor
    bucket_reuse_rate: Tensor
    table_utilization: Tensor
    maximum_address_fraction: Tensor
    gate_mean: Tensor
    value_norm: Tensor
    hidden_norm: Tensor


def _diagnostics(
    addresses: Tensor,
    gate: Tensor,
    values: Tensor,
    hidden: Tensor,
    table_size: int,
) -> MemoryDiagnostics:
    counts = torch.bincount(addresses.flatten(), minlength=table_size)
    lookup_count = torch.tensor(addresses.numel(), device=addresses.device)
    unique = (counts > 0).sum()
    collisions = lookup_count - unique
    denominator = lookup_count.clamp_min(1)
    return MemoryDiagnostics(
        lookup_count=lookup_count.detach(),
        unique_addresses=unique.detach(),
        collision_count=collisions.detach(),
        bucket_reuse_rate=(collisions.float() / denominator).detach(),
        table_utilization=(unique.float() / table_size).detach(),
        maximum_address_fraction=(counts.max().float() / denominator).detach(),
        gate_mean=gate.mean().detach(),
        value_norm=values.norm(dim=-1).mean().detach(),
        hidden_norm=hidden.norm(dim=-1).mean().detach(),
    )


class TokenNgramMemory(nn.Module):
    """Map causal token-ID suffixes into trainable memory values."""

    def __init__(
        self,
        hidden_dim: int,
        table_size: int,
        ngram_size: int,
        value_dim: int,
        ngram_orders: tuple[int, ...] = (),
        hash_heads: int = 1,
    ) -> None:
        super().__init__()
        self.table_size = table_size
        self.ngram_size = ngram_size
        self.ngram_orders = ngram_orders or (ngram_size,)
        self.hash_heads = hash_heads
        self.table = nn.Embedding(table_size, value_dim)
        self.extra_tables = nn.ModuleList(
            nn.Embedding(table_size, value_dim)
            for _ in range(len(self.ngram_orders) * hash_heads - 1)
        )
        self.output = nn.Linear(value_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, 1, bias=False)
        self.last_diagnostics: MemoryDiagnostics | None = None

    def addresses(
        self, input_ids: Tensor, order: int | None = None, seed: int = 0
    ) -> Tensor:
        """Hash only IDs at or before each position for one order/hash head."""
        batch, length = input_ids.shape
        address = torch.full(
            (batch, length), seed + 1, dtype=torch.long, device=input_ids.device
        )
        order = self.ngram_size if order is None else order
        for offset in range(order):
            shifted = torch.zeros_like(input_ids)
            if offset == 0:
                shifted = input_ids
            elif offset < length:
                shifted[:, offset:] = input_ids[:, :-offset]
            address = torch.remainder(
                address * (257 + seed * 2) + shifted, self.table_size
            )
        return address

    def forward(self, hidden: Tensor, input_ids: Tensor) -> Tensor:
        addresses = [
            self.addresses(input_ids, order, head)
            for order in self.ngram_orders
            for head in range(self.hash_heads)
        ]
        tables = (self.table, *self.extra_tables)
        values = torch.stack(
            [
                self.output(table(address))
                for table, address in zip(tables, addresses, strict=True)
            ]
        ).mean(dim=0)
        gate = torch.sigmoid(self.gate(hidden))
        self.last_diagnostics = _diagnostics(
            torch.cat([address.flatten() for address in addresses]),
            gate,
            values,
            hidden,
            self.table_size,
        )
        return hidden + gate * values


class ByteAddressMemory(nn.Module):
    """Apply precomputed causal byte addresses supplied by prepared data."""

    def __init__(self, hidden_dim: int, table_size: int, value_dim: int) -> None:
        super().__init__()
        self.table_size = table_size
        self.table = nn.Embedding(table_size, value_dim)
        self.output = nn.Linear(value_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, 1, bias=False)
        self.last_diagnostics: MemoryDiagnostics | None = None

    def forward(self, hidden: Tensor, addresses: Tensor) -> Tensor:
        if addresses.shape != hidden.shape[:2]:
            raise ValueError("byte addresses must have shape [batch, sequence]")
        if addresses.dtype != torch.long:
            raise ValueError("byte addresses must be int64")
        if addresses.numel() and (
            addresses.min() < 0 or addresses.max() >= self.table_size
        ):
            raise ValueError("byte address is outside configured memory table")
        values = self.output(self.table(addresses))
        gate = torch.sigmoid(self.gate(hidden))
        self.last_diagnostics = _diagnostics(
            addresses, gate, values, hidden, self.table_size
        )
        return hidden + gate * values
