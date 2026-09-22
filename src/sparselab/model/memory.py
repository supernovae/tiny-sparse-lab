"""Local causal token n-gram memory adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MemoryDiagnostics:
    unique_addresses: Tensor
    maximum_address_fraction: Tensor


class TokenNgramMemory(nn.Module):
    """Map causal token-ID suffixes into trainable memory values."""

    def __init__(
        self, hidden_dim: int, table_size: int, ngram_size: int, value_dim: int
    ) -> None:
        super().__init__()
        self.table_size = table_size
        self.ngram_size = ngram_size
        self.table = nn.Embedding(table_size, value_dim)
        self.output = nn.Linear(value_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, 1, bias=False)
        self.last_diagnostics: MemoryDiagnostics | None = None

    def addresses(self, input_ids: Tensor) -> Tensor:
        """Hash only IDs at or before each position, left-padding early suffixes with zero."""
        batch, length = input_ids.shape
        address = torch.zeros(
            (batch, length), dtype=torch.long, device=input_ids.device
        )
        for offset in range(self.ngram_size):
            shifted = torch.zeros_like(input_ids)
            if offset == 0:
                shifted = input_ids
            elif offset < length:
                shifted[:, offset:] = input_ids[:, :-offset]
            address = torch.remainder(address * 257 + shifted, self.table_size)
        return address

    def forward(self, hidden: Tensor, input_ids: Tensor) -> Tensor:
        address = self.addresses(input_ids)
        values = self.output(self.table(address))
        gate = torch.sigmoid(self.gate(hidden))
        counts = torch.bincount(address.flatten(), minlength=self.table_size)
        self.last_diagnostics = MemoryDiagnostics(
            (counts > 0).sum().detach(),
            (counts.max().float() / address.numel()).detach(),
        )
        return hidden + gate * values
