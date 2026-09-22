"""Tokenizer-agnostic byte-addressed Engram package and backbone adapter."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from sparselab.model.memory import MemoryDiagnostics, _diagnostics

FORMAT_VERSION = 1


@dataclass(frozen=True)
class PortableEngramManifest:
    format_version: int
    normalization: str
    hashing: str
    ngram_size: int
    table_size: int
    embedding_dim: int
    table_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "normalization": self.normalization,
            "hashing": self.hashing,
            "ngram_size": self.ngram_size,
            "table_size": self.table_size,
            "embedding_dim": self.embedding_dim,
            "table_sha256": self.table_sha256,
        }


def _digest(table: Tensor) -> str:
    return hashlib.sha256(
        table.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def export_portable_engram(
    table: Tensor,
    path: Path,
    *,
    ngram_size: int,
    normalization: str = "raw-utf8-v1",
    hashing: str = "poly257-terminal-v1",
) -> PortableEngramManifest:
    """Write immutable latent table weights without a backbone-specific adapter."""
    if table.ndim != 2:
        raise ValueError(
            "portable Engram table must have shape [table_size, embedding_dim]"
        )
    if ngram_size < 1:
        raise ValueError("portable Engram ngram_size must be positive")
    weights = table.detach().cpu().contiguous()
    manifest = PortableEngramManifest(
        FORMAT_VERSION,
        normalization,
        hashing,
        ngram_size,
        weights.shape[0],
        weights.shape[1],
        _digest(weights),
    )
    payload = {"manifest": manifest.as_dict(), "table": weights}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    if path.exists():
        loaded = load_portable_engram(path)
        if loaded.manifest == manifest and torch.equal(loaded.table, weights):
            temporary.unlink(missing_ok=True)
            return manifest
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"conflicting portable Engram package: {path}")
    temporary.replace(path)
    return manifest


class PortableEngram:
    """Validated immutable latent table shared independently of model width."""

    def __init__(self, manifest: PortableEngramManifest, table: Tensor) -> None:
        if table.shape != (manifest.table_size, manifest.embedding_dim):
            raise ValueError("portable Engram table shape conflicts with manifest")
        if _digest(table) != manifest.table_sha256:
            raise ValueError("portable Engram table hash mismatch")
        self.manifest = manifest
        self.table = table


def load_portable_engram(path: Path) -> PortableEngram:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"manifest", "table"}:
        raise ValueError("invalid portable Engram package")
    raw_manifest, table = payload["manifest"], payload["table"]
    if not isinstance(raw_manifest, dict) or not isinstance(table, Tensor):
        raise TypeError("invalid portable Engram package payload")
    manifest = PortableEngramManifest(**raw_manifest)
    if manifest.format_version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported portable Engram version: {manifest.format_version}"
        )
    return PortableEngram(manifest, table)


class PortableEngramAdapter(nn.Module):
    """Frozen portable latent table plus trainable adapter for one backbone."""

    def __init__(self, package: PortableEngram, hidden_dim: int) -> None:
        super().__init__()
        self.table_size = package.manifest.table_size
        self.embedding = nn.Embedding.from_pretrained(package.table, freeze=True)
        self.output = nn.Linear(package.manifest.embedding_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, 1, bias=False)
        self.last_diagnostics: MemoryDiagnostics | None = None

    def forward(self, hidden: Tensor, addresses: Tensor) -> Tensor:
        if addresses.shape != hidden.shape[:2] or addresses.dtype != torch.long:
            raise ValueError(
                "portable addresses must be int64 with shape [batch, sequence]"
            )
        if addresses.numel() and (
            addresses.min() < 0 or addresses.max() >= self.table_size
        ):
            raise ValueError("portable address is outside package table")
        values = self.output(self.embedding(addresses))
        gate = torch.sigmoid(self.gate(hidden))
        self.last_diagnostics = _diagnostics(
            addresses, gate, values, hidden, self.table_size
        )
        return hidden + gate * values
