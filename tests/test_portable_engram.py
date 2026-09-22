from __future__ import annotations

import pytest
import torch

from sparselab.model.portable_engram import (
    PortableEngramAdapter,
    export_portable_engram,
    load_portable_engram,
)


def test_portable_package_freezes_table_and_trains_adapter(tmp_path) -> None:
    table = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    path = tmp_path / "memory.engram"
    manifest = export_portable_engram(table, path, ngram_size=3)
    package = load_portable_engram(path)
    assert package.manifest == manifest
    adapter = PortableEngramAdapter(package, hidden_dim=8)
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    output = adapter(hidden, torch.tensor([[0, 1, 2], [3, 4, 5]]))
    output.square().mean().backward()
    assert adapter.embedding.weight.requires_grad is False
    assert adapter.output.weight.grad is not None
    assert adapter.last_diagnostics is not None
    assert int(adapter.last_diagnostics.lookup_count) == 6


def test_portable_package_rejects_conflicting_rewrite(tmp_path) -> None:
    path = tmp_path / "memory.engram"
    export_portable_engram(torch.zeros(4, 2), path, ngram_size=2)
    with pytest.raises(FileExistsError, match="conflicting"):
        export_portable_engram(torch.ones(4, 2), path, ngram_size=2)
