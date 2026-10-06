"""Queue authoring rejects retired inputs before opening or mutating a store."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from sparselab.workers import cli


@pytest.mark.parametrize("source", ["tinystories", "local_stories"])
@pytest.mark.parametrize("matrix", [False, True])
def test_retired_queue_submission_rejected_before_controller(
    monkeypatch, source, matrix
):
    retired = SimpleNamespace(dataset=SimpleNamespace(source=source))
    current = SimpleNamespace(dataset=SimpleNamespace(source="snapshot"))
    monkeypatch.setattr(cli, "load_config", lambda path: retired)
    monkeypatch.setattr(
        cli,
        "_matrix_requests",
        lambda args: [{"config": current}, {"config": retired}],
    )
    monkeypatch.setattr(
        cli,
        "_controller",
        lambda path: pytest.fail("retired inputs opened a submission store"),
    )
    args = Namespace(
        config=None if matrix else "old.yaml",
        matrix="matrix.yaml" if matrix else None,
        dry_run=False,
    )
    with pytest.raises(ValueError, match="retired for new execution"):
        cli._experiment_submit(args)


def test_historical_matrix_dry_run_remains_read_only(monkeypatch):
    observed = []
    monkeypatch.setattr(cli, "_matrix_requests", lambda args: observed.append(args))
    monkeypatch.setattr(
        cli, "_controller", lambda path: pytest.fail("dry run opened a store")
    )
    args = Namespace(config=None, matrix="historical.yaml", dry_run=True)
    cli._experiment_submit(args)
    assert observed == [args]
