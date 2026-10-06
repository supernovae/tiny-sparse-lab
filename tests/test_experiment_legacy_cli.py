"""Retirement gates authoring and dispatch without disabling historical readers."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest
import yaml
from test_experiment_story_inputs import story_plan

from sparselab.experiments import cli
from sparselab.experiments.lock import open_lock, publish_lock, resolve_plan


@pytest.mark.parametrize("source", ["local_stories", "tinystories"])
@pytest.mark.parametrize("command", ["lock", "prepare", "bind-inputs", "run"])
def test_retired_experiment_commands_reject_before_publication(
    tmp_path, monkeypatch, source, command
):
    plan, declaration, prepared = story_plan(tmp_path, monkeypatch, source)
    monkeypatch.setattr(cli, "_verification", lambda args: {})
    args = Namespace(source=str(declaration), max_runs=10)
    if command == "bind-inputs":
        run = tmp_path / "run.yaml"
        run.write_text(yaml.safe_dump(plan.base_run.model_dump(mode="json")))
        args = Namespace(
            run_config=str(run),
            template="unused.yaml",
            prepared_root=prepared.root,
            output=tmp_path / "new.yaml",
        )
    elif command == "run":
        historical = publish_lock(resolve_plan(plan, declaration), tmp_path / "history")
        assert open_lock(historical).cells[0].config.dataset.source == source
        args = Namespace(lock=str(historical), cell=None, phase=None, binding=None)
    handler = getattr(
        cli, "_bind_inputs" if command == "bind-inputs" else f"_{command}"
    )
    with pytest.raises(ValueError, match="retired for new execution"):
        handler(args)
    assert not (tmp_path / "work/experiments").exists()
    assert not (tmp_path / "new.yaml").exists()


def test_lock_checks_effective_cell_sources_before_publishing(tmp_path, monkeypatch):
    from sparselab.experiments import lock

    current = SimpleNamespace(dataset=SimpleNamespace(source="snapshot"))
    retired = SimpleNamespace(dataset=SimpleNamespace(source="tinystories"))
    monkeypatch.setattr(
        cli,
        "_declaration",
        lambda args: (SimpleNamespace(id="test"), tmp_path / "plan.yaml"),
    )
    monkeypatch.setattr(cli, "base_run_config", lambda *args: current)
    monkeypatch.setattr(cli, "_verification", lambda args: {})
    monkeypatch.setattr(
        lock,
        "resolve_plan",
        lambda *args, **kwargs: SimpleNamespace(
            cells=[SimpleNamespace(config=retired)]
        ),
    )
    monkeypatch.setattr(
        lock,
        "publish_lock",
        lambda *args, **kwargs: pytest.fail("published retired cell"),
    )
    with pytest.raises(ValueError, match="retired for new execution"):
        cli._lock(Namespace(max_runs=10))
