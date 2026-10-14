"""Fence work-root initialization before monitor handlers; no command execution."""

import sys

import pytest


@pytest.mark.parametrize("verb", ["monitor", "monitor-baseline"])
@pytest.mark.parametrize("selector", ["environment", "global-prefix"])
def test_monitor_initialization_honors_explicit_work_root(
    tmp_path, monkeypatch, verb, selector
):
    from sparselab.cli import main as cli

    selected = tmp_path / "selected"
    ambient = tmp_path / "ambient"
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(ambient))
    monkeypatch.setenv("SPARSELAB_WORKER_LEASE_DIR", str(selected / "worker-leases"))
    prefix = []
    if selector == "environment":
        monkeypatch.setenv("SPARSELAB_WORK_DIR", str(selected))
    else:
        prefix = ["--work-dir", str(selected)]
    # Deliberately different: monitored workspace is not the global work root.
    monitored = tmp_path / "monitored"
    if verb == "monitor":
        args = [
            verb,
            "--policy",
            "unused.json",
            "--workspace",
            str(monitored),
            "--log-dir",
            str(monitored / "logs"),
            "--",
            "never-execute",
        ]
    else:
        args = [verb, str(monitored), "--output", str(monitored / "baseline.json")]
    monkeypatch.setattr(sys, "argv", ["sparselab", *prefix, *args])
    monkeypatch.setattr(cli, "set_token_file", lambda _: None)
    monkeypatch.setattr(cli, "_command_storage_checks", lambda _: [])

    class BeforeHandler(Exception):
        pass

    observed = []

    def ensure(path):
        observed.append(path)
        raise BeforeHandler

    monkeypatch.setattr(cli, "ensure_work_dir", ensure)
    with pytest.raises(BeforeHandler):
        cli.main()
    assert observed == [selected]
    assert list(tmp_path.iterdir()) == []
