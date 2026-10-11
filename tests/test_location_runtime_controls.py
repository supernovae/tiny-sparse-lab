"""Cross-contract checks for the retained location/runtime selectors."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sparselab.explore_config import load_explore_config
from sparselab.probes.config import load_probe_config
from sparselab.try_config import load_try_config


@pytest.mark.parametrize(
    ("loader", "fields"),
    [
        (load_try_config, ("lab_dir", "runtime_profile", "resource_envelope")),
        (
            load_probe_config,
            ("lab_dir", "runs_dir", "runtime_profile", "resource_envelope"),
        ),
        (load_explore_config, ("lab_dir", "runs_dir")),
    ],
)
def test_declaration_locations_do_not_follow_cwd_or_work_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader, fields: tuple[str, ...]
) -> None:
    source = tmp_path / "declarations" / "request.yaml"
    source.parent.mkdir()
    source.write_text("\n".join(f"{field}: ../retained/{field}" for field in fields))
    expected = {field: tmp_path / "retained" / field for field in fields}
    before = source.read_bytes()
    for name in ("first", "second"):
        cwd = tmp_path / name
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        monkeypatch.setenv("SPARSELAB_WORK_DIR", str(cwd / "work"))
        config = loader(str(source) if loader != load_explore_config else source)
        assert {field: getattr(config, field) for field in fields} == expected
        assert not (cwd / "work").exists()
    assert source.read_bytes() == before
    assert not (tmp_path / "retained").exists()


def test_runtime_registry_and_install_root_do_not_follow_scientific_work_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.runtime_environments import (
        RuntimeEntry,
        profile_for_id,
        register_runtime,
        resolve_registry_path,
        resolve_runtime_dir,
    )

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("SPARSELAB_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "first-work"))
    register_runtime(
        "retained-cpu",
        RuntimeEntry(
            python=Path(sys.executable).absolute(),
            engine="pytorch",
            backend="cpu",
            device_index=0,
        ),
    )
    registry = resolve_registry_path()
    before = registry.read_bytes()
    profile = profile_for_id("retained-cpu")
    for name in ("first-work", "second-work"):
        monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / name))
        assert resolve_registry_path() == registry
        assert resolve_runtime_dir() == tmp_path / "data/sparselab/runtimes"
        assert profile_for_id("retained-cpu") == profile
        assert not (tmp_path / name).exists()
    assert registry.read_bytes() == before
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    "command",
    [
        "list",
        "discover",
        "show",
        "register",
        "unregister",
        "doctor",
        "profile",
        "provision",
    ],
)
def test_all_environment_help_routes_avoid_framework_and_execution(
    tmp_path: Path, command: str
) -> None:
    # A fresh process catches eager imports hidden by pytest's application imports.
    script = """
import importlib.abc
import sys

class RefuseExecutionImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname == 'sparselab.cli.main':
            raise AssertionError('environment help imported execution: ' + fullname)

sys.meta_path.insert(0, RefuseExecutionImports())
def refuse_children(event, args):
    if event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec'}:
        raise AssertionError('environment help started execution: ' + event)
sys.addaudithook(refuse_children)
from sparselab.cli.entry import main
sys.argv = ['sparselab', 'runtime', 'env', sys.argv[1], '--help']
main()
"""
    settings = {
        "SPARSELAB_WORK_DIR": str(tmp_path / "work"),
        "SPARSELAB_RUNTIME_DIR": str(tmp_path / "runtimes"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
    }
    result = subprocess.run(
        [sys.executable, "-c", script, command],
        env={**os.environ, **settings, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"sparselab runtime env {command}" in result.stdout
    assert all(not Path(path).exists() for path in settings.values())
