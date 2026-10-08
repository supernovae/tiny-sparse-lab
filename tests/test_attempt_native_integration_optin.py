"""Zero-update checks for the opt-in native CPU qualification node."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

NODE = (
    "tests/test_attempt_native_integration.py::test_bounded_cpu_final_mask_and_ledger"
)
CHECKOUT = Path(__file__).resolve().parent.parent


def _pytest(
    tmp_path: Path, env: dict[str, str], *, collect_only: bool = False
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(tmp_path / ("collect" if collect_only else "skipped")),
    ]
    if collect_only:
        command.append("--collect-only")
    command.append(NODE)
    return subprocess.run(
        command,
        cwd=CHECKOUT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_ordinary_pytest_skips_optimizer_node_before_setup(tmp_path: Path) -> None:
    for context in ("none", "flag_only", "root_only", "no_baseline", "no_policy"):
        case = tmp_path / context
        root = case / "qualification"
        root.mkdir(parents=True)
        if context != "no_baseline":
            (root / "baseline.json").write_text("{}")
        if context != "no_policy":
            (root / "outer-policy.json").write_text("{}")
        env = os.environ.copy()
        env.pop("KML_NATIVE_CPU_QUALIFICATION", None)
        env.pop("KML_QUAL_ROOT", None)
        if context in ("flag_only", "no_baseline", "no_policy"):
            env["KML_NATIVE_CPU_QUALIFICATION"] = "1"
        if context in ("root_only", "no_baseline", "no_policy"):
            env["KML_QUAL_ROOT"] = str(root)
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["UV_OFFLINE"] = "1"
        result = _pytest(case, env)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1 skipped" in result.stdout
        assert not (root / "ledger.sqlite").exists()
        assert not (root / "fixture").exists()


def test_harness_context_selects_node_without_running_it(tmp_path: Path) -> None:
    root = tmp_path / "unused-qualification-root"
    root.mkdir()
    (root / "baseline.json").write_text("{}")
    (root / "outer-policy.json").write_text("{}")
    env = os.environ.copy()
    env["KML_NATIVE_CPU_QUALIFICATION"] = "1"
    env["KML_QUAL_ROOT"] = str(root)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["UV_OFFLINE"] = "1"
    marker = subprocess.run(
        [
            sys.executable,
            "-c",
            """import sys
sys.path.insert(0, "tests")
from test_attempt_native_integration import test_bounded_cpu_final_mask_and_ledger as node
print(next(mark.args[0] for mark in node.pytestmark if mark.name == "skipif"))
""",
        ],
        cwd=CHECKOUT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert marker.returncode == 0, marker.stdout + marker.stderr
    assert marker.stdout.strip() == "False"
    selection = _pytest(tmp_path, env, collect_only=True)
    assert selection.returncode == 0, selection.stdout + selection.stderr
    assert "1 test collected" in selection.stdout
    assert not (root / "ledger.sqlite").exists()
    assert not (root / "fixture").exists()
