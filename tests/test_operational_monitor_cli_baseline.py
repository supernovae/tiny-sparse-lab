"""Zero-model CLI coverage for the existing persistent monitor baseline API."""

import json
from pathlib import Path
from types import SimpleNamespace

from sparselab import operational_monitor_cli
from sparselab.cli.main import build_parser
from sparselab.operational_monitor import load_workspace_baseline


def test_capture_baseline_cli_uses_native_persistent_receipt(tmp_path, capsys):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "retained.txt").write_text("retained")
    output = tmp_path / "baseline.json"
    args = build_parser().parse_args(
        [
            "monitor-baseline",
            str(root),
            "--output",
            str(output),
            "--seconds",
            "2",
            "--json",
        ]
    )
    args.handler(args)
    result = json.loads(capsys.readouterr().out)
    assert result["sha256"] == load_workspace_baseline(output, root).sha256
    assert result["apparent_bytes"] == len("retained")
    assert result["inodes"] >= 2


def test_monitor_cli_forwards_baseline_without_changing_legacy_optional_path(
    monkeypatch,
):
    observed = []

    def monitor(command, policy, **kwargs):
        observed.append((command, kwargs["baseline_path"]))
        return SimpleNamespace(
            status="COMPLETE", returncode=0, model_dump=lambda **_: {}
        )

    monkeypatch.setattr(operational_monitor_cli, "monitor_command", monitor)
    monkeypatch.setattr(operational_monitor_cli, "load_monitor_policy", lambda _: None)
    parser = build_parser()
    common = [
        "monitor",
        "--policy",
        "/policy",
        "--log-dir",
        "/logs",
        "--workspace",
        "/workspace",
    ]
    with_baseline = parser.parse_args(
        [*common, "--baseline", "/baseline", "--", "echo", "safe"]
    )
    without_baseline = parser.parse_args([*common, "--", "echo", "safe"])
    assert operational_monitor_cli.execute_monitor(with_baseline) == 0
    assert operational_monitor_cli.execute_monitor(without_baseline) == 0
    assert observed == [(["echo", "safe"], Path("/baseline")), (["echo", "safe"], None)]
