"""Mock-only checks for the public cumulative attempt-contract adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparselab.cli.main import build_parser
from sparselab.training import attempt_contract_cli as cli

_SHA = "a" * 64


def _run_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "attempt",
        "run",
        "--ledger",
        str(tmp_path / "ledger.sqlite"),
        "--label",
        "phase_1",
        "--activity",
        "evaluate",
        "--content-identity-sha256",
        _SHA,
        "--policy",
        str(tmp_path / "policy.json"),
        "--baseline",
        str(tmp_path / "baseline.json"),
        "--workspace",
        str(tmp_path),
        "--completion",
        str(tmp_path / "completion.json"),
        "--updates",
        "0",
        "--target-positions",
        "0",
        "--generation-calls",
        "0",
        "--generated-tokens",
        "0",
        *extra,
        "--",
        "sparselab",
        "evidence",
    ]


def test_init_and_status_use_existing_budget_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[object] = []

    class FakeBudget:
        def __init__(self, path: Path):
            calls.append(("open", path))

        @classmethod
        def create_contract(
            cls, path: Path, *, contract_path: Path, expected_sha256: str
        ) -> FakeBudget:
            calls.append(("init", path, contract_path, expected_sha256))
            return cls(path)

        def status(self) -> dict[str, int]:
            return {"charged_updates": 0}

    monkeypatch.setattr(cli, "AttemptBudget", FakeBudget)
    parser = build_parser(tmp_path)
    contract = tmp_path / "contract.json"
    args = parser.parse_args(
        [
            "attempt",
            "init",
            "--ledger",
            str(tmp_path / "ledger.sqlite"),
            "--contract",
            str(contract),
            "--contract-sha256",
            _SHA,
        ]
    )
    args.handler(args)
    assert json.loads(capsys.readouterr().out) == {"charged_updates": 0}
    assert calls[0] == ("init", tmp_path / "ledger.sqlite", contract, _SHA)
    args = parser.parse_args(
        ["attempt", "status", "--ledger", str(tmp_path / "ledger.sqlite")]
    )
    args.handler(args)
    assert json.loads(capsys.readouterr().out) == {"charged_updates": 0}


def test_run_passes_exact_native_and_campaign_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    class FakeBudget:
        def __init__(self, path: Path):
            assert path == tmp_path / "ledger.sqlite"

        def run_contract(self, command: list[str], **kwargs: object) -> int:
            calls.append((command, kwargs))
            return 0

    monkeypatch.setattr(cli, "AttemptBudget", FakeBudget)
    args = build_parser(tmp_path).parse_args(
        _run_args(
            tmp_path,
            "--generation-calls",
            "2",
            "--generated-tokens",
            "128",
            "--receipt-kind",
            "campaign_panel",
            "--receipt-path",
            str(tmp_path / "campaign.yaml"),
            "--campaign-stage",
            "panel_1",
            "--campaign-work-dir",
            str(tmp_path / "campaign-work"),
            "--parent-checkpoint",
            str(tmp_path / "parent-checkpoint"),
        )
    )
    args.handler(args)
    assert calls == [
        (
            ["sparselab", "evidence"],
            {
                "activity": "evaluate",
                "label": "phase_1",
                "content_identity_sha256": _SHA,
                "monitor_policy_path": tmp_path / "policy.json",
                "workspace_baseline_path": tmp_path / "baseline.json",
                "workspace_root": tmp_path,
                "completion": tmp_path / "completion.json",
                "updates": 0,
                "target_positions": 0,
                "generation_calls": 2,
                "generated_tokens": 128,
                "native_receipt_kind": "campaign_panel",
                "native_receipt_path": tmp_path / "campaign.yaml",
                "campaign_stage": "panel_1",
                "campaign_work_dir": tmp_path / "campaign-work",
                "parent_checkpoint_path": tmp_path / "parent-checkpoint",
            },
        )
    ]


@pytest.mark.parametrize(
    "extra,reason",
    [
        (["--receipt-kind", "none", "--generation-calls", "1"], "native receipt"),
        (["--receipt-kind", "panel"], "receipt path"),
        (["--receipt-kind", "campaign_run", "--receipt-path", "/x"], "campaign-stage"),
        (
            [
                "--receipt-kind",
                "campaign_run",
                "--receipt-path",
                "/x",
                "--campaign-stage",
                "run_1",
            ],
            "campaign-work-dir",
        ),
        (["--receipt-kind", "none", "--updates", "1"], "native receipt"),
        (["--receipt-kind", "none", "--target-positions", "1"], "native receipt"),
        (["--receipt-kind", "none", "--generated-tokens", "1"], "native receipt"),
        (["--receipt-kind", "none", "--updates", "-1"], "nonnegative"),
    ],
)
def test_invalid_receipt_binding_refuses_before_budget_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
    reason: str,
) -> None:
    def unexpected_budget(path: Path) -> None:
        pytest.fail(f"budget opened before rejecting invalid binding: {path}")

    monkeypatch.setattr(cli, "AttemptBudget", unexpected_budget)
    args = build_parser(tmp_path).parse_args(_run_args(tmp_path, *extra))
    with pytest.raises(ValueError, match=reason):
        args.handler(args)


def test_failed_native_phase_exit_is_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeBudget:
        def __init__(self, path: Path):
            pass

        def run_contract(self, command: list[str], **kwargs: object) -> int:
            return 7

    monkeypatch.setattr(cli, "AttemptBudget", FakeBudget)
    args = build_parser(tmp_path).parse_args(
        _run_args(
            tmp_path,
            "--receipt-kind",
            "train",
            "--receipt-path",
            str(tmp_path / "run"),
        )
    )
    with pytest.raises(SystemExit) as error:
        args.handler(args)
    assert error.value.code == 7
