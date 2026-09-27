from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sparselab.batch_calibration import _summarize_pilot, batch_candidates
from sparselab.config.loading import load_config
from sparselab.readiness import smoke_readiness
from sparselab.workspace_preflight import check_storage, require_storage

_ROOT = Path(__file__).resolve().parents[1]


def test_workspace_preflight_rejects_bytes_and_inodes_before_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "unpublished" / "run"
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda _: SimpleNamespace(
            f_bavail=100, f_frsize=1024, f_favail=100, f_files=1000
        ),
    )
    check = check_storage(
        destination,
        projected_bytes=101 * 1024,
        projected_inodes=101,
        reserve_bytes=0,
        reserve_inodes=0,
    )
    assert check.status == "insufficient"
    with pytest.raises(OSError, match="preflight failed before publication"):
        require_storage([check])
    assert not destination.parent.exists()


def test_batch_candidates_preserve_effective_batch() -> None:
    config = load_config(_ROOT / "configs" / "scale_dense_160m_cuda.yaml")
    candidates = batch_candidates(config, max_candidates=3)
    assert config.training.micro_batch_size in candidates
    assert len(candidates) == 3
    assert all(
        config.training.micro_batch_size * config.training.gradient_accumulation % micro
        == 0
        for micro in candidates
    )


def test_batch_calibration_rejects_unstable_or_low_headroom() -> None:
    stable = {
        "update_observations": [
            {"targets": 128, "update_seconds": seconds}
            for seconds in (4.0, 3.0, 1.0, 1.1, 0.9, 1.0, 1.0, 1.1)
        ],
        "observed_peak_bytes": 40,
        "runtime": {"backend": "cpu"},
    }
    accepted = _summarize_pilot(stable, 100)
    assert accepted["status"] == "eligible"
    assert accepted["targets_per_second"] > 100
    assert accepted["backend"] == "cpu"

    low_headroom = _summarize_pilot({**stable, "observed_peak_bytes": 95}, 100)
    assert low_headroom["reason"] == "insufficient measured headroom"
    unstable = {
        **stable,
        "update_observations": [
            {"targets": 128, "update_seconds": seconds}
            for seconds in (4.0, 3.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0)
        ],
    }
    assert _summarize_pilot(unstable, 100)["reason"] == "unstable update timing"


def test_readiness_rejects_non_cpu_config_before_creating_output(
    tmp_path: Path,
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    shutil.copyfile(
        _ROOT / "configs/tokenizer_smoke.yaml", configs / "tokenizer_smoke.yaml"
    )
    source = yaml.safe_load((_ROOT / "configs/smoke_cpu.yaml").read_text())
    source["runtime"]["backend"] = "cuda"
    (configs / "smoke_cpu.yaml").write_text(yaml.safe_dump(source))
    output = tmp_path / "new-workspace"

    with pytest.raises(ValueError, match="compatible CPU synthetic"):
        smoke_readiness(configs, output, families=("dense",))
    assert not output.exists()
