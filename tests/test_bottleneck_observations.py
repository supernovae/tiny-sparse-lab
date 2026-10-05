"""Operational observations never become prepared/stage authority."""

from __future__ import annotations

import json
from pathlib import Path

from test_training import config as training_config

from sparselab.bottleneck_observations import BottleneckObserver, _classify
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer
from sparselab.staging import _read_sealed, materialize_prepared_inputs, stage


def _labels(**changes: object) -> tuple[str, str]:
    counters = {
        "seconds": 10.0,
        "cpu_seconds": None,
        "io_bytes": None,
        "swap_bytes": None,
        "available_bytes": None,
        "total_bytes": None,
        "accelerator_utilization_percent": None,
        "host_kind": None,
        "cache_event": None,
    }
    counters.update(changes)
    return _classify(**counters)


def test_classifications_require_counters_and_provenance() -> None:
    assert _labels(host_kind="copy_bound", cache_event="hit") == ("unknown", "unknown")
    assert _labels(cpu_seconds=9.0) == ("unknown", "cpu_bound")
    assert _labels(cpu_seconds=1.0, io_bytes=123) == ("unknown", "io_bound")
    assert _labels(cpu_seconds=1.0, io_bytes=123, host_kind="copy_bound") == (
        "unknown",
        "copy_bound",
    )
    assert _labels(cpu_seconds=1.0, host_kind="verification_bound") == (
        "unknown",
        "verification_bound",
    )
    assert _labels(cpu_seconds=1.0, host_kind="serialization_bound") == (
        "unknown",
        "serialization_bound",
    )
    assert _labels(cpu_seconds=0.0, cache_event="miss") == ("unknown", "cache_miss")
    assert _labels(cpu_seconds=0.0, cache_event="hit") == ("unknown", "cache_hit")
    assert _labels(cpu_seconds=9.0, accelerator_utilization_percent=90.0) == (
        "accelerator",
        "cpu_bound",
    )
    assert _labels(cpu_seconds=1.0, accelerator_utilization_percent=10.0) == (
        "input_host",
        "unknown",
    )
    assert _labels(swap_bytes=4096, accelerator_utilization_percent=90.0) == (
        "memory_pressure",
        "unknown",
    )
    assert _labels(available_bytes=2, total_bytes=100) == (
        "memory_pressure",
        "unknown",
    )


def test_missing_counters_and_probe_remain_unmeasured(monkeypatch) -> None:
    import sparselab.bottleneck_observations as observations

    monkeypatch.setattr(
        observations,
        "_snapshot",
        lambda: {
            "cpu": {},
            "io": {},
            "rss": None,
            "swap": None,
            "available": None,
            "total": None,
        },
    )
    observer = BottleneckObserver(accelerator_probe=lambda: None)
    with observer.phase("unavailable", host_kind="verification_bound"):
        pass
    record = observer.records[0]
    assert record["cpu_user_seconds"] is None
    assert record["cpu_system_seconds"] is None
    assert record["process_read_bytes"] is None
    assert record["process_write_bytes"] is None
    assert record["observed_tree_rss_max_bytes"] is None
    assert record["observed_tree_swap_max_bytes"] is None
    assert record["accelerator_utilization_percent"] is None
    assert record["host_bottleneck"] == "unknown"
    assert record["accelerator_bottleneck"] == "unknown"


def test_endpoint_deltas_and_sampled_memory_are_not_lifetime_peaks(
    monkeypatch,
) -> None:
    import sparselab.bottleneck_observations as observations

    samples = iter(
        [
            {
                "cpu": {(10, 1.0): (1.0, 2.0)},
                "io": {(10, 1.0): (100, 200)},
                "rss": 100,
                "swap": 0,
                "available": 900,
                "total": 1000,
            },
            {
                "cpu": {(10, 1.0): (1.2, 2.1), (11, 2.0): (20.0, 10.0)},
                "io": {(10, 1.0): (150, 280), (11, 2.0): (100, 100)},
                "rss": 200,
                "swap": 0,
                "available": 800,
                "total": 1000,
            },
        ]
    )
    monkeypatch.setattr(observations, "_snapshot", lambda: next(samples))
    observer = BottleneckObserver(accelerator_probe=lambda: 90.0)
    with observer.phase("copy", host_kind="copy_bound"):
        pass
    record = observer.records[0]
    assert round(record["cpu_user_seconds"], 6) == 0.2
    assert round(record["cpu_system_seconds"], 6) == 0.1
    assert record["cpu_matched_processes"] == 1
    assert record["process_read_bytes"] == 50
    assert record["process_write_bytes"] == 80
    assert record["observed_tree_rss_max_bytes"] == 200
    assert record["memory_sampling"] == "phase_endpoints_not_lifetime_peak"
    assert record["accelerator_bottleneck"] == "accelerator"
    assert record["host_bottleneck"] == "copy_bound"


def test_observation_failure_preserves_real_preparation_identity(
    tmp_path: Path, monkeypatch
) -> None:
    import sparselab.bottleneck_observations as observations

    config = training_config(tmp_path)
    expected = prepare_data(config, load_tokenizer(config.tokenizer.path))

    def unavailable() -> None:
        raise RuntimeError("optional sampling unavailable")

    monkeypatch.setattr(observations, "_snapshot", unavailable)
    observer = BottleneckObserver(accelerator_probe=unavailable)
    actual = prepare_data(
        config, load_tokenizer(config.tokenizer.path), observer=observer
    )
    assert actual.manifest == expected.manifest
    assert actual.receipt.manifest_sha256 == expected.receipt.manifest_sha256
    assert observer.records[0]["accelerator_utilization_percent"] is None


def test_preparation_and_stage_records_are_outside_sealed_identities(
    tmp_path: Path,
) -> None:
    config = training_config(tmp_path)
    observer = BottleneckObserver()
    prepared = prepare_data(
        config, load_tokenizer(config.tokenizer.path), observer=observer
    )
    manifest_before = (prepared.root / "manifest.json").read_bytes()
    prepared_again = prepare_data(
        config, load_tokenizer(config.tokenizer.path), observer=observer
    )
    assert prepared_again.root == prepared.root
    assert (prepared.root / "manifest.json").read_bytes() == manifest_before
    assert any(
        record["phase"] == "prepared_cache_validation" for record in observer.records
    )
    receipt_path = prepared.root.with_name(prepared.root.name + ".preparation.json")
    receipt = json.loads(receipt_path.read_text())
    assert receipt["manifest_sha256"] == prepared.manifest["manifest_sha256"]

    inputs = materialize_prepared_inputs(
        config, tmp_path / "prepared-inputs", observer=observer
    )
    stage_root = stage(
        config,
        tmp_path / "stage",
        through="validate",
        prepared_inputs=inputs,
        observer=observer,
    )
    bundle = _read_sealed(stage_root / "bundle.json")
    assert bundle["status"] == "complete"
    assert "bottleneck_observations" not in bundle
    assert "bottleneck_observations" not in _read_sealed(stage_root / "inputs.json")
    assert any(record["phase"] == "stage_owned_copy" for record in observer.records)
    assert any(
        record["phase"] == "stage_prepared_input_verification"
        for record in observer.records
    )
