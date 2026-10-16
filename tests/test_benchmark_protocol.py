"""Offline metadata/fake-harness fixtures: no model, dataset or benchmark work."""

import copy
import json
import os
import sys
from types import SimpleNamespace

import pytest

from sparselab.probes import benchmark_protocol as bp
from sparselab.probes import lm_eval_adapter as adapter
from sparselab.probes import runner
from sparselab.probes.config import ProbeConfig, load_probe_config


def protocol():
    return {
        "harness_version": "0.4.13",
        "dependency_versions": dict.fromkeys(
            ("datasets", "tokenizers", "torch", "numpy"), "fixture"
        ),
        "tasks": {
            "piqa": {
                "config_sha256": bp.digest({"test_split": "test", "num_fewshot": None}),
                "dataset_fingerprints": {"test": "fixture-fingerprint"},
                "evaluation_split": "test",
                "primary_metric": "acc,none",
            }
        },
        "limit": 2,
        "num_fewshot": 3,
        "seed": 7,
        "fewshot_seed": 9,
        "bootstrap_iters": 100,
    }


def raw():
    return {
        "results": {
            "piqa": {"acc,none": 0.5, "acc_norm,none": 0.75, "acc_stderr,none": 0.2}
        },
        "n-samples": {"piqa": {"original": 10, "effective": 2}},
        "n-shot": {"piqa": 3},
        "versions": {"piqa": 1},
        "samples": {
            "piqa": [
                {
                    "doc_id": i,
                    "doc_hash": str(i),
                    "arguments": ["prompt", "answer"],
                    "target": "0",
                    "acc": i,
                }
                for i in range(2)
            ]
        },
    }


def loaded(checkpoint="a"):
    return SimpleNamespace(
        identity={
            "checkpoint_sha256": checkpoint * 64,
            "tokenizer_sha256": "b" * 64,
            "tokens_seen": 1234,
            "step": 4,
        },
        config=SimpleNamespace(model=SimpleNamespace(max_seq_len=128)),
        device="cpu",
        model=SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(dtype="float32")])
        ),
    )


@pytest.fixture
def fake(monkeypatch):
    calls = []
    task = SimpleNamespace(
        dump_config=lambda: {"test_split": "test", "num_fewshot": None},
        dataset={"test": SimpleNamespace(_fingerprint="fixture-fingerprint")},
        UNSAFE_CODE=False,
    )
    manager = SimpleNamespace(load=lambda names: {"tasks": {"piqa": task}})
    monkeypatch.setitem(
        sys.modules, "lm_eval.tasks", SimpleNamespace(TaskManager=lambda: manager)
    )
    monkeypatch.setattr(
        bp, "version", lambda name: "0.4.13" if name == "lm-eval" else "fixture"
    )

    def evaluate(**kwargs):
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        import datasets.config
        import huggingface_hub.constants

        assert (
            datasets.config.HF_DATASETS_OFFLINE
            and huggingface_hub.constants.HF_HUB_OFFLINE
        )
        calls.append(kwargs)
        return raw()

    monkeypatch.setattr(
        adapter, "_harness", lambda: (SimpleNamespace(simple_evaluate=evaluate), object)
    )
    monkeypatch.setattr(adapter, "build_lm", lambda _: "fake-model")
    return calls, task


def test_public_declaration_and_spec(tmp_path):
    path = tmp_path / "protocol.yaml"
    path.write_text(json.dumps({"tier": "full", "lm_eval_protocol": protocol()}))
    config = load_probe_config(str(path))
    spec = runner.lm_eval_spec(protocol=config.lm_eval_protocol)
    assert spec.params["limit"] == 2 and spec.params["protocol"]["num_fewshot"] == 3
    assert runner.lm_eval_spec().params["limit"] == 50


@pytest.mark.parametrize(
    "change",
    [
        {"limit": True},
        {"limit": 0},
        {"num_fewshot": -1},
        {"seed": True},
        {"harness_version": "latest"},
        {"cache_policy": "download"},
        {"unknown": 1},
    ],
)
def test_invalid_protocol(change):
    with pytest.raises(ValueError):
        bp.BenchmarkProtocol.model_validate(protocol() | change)


def test_protocol_config_requires_full_and_no_smoke_overrides():
    for extra in (
        {},
        {"tier": "full", "lm_eval_limit": 50},
        {"tier": "full", "lm_eval_tasks": ["piqa"]},
    ):
        with pytest.raises(ValueError):
            ProbeConfig.model_validate({"lm_eval_protocol": protocol(), **extra})


def test_full_metrics_counts_provenance_and_compatibility(fake):
    import datasets.config
    import huggingface_hub.constants

    old_flags = (
        datasets.config.HF_DATASETS_OFFLINE,
        huggingface_hub.constants.HF_HUB_OFFLINE,
    )
    before = os.environ.get("HF_DATASETS_OFFLINE")
    a = adapter.run_lm_eval(loaded(), ["piqa"], 2, protocol=protocol())
    b = adapter.run_lm_eval(loaded("c"), ["piqa"], 2, protocol=protocol())
    assert a["benchmark_group"] == b["benchmark_group"]
    c = a["controlled"]
    assert (
        c["subject"]["checkpoint_sha256"]
        != b["controlled"]["subject"]["checkpoint_sha256"]
    )
    assert c["subject"]["tokens_seen"] == 1234
    assert c["tasks"]["piqa"]["metrics"] == raw()["results"]["piqa"]
    assert c["tasks"]["piqa"]["uncertainty"]["reported"] == {"acc_stderr,none": 0.2}
    assert c["evidence_complete"] and c["scientific_result"] == "not_assessed"
    assert c["cost"] is None and c["training_time_seconds"] is None
    assert c["raw_result_sha256"] == bp.digest(raw())
    assert fake[0][0]["num_fewshot"] == 3 and fake[0][0]["fewshot_random_seed"] == 9
    assert (
        fake[0][0]["bootstrap_iters"] == 100
        and not fake[0][0]["confirm_run_unsafe_code"]
    )
    assert os.environ.get("HF_DATASETS_OFFLINE") == before
    assert old_flags == (
        datasets.config.HF_DATASETS_OFFLINE,
        huggingface_hub.constants.HF_HUB_OFFLINE,
    )
    changed = protocol() | {"seed": 8}
    assert (
        adapter.run_lm_eval(loaded(), ["piqa"], 2, protocol=changed)["benchmark_group"]
        != a["benchmark_group"]
    )


@pytest.mark.parametrize(
    "kind", ["missing", "partial", "count", "duplicate", "shots", "metric"]
)
def test_partial_evidence_keeps_metrics(kind):
    r = raw()
    if kind == "missing":
        r["samples"] = {}
    if kind == "partial":
        r["samples"]["piqa"].pop()
    if kind == "count":
        r["n-samples"]["piqa"]["effective"] = 1
    if kind == "duplicate":
        r["samples"]["piqa"][1]["doc_id"] = 0
    if kind == "shots":
        r["n-shot"]["piqa"] = 0
    if kind == "metric":
        del r["results"]["piqa"]["acc,none"]
    c = bp.controlled_results(
        r, bp.BenchmarkProtocol.model_validate(protocol()), loaded(), 0.1
    )
    assert c["missing_tasks"] == ["piqa"] and not c["evidence_complete"]
    assert c["tasks"]["piqa"]["metrics"] == r["results"]["piqa"]


@pytest.mark.parametrize(
    "kind", ["version", "config", "data", "split", "zero-shot", "unsafe"]
)
def test_pins_refuse_before_scoring(fake, monkeypatch, kind):
    calls, task = fake
    p = protocol()
    if kind == "version":
        monkeypatch.setattr(bp, "version", lambda _: "wrong")
    if kind == "config":
        p["tasks"]["piqa"]["config_sha256"] = "0" * 64
    if kind == "data":
        task.dataset["test"]._fingerprint = "changed"
    if kind == "split":
        p["tasks"]["piqa"].update(
            evaluation_split="other",
            dataset_fingerprints={"test": "fixture-fingerprint", "other": "x"},
        )
    if kind == "zero-shot":
        task.dump_config = lambda: {"test_split": "test", "num_fewshot": 0}
        p["tasks"]["piqa"]["config_sha256"] = bp.digest(task.dump_config())
    if kind == "unsafe":
        task.UNSAFE_CODE = True
    with pytest.raises(ValueError):
        adapter.run_lm_eval(loaded(), ["piqa"], 2, protocol=p)
    assert not calls


def test_missing_cache_or_dependency_stays_explicit(fake, monkeypatch):
    def unavailable(_):
        raise FileNotFoundError("offline cache miss")

    monkeypatch.setattr(bp, "pinned_tasks", unavailable)
    with pytest.raises(FileNotFoundError, match="cache miss"):
        adapter.run_lm_eval(loaded(), ["piqa"], 2, protocol=protocol())

    def missing():
        raise adapter.LmEvalUnavailable("missing dependency")

    monkeypatch.setattr(adapter, "_harness", missing)
    with pytest.raises(adapter.LmEvalUnavailable):
        adapter.run_lm_eval(loaded(), ["piqa"], 2, protocol=protocol())
    assert not fake[0]


def test_sciq_requires_support_passage_label():
    p = protocol()
    p["tasks"] = {"sciq": p["tasks"]["piqa"]}
    with pytest.raises(ValueError, match="support-passage"):
        bp.BenchmarkProtocol.model_validate(p)
    p["tasks"]["sciq"]["context"] = "support_passage"
    bp.BenchmarkProtocol.model_validate(p)


def test_controlled_judgement_preserves_partial_results(fake):
    spec = runner.lm_eval_spec(protocol=protocol())
    a = adapter.run_lm_eval(loaded(), ["piqa"], 2, protocol=protocol())
    row = runner._judge(spec, a, None, {}, {})
    assert (
        row["status"] == "info"
        and row["details"]["controlled"]["scientific_result"] == "not_assessed"
    )
    b = copy.deepcopy(a)
    b["benchmark_group"] = "different"
    assert runner._judge(spec, a, b, {}, {})["status"] == "not_comparable"
    a["controlled"]["evidence_complete"] = False
    row = runner._judge(spec, a, None, {}, {})
    assert row["status"] == "error" and row["details"]["controlled"]["tasks"]


def test_public_cli_seals_protocol_and_never_overwrites(tmp_path, monkeypatch, capsys):
    from test_lab_results import _cli, _record, _run

    from sparselab.lab_records import read_lab_record
    from sparselab.probes import cli

    target = _record(tmp_path / "lab", _run(tmp_path / "lab"))
    request = tmp_path / "probe.yaml"
    request.write_text(json.dumps({"tier": "full", "lm_eval_protocol": protocol()}))
    seen = []

    def battery(*args, **kwargs):
        seen.append(kwargs["lm_eval"].params["protocol"])
        return {
            "format": "sparselab-probe-v1",
            "stop": {"kind": "complete"},
            "probes": [],
            "verdict": {"action": "rerun"},
        }

    monkeypatch.setattr(cli, "run_battery", battery)
    _cli(
        monkeypatch, tmp_path, "probe", str(target), "--config", str(request), "--json"
    )
    first = json.loads(capsys.readouterr().out)
    path = tmp_path / "lab/probes" / first["probe_id"] / "probe.json"
    original = path.read_bytes()
    request.write_text(
        json.dumps({"tier": "full", "lm_eval_protocol": protocol() | {"seed": 8}})
    )
    _cli(
        monkeypatch, tmp_path, "probe", str(target), "--config", str(request), "--json"
    )
    second = json.loads(capsys.readouterr().out)
    assert first["probe_id"] != second["probe_id"] and path.read_bytes() == original
    assert seen[0]["seed"] == 7 and seen[1]["seed"] == 8
    _, reread = read_lab_record(path, "probe")
    assert reread["request"]["lm_eval_protocol"]["seed"] == 7


@pytest.mark.parametrize("value", [None, True, "N/A", float("inf")])
def test_non_numeric_primary_never_completes(value):
    r = raw()
    r["results"]["piqa"]["acc,none"] = value
    if value == float("inf"):
        with pytest.raises(ValueError):
            bp.controlled_results(
                r, bp.BenchmarkProtocol.model_validate(protocol()), loaded(), 0
            )
    else:
        assert not bp.controlled_results(
            r, bp.BenchmarkProtocol.model_validate(protocol()), loaded(), 0
        )["evidence_complete"]


def test_runtime_scoring_semantics_change_group(fake):
    a = adapter.run_lm_eval(loaded(), ["piqa"], 2, protocol=protocol())
    other = loaded()
    other.config.model.max_seq_len = 256
    b = adapter.run_lm_eval(other, ["piqa"], 2, protocol=protocol())
    assert a["benchmark_group"] != b["benchmark_group"]


def test_missing_item_identity_stays_incomplete():
    r = raw()
    del r["samples"]["piqa"][0]["doc_hash"]
    assert not bp.controlled_results(
        r, bp.BenchmarkProtocol.model_validate(protocol()), loaded(), 0
    )["evidence_complete"]
