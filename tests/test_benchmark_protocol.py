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
                "config_sha256": bp.digest(
                    {
                        "test_split": "test",
                        "training_split": "train",
                        "num_fewshot": None,
                    }
                ),
                "dataset_fingerprints": {
                    "test": "fixture-fingerprint",
                    "train": "train-fingerprint",
                },
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
        dump_config=lambda: {
            "test_split": "test",
            "training_split": "train",
            "num_fewshot": None,
        },
        dataset={
            "test": SimpleNamespace(_fingerprint="fixture-fingerprint"),
            "train": SimpleNamespace(_fingerprint="train-fingerprint"),
        },
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
            and datasets.config.HF_HUB_OFFLINE
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
        datasets.config.HF_HUB_OFFLINE,
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
    assert fake[0][0]["num_fewshot"] == 3 and fake[0][0]["fewshot_random_seed"] == 9
    assert (
        fake[0][0]["bootstrap_iters"] == 100
        and not fake[0][0]["confirm_run_unsafe_code"]
    )
    assert os.environ.get("HF_DATASETS_OFFLINE") == before
    assert old_flags == (
        datasets.config.HF_DATASETS_OFFLINE,
        datasets.config.HF_HUB_OFFLINE,
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
        task.dataset["validation"] = SimpleNamespace(
            _fingerprint="validation-fingerprint"
        )
        p["tasks"]["piqa"]["dataset_fingerprints"]["validation"] = (
            "validation-fingerprint"
        )
        p["tasks"]["piqa"]["evaluation_split"] = "validation"
    if kind == "zero-shot":
        task.dump_config = lambda: {"test_split": "test", "num_fewshot": 0}
        p["tasks"]["piqa"]["config_sha256"] = bp.digest(task.dump_config())
    if kind == "unsafe":
        task.UNSAFE_CODE = True
    with pytest.raises(
        ValueError, match="evaluation split" if kind == "split" else None
    ):
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


@pytest.mark.parametrize(
    "config",
    [
        {"test_split": "test"},
        {"validation_split": "test"},
        {"test_split": "test", "training_split": "train", "fewshot_split": "test"},
        {"test_split": "test", "fewshot_config": {"split": "test", "samples": []}},
    ],
)
def test_fewshot_evaluation_leakage_refuses_before_scoring(fake, config):
    calls, task = fake
    task.dump_config = lambda: config
    p = protocol()
    p["tasks"]["piqa"]["config_sha256"] = bp.digest(config)
    if "training_split" not in config:
        del task.dataset["train"]
        del p["tasks"]["piqa"]["dataset_fingerprints"]["train"]
    with pytest.raises(ValueError, match="overlaps evaluation"):
        adapter.run_lm_eval(loaded(), ["piqa"], 2, protocol=p)
    assert calls == []
    # Zero-shot does not sample the evaluation split.
    adapter.run_lm_eval(loaded(), ["piqa"], 2, protocol=p | {"num_fewshot": 0})
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kind",
    [
        "fingerprints",
        "split",
        "tasks",
        "lm-eval",
        "empty-version",
        "datasets",
        "tokenizers",
        "torch",
        "numpy",
    ],
)
def test_required_pins(kind):
    p = protocol()
    if kind == "fingerprints":
        p["tasks"]["piqa"]["dataset_fingerprints"] = {}
    elif kind == "split":
        p["tasks"]["piqa"]["evaluation_split"] = "missing"
    elif kind == "tasks":
        p["tasks"] = {}
    elif kind == "lm-eval":
        p["dependency_versions"]["lm-eval"] = "0.4.13"
    elif kind == "empty-version":
        p["dependency_versions"]["torch"] = ""
    else:
        del p["dependency_versions"][kind]
    with pytest.raises(ValueError):
        bp.BenchmarkProtocol.model_validate(p)


@pytest.mark.parametrize("kwargs", [{"tasks": ["piqa"]}, {"limit": 2}])
def test_spec_rejects_legacy_overrides(kwargs):
    with pytest.raises(ValueError):
        runner.lm_eval_spec(protocol=protocol(), **kwargs)


def test_task_expansion_refuses_before_scoring(fake, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "lm_eval.tasks",
        SimpleNamespace(
            TaskManager=lambda: SimpleNamespace(load=lambda names: {"tasks": {}})
        ),
    )
    with pytest.raises(ValueError, match="expansion"):
        adapter.run_lm_eval(loaded(), ["piqa"], 2, protocol=protocol())
    assert not fake[0]


@pytest.mark.parametrize(
    "kind", ["extra-duplicate", "bool-id", "string-id", "arguments", "target"]
)
def test_partial_adapter_withholds_comparison_group(fake, monkeypatch, kind):
    evidence = raw()
    samples = evidence["samples"]["piqa"]
    if kind == "extra-duplicate":
        samples.append(copy.deepcopy(samples[1]))
    elif kind == "bool-id":
        samples[0]["doc_id"] = False
    elif kind == "string-id":
        samples[0]["doc_id"] = "0"
    else:
        del samples[0][kind]
    monkeypatch.setattr(
        adapter,
        "_harness",
        lambda: (SimpleNamespace(simple_evaluate=lambda **kw: evidence), object),
    )
    result = adapter.run_lm_eval(loaded(), ["piqa"], 2, protocol=protocol())
    assert result["benchmark_group"] is None
    assert not result["controlled"]["evidence_complete"]
    assert (
        result["controlled"]["tasks"]["piqa"]["metrics"] == evidence["results"]["piqa"]
    )


@pytest.mark.parametrize("kind", ["backend", "dtype"])
def test_runtime_identity_independent_fields(fake, kind):
    baseline = adapter.run_lm_eval(loaded(), ["piqa"], 2, protocol=protocol())
    changed = loaded()
    if kind == "backend":
        changed.device = "cuda"
    else:
        changed.model.parameters = lambda: iter([SimpleNamespace(dtype="float16")])
    result = adapter.run_lm_eval(changed, ["piqa"], 2, protocol=protocol())
    assert result["benchmark_group"] != baseline["benchmark_group"]


def test_measure_forwards_controlled_protocol(fake):
    result = runner._measure(
        runner.lm_eval_spec(protocol=protocol()), loaded(), None, {}
    )
    assert result["controlled"]["protocol"] == bp.BenchmarkProtocol.model_validate(
        protocol()
    ).model_dump(mode="json")
    call = fake[0][0]
    assert (
        call["random_seed"]
        == call["numpy_random_seed"]
        == call["torch_random_seed"]
        == 7
    )
    assert call["fewshot_random_seed"] == 9
    assert call["bootstrap_iters"] == 100
    assert call["apply_chat_template"] is False


def test_baseline_incomplete_and_legacy_are_distinct(fake):
    spec = runner.lm_eval_spec(protocol=protocol())
    target = adapter.run_lm_eval(loaded(), ["piqa"], 2, protocol=protocol())
    baseline = copy.deepcopy(target)
    baseline["controlled"]["evidence_complete"] = False
    assert runner._judge(spec, target, baseline, {}, {})["status"] == "error"
    del baseline["controlled"]
    baseline["benchmark_group"] = "legacy"
    assert runner._judge(spec, target, baseline, {}, {})["status"] == "not_comparable"


@pytest.mark.parametrize("where", ["primary", "stderr", "sample"])
def test_nonfinite_adapter_preserves_other_evidence(fake, monkeypatch, where):
    evidence = raw()
    if where == "sample":
        evidence["samples"]["piqa"][0]["acc"] = float("nan")
    else:
        evidence["results"]["piqa"][
            "acc,none" if where == "primary" else "acc_stderr,none"
        ] = float("inf")
    monkeypatch.setattr(
        adapter,
        "_harness",
        lambda: (SimpleNamespace(simple_evaluate=lambda **kw: evidence), object),
    )
    result = adapter.run_lm_eval(loaded(), ["piqa"], 2, protocol=protocol())
    json.dumps(result, allow_nan=False)
    assert result["benchmark_group"] is None
    task = result["controlled"]["tasks"]["piqa"]
    assert task["nonfinite_evidence"] and not task["complete"]
    assert task["metrics"]["acc_norm,none"] == 0.75
    assert len(task["items"]) == 2
