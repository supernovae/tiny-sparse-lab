"""Cache identity regressions with model-free native-loader fakes."""

from types import SimpleNamespace

import pytest

from sparselab import explorer, lab_results
from sparselab.lab_records import is_sealed, seal, write_json_atomic


def runtime(**changes):
    return {
        "engine": "pytorch",
        "backend": "cpu",
        "device_index": 0,
        "precision": "fp32",
        "framework_version": "fixture",
        **changes,
    }


@pytest.fixture
def fake(monkeypatch, tmp_path):
    state = SimpleNamespace(
        runtime=runtime(),
        sha="c" * 64,
        calls=0,
        loads=0,
        engine=None,
        semantic_memories=None,
    )
    monkeypatch.setattr(
        lab_results, "resolve_checkpoint", lambda *a, **k: SimpleNamespace(run=tmp_path)
    )
    monkeypatch.setattr(explorer, "run_config", lambda run: None)
    monkeypatch.setattr(
        explorer, "architecture", lambda config: {"inventory": {"total": 1}}
    )
    monkeypatch.setattr(explorer, "release_memory", lambda: None)

    def load():
        state.loads += 1
        return SimpleNamespace(
            identity={"checkpoint_sha256": state.sha, "runtime": state.runtime},
            engine=state.engine,
            model=SimpleNamespace(semantic_memories=state.semantic_memories),
        )

    def measure(loaded, text, **kwargs):
        state.calls += 1
        return {
            "format": explorer.EXPLORER_FORMAT,
            "target": {"checkpoint_sha256": state.sha},
            "text": text,
            "runtime": dict(state.runtime),
            "measurement": state.calls,
        }

    monkeypatch.setattr(explorer, "explore_loaded", measure)
    state.explore = lambda **kwargs: explorer.explore(
        "fake", lab_dir=tmp_path, load=load, **kwargs
    )
    return state


def test_actual_backend_change_cannot_reuse_diagnostics(fake):
    first, first_path = fake.explore(backend="auto")
    fake.runtime = runtime(backend="rocm")
    second, second_path = fake.explore(backend="auto")
    assert fake.calls == 2
    assert second["runtime"]["backend"] == "rocm"
    assert first_path != second_path
    assert first["measurement"] != second["measurement"]


def test_same_actual_runtime_hits_and_keeps_original_provenance(fake):
    first, path = fake.explore(backend="auto")
    original = path.read_bytes()
    # Extra provenance is intentionally outside the documented canonical subset.
    fake.runtime = runtime(framework_version="other", os="other", device_name="other")
    second, second_path = fake.explore(backend="cpu")
    assert path == second_path and path.read_bytes() == original
    assert fake.calls == 1 and fake.loads == 2
    assert second["measurement"] == first["measurement"]
    assert second["runtime"] == first["runtime"]
    assert second["cache_reuse"] == {"status": "hit"}
    assert is_sealed(second)


@pytest.mark.parametrize(
    "change",
    [
        {"backend": "cuda"},
        {"backend": "rocm"},
        {"device_index": 1},
        {"precision": "bf16"},
        {"precision": "fp16"},
    ],
)
def test_each_actual_runtime_field_binds_reuse(fake, change):
    _, first = fake.explore()
    fake.runtime = runtime(**change)
    result, second = fake.explore()
    assert first != second and fake.calls == 2
    assert result["cache_reuse"]["status"] == "miss"


def test_cuda_and_rocm_are_distinct_even_with_the_same_requested_backend(fake):
    fake.runtime = runtime(backend="cuda")
    _, cuda_path = fake.explore(backend="auto")
    fake.runtime = runtime(backend="rocm")
    result, rocm_path = fake.explore(backend="auto")
    assert cuda_path != rocm_path and fake.calls == 2
    assert result["runtime"]["backend"] == "rocm"


def test_refresh_recomputes_only_the_selected_runtime(fake):
    _, cpu_path = fake.explore()
    cpu_bytes = cpu_path.read_bytes()
    fake.runtime = runtime(backend="rocm")
    _, rocm_path = fake.explore()
    result, refreshed_path = fake.explore(refresh=True)
    assert fake.calls == 3 and refreshed_path == rocm_path
    assert cpu_path.read_bytes() == cpu_bytes
    assert result["cache_reuse"] == {"status": "miss", "reason": "refresh requested"}


@pytest.mark.parametrize("change", ["text", "checkpoint"])
def test_checkpoint_and_text_still_bind_reuse(fake, change):
    _, first = fake.explore(text="first")
    if change == "checkpoint":
        fake.sha = "d" * 64
    _, second = fake.explore(text="second" if change == "text" else "first")
    assert first != second and fake.calls == 2


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        [],
        "cpu",
        runtime(backend=[]),
        runtime(device_index=True),
        runtime(device_index=-1),
        runtime(precision="auto"),
        runtime(precision=[]),
        runtime(engine="mlx"),
    ],
)
def test_missing_or_malformed_producer_never_hits(fake, bad):
    _, path = fake.explore()
    value = explorer.read_cached(path)
    write_json_atomic(path, seal({**value, "runtime": bad}))
    result, _ = fake.explore()
    assert fake.calls == 2 and result["cache_reuse"]["status"] == "miss"
    assert "actual-runtime provenance" in result["cache_reuse"]["reason"]


@pytest.mark.parametrize("field", ["text", "checkpoint", "runtime"])
def test_resealed_payload_must_match_its_cache_key(fake, field):
    _, path = fake.explore()
    value = explorer.read_cached(path)
    if field == "text":
        value["text"] = "different"
    elif field == "checkpoint":
        value["target"]["checkpoint_sha256"] = "e" * 64
    else:
        value["runtime"] = runtime(backend="rocm")
    write_json_atomic(path, seal(value))
    listing = explorer.cached_explorations(path.parents[2])
    assert listing[0]["cache_issue"] == "sealed payload/cache key mismatch"
    result, _ = fake.explore()
    assert fake.calls == 2
    assert result["cache_reuse"]["reason"] == "sealed payload/cache key mismatch"


def test_unsealed_cache_is_an_explicit_miss(fake):
    _, path = fake.explore()
    value = explorer.read_cached(path)
    write_json_atomic(path, {**value, "measurement": "edited"})
    result, _ = fake.explore()
    assert fake.calls == 2
    assert (
        result["cache_reuse"]["reason"] == "unreadable, unsealed or wrong-format cache"
    )


def test_legacy_bytes_remain_readable_but_never_backend_verified(fake):
    _, path = fake.explore()
    lab = path.parents[2]
    legacy_path = explorer.cache_path(lab, fake.sha, explorer.DEFAULT_TEXT)
    value = explorer.read_cached(path)
    del value["runtime"]
    write_json_atomic(legacy_path, seal(value))
    original = legacy_path.read_bytes()
    path.unlink()  # fixture: leave only the historical cache
    result, new_path = fake.explore()
    assert fake.calls == 2 and new_path != legacy_path
    assert legacy_path.read_bytes() == original
    assert result["cache_reuse"]["reason"] == "missing actual-runtime provenance"
    listing = explorer.cached_explorations(lab)
    assert any(v["cache_issue"] == "missing actual-runtime provenance" for v in listing)


@pytest.mark.parametrize(
    "change, message",
    [("engine", "PyTorch engine"), ("semantic_memories", "semantic packs")],
)
def test_unsupported_loaded_models_cannot_hit_existing_cache(fake, change, message):
    fake.explore()
    setattr(fake, change, "unsupported")
    with pytest.raises(explorer.ExplorerUnavailable, match=message):
        fake.explore()
    assert fake.calls == 1 and fake.loads == 2


@pytest.mark.parametrize("bad", [None, runtime(engine="mlx")])
def test_invalid_native_runtime_is_explicit_before_measurement(fake, bad):
    fake.runtime = bad
    with pytest.raises(ValueError, match="actual-runtime provenance"):
        fake.explore()
    assert fake.calls == 0


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_reports_producer_runtime_and_reuse_status(
    monkeypatch, tmp_path, capsys, as_json
):
    import json

    from sparselab.cli.main import _explore

    result = {
        "target": {"run_id": "fake"},
        "runtime": runtime(backend="rocm"),
        "cache_reuse": {
            "status": "miss",
            "reason": "missing actual-runtime provenance",
        },
        "architecture": {
            "inventory": {"total": 1, "active_per_token": 1},
            "layers": [],
        },
        "tokens": [{"loss": 1.0}],
        "attention": {"layers": {}},
        "seconds": 0,
    }
    monkeypatch.setattr(
        explorer, "explore", lambda *a, **k: (result, tmp_path / "cache.json")
    )
    _explore(
        SimpleNamespace(
            target="fake",
            work_dir=tmp_path,
            lab_dir=tmp_path,
            runs_dir=None,
            text=None,
            backend="cpu",
            runtime_authorization=None,
            resource_envelope_value=None,
            refresh=False,
            json=as_json,
        )
    )
    output = capsys.readouterr().out
    if as_json:
        value = json.loads(output)
        assert value["runtime"] == result["runtime"]
        assert value["cache_reuse"] == result["cache_reuse"]
    else:
        assert "diagnostics runtime: pytorch / rocm:0 / fp32" in output
        assert "cache reuse: miss · missing actual-runtime provenance" in output
