"""Paired explorer guard coverage on tiny CPU fixtures; no training or downloads."""

from copy import deepcopy

import pytest
from test_explorer import TEXT, _loaded

from sparselab import explorer
from sparselab.lab_records import jsonable, seal


@pytest.fixture(scope="module")
def pair():
    left = jsonable(explorer.explore_loaded(_loaded(ffn="dense"), TEXT))
    right = deepcopy(left)
    right["target"]["checkpoint_sha256"] = "d" * 64
    return left, right


@pytest.mark.parametrize(
    "case,message",
    [
        ("runtime", "unverified runtime identity"),
        ("checkpoint-none", "require target.checkpoint_sha256"),
        ("checkpoint-empty", "require target.checkpoint_sha256"),
        ("tokenizer-none", "require target.tokenizer_sha256"),
        ("tokenizer-empty", "require target.tokenizer_sha256"),
        ("left-truncated", "cannot compare truncated"),
        ("one-token-left", "missing token sequences"),
        ("one-token-right", "missing token sequences"),
        ("architecture", "missing architecture"),
        ("layers", "missing architecture"),
        ("layer-object", "malformed architecture layers"),
        ("layer-index", "malformed architecture layers"),
        ("metric", "missing activation RMS"),
        ("sites", "missing activation RMS"),
        ("short", "invalid activation length"),
        ("bool", "non-numeric activation"),
        ("string", "non-numeric activation"),
    ],
)
def test_resealed_comparison_guard_refusals(pair, case, message):
    left, right = deepcopy(pair)
    if case == "runtime":
        for side in (left, right):
            del side["runtime"]["precision"]
    elif case.startswith(("checkpoint-", "tokenizer-")):
        field, value = case.split("-")
        left["target"][field + "_sha256"] = None if value == "none" else ""
        # Equal absent tokenizer identities must still fail the presence guard.
        if field == "tokenizer":
            right["target"][field + "_sha256"] = left["target"][field + "_sha256"]
    elif case == "left-truncated":
        left["truncated"] = True
    elif case.startswith("one-token"):
        for side in (left,) if case.endswith("left") else (right,):
            side["tokens"] = side["tokens"][:1]
            for layer in side["activation"]["layers"].values():
                for site in explorer.ACTIVATION_SITES:
                    layer[site] = layer[site][:1]
    elif case == "architecture":
        left["architecture"] = None
    elif case == "layers":
        left["architecture"]["layers"] = {}
    elif case == "layer-object":
        left["architecture"]["layers"][0] = None
    elif case == "layer-index":
        del left["architecture"]["layers"][0]["index"]
    elif case == "metric":
        left["activation"]["metric"] = "l2"
    elif case == "sites":
        left["activation"]["sites"].reverse()
    else:
        values = left["activation"]["layers"]["0"]["residual_pre"]
        if case == "short":
            values.pop()
        else:
            values[0] = {
                "bool": True,
                "string": "1.0",
                "nan": float("nan"),
                "inf": float("inf"),
            }[case]
    with pytest.raises(explorer.ExplorerUnavailable, match=message):
        explorer.compare_explorations(seal(left), seal(right))


@pytest.mark.parametrize("field", ["memory", "tie_embeddings"])
def test_architecture_difference_hides_delta_even_with_identical_layers(pair, field):
    left, right = deepcopy(pair)
    if field == "memory":
        right["architecture"][field] = {"kind": "ngram", "tables": 1}
    else:
        right["architecture"][field] = not left["architecture"][field]
    compared = explorer.compare_explorations(seal(left), seal(right))
    assert compared["common_layers"]
    assert not compared["architecture_matches"]
    assert compared["delta"] is None


def test_nonfinite_residual_refuses_and_removes_only_owned_hooks():
    loaded = _loaded(ffn="dense")
    before = {
        id(module): (len(module._forward_hooks), len(module._forward_pre_hooks))
        for module in loaded.model.modules()
    }

    def corrupt(module, args, output):
        return output + float("inf")

    handle = loaded.model.blocks[0].attention.register_forward_hook(corrupt)
    try:
        with pytest.raises(
            explorer.ExplorerUnavailable, match="non-finite activation RMS"
        ):
            explorer.explore_loaded(loaded, TEXT)
        # The caller's injection hook survives; explorer-owned hooks are removed.
        assert list(loaded.model.blocks[0].attention._forward_hooks) == [handle.id]
    finally:
        handle.remove()
    assert {
        id(module): (len(module._forward_hooks), len(module._forward_pre_hooks))
        for module in loaded.model.modules()
    } == before
    # The same model remains usable after failed exploration.
    assert explorer.explore_loaded(loaded, TEXT)["activation"]["metric"] == "rms"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_activation_is_unsealable_and_validator_refuses(pair, value):
    left = deepcopy(pair[0])
    left["activation"]["layers"]["0"]["residual_pre"][0] = value
    # Canonical JSON cannot produce a sealed non-finite record. Exercise the
    # lower validator separately rather than bypassing the provenance check.
    with pytest.raises(ValueError, match="Out of range float"):
        seal(left)
    with pytest.raises(explorer.ExplorerUnavailable, match="non-finite activation"):
        explorer._validated_activation_layers(left, len(left["tokens"]), "left")
