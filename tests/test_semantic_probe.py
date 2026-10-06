"""Consumer-visible supplied-vector semantic probe coverage."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch
import yaml
from test_training import config

from sparselab.engram.semantic_probe import _state_digest, run_semantic_probe
from sparselab.research.scaffold import scaffold_lesson
from sparselab.training.manifest import canonical_json
from sparselab.training.trainer import train

pytest_plugins = ("test_inference",)


def test_scaffold_probe_runs_all_native_retrieval_outcomes(tmp_path: Path) -> None:
    lesson = scaffold_lesson("semantic-retrieval", tmp_path / "semantic")
    report = run_semantic_probe(lesson / "probe.yaml")

    assert report["format"] == "sparselab-semantic-probe-report-v1"
    assert report["model"]["kind"] == "initialized"
    assert report["logits_shape"] == [1, 6, 260]
    attachment = report["attachments"][0]
    positions = attachment["traces"]
    assert [
        item["trace"]["status"] if item["trace"] else "masked" for item in positions
    ] == ["hit", "unknown", "conflict", "hit", "temporal_miss", "masked"]
    assert positions[2]["trace"]["tied_record_ids"] == [
        "lesson-fork-left",
        "lesson-fork-right",
    ]
    assert attachment["weights"]["kind"] == "initialized"


def test_checkpoint_probe_authenticates_native_generation_and_addon(
    tmp_path: Path, trained_run
) -> None:
    from sparselab.evaluation.inference import load_run

    native = load_run(
        "original", trained_run.logging.root_dir, "latest.json", backend="cpu"
    )
    lesson = scaffold_lesson("semantic-retrieval", tmp_path / "semantic")
    source = lesson / "probe.yaml"
    declaration = yaml.safe_load(source.read_text(encoding="utf-8"))
    declaration["model"] = {
        "kind": "checkpoint",
        "run_id": "original",
        "runs_dir": str(trained_run.logging.root_dir),
        "checkpoint": native.identity["checkpoint_relative_path"],
    }
    source.write_text(yaml.safe_dump(declaration), encoding="utf-8")

    report = run_semantic_probe(source)

    assert report["model"]["kind"] == "checkpoint"
    assert report["model"]["identity"] == native.identity
    assert report["attachments"][0]["weights"]["kind"] == "initialized"
    assert report["attachments"][0]["weights"]["state_sha256"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["queries"].append(copy.deepcopy(value["queries"][0])),
            "query ids must be unique",
        ),
        (
            lambda value: value["attachments"].append(
                copy.deepcopy(value["attachments"][0])
            ),
            "attachment names must be unique",
        ),
        (
            lambda value: value["queries"][0].__setitem__("mask", [[True]]),
            "mask shape",
        ),
        (
            lambda value: value["queries"][0].__setitem__("vectors", [[[1.0], [1.0]]]),
            "sequence shape",
        ),
        (
            lambda value: value["input_tokens"].__setitem__(0, [999]),
            "out-of-vocabulary",
        ),
        (
            lambda value: value["attachments"][0].__setitem__("site", "final"),
            "block_index is valid only",
        ),
        (
            lambda value: value["attachments"][0].__setitem__(
                "expected_pack_id", "0" * 64
            ),
            "pack",
        ),
        (
            lambda value: value["queries"][0]["encoder"].__setitem__(
                "name", "same-sha-different-identity"
            ),
            "does not match pack key encoder",
        ),
        (
            lambda value: value["queries"][0].__setitem__("as_of", [None]),
            "as_of sequence",
        ),
    ],
)
def test_probe_rejects_invalid_consumer_declarations_before_forward(
    tmp_path: Path, mutate, message: str, monkeypatch
) -> None:
    from sparselab.model.transformer import DenseLM

    def forbidden_forward(*args, **kwargs):
        raise AssertionError("invalid input reached model forward")

    monkeypatch.setattr(DenseLM, "forward", forbidden_forward)
    lesson = scaffold_lesson("semantic-retrieval", tmp_path / "semantic")
    source = lesson / "probe.yaml"
    declaration = yaml.safe_load(source.read_text(encoding="utf-8"))
    mutate(declaration)
    source.write_text(yaml.safe_dump(declaration), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        run_semantic_probe(source)


def test_probe_identity_is_canonical_and_initialized_rng_isolated(
    tmp_path: Path,
) -> None:
    lesson = scaffold_lesson("semantic-retrieval", tmp_path / "semantic")
    source = lesson / "probe.yaml"
    before = torch.random.get_rng_state()
    initial = run_semantic_probe(source)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)

    declaration = yaml.safe_load(source.read_text(encoding="utf-8"))
    declaration["queries"][0]["vectors"] = [declaration["queries"][0]["vectors"][0][0]]
    declaration["queries"][0]["as_of"] = [None]
    declaration["queries"][0]["mask"] = [True]
    declaration["input_tokens"] = [declaration["input_tokens"][0][:1]]
    source.write_text(yaml.safe_dump(declaration), encoding="utf-8")
    explicit = run_semantic_probe(source)
    declaration["queries"][0].pop("mask")
    source.write_text(yaml.safe_dump(declaration), encoding="utf-8")
    omitted = run_semantic_probe(source)
    assert initial["queries"]["lesson-keys"]["vectors"]["dtype"] == "float32"
    assert omitted["queries"]["lesson-keys"]["mask"]["shape"] == [1]
    assert (
        explicit["queries"]["lesson-keys"]["identity_sha256"]
        == omitted["queries"]["lesson-keys"]["identity_sha256"]
    )


def test_state_digest_preserves_tensor_dtype_raw_bytes() -> None:
    module = torch.nn.Module()
    module.register_buffer("count", torch.tensor([1, 256], dtype=torch.int64))
    raw = module.state_dict()["count"].contiguous()
    row = {
        "name": "count",
        "dtype": str(raw.dtype),
        "shape": [2],
        "sha256": hashlib.sha256(
            raw.numpy().astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }
    assert _state_digest(module) == hashlib.sha256(canonical_json([row])).hexdigest()


def test_probe_rejects_bad_inputs_before_model_forward(tmp_path: Path) -> None:
    lesson = scaffold_lesson("semantic-retrieval", tmp_path / "semantic")
    source = lesson / "probe.yaml"
    declaration = yaml.safe_load(source.read_text(encoding="utf-8"))

    malformed = copy.deepcopy(declaration)
    malformed["input_tokens"] = [[True]]
    source.write_text(yaml.safe_dump(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="strict integers"):
        run_semantic_probe(source)

    malformed = copy.deepcopy(declaration)
    malformed["queries"][0]["vectors"][0][0][0] = float("inf")
    source.write_text(yaml.safe_dump(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        run_semantic_probe(source)

    malformed = copy.deepcopy(declaration)
    malformed["attachments"][0]["query"] = "missing"
    source.write_text(yaml.safe_dump(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown query"):
        run_semantic_probe(source)


@pytest.fixture(scope="module")
def allocation_checkpoint(tmp_path_factory):
    import json

    import numpy as np
    from test_semantic import _retriever

    from sparselab.config.models import TokenizerTrainConfig
    from sparselab.data.allocation import build_allocation_manifest
    from sparselab.data.packing import _tokenizer_sha256, prepare_data
    from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
    from sparselab.evaluation.inference import load_run
    from sparselab.training.manifest import sha256_file, source_identity

    root = tmp_path_factory.mktemp("real-semantic-allocation")
    base = config(root)
    for split, count in (("train", 4), ("validation", 2)):
        (root / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"{split} question {index}: where is the fox?",
                            },
                            {
                                "role": "assistant",
                                "content": f"{split} answer {index}: the fox is in a quiet forest.",
                            },
                        ]
                    }
                )
                + "\n"
                for index in range(count)
            )
        )
    dataset = base.dataset.model_copy(
        update={
            "source": "local_chat",
            "train_path": root / "train.jsonl",
            "validation_path": root / "validation.jsonl",
            "license": "CC0-1.0 synthetic test",
            "train_max_documents": 4,
            "validation_max_documents": 2,
        }
    )
    tokenizer_path = train_tokenizer(
        TokenizerTrainConfig(
            schema_version=1,
            vocab_size=260,
            min_frequency=1,
            max_documents=4,
            output_dir=root / "chat-tokenizer",
            dataset=dataset,
        )
    )
    configured = base.model_copy(
        update={
            "dataset": dataset,
            "model": base.model.model_copy(update={"vocab_size": 260}),
            "tokenizer": base.tokenizer.model_copy(update={"path": tokenizer_path}),
            "training": base.training.model_copy(update={"max_steps": 1}),
            "optimizer": base.optimizer.model_copy(update={"warmup_steps": 0}),
        }
    )
    tokenizer = load_tokenizer(tokenizer_path)
    plain = prepare_data(configured, tokenizer)
    retriever = _retriever(
        root / "allocation" / "assets",
        name="probe-allocation",
        record_ids=("allocated-hit",),
        keys=[[1, 0, 0]],
        values=[[1, 2, 3, 4, 5]],
    )
    pack = root / "allocation" / "assets" / "pack.enpack"
    owners, vectors, masks = [], [], []
    for tokens in (plain.train, plain.validation):
        owner = np.zeros(len(tokens), dtype=np.uint8)
        owner[1::2] = 2
        owners.append(owner)
        vectors.append(
            np.tile(np.asarray([1, 0, 0], dtype=np.float32), (len(tokens), 1))
        )
        mask = np.zeros(len(tokens), dtype=bool)
        mask[:-1] = owner[1:] == 2
        masks.append(mask)
    manifest = build_allocation_manifest(
        root / "allocation" / "allocation.json",
        source_identity_sha256=source_identity()["sha256"],
        tokenizer_sha256=_tokenizer_sha256(tokenizer),
        train_jsonl_sha256=sha256_file(dataset.train_path),
        validation_jsonl_sha256=sha256_file(dataset.validation_path),
        train_owner=owners[0],
        validation_owner=owners[1],
        train_queries=vectors[0],
        validation_queries=vectors[1],
        train_mask=masks[0],
        validation_mask=masks[1],
        semantic={
            "pack_path": "assets/pack.enpack",
            "pack_sha256": sha256_file(pack / "manifest.json"),
            "pack_id": retriever.pack_id,
            "key_encoder": retriever.key_encoder.model_dump(mode="json"),
        },
    )
    configured = configured.model_copy(
        update={
            "dataset": dataset.model_copy(
                update={"allocation_manifest_path": manifest.path}
            ),
            "model": configured.model.model_copy(update={"semantic_memory_dim": 5}),
        }
    )
    train(configured, run_id="allocation-probe")
    loaded = load_run(
        "allocation-probe", configured.logging.root_dir, "latest.json", backend="cpu"
    )
    assert set(loaded.model.semantic_memories) == {"allocation"}
    declaration = {
        "format": "sparselab-semantic-probe-v1",
        "model": {
            "kind": "checkpoint",
            "run_id": "allocation-probe",
            "runs_dir": str(configured.logging.root_dir),
            "checkpoint": loaded.identity["checkpoint_relative_path"],
        },
        "input_tokens": [[1, 2]],
        "queries": [
            {
                "id": "allocation-query",
                "encoder": retriever.key_encoder.model_dump(mode="json"),
                "vectors": [[[1, 0, 0], [0, 0, 0]]],
            }
        ],
        "attachments": [
            {
                "name": "allocation",
                "pack": str(pack),
                "expected_pack_id": retriever.pack_id,
                "query": "allocation-query",
                "site": "final",
                "weights": {"kind": "checkpoint"},
            }
        ],
    }
    return root, declaration, loaded.identity


def test_real_restored_allocation_checkpoint_probe(tmp_path, allocation_checkpoint):
    _, declaration, native_identity = allocation_checkpoint
    source = tmp_path / "probe.yaml"
    source.write_text(yaml.safe_dump(declaration))
    report = run_semantic_probe(source)
    assert report["model"]["identity"] == native_identity
    adapter = report["attachments"][0]
    assert adapter["weights"] == {"kind": "checkpoint"}
    assert (
        adapter["pack_id"] == native_identity["allocation"]["semantic_pack"]["pack_id"]
    )
    assert adapter["traces"][0]["trace"]["best_record_id"] == "allocated-hit"
    assert adapter["traces"][0]["trace"]["status"] == "hit"
    assert [position["trace"]["best_score"] for position in adapter["traces"]] == [
        1.0,
        0.0,
    ]
    assert report["logits_shape"] == [1, 2, 260]


@pytest.mark.parametrize(
    "change", ["missing", "renamed", "site", "threshold", "reinitialize"]
)
def test_restored_allocation_declaration_mismatch_rejected(
    tmp_path, allocation_checkpoint, change
):
    _, original, _ = allocation_checkpoint
    declaration = copy.deepcopy(original)
    if change == "missing":
        # A different initialized attachment does not authorize dropping allocation.
        declaration["attachments"][0]["name"] = "addon"
        declaration["attachments"][0]["weights"] = {"kind": "initialized", "seed": 1}
    elif change == "renamed":
        declaration["attachments"][0]["name"] = "renamed"
    elif change == "site":
        declaration["attachments"][0]["site"] = "embedding"
    elif change == "threshold":
        declaration["attachments"][0]["min_score"] = 0.5
    else:
        declaration["attachments"][0]["weights"] = {"kind": "initialized", "seed": 1}
    source = tmp_path / "probe.yaml"
    source.write_text(yaml.safe_dump(declaration))
    with pytest.raises(ValueError, match="restored|explicitly declared|match|collides"):
        run_semantic_probe(source)


@pytest.mark.parametrize("member", ["config", "manifest", "weights"])
def test_real_allocation_checkpoint_tampering_rejected(
    tmp_path, allocation_checkpoint, member
):
    root, declaration, identity = allocation_checkpoint
    run = root / "runs" / "allocation-probe"
    generation = run / identity["checkpoint_relative_path"]
    if member == "config":
        target = run / "resolved_config.yaml"
    elif member == "manifest":
        target = generation / "manifest.json"
    else:
        import json

        metadata = json.loads((generation / "manifest.json").read_text())
        target = generation / next(
            row["name"]
            for row in metadata["files"]
            if row["name"].endswith(".safetensors")
        )
    original = target.read_bytes()
    target.write_bytes(original + b"tampered")
    source = tmp_path / "probe.yaml"
    source.write_text(yaml.safe_dump(declaration))
    try:
        with pytest.raises((ValueError, RuntimeError)):
            run_semantic_probe(source)
    finally:
        target.write_bytes(original)


@pytest.mark.parametrize(
    "change",
    [
        "overflow",
        "huge_integer",
        "boolean_vector",
        "rank4",
        "ragged",
        "wrong_batch",
        "wrong_width",
        "boolean_mask",
        "all_masked_bad_time",
        "trace_limit",
        "unknown_query",
        "missing_queries",
        "unused_query",
        "conflicting_sha",
        "negative_block",
        "missing_block",
        "checkpoint_weights_initialized",
    ],
)
def test_semantic_input_boundaries_reject_before_forward(tmp_path, monkeypatch, change):
    from sparselab.model.transformer import DenseLM

    source = scaffold_lesson("semantic-retrieval", tmp_path / "lesson") / "probe.yaml"
    value = yaml.safe_load(source.read_text())
    query = value["queries"][0]
    if change in {"overflow", "huge_integer", "boolean_vector"}:
        query["vectors"][0][0][0] = {
            "overflow": 1e100,
            "huge_integer": 10**1000,
            "boolean_vector": True,
        }[change]
    elif change == "rank4":
        query["vectors"] = [query["vectors"]]
    elif change == "ragged":
        query["vectors"][0][0].append(0)
    elif change == "wrong_batch":
        query["vectors"].append(copy.deepcopy(query["vectors"][0]))
    elif change == "wrong_width":
        query["vectors"][0] = [row + [0] for row in query["vectors"][0]]
    elif change == "boolean_mask":
        query["mask"][0][0] = 1
    elif change == "all_masked_bad_time":
        query["mask"] = [[False] * 6]
        query["as_of"] = ["not-a-date"] * 6
    elif change == "trace_limit":
        value["input_tokens"] = [[1] for _ in range(65)]
    elif change == "unknown_query":
        value["attachments"][0]["query"] = "not-declared"
    elif change == "missing_queries":
        value["queries"] = []
    elif change in {"unused_query", "conflicting_sha"}:
        other = copy.deepcopy(query)
        other["id"] = "unused"
        other["encoder"]["name"] = "different-encoder"
        if change == "unused_query":
            other["encoder"]["sha256"] = "a" * 64
        value["queries"].append(other)
    elif change == "negative_block":
        value["attachments"][0]["block_index"] = -1
    elif change == "missing_block":
        value["attachments"][0].pop("block_index")
    else:
        value["attachments"][0]["weights"] = {"kind": "checkpoint"}
    source.write_text(yaml.safe_dump(value))

    def forbidden_forward(*args, **kwargs):
        raise AssertionError("invalid declaration reached inference")

    monkeypatch.setattr(DenseLM, "forward", forbidden_forward)
    with pytest.raises(ValueError):
        run_semantic_probe(source)


@pytest.mark.parametrize("change", ["extra", "missing", "tampered"])
def test_pack_asset_inventory_rejected_before_forward(tmp_path, monkeypatch, change):
    import json

    from sparselab.model.transformer import DenseLM

    source = scaffold_lesson("semantic-retrieval", tmp_path / "lesson") / "probe.yaml"
    declaration = yaml.safe_load(source.read_text())
    pack = source.parent / declaration["attachments"][0]["pack"]
    manifest = json.loads((pack / "manifest.json").read_text())
    asset = pack / manifest["semantic"]["keys_path"]
    if change == "extra":
        (pack / "unexpected").write_text("extra")
    elif change == "missing":
        asset.unlink()
    else:
        asset.write_bytes(asset.read_bytes() + b"tampered")

    def forbidden_forward(*args, **kwargs):
        raise AssertionError("unauthenticated pack reached inference")

    monkeypatch.setattr(DenseLM, "forward", forbidden_forward)
    with pytest.raises(ValueError):
        run_semantic_probe(source)


@pytest.mark.parametrize(
    "change", ["pack", "query", "declaration", "declaration_symlink"]
)
def test_semantic_bindings_rechecked_after_forward(tmp_path, monkeypatch, change):
    import json

    from sparselab.model.transformer import DenseLM

    source = scaffold_lesson("semantic-retrieval", tmp_path / "lesson") / "probe.yaml"
    declaration = yaml.safe_load(source.read_text())
    original = DenseLM.forward

    def changing_forward(model, tokens, **kwargs):
        output = original(model, tokens, **kwargs)
        if change == "query":
            next(iter(kwargs["semantic_queries"].values())).vectors[0, 0, 0] += 1
        elif change == "pack":
            pack = source.parent / declaration["attachments"][0]["pack"]
            metadata = json.loads((pack / "manifest.json").read_text())
            asset = pack / metadata["semantic"]["keys_path"]
            asset.write_bytes(asset.read_bytes() + b"tampered")
        elif change == "declaration":
            source.write_text(source.read_text() + "\n# changed")
        else:
            replacement = source.with_name("replacement.yaml")
            source.replace(replacement)
            source.symlink_to(replacement)
        return output

    monkeypatch.setattr(DenseLM, "forward", changing_forward)
    with pytest.raises(ValueError):
        run_semantic_probe(source)
