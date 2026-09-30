from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_training import config

from sparselab.data import packing
from sparselab.data import preparation_chunks as chunks
from sparselab.data.encoding import PreparationEncoder
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer
from sparselab.training.manifest import canonical_json


class Interrupted(Exception):
    pass


def fixture_run(tmp_path: Path):
    run = config(tmp_path)
    paths = {}
    for split, count in (("train", 13), ("validation", 7)):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "text": f"{split} record-{n}: café 🦊\ndef sum_{n}(x): return x + {n}\n"
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for n in range(count)
            ),
            encoding="utf-8",
        )
        paths[f"{split}_path"] = path
    dataset = run.dataset.model_copy(
        update={
            "source": "local_text",
            "license": "fixture",
            "train_max_documents": 13,
            "validation_max_documents": 7,
            "train_max_tokens": 100_000,
            "validation_max_tokens": 100_000,
            **paths,
        }
    )
    model = run.model.model_copy(
        update={
            "memory": "byte",
            "memory_table_size": 1024,
            "memory_dim": 8,
            "memory_ngram_size": 3,
        }
    )
    return run.model_copy(update={"dataset": dataset, "model": model})


def scientific(prepared):
    return prepared.manifest, {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in prepared.root.glob("*.npy")
    }


def bounded_chunks(monkeypatch, limit=4):
    original = chunks.PreparationChunks.__init__

    def small(self, *args, **kwargs):
        kwargs["record_limit"] = limit
        return original(self, *args, **kwargs)

    monkeypatch.setattr(chunks.PreparationChunks, "__init__", small)


def interrupt_after_first(monkeypatch, run, *, split="train"):
    def fail(actual_split, index, _receipt):
        if actual_split == split and index == 0:
            raise Interrupted

    monkeypatch.setattr(chunks, "_after_sealed_chunk", fail)
    with pytest.raises(Interrupted):
        prepare_data(
            run, load_tokenizer(run.tokenizer.path), tokenizer_batch_documents=1
        )
    monkeypatch.setattr(chunks, "_after_sealed_chunk", None)
    stages = list(run.dataset.cache_dir.glob("*.tmp"))
    assert len(stages) == 1
    return stages[0]


@pytest.mark.parametrize("interrupted_split", ["train", "validation"])
def test_sealed_resume_skips_committed_records_and_preserves_exact_outputs(
    tmp_path, monkeypatch, interrupted_split
):
    run = fixture_run(tmp_path)
    bounded_chunks(monkeypatch)
    baseline = prepare_data(
        run.model_copy(
            update={
                "dataset": run.dataset.model_copy(
                    update={"cache_dir": tmp_path / "baseline"}
                )
            }
        ),
        load_tokenizer(run.tokenizer.path),
        tokenizer_batch_documents=1,
    )
    stage = interrupt_after_first(monkeypatch, run, split=interrupted_split)
    receipts = {
        split: [
            json.loads(path.read_text())
            for path in sorted(stage.glob(f"{split}-*.receipt.json"))
        ]
        for split in ("train", "validation")
    }
    acquired = {
        split: sum(receipt["acquired_documents"] for receipt in items)
        for split, items in receipts.items()
    }
    assert acquired[interrupted_split] == 4
    if interrupted_split == "validation":
        assert acquired["train"] == 13
        # An unfinished finalized array is not authority on restart.
        (stage / "train.npy").write_bytes(b"untrusted interrupted final array")
    texts = []
    original_encode = PreparationEncoder.encode

    def record(self, batch):
        texts.extend(batch)
        return original_encode(self, batch)

    monkeypatch.setattr(PreparationEncoder, "encode", record)
    resumed = prepare_data(
        run, load_tokenizer(run.tokenizer.path), tokenizer_batch_documents=1
    )
    expected = []
    for split, count in (("train", 13), ("validation", 7)):
        rows = [
            json.loads(line)["text"]
            for line in getattr(run.dataset, f"{split}_path").read_text().splitlines()
        ]
        expected.extend(rows[acquired[split] : count])
    assert texts == expected
    assert scientific(resumed) == scientific(baseline)
    assert {path.name for path in resumed.root.iterdir()} - {
        ".sparselab-cache-owner.json"
    } == {
        "manifest.json",
        "train.npy",
        "validation.npy",
        "train_byte_addresses.npy",
        "validation_byte_addresses.npy",
    }
    assert not stage.exists()


@pytest.mark.parametrize(
    "tamper", ["raw", "source", "tokenizer", "owner", "unexpected", "symlink"]
)
def test_resume_rejects_changed_sealed_inputs_without_publishing(
    tmp_path, monkeypatch, tamper
):
    run = fixture_run(tmp_path)
    bounded_chunks(monkeypatch)
    stage = interrupt_after_first(monkeypatch, run)
    if tamper == "raw":
        path = next(stage.glob("*.ids.raw"))
        original = path.read_bytes()
        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    elif tamper == "source":
        with run.dataset.train_path.open("a") as output:
            output.write(json.dumps({"text": "changed source"}) + "\n")
    elif tamper == "tokenizer":
        tokenizer = load_tokenizer(run.tokenizer.path)
        tokenizer.add_tokens(["new-tokenizer-identity"])
        tokenizer.save(str(run.tokenizer.path))
    elif tamper == "owner":
        path = stage / "staging.json"
        payload = json.loads(path.read_text())
        payload.pop("checksum")
        payload["cache_identity"]["tokenizer_sha256"] = "0" * 64
        payload["checksum"] = hashlib.sha256(canonical_json(payload)).hexdigest()
        path.write_bytes(canonical_json(payload) + b"\n")
    elif tamper == "unexpected":
        (stage / "notes.txt").write_text("not a chunk")
    else:
        (stage / "unowned").symlink_to(run.tokenizer.path)
    with pytest.raises(ValueError):
        prepare_data(
            run, load_tokenizer(run.tokenizer.path), tokenizer_batch_documents=1
        )
    assert stage.exists()
    assert not [
        path
        for path in run.dataset.cache_dir.iterdir()
        if path.is_dir() and not path.name.endswith(".tmp")
    ]
    if tamper == "unexpected":
        assert (stage / "notes.txt").read_text() == "not a chunk"


def test_chunks_bound_consumed_records_raw_bytes_and_whole_documents(tmp_path):
    owner = chunks.PreparationChunks(
        tmp_path / "chunks.tmp",
        cache_identity={"id": "fixture"},
        binding={"source": "generated"},
        arrays={"ids": np.dtype(np.int32), "byte_addresses": np.dtype(np.int32)},
        record_limit=3,
        raw_byte_limit=24,
    )
    split = owner.open_split("train")
    split.append_record(1, {"ids": [1, 2], "byte_addresses": [3, 4]})
    split.append_record(2, None)
    split.append_record(3, {"ids": [5, 6], "byte_addresses": [7, 8]})
    split.append_record(4, None)
    split.append_record(5, None)
    split.append_record(6, None)
    split.seal()
    receipts = [
        json.loads(path.read_text())
        for path in sorted(owner.root.glob("train-*.receipt.json"))
    ]
    assert [
        (
            item["first_acquired_index"],
            item["acquired_documents"],
            item["retained_documents"],
            item["output_tokens"],
        )
        for item in receipts
    ] == [(1, 2, 1, 2), (3, 3, 1, 2), (6, 1, 0, 0)]
    assert all(
        sum(array["length"] for array in item["arrays"].values()) <= 24
        for item in receipts
    )
    with pytest.raises(ValueError, match="raw-byte cap"):
        split.append_record(7, {"ids": [1, 2, 3, 4], "byte_addresses": [1, 2, 3, 4]})
    assert split.state["acquired_documents"] == 6


@pytest.mark.skipif(os.name != "posix", reason="subprocess SIGKILL scenario")
def test_killed_unsealed_chunk_resumes_only_committed_source_records(
    tmp_path, monkeypatch
):
    run = fixture_run(tmp_path)
    bounded_chunks(monkeypatch)
    baseline = prepare_data(
        run.model_copy(
            update={
                "dataset": run.dataset.model_copy(
                    update={"cache_dir": tmp_path / "baseline"}
                )
            }
        ),
        load_tokenizer(run.tokenizer.path),
        tokenizer_batch_documents=1,
    )
    config_path = tmp_path / "run.json"
    config_path.write_text(run.model_dump_json())
    script = """
import json, sys
from pathlib import Path
from sparselab.config.models import RunConfig
from sparselab.data import preparation_chunks as chunks
from sparselab.data.encoding import PreparationEncoder
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer
run = RunConfig.model_validate_json(Path(sys.argv[1]).read_text())
original = chunks.PreparationChunks.__init__
def small(self,*args,**kwargs):
    kwargs['record_limit'] = 4
    return original(self,*args,**kwargs)
chunks.PreparationChunks.__init__ = small
encode = PreparationEncoder.encode
def record(self,texts):
    with Path(sys.argv[2]).open('a') as log:
        for text in texts:
            log.write(json.dumps(text) + '\\n')
    return encode(self,texts)
PreparationEncoder.encode = record
def pause(split,index,path):
    if split == 'train' and index == 1:
        print('UNSEALED',flush=True)
        sys.stdin.readline()
chunks._after_unsealed_chunk_write = pause
prepare_data(run,load_tokenizer(run.tokenizer.path),tokenizer_batch_documents=1)
"""
    log = tmp_path / "encoded.jsonl"
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        UV_OFFLINE="1",
        SPARSELAB_WORK_DIR=str(tmp_path),
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(config_path), str(log)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    try:
        assert process.stdout.readline().strip() == "UNSEALED"
        process.send_signal(signal.SIGKILL)
        assert process.wait(timeout=15) == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()
    stage = next(run.dataset.cache_dir.glob("*.tmp"))
    partial = list(stage.glob("train-000001.*.raw.tmp"))
    assert partial
    sealed = json.loads((stage / "train-000000.receipt.json").read_text())
    assert sealed["acquired_documents"] == 4
    before = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(before) == 5
    resumed_texts = []
    encode = PreparationEncoder.encode

    def record(self, texts):
        resumed_texts.extend(texts)
        return encode(self, texts)

    monkeypatch.setattr(PreparationEncoder, "encode", record)
    resumed = prepare_data(
        run, load_tokenizer(run.tokenizer.path), tokenizer_batch_documents=1
    )
    all_train = [
        json.loads(line)["text"]
        for line in run.dataset.train_path.read_text().splitlines()
    ]
    all_validation = [
        json.loads(line)["text"]
        for line in run.dataset.validation_path.read_text().splitlines()
    ]
    assert resumed_texts == all_train[4:] + all_validation
    assert scientific(resumed) == scientific(baseline)
    assert not any(path.exists() for path in partial)


def test_atomic_publication_never_replaces_published_cache(tmp_path):
    staged, published = tmp_path / "staged", tmp_path / "published"
    staged.mkdir()
    published.mkdir()
    (staged / "manifest.json").write_text("new")
    (published / "manifest.json").write_text("verified original")
    with pytest.raises(FileExistsError):
        packing._publish_prepared_directory(staged, published)
    assert (published / "manifest.json").read_text() == "verified original"
    assert (staged / "manifest.json").read_text() == "new"


@pytest.mark.parametrize("crash_window", ["published", "cleanup"])
def test_published_cleanup_resume_hashes_arrays_once_and_never_encodes(
    tmp_path, monkeypatch, crash_window
):
    from sparselab.training import manifest as manifest_module

    run = fixture_run(tmp_path)
    bounded_chunks(monkeypatch)
    baseline = prepare_data(
        run.model_copy(
            update={
                "dataset": run.dataset.model_copy(
                    update={"cache_dir": tmp_path / "baseline"}
                )
            }
        ),
        load_tokenizer(run.tokenizer.path),
        tokenizer_batch_documents=1,
    )
    if crash_window == "published":
        publish = packing._publish_prepared_directory

        def fail(staged, destination):
            publish(staged, destination)
            raise Interrupted

        monkeypatch.setattr(packing, "_publish_prepared_directory", fail)
    else:
        cleanup = chunks.PreparationChunks._finish_cleanup

        def fail(self, root, document, *, cold):
            (root / min(document["files"])).unlink()
            raise Interrupted

        monkeypatch.setattr(chunks.PreparationChunks, "_finish_cleanup", fail)
    with pytest.raises(Interrupted):
        prepare_data(
            run, load_tokenizer(run.tokenizer.path), tokenizer_batch_documents=1
        )
    if crash_window == "published":
        monkeypatch.setattr(packing, "_publish_prepared_directory", publish)
    else:
        monkeypatch.setattr(chunks.PreparationChunks, "_finish_cleanup", cleanup)
    reads = []
    sha256 = manifest_module.sha256_file

    def counted(path):
        if Path(path).suffix == ".npy":
            reads.append(Path(path).name)
        return sha256(path)

    def never_encode(*_args):
        pytest.fail("a published cache must not retokenize")

    monkeypatch.setattr(manifest_module, "sha256_file", counted)
    monkeypatch.setattr(PreparationEncoder, "encode", never_encode)
    resumed = prepare_data(run, load_tokenizer(run.tokenizer.path))
    assert scientific(resumed) == scientific(baseline)
    assert sorted(reads) == sorted(
        [
            "train.npy",
            "validation.npy",
            "train_byte_addresses.npy",
            "validation_byte_addresses.npy",
        ]
    )
    assert not (resumed.root / "staging.json").exists()
    assert not list(resumed.root.glob("*.raw"))


def test_external_cache_writer_cannot_be_locked_by_another_process(
    tmp_path, monkeypatch
):
    run = fixture_run(tmp_path)
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "separate-workspace"))
    observed = []

    def check_live_lock(_split, _index, _path):
        if observed:
            return
        script = """
import fcntl,sys
with open(sys.argv[1],'a+b') as lock:
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('active writer protected')
    else:
        raise SystemExit('active writer is unprotected')
"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(run.dataset.cache_dir / ".cleanup.lock"),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        observed.append(result.stdout.strip())

    monkeypatch.setattr(chunks, "_after_unsealed_chunk_write", check_live_lock)
    prepared = prepare_data(
        run, load_tokenizer(run.tokenizer.path), tokenizer_batch_documents=1
    )
    assert observed == ["active writer protected"]
    assert prepared.manifest["train"]["retained_documents"] == 13
