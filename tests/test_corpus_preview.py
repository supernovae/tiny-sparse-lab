import hashlib
import json
import tracemalloc

import pytest

from sparselab.corpus.preview import Reader, preview
from sparselab.training.manifest import canonical_json


def fixture(tmp_path, count=20, text="hello world"):
    snapshot = "a" * 64
    base = tmp_path / "input" / snapshot
    (base / "files").mkdir(parents=True)
    manifest = base / "manifest.json"
    manifest.write_text('{"fixture": true}')
    data = base / "files" / "sample.jsonl"
    with data.open("wb") as stream:
        for i in range(count):
            value = {"text": text, "metadata": {"i": i}}
            digest = hashlib.sha256(canonical_json(value)).hexdigest()
            value["_sparselab_source"] = {
                "source_row_index": i,
                "source_row_sha256": digest,
            }
            stream.write(canonical_json(value) + b"\n")
    root = tmp_path / "input"

    def pin(path):
        return {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    spec = {
        "format": "sparselab-corpus-preview-v1",
        "seed": "fixture",
        "sources": [
            {
                "source_id": "sample",
                "snapshot_sha256": snapshot,
                "manifest": pin(manifest),
                "data": pin(data),
            }
        ],
        "max_read_bytes": 64 * 1024**2,
        "max_record_bytes": 1024**2,
        "samples_per_source": 3,
        "max_excerpt_bytes": 32,
        "max_output_bytes": 65536,
        "max_wall_seconds": 60,
    }
    declaration = tmp_path / "preview.json"
    declaration.write_text(json.dumps(spec))
    return declaration, root, spec, data


def run(tmp_path, declaration, root):
    out = tmp_path / "out"
    preview(declaration, root, out)
    return json.loads((out / "report.json").read_text())


def test_complete_deterministic_counted_no_clobber(tmp_path):
    declaration, root, spec, _ = fixture(tmp_path)
    report = run(tmp_path, declaration, root)
    expected = declaration.stat().st_size + sum(
        p["bytes"] for p in [spec["sources"][0]["manifest"], spec["sources"][0]["data"]]
    )
    assert report["read_bytes"] == expected
    assert report["sources"][0]["rows"] == 20
    assert len(report["sources"][0]["items"]) == 3
    assert report["admission"] is False
    preview(declaration, root, tmp_path / "again")
    again = json.loads((tmp_path / "again/report.json").read_text())
    assert report["sources"] == again["sources"]
    with pytest.raises(FileExistsError):
        preview(declaration, root, tmp_path / "out")


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("bytes", "size differs from pin"),
        ("hash", "SHA256 differs"),
        ("record", "record-byte cap"),
        ("symlink", "symlinked preview input"),
        ("escape", "canonical relative path"),
        ("aggregate", "pinned input bytes exceed"),
        ("unknown", "Extra inputs are not permitted"),
    ],
)
def test_refuses_unsafe_or_over_budget(tmp_path, mutation, message):
    declaration, root, spec, data = fixture(tmp_path)
    source = spec["sources"][0]
    if mutation == "bytes":
        source["data"]["bytes"] += 1
    if mutation == "hash":
        source["data"]["sha256"] = "b" * 64
    if mutation == "record":
        spec["max_record_bytes"] = 10
    if mutation == "aggregate":
        spec["max_read_bytes"] = 2000
    if mutation == "escape":
        source["data"]["path"] = "../outside"
    if mutation == "unknown":
        spec["ignored"] = True
    if mutation == "symlink":
        target = data.with_suffix(".real")
        data.rename(target)
        data.symlink_to(target)
    declaration.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=message):
        run(tmp_path, declaration, root)
    assert not (tmp_path / "out/report.json").exists()
    if (tmp_path / "out").exists():
        assert (
            json.loads((tmp_path / "out/failure.json").read_text())["status"]
            == "incomplete"
        )


def test_no_all_row_retention(tmp_path):
    declaration, root, _, _ = fixture(tmp_path, count=4000, text="x" * 4096)
    tracemalloc.start()
    try:
        report = run(tmp_path, declaration, root)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024**2
    assert report["sources"][0]["rows"] == 4000


def test_deadline_and_partial_counter(tmp_path, monkeypatch):
    declaration, root, _, _ = fixture(tmp_path)
    original = Reader.read

    def expire(self, stream, count, **kwargs):
        value = original(self, stream, count, **kwargs)
        self.deadline = 0
        return value

    monkeypatch.setattr(Reader, "read", expire)
    with pytest.raises(TimeoutError):
        run(tmp_path, declaration, root)
    failure = json.loads((tmp_path / "out/failure.json").read_text())
    assert failure["read_bytes"] > declaration.stat().st_size
    assert not (tmp_path / "out/report.json").exists()


def test_changed_row_envelope_fails_even_with_pinned_file(tmp_path):
    declaration, root, spec, data = fixture(tmp_path, count=1)
    value = json.loads(data.read_text())
    value["text"] = "changed"
    data.write_bytes(canonical_json(value) + b"\n")
    spec["sources"][0]["data"].update(
        bytes=data.stat().st_size, sha256=hashlib.sha256(data.read_bytes()).hexdigest()
    )
    declaration.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="provenance"):
        run(tmp_path, declaration, root)


def test_attempt_command_public_contract():
    from sparselab.training.attempt_commands import classify_attempt_command

    command = classify_attempt_command(
        [
            "sparselab",
            "corpus",
            "preview",
            "input.json",
            "--input-root",
            "/input",
            "--output",
            "/output",
        ]
    )
    assert command.operation == "corpus preview"
    assert command.effect == "preparation"


def test_wheel_distribution_public_cli(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    import zipfile
    from pathlib import Path

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("existing uv backend required; test never installs it")
    declaration, root, _, _ = fixture(tmp_path)
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    subprocess.run(
        [uv, "build-backend", "build-wheel", "--offline", str(wheel_dir)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
    )
    distribution = tmp_path / "distribution"
    with zipfile.ZipFile(next(wheel_dir.glob("*.whl"))) as archive:
        archive.extractall(distribution)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    command = [
        sys.executable,
        "-B",
        "-m",
        "sparselab",
        "corpus",
        "preview",
        str(declaration),
        "--input-root",
        str(root),
        "--output",
        str(tmp_path / "packaged-output"),
    ]
    result = subprocess.run(
        command,
        cwd=distribution,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "complete"
    loaded = subprocess.check_output(
        [
            sys.executable,
            "-B",
            "-c",
            "import sparselab.corpus.preview as p; print(p.__file__)",
        ],
        cwd=distribution,
        env=env,
        text=True,
    )
    assert str(distribution) in loaded


def test_duplicate_spec_and_output_inside_snapshot_refuse(tmp_path):
    declaration, root, spec, data = fixture(tmp_path)
    declaration.write_text(
        declaration.read_text().replace('"seed":', '"seed": "duplicate", "seed":', 1)
    )
    with pytest.raises(ValueError, match="duplicate"):
        run(tmp_path, declaration, root)
    declaration.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="outside acquired"):
        preview(declaration, root, data.parent / "output")


def test_aggregate_counter_never_reads_past_remaining():
    import io
    import time

    reader = Reader(3, time.monotonic() + 60, 0)
    stream = io.BytesIO(b"abcdef")
    assert reader.read(stream, 100) == b"abc"
    with pytest.raises(ValueError, match="exhausted"):
        reader.read(stream, 1)
    assert stream.tell() == reader.used == 3


def repin(declaration, spec, data):
    spec["sources"][0]["data"].update(
        bytes=data.stat().st_size, sha256=hashlib.sha256(data.read_bytes()).hexdigest()
    )
    declaration.write_text(json.dumps(spec))


@pytest.mark.parametrize(
    "text, excerpt, truncated", [("abé終", "ab", True), ("abc", "abc", False)]
)
def test_utf8_excerpt_byte_bound(tmp_path, text, excerpt, truncated):
    declaration, root, spec, _ = fixture(tmp_path, count=1, text=text)
    spec["max_excerpt_bytes"] = 3
    declaration.write_text(json.dumps(spec))
    item = run(tmp_path, declaration, root)["sources"][0]["items"][0]
    assert item["excerpt"] == excerpt
    assert len(item["excerpt"].encode()) <= 3
    assert item["truncated"] is truncated
    assert item["text_bytes"] == len(text.encode())


def test_output_overflow_retains_failure(tmp_path):
    declaration, root, spec, _ = fixture(tmp_path, count=20, text="x" * 4096)
    spec.update(samples_per_source=20, max_excerpt_bytes=4096, max_output_bytes=8192)
    declaration.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="output-byte cap"):
        run(tmp_path, declaration, root)
    out = tmp_path / "out"
    assert not (out / "report.json").exists()
    failure = json.loads((out / "failure.json").read_text())
    assert "output-byte cap" in failure["error"]
    assert failure["completed_source_count"] == 1
    assert sum(p.stat().st_size for p in out.iterdir()) <= 8192


@pytest.mark.parametrize("failure_kind", ["hash", "escaped_error"])
def test_long_source_ids_and_escaped_error_fit_failure_reserve(
    tmp_path, monkeypatch, failure_kind
):
    from sparselab.corpus import preview as module

    declaration, root, spec, _ = fixture(tmp_path, count=1)
    original = spec["sources"][0]
    spec["sources"] = []
    for i in range(64):
        source = json.loads(json.dumps(original))
        source["source_id"] = "s" + str(i).zfill(3) + "x" * 124
        for kind in ("manifest", "data"):
            old_path = root / source[kind]["path"]
            source[kind]["path"] = str(i) + "/" + source[kind]["path"]
            new_path = root / source[kind]["path"]
            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.write_bytes(old_path.read_bytes())
        spec["sources"].append(source)
    spec["max_output_bytes"] = 65536
    declaration.write_text(json.dumps(spec))

    def fail(root, source, spec, reader):
        if source == spec.sources[-1]:
            raise ValueError("\x00" * 380 + "終" * 1000)
        return {"source_id": source.source_id}

    if failure_kind == "escaped_error":
        monkeypatch.setattr(module, "_source", fail)
        message = "終"
    else:
        spec["sources"][-1]["data"]["sha256"] = "b" * 64
        declaration.write_text(json.dumps(spec))
        message = "SHA256 differs from pin"
    with pytest.raises(ValueError):
        preview(declaration, root, tmp_path / "measure")
    started_size = (tmp_path / "measure/started.json").stat().st_size
    # Same digit width preserves the started receipt size after rebinding.
    spec["max_output_bytes"] = started_size + 4096
    declaration.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=message):
        run(tmp_path, declaration, root)
    out = tmp_path / "out"
    failure = json.loads((out / "failure.json").read_text())
    assert failure["completed_source_count"] == 63
    assert (
        failure["completed_sources_sha256"]
        == hashlib.sha256(
            canonical_json([s["source_id"] for s in spec["sources"][:-1]])
        ).hexdigest()
    )
    assert len(failure["error"].encode()) <= 384
    assert (out / "failure.json").stat().st_size <= 4096
    assert sum(p.stat().st_size for p in out.iterdir()) <= spec["max_output_bytes"]


def test_newline_free_record_is_bounded(tmp_path):
    declaration, root, spec, data = fixture(tmp_path, count=0)
    with data.open("wb") as stream:
        for _ in range(32):
            stream.write(b"x" * 1024**2)
    repin(declaration, spec, data)
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="record-byte cap"):
            run(tmp_path, declaration, root)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024**2


def test_nonregular_input_refused_before_open(tmp_path, monkeypatch):
    from pathlib import Path

    declaration, root, _, data = fixture(tmp_path)
    data.unlink()
    data.mkdir()
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path != data, "nonregular input must refuse before open"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="regular file"):
        run(tmp_path, declaration, root)


@pytest.mark.parametrize("target", ["root", "output"])
def test_symlinked_ancestors_refuse(tmp_path, target):
    declaration, root, _, _ = fixture(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked preview root or output"):
        preview(
            declaration,
            link / "input" if target == "root" else root,
            link / "out" if target == "output" else tmp_path / "out",
        )


def test_changed_during_read_refuses(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace

    declaration, root, _, _ = fixture(tmp_path)
    original, calls = os.fstat, 0

    def changed(fd):
        nonlocal calls
        calls += 1
        result = original(fd)
        if calls == 2:
            fields = {
                name: getattr(result, name)
                for name in (
                    "st_dev",
                    "st_ino",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            }
            fields["st_mtime_ns"] += 1
            return SimpleNamespace(**fields)
        return result

    monkeypatch.setattr(os, "fstat", changed)
    with pytest.raises(ValueError, match="changed while reading"):
        run(tmp_path, declaration, root)


@pytest.mark.parametrize("indices", [[0, 0], [1, 0], [True], [1.0]])
def test_invalid_indices_refuse(tmp_path, indices):
    declaration, root, spec, data = fixture(tmp_path, count=len(indices))
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    for row, index in zip(rows, indices, strict=True):
        row["_sparselab_source"]["source_row_index"] = index
    data.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    repin(declaration, spec, data)
    with pytest.raises(ValueError, match="unordered index"):
        run(tmp_path, declaration, root)


@pytest.mark.parametrize(
    "mutation, error, message",
    [
        ("text", TypeError, "must contain text"),
        ("envelope", ValueError, "unsupported acquisition envelope"),
    ],
)
def test_unsupported_row_refuses(tmp_path, mutation, error, message):
    declaration, root, spec, data = fixture(tmp_path, count=1)
    row = json.loads(data.read_text())
    if mutation == "text":
        del row["text"]
        value = {k: v for k, v in row.items() if k != "_sparselab_source"}
        row["_sparselab_source"]["source_row_sha256"] = hashlib.sha256(
            canonical_json(value)
        ).hexdigest()
    else:
        row["_sparselab_source"] = {"page_id": 123}
    data.write_bytes(canonical_json(row) + b"\n")
    repin(declaration, spec, data)
    with pytest.raises(error, match=message):
        run(tmp_path, declaration, root)


def test_lowest_ranks_and_blank_lines(tmp_path):
    declaration, root, spec, data = fixture(tmp_path, count=20)
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    data.write_bytes(b"\n" + data.read_bytes())
    repin(declaration, spec, data)
    source = spec["sources"][0]
    ranked = []
    for row in rows:
        e = row["_sparselab_source"]
        index = e["source_row_index"]
        rank = hashlib.sha256(
            canonical_json(
                [
                    spec["seed"],
                    source["source_id"],
                    source["snapshot_sha256"],
                    index,
                    e["source_row_sha256"],
                ]
            )
        ).hexdigest()
        ranked.append((rank, index))
    report = run(tmp_path, declaration, root)["sources"][0]
    assert report["rows"] == 20
    assert report["blank_lines"] == 1
    assert [item["source_row_index"] for item in report["items"]] == [
        i for _, i in sorted(ranked)[:3]
    ]
    assert all(
        item["physical_line"] == item["source_row_index"] + 2
        for item in report["items"]
    )


def test_preview_dispatch_requires_output():
    from sparselab.training.attempt_commands import classify_attempt_command

    with pytest.raises(ValueError, match="required native option"):
        classify_attempt_command(
            ["sparselab", "corpus", "preview", "input.json", "--input-root", "/input"]
        )


def test_success_receipt_cannot_consume_failure_reserve(tmp_path):
    declaration, root, spec, _ = fixture(tmp_path, count=3, text="x" * 4096)
    spec.update(max_excerpt_bytes=4096)
    declaration.write_text(json.dumps(spec))
    preview(declaration, root, tmp_path / "measure")
    report_bytes = sum(p.stat().st_size for p in (tmp_path / "measure").iterdir())
    spec["max_output_bytes"] = report_bytes + 1000
    declaration.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="output-byte cap"):
        run(tmp_path, declaration, root)
    assert (tmp_path / "out/failure.json").exists()
    assert not (tmp_path / "out/report.json").exists()


@pytest.mark.parametrize("include_text", [False, True])
def test_declared_text_field_preserves_original_row_hash(tmp_path, include_text):
    declaration, root, spec, data = fixture(tmp_path, count=1)
    value = {"code": "print('hello')"}
    if include_text:
        value["text"] = "not the selected content"
    digest = hashlib.sha256(canonical_json(value)).hexdigest()
    value["_sparselab_source"] = {"source_row_index": 0, "source_row_sha256": digest}
    data.write_bytes(canonical_json(value) + b"\n")
    spec["sources"][0]["text_field"] = "code"
    repin(declaration, spec, data)
    source = run(tmp_path, declaration, root)["sources"][0]
    assert source["text_field"] == "code"
    assert source["items"][0]["excerpt"] == "print('hello')"
    assert source["items"][0]["source_row_sha256"] == digest
    # Re-pinning the file does not excuse mutation of the original row envelope.
    value["code"] = "changed code"
    data.write_bytes(canonical_json(value) + b"\n")
    repin(declaration, spec, data)
    with pytest.raises(ValueError, match="provenance"):
        preview(declaration, root, tmp_path / "changed")


@pytest.mark.parametrize("value", [None, 17, {"nested": "content"}])
def test_selected_text_field_must_be_string_without_fallback(tmp_path, value):
    declaration, root, spec, data = fixture(tmp_path, count=1)
    row = {"text": "a valid default must not be used", "code": value}
    row["_sparselab_source"] = {
        "source_row_index": 0,
        "source_row_sha256": hashlib.sha256(canonical_json(row)).hexdigest(),
    }
    data.write_bytes(canonical_json(row) + b"\n")
    spec["sources"][0]["text_field"] = "code"
    repin(declaration, spec, data)
    with pytest.raises(TypeError, match="must contain code as a string"):
        run(tmp_path, declaration, root)


@pytest.mark.parametrize("field", ["", None, 1, "x" * 129])
def test_invalid_text_field_declaration_refuses(tmp_path, field):
    declaration, root, spec, _ = fixture(tmp_path)
    spec["sources"][0]["text_field"] = field
    declaration.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="text_field"):
        run(tmp_path, declaration, root)
    assert not (tmp_path / "out").exists()
