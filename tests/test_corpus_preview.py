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
    "mutation", ["bytes", "hash", "record", "symlink", "escape", "aggregate", "unknown"]
)
def test_refuses_unsafe_or_over_budget(tmp_path, mutation):
    declaration, root, spec, data = fixture(tmp_path)
    source = spec["sources"][0]
    if mutation == "bytes":
        source["data"]["bytes"] += 1
    if mutation == "hash":
        source["data"]["sha256"] = "b" * 64
    if mutation == "record":
        spec["max_record_bytes"] = 10
    if mutation == "aggregate":
        spec["max_read_bytes"] = 10
    if mutation == "escape":
        source["data"]["path"] = "../outside"
    if mutation == "unknown":
        spec["ignored"] = True
    if mutation == "symlink":
        target = data.with_suffix(".real")
        data.rename(target)
        data.symlink_to(target)
    declaration.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
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
