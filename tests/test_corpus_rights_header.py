"""Synthetic rights-only regressions; total input fixtures stay below 2 MiB.

No acquisition, model, tokenizer, process pool or normalization workload. The
reuse test seeds an empty native prepared receipt with a stub record parser,
then exercises build_large through actual receipt verification, stopping before
indexing. Snapshot verification is stubbed; this does not certify acquisition.
"""

from pathlib import Path

import pytest

from sparselab.corpus import acquisition, large_build, pipeline
from sparselab.corpus.large_build import _RIGHTS_HEADER_MAX_BYTES, _read_rights_header
from sparselab.corpus.progress import BuildProgress
from sparselab.corpus.project import load_project
from sparselab.corpus.rights import RightsPolicy, resolve_file_rights
from sparselab.training.manifest import sha256_file


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"ordinary text",
        b"# SPDX-License-Identifier: MIT\nCopyright Example\nbody\n",
        b"# SPDX-License-Identifier: MIT\r\nCopyright Example\r\nbody\r\n",
        b"# SPDX-License-Identifier: MIT\nCopyright Example",
        b"Copyright \xff\nSPDX-License-Identifier: MIT\n",
        b"SPDX-License-Identifier: MIT\nSPDX-License-Identifier: GPL-3.0-only\n",
        b"\n" * 29 + b"SPDX-License-Identifier: GPL-3.0-only\n",
        b"\n" * 30 + b"SPDX-License-Identifier: GPL-3.0-only\n",
        b"Copyright Example\rSPDX-License-Identifier: MIT\n",
        b"x" * 8192 + b"\nSPDX-License-Identifier: MIT\n",
    ],
)
def test_header_preserves_full_bytes_rights(tmp_path, raw):
    path = tmp_path / "source.md"
    path.write_bytes(raw)
    policy = RightsPolicy(
        training_eligibility="eligible",
        redistribution_mode="redistributable_under_source_terms",
        spdx_expression="MIT",
    )
    assert resolve_file_rights(
        policy, path.name, _read_rights_header(path, path.name)
    ) == (resolve_file_rights(policy, path.name, raw))


@pytest.mark.parametrize("suffix", [".json", ".jsonl", ".parquet"])
def test_dataset_header_does_not_open_file(tmp_path, suffix):
    path = tmp_path / ("missing" + suffix)
    assert _read_rights_header(path, path.name) == b""


@pytest.mark.parametrize(
    ("prefix", "tail", "accepted"),
    [
        (b"", b"", True),
        (b"", b"x", False),
        (b"", b"\n", False),
        (b"\n" * 29, b"", True),
        (b"\n" * 29, b"x", False),
        (b"\n" * 29, b"\n", False),
    ],
)
def test_header_byte_ceiling(tmp_path, prefix, tail, accepted):
    path = tmp_path / "source.md"
    raw = prefix + b"x" * (_RIGHTS_HEADER_MAX_BYTES - len(prefix)) + tail
    path.write_bytes(raw)
    if accepted:
        assert _read_rights_header(path, path.name) == raw
    else:
        with pytest.raises(ValueError, match="rights header exceeds 65536 bytes"):
            _read_rights_header(path, path.name)


@pytest.mark.parametrize("lines", [29, 30])
def test_lf_at_ceiling_requires_eof_or_thirty_lines(tmp_path, lines):
    path = tmp_path / "source.md"
    header = b"x" * (_RIGHTS_HEADER_MAX_BYTES - lines) + b"\n" * lines
    path.write_bytes(header + b"SPDX-License-Identifier: GPL-3.0-only\n")
    if lines == 30:
        assert _read_rights_header(path, path.name) == header
    else:
        with pytest.raises(ValueError, match="rights header exceeds"):
            _read_rights_header(path, path.name)
    path.write_bytes(header)
    assert _read_rights_header(path, path.name) == header


def test_binary_reads_have_a_total_byte_bound(tmp_path, monkeypatch):
    import io

    raw = b"SPDX-License-Identifier: MIT\n" + b"x" * _RIGHTS_HEADER_MAX_BYTES
    sizes = []

    class Reader(io.BytesIO):
        def readline(self, size=-1):
            assert 0 < size <= _RIGHTS_HEADER_MAX_BYTES + 1
            line = super().readline(size)
            sizes.append(len(line))
            return line

    monkeypatch.setattr(Path, "open", lambda *a, **kw: Reader(raw))
    with pytest.raises(ValueError, match="rights header exceeds"):
        _read_rights_header(tmp_path / "source.md", "source.md")
    assert sum(sizes) == _RIGHTS_HEADER_MAX_BYTES + 1


@pytest.mark.parametrize(
    "suffix,ceiling",
    [
        (".json", None),
        (".jsonl", None),
        (".parquet", None),
        (".md", None),
        (".JSON", None),
        (".md", 29),
    ],
)
def test_build_rights_before_actual_prepared_reuse(
    tmp_path, monkeypatch, suffix, ceiling
):
    project = load_project("corpora/devmind-v2/corpus.yaml")
    source = next(s for s in project.sources if s.id == "fineweb_edu_sample_10bt")
    project = project.model_copy(update={"sources": (source,)})
    snapshot = tmp_path / "snapshot"
    (snapshot / "files").mkdir(parents=True)
    path = snapshot / "files" / ("source" + suffix)
    raw = b"\n" * 30 + b"SPDX-License-Identifier: GPL-3.0-only\n"
    path.write_bytes(raw)
    file = {"path": path.name, "sha256": sha256_file(path), "size": len(raw)}
    monkeypatch.setattr(
        acquisition, "verify_snapshot", lambda *a, **kw: {"files": [file]}
    )
    monkeypatch.setattr(pipeline, "_records_for_file", lambda *a, **kw: [])
    decision = resolve_file_rights(
        source.rights, path.name, raw, prospective_private_research=True
    )
    progress = BuildProgress(tmp_path / "progress.jsonl", "header-reuse")
    large_build._prepare_file(
        build_id="header-reuse",
        prepared_root=tmp_path / "builds/prepared/header-reuse",
        source=source,
        file=file,
        snapshot_sha="0" * 64,
        source_path=path,
        decision=decision,
        progress=progress,
    )
    if ceiling is not None:
        monkeypatch.setattr(large_build, "_RIGHTS_HEADER_MAX_BYTES", ceiling)
    original_open = Path.open
    opens = []

    def guarded_open(self, *args, **kwargs):
        if self == path:
            assert suffix not in {".json", ".jsonl", ".parquet"}
            opens.append(self)
        return original_open(self, *args, **kwargs)

    original_read = Path.read_bytes

    def guarded_read(self):
        assert self != path, "whole-source read before prepared reuse"
        return original_read(self)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    actual_resolve = large_build.resolve_file_rights

    def resolve(*args, **kwargs):
        result = actual_resolve(*args, **kwargs)
        assert result == decision
        return result

    monkeypatch.setattr(large_build, "resolve_file_rights", resolve)

    def stop(*args, **kwargs):
        raise RuntimeError("stop before indexing")

    monkeypatch.setattr(large_build, "_index_prepared", stop)
    error = RuntimeError if ceiling is None else ValueError
    message = "stop before indexing" if ceiling is None else "rights header exceeds"
    with pytest.raises(error, match=message):
        large_build.build_large(
            project,
            tmp_path,
            {
                "sources": {
                    source.id: {
                        "snapshot_path": str(snapshot),
                        "snapshot_sha256": "0" * 64,
                    }
                }
            },
            {},
            "header-reuse",
            tmp_path / "result",
            progress,
        )
    assert len(opens) == (0 if suffix in {".json", ".jsonl", ".parquet"} else 1)
    events = (tmp_path / "progress.jsonl").read_text()
    assert ('"reused_verified_shard"' in events) == (ceiling is None)
