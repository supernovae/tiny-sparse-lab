"""Snapshot verification belongs in semantic authorities, not array hash proofs."""

from __future__ import annotations

import pytest

from sparselab import verifier_authority as authority


@pytest.mark.parametrize("kind", ["file", "prepared_array", "prepared_data"])
def test_packed_inventory_authority_excludes_source_preparation(kind):
    modules = authority.verifier_authority(kind, 1)["modules"]
    assert "data.packing" in modules
    assert "data.verification" in modules
    assert "data.sources" not in modules
    assert "corpus.acquisition" not in modules


@pytest.mark.parametrize(
    "module", ["data.sources", "corpus.acquisition", "corpus.project"]
)
def test_tokenizer_authority_invalidates_on_snapshot_verifier_chain_change(
    monkeypatch, module
):
    before = authority.verifier_authority("tokenizer", 1)
    array_before = authority.verifier_authority("prepared_array", 1)
    assert {
        "data.sources",
        "corpus.acquisition",
        "corpus.project",
        "data.tokenizer",
        "training.manifest",
        "hf_auth",
        "engram.packs",
    } <= set(before["modules"])
    target = authority.Path(authority.__file__).parent / (
        module.replace(".", "/") + ".py"
    )
    read = authority.Path.read_bytes

    def changed(path):
        raw = read(path)
        return (
            raw + b"\n# changed snapshot verifier semantics\n"
            if path == target
            else raw
        )

    monkeypatch.setattr(authority.Path, "read_bytes", changed)
    assert authority.verifier_authority("tokenizer", 1)["sha256"] != before["sha256"]
    assert (
        authority.verifier_authority("prepared_array", 1)["sha256"]
        == array_before["sha256"]
    )


@pytest.mark.parametrize(
    "kind", ["tokenizer", "source_snapshot", "stage_bundle", "corpus_export"]
)
def test_semantic_domains_retain_snapshot_verifier(kind):
    assert "data.sources" in authority.verifier_authority(kind, 1)["modules"]


@pytest.mark.parametrize(
    "module, function",
    [("sources", "verify_snapshot"), ("tokenizer", "verify_tokenizer_artifact")],
)
def test_new_snapshot_use_in_array_verifier_requires_reaudit(
    monkeypatch, module, function
):
    read = authority.Path.read_bytes

    def changed(path):
        raw = read(path)
        if path.name == "packing.py":
            return (
                raw
                + f"""\ndef _new_array_verifier(config):
    from sparselab.data.{module} import {function} as verify_generic_snapshot
    return verify_generic_snapshot(config)
""".encode()
            )
        return raw

    monkeypatch.setattr(authority.Path, "read_bytes", changed)
    with pytest.raises(
        ValueError, match="verifier exclusion needs review.*data.packing"
    ):
        authority.verifier_authority("prepared_array", 1)


def test_all_declared_domains_have_reviewed_closures():
    for kind in authority._CLOSURES:
        result = authority.verifier_authority(kind, 1)
        assert result["kind"] == kind
        assert result["modules"]


def test_tokenizer_authority_follows_new_snapshot_import_helper(monkeypatch):
    package = authority.Path(authority.__file__).parent
    source = package / "data" / "sources.py"
    helper = package / "data" / "snapshot_import.py"
    read = authority.Path.read_bytes
    is_file = authority.Path.is_file
    helper_bytes = b"def verify_import(value):\n    return value\n"
    array_before = authority.verifier_authority("prepared_array", 1)

    def files(path):
        return path == helper or is_file(path)

    def content(path):
        if path == helper:
            return helper_bytes
        raw = read(path)
        if path == source:
            return (
                raw
                + b"""\ndef _verify_imported_snapshot(value):
    from sparselab.data.snapshot_import import verify_import
    return verify_import(value)
"""
            )
        return raw

    monkeypatch.setattr(authority.Path, "is_file", files)
    monkeypatch.setattr(authority.Path, "read_bytes", content)
    before = authority.verifier_authority("tokenizer", 1)
    assert "data.snapshot_import" in before["modules"]
    helper_bytes += b"\n# revised imported-snapshot verification\n"
    after = authority.verifier_authority("tokenizer", 1)
    assert (
        after["modules"]["data.snapshot_import"]
        != before["modules"]["data.snapshot_import"]
    )
    assert after["sha256"] != before["sha256"]
    assert (
        authority.verifier_authority("prepared_array", 1)["sha256"]
        == array_before["sha256"]
    )
