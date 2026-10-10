"""Exercise cold input authentication and unsafe path rejection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE

from sparselab.corpus import token_denominator_identity as identity
from sparselab.corpus.release import _verification_operation, verify_release
from sparselab.training.manifest import sha256_file

pytest_plugins = ("test_corpus_token_denominator",)


@pytest.fixture
def lm_release(frozen: tuple[Path, Path, Path]) -> Path:
    """Use the same tiny cold-built input as the canonical measurement tests."""
    return frozen[0]


def _tokenizer(root: Path, vocab: int) -> Path:
    root.mkdir(parents=True)
    path = root / "tokenizer.json"
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.add_special_tokens(["<unk>"])
    tokenizer.add_tokens([f"word_{index}" for index in range(vocab - 1)])
    tokenizer.save(str(path))
    return path


def test_normal_path_fully_authenticates_inputs(
    lm_release: Path, tmp_path: Path
) -> None:
    tokenizer = _tokenizer(tmp_path / "identity-tokenizer", 2)
    (tokenizer.parent / "tokenizer_manifest.json").write_text(
        json.dumps(
            {
                "sha256": sha256_file(tokenizer),
                "source": "synthetic",
                "revision": None,
                "vocab_size": 2,
            }
        )
    )
    with _verification_operation():
        verify_release(
            lm_release
        )  # authentic upstream proof reused inside this operation
        manifest, bindings = identity._authenticate_inputs(lm_release, tokenizer)
    assert manifest["release_id"] == lm_release.name
    assert bindings == {
        "release_manifest_sha256": sha256_file(lm_release / "manifest.json"),
        "tokenizer_sha256": sha256_file(tokenizer),
        "tokenizer_manifest_sha256": sha256_file(
            tokenizer.with_name("tokenizer_manifest.json")
        ),
        "evidence": None,
    }
    tokenizer.write_bytes(tokenizer.read_bytes() + b" ")
    with pytest.raises(ValueError, match="provenance or digest"):
        identity._authenticate_inputs(lm_release, tokenizer)
    copied = tmp_path / "releases" / lm_release.name
    shutil.copytree(lm_release, copied)
    tampered = json.loads((copied / "manifest.json").read_bytes())
    tampered["release_id"] = "0" * 64
    (copied / "manifest.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="directory identity"):
        identity._authenticate_inputs(copied, tokenizer)


def test_normal_operation_rejects_changed_manifest(
    lm_release: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "releases" / lm_release.name
    shutil.copytree(lm_release, copied)
    shutil.copytree(lm_release.parent.parent / "snapshots", tmp_path / "snapshots")
    tokenizer = _tokenizer(tmp_path / "identity-tokenizer", 2)
    (tokenizer.parent / "tokenizer_manifest.json").write_text(
        json.dumps(
            {
                "sha256": sha256_file(tokenizer),
                "source": "synthetic",
                "revision": None,
                "vocab_size": 2,
            }
        )
    )
    with _verification_operation():
        verify_release(copied)
        with (copied / "manifest.json").open("ab") as stream:
            stream.write(b"\n")
        with pytest.raises(ValueError, match="changed within verification operation"):
            identity._authenticate_inputs(copied, tokenizer)


def test_unsafe_paths(tmp_path: Path) -> None:
    directory = tmp_path / "actual"
    directory.mkdir()
    (tmp_path / "linked").symlink_to(directory)
    with pytest.raises(ValueError, match="symlink"):
        identity._safe_path(tmp_path / "linked" / "not-yet-created")
    with pytest.raises(ValueError, match="parent traversal"):
        identity._safe_path(tmp_path / "actual" / ".." / "other")
