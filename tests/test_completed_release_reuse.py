"""Tiny native payloads exercise explicit no-copy release publication.

The shared fixture substitutes only historical inventory-executor adoption;
native rendering, completion storage and output verification remain real.
"""

import json
import os
import sys
from pathlib import Path

import pytest
from test_completed_preparation import fixture

from sparselab.corpus import completed_preparation, completed_release, pipeline
from sparselab.corpus.admission_inspection_cache import InspectionCache
from sparselab.corpus.export import _release
from sparselab.corpus.release import _digest, _manifest_payload, freeze, verify_release
from sparselab.corpus.split_runs import _stamp
from sparselab.training.manifest import canonical_json


def built(tmp_path, monkeypatch):
    root, _recipe, project = fixture(tmp_path, monkeypatch)
    path = pipeline.build(project, root, offline=True, reuse_completed_preparation=True)
    manifest = json.loads((path / "build.json").read_bytes())
    return root, path, manifest


def test_in_place_freeze_preserves_payload_and_original_completion(
    tmp_path, monkeypatch
):
    root, path, build = built(tmp_path, monkeypatch)
    before = {name: _stamp(path / name) for name in (*build["files"], "build.json")}
    original = (path / "build.json").read_bytes()

    def copied(*_args, **_kwargs):
        pytest.fail("release freeze copied a payload file")

    monkeypatch.setattr("shutil.copyfile", copied)
    result = freeze(path, root, reuse_completed_preparation=True)
    assert result == path
    assert (path / "build.json").read_bytes() == original
    assert before == {name: _stamp(path / name) for name in before}
    assert all((path / name).stat().st_nlink == 1 for name in before)
    manifest = verify_release(result)
    assert manifest["release_id"] == _digest(_manifest_payload(build))
    assert _release(result) == manifest  # Real export consumer, not path.name identity.
    assert completed_preparation.verify_completed_build(path) == build
    assert freeze(path, root, reuse_completed_preparation=True) == result
    note = json.loads((path / "release-publication.json").read_bytes())
    assert note["kind"] == "adopted_execution"


def test_explicit_release_uses_historical_signature_without_relaxing_build_reuse(
    tmp_path, monkeypatch
):
    root, path, _build = built(tmp_path, monkeypatch)
    original = InspectionCache.__init__

    def changed_authority(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.authority = {
            **self.authority,
            "corpus": {"fixture_changed_verifier": True},
        }

    monkeypatch.setattr(InspectionCache, "__init__", changed_authority)
    with pytest.raises(ValueError, match="completed native build evidence unavailable"):
        completed_preparation.verify_completed_build(path)
    assert freeze(path, root, reuse_completed_preparation=True) == path
    assert verify_release(path)["build_id"] == path.name


@pytest.mark.parametrize("change", ["payload", "producer", "proof"])
def test_missing_or_changed_native_evidence_cannot_publish(
    tmp_path, monkeypatch, change
):
    root, path, _build = built(tmp_path, monkeypatch)
    if change == "payload":
        p = path / "lm/train.jsonl"
        info = p.stat()
        raw = p.read_bytes()
        p.write_bytes(raw.replace(b"Generated", b"Changed!!", 1))
        os.utime(p, ns=(info.st_atime_ns, info.st_mtime_ns))
    elif change == "producer":
        monkeypatch.setattr(
            completed_release, "_producer", lambda: {"changed": "f" * 64}
        )
    else:
        from sparselab.verification_proofs import ProofStore

        monkeypatch.setattr(ProofStore, "lookup", lambda *_args, **_kwargs: False)
    with pytest.raises((ValueError, FileNotFoundError)):
        freeze(path, root, reuse_completed_preparation=True)
    assert not (path / "manifest.json").exists()


@pytest.mark.parametrize("change", ["payload", "missing", "manifest", "note"])
def test_later_mutation_or_missing_payload_refuses_release_use(
    tmp_path, monkeypatch, change
):
    root, path, _build = built(tmp_path, monkeypatch)
    freeze(path, root, reuse_completed_preparation=True)
    if change == "payload":
        p = path / "lm/train.jsonl"
        info = p.stat()
        p.write_bytes(p.read_bytes().replace(b"Generated", b"Changed!!", 1))
        os.utime(p, ns=(info.st_atime_ns, info.st_mtime_ns))
    elif change == "missing":
        (path / "lm/train.jsonl").unlink()
    elif change == "manifest":
        p = path / "manifest.json"
        value = json.loads(p.read_bytes())
        value["release_id"] = "0" * 64
        p.write_bytes(canonical_json(value) + b"\n")
    else:
        (path / "release-publication.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_release(path)


def test_metadata_interruption_repeats_without_rebuilding_payload(
    tmp_path, monkeypatch
):
    root, path, build = built(tmp_path, monkeypatch)
    before = {name: _stamp(path / name) for name in (*build["files"], "build.json")}
    original = completed_release._publish_metadata

    def interrupted(target, *args):
        if target.name == "manifest.json":
            raise OSError("fixture interruption before final commit")
        return original(target, *args)

    with monkeypatch.context() as patched:
        patched.setattr(completed_release, "_publish_metadata", interrupted)
        with pytest.raises(OSError, match="fixture interruption"):
            freeze(path, root, reuse_completed_preparation=True)
    assert not (path / "manifest.json").exists()
    assert freeze(path, root, reuse_completed_preparation=True) == path
    assert before == {name: _stamp(path / name) for name in before}


def test_public_cli_explicit_selection(tmp_path, monkeypatch, capsys):
    from sparselab.cli.main import main

    root, path, _build = built(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            "--work-dir",
            str(root),
            "corpus",
            "freeze",
            str(path),
            "--reuse-completed-preparation",
        ],
    )
    main()
    assert Path(capsys.readouterr().out.strip().splitlines()[-1]) == path
    assert verify_release(path)["release_id"]


def test_in_place_release_exports_fits_and_packs_native_inputs(tmp_path, monkeypatch):
    from sparselab.config.loading import load_config, load_tokenizer_config
    from sparselab.corpus.export import export_release, verify_release_export
    from sparselab.data.packing import prepare_data
    from sparselab.data.tokenizer import load_tokenizer, train_tokenizer
    from sparselab.experiments.artifacts import verify_artifact
    from sparselab.experiments.plan import Artifact

    root, path, _build = built(tmp_path, monkeypatch)
    freeze(path, root, reuse_completed_preparation=True)
    release = verify_release(path)
    exported = export_release(
        path, "lm", Path("configs/runtime_smoke_cpu.yaml"), 300, root
    )
    run = load_config(exported / "run.yaml")
    assert verify_release_export(run.dataset)["release_id"] == release["release_id"]
    tokenizer = train_tokenizer(load_tokenizer_config(exported / "tokenizer.yaml"))
    packed = prepare_data(run, load_tokenizer(tokenizer))
    assert packed.manifest["train"]["retained_documents"] > 0
    artifact = Artifact(
        kind="corpus_release",
        version=1,
        producer="sparselab",
        identifier=release["release_id"],
        sha256=release["release_id"],
        path=str(path),
    )
    assert (
        verify_artifact(artifact, path / "manifest.json")["identifier"]
        == release["release_id"]
    )
