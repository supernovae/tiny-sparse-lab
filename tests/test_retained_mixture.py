"""Tiny authenticated-stage fixtures for additive-mixture metadata reuse."""

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sparselab.corpus import mixture
from sparselab.corpus.admission_inspection_cache import capture
from sparselab.corpus.completed_preparation import _producer
from sparselab.corpus.mixture import MixtureDeclaration, _rank, _schedule
from sparselab.corpus.mixture_index import MixtureIndex
from sparselab.corpus.retained_mixture import (
    RetainedMixtureIndex,
    _cross_release_isolation,
    _phase,
    prepared_release,
    verify_additive_anchor,
)
from sparselab.corpus.split_runs import SplitStages, rows, write_rows


def test_multiple_additional_releases_are_refused_before_any_io(tmp_path):
    declaration = {
        "schema_version": 1,
        "release_path": str(tmp_path / "base"),
        "additional_releases": [
            {"release_path": str(tmp_path / "delta1")},
            {"release_path": str(tmp_path / "delta2")},
        ],
        "tokenizer_config": str(tmp_path / "tokenizer.yaml"),
        "family_inventory": str(tmp_path / "families.jsonl"),
        "source_strata": {"code": "code"},
        "target_quotas": {"code": 10},
        "seed": 86,
        "max_exposures": 1,
    }
    with pytest.raises(ValueError, match="at most 1"):
        MixtureDeclaration.model_validate(declaration)
    with pytest.raises(ValueError, match="at most one additional release"):
        RetainedMixtureIndex(tmp_path, tmp_path, [{}, {}, {}], 86, {})


def test_retained_phase_reuses_signed_execution_and_refuses_mutation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    request = {"producer": _producer(), "inputs": {}}
    stages = SplitStages(tmp_path / "stages", request, work_root=tmp_path)
    result = stages.phase(
        "build-frozen-family-bindings",
        lambda d: {
            "files": [write_rows(d / "refs.jsonl", [["row"]])],
            "references": str(d / "refs.jsonl"),
        },
    )
    loaded, evidence = _phase(tmp_path, stages.root, "build-frozen-family-bindings")
    assert loaded == result
    assert len(evidence["historical_proof_sha256"]) == 64
    Path(result["references"]).write_text("changed\n")
    with pytest.raises(
        ValueError,
        match="historical proof invalid|semantic binding differs|output changed",
    ):
        _phase(tmp_path, stages.root, "build-frozen-family-bindings")


@pytest.mark.parametrize("legacy", [False, True])
def test_prepared_release_reads_only_authenticated_metadata(
    tmp_path, monkeypatch, legacy
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    workspace = tmp_path / "corpora" / "fixture"
    release = workspace / "builds" / "release"
    release.mkdir(parents=True)
    producer = {**_producer(), **({"pipeline.py": "0" * 64} if legacy else {})}
    request = {"producer": producer, "inputs": {}}
    stages = SplitStages(workspace / "build-stages", request, work_root=tmp_path)
    stages.phase(
        "build-frozen-family-bindings",
        lambda d: {
            "files": [write_rows(d / "refs", [["row"]])],
            "references": str(d / "refs"),
            "documents": 1,
        },
    )
    stages.phase(
        "build-deduplicate",
        lambda d: {
            "files": [write_rows(d / "parents", [[0]])],
            "parents": str(d / "parents"),
        },
    )
    stages.phase(
        "build-duplicate-keys",
        lambda d: {
            "files": [write_rows(d / "origins", [])],
            "origins": str(d / "origins"),
        },
    )
    source_stages = SplitStages(
        workspace / "source-build-stages" / "source", request, work_root=tmp_path
    )

    def render(directory):
        prepared = directory / "prepared"
        prepared.mkdir()
        docs = write_rows(prepared / "docs.jsonl", [{"text": "never read"}])
        receipt = write_rows(
            prepared / "receipt.json",
            [
                {
                    "identity": {
                        "source_id": "source",
                        "snapshot_sha256": "snapshot",
                        **(
                            {
                                "build_id": source_stages.identity,
                                "admission_sha256": "admission",
                            }
                            if legacy
                            else {}
                        ),
                    },
                    **(
                        {}
                        if legacy
                        else {
                            "provenance": {
                                "build_id": source_stages.identity,
                                "admission_sha256": "admission",
                            }
                        }
                    ),
                }
            ],
        )
        return {"files": [docs, receipt], "prepared": str(prepared)}

    source_stages.phase("native-shard-render", render)
    manifest = {
        "release_id": "release",
        "build_identity": {
            "completed_preparation": {
                "request": request,
                **(
                    {}
                    if legacy
                    else {"source_stage_ids": {"source": source_stages.identity}}
                ),
            },
            "release": {"record_admission": {"sha256": "admission"}},
            "lock": {"source": {"snapshot_sha256": "snapshot"}},
        },
    }
    monkeypatch.setattr(
        "sparselab.corpus.retained_mixture.verify_retained_release", lambda _: manifest
    )
    from sparselab import verification_proofs

    original_authority = verification_proofs.verifier_authority
    if legacy:
        monkeypatch.setattr(
            verification_proofs,
            "verifier_authority",
            lambda kind, version: {
                **original_authority(kind, version),
                "fixture_new_verifier": True,
            },
        )
    original = Path.open

    def no_document_body(path, *args, **kwargs):
        if path.name == "docs.jsonl":
            raise AssertionError("document body was opened")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", no_document_body)
        cohort = prepared_release(tmp_path, release)
    assert cohort["documents"] == 1
    assert cohort["source_ids"] == ["source"]
    assert len(cohort["evidence"]["sources"]["source"]["historical_proof_sha256"]) == 64
    proof_path = Path(cohort["evidence"]["sources"]["source"]["proof_path"])
    saved = json.loads(proof_path.read_bytes())
    saved["proof"]["binding"]["upstream_identities"]["authority"]["streaming_producer"][
        "pipeline.py"
    ] = "f" * 64
    proof_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="historical proof invalid"):
        prepared_release(tmp_path, release)


def test_retained_offsets_rank_without_reading_old_document_bodies(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    docs = tmp_path / "docs.jsonl"
    entries = [
        {
            "document_id": "a",
            "source_id": "code-source",
            "content_sha256": "x",
            "domains": ["code"],
            "rights": {"training_eligibility": "eligible"},
            "text": "a",
        },
        {
            "document_id": "b",
            "source_id": "code-source",
            "content_sha256": "y",
            "domains": ["code"],
            "rights": {"training_eligibility": "eligible"},
            "text": "b",
        },
        {
            "document_id": "c",
            "source_id": "code-source",
            "content_sha256": "x",
            "domains": ["code"],
            "rights": {"training_eligibility": "eligible"},
            "text": "c",
        },
    ]
    raw = [(json.dumps(x) + "\n").encode() for x in entries]
    docs.write_bytes(b"".join(raw))
    references = tmp_path / "references.jsonl"
    offsets = [0, len(raw[0]), len(raw[0]) + len(raw[1])]
    write_rows(
        references,
        [
            [
                x["document_id"],
                0,
                offsets[i],
                len(raw[i]),
                "validation" if i == 1 else "train",
                "raw",
                x["content_sha256"],
                "rights",
                False,
                "source",
                ["code"],
                "family",
                "code",
            ]
            for i, x in enumerate(entries)
        ],
    )
    parents = tmp_path / "parents.bin"
    parents.write_bytes(struct.pack("<QQQ", 0, 1, 0))
    origins = tmp_path / "origins.jsonl"
    write_rows(origins, [[1, "x", 0, "a", "train", "rights"]])
    cache = tmp_path / "cache.json"
    proof = tmp_path / "proof.json"
    cache.write_text("cache")
    proof.write_text("proof")
    evidence = {
        "completion_path": str(cache),
        "cache_path": str(cache),
        "proof_path": str(proof),
    }
    cohort = {
        "release_id": "release",
        "manifest": {"corpus_id": "fixture"},
        "references": str(references),
        "parents": str(parents),
        "origins": str(origins),
        "documents": 3,
        "sources": [str(docs)],
        "source_ids": ["code-source"],
        "evidence": {
            "references": evidence,
            "deduplication": evidence,
            "origins": evidence,
            "sources": {"code-source": evidence},
        },
    }
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    original = Path.open

    def no_body_before_schedule(path, *args, **kwargs):
        if path == docs:
            raise AssertionError("document body read during metadata indexing")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", no_body_before_schedule)
        index = RetainedMixtureIndex(
            scratch, tmp_path, [cohort], 86, {"code-source": "code"}
        )
    with index:
        candidates = list(index.candidates("code"))
        assert [(doc["document_id"], family) for doc, family in candidates] == [
            ("a", "family")
        ]
        assert candidates[0][0]["split"] == "train"
        index.verify_inputs()


@pytest.mark.parametrize(
    "other_split,other_rights,match",
    [
        ("validation", "rights", "different splits"),
        ("train", "different", "incompatible rights"),
    ],
)
def test_cross_release_overlap_refuses_split_and_rights_conflicts(
    tmp_path, other_split, other_rights, match
):
    left, right = tmp_path / "left.jsonl", tmp_path / "right.jsonl"
    write_rows(left, [[1, "digest", 0, "one", "train", "rights"]])
    write_rows(right, [[1, "digest", 0, "two", other_split, other_rights]])
    directory = tmp_path / "check"
    directory.mkdir()
    with pytest.raises(ValueError, match=match):
        _cross_release_isolation(
            directory, [{"origins": str(left)}, {"origins": str(right)}]
        )


def test_additive_release_must_bind_exact_prior_inventory(tmp_path):
    receipt = tmp_path / "splits.receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "prior_family_inventory": {
                    "release_id": "old",
                    "inventory_sha256": "digest",
                }
            }
        )
    )
    base = {
        "release_id": "old",
        "manifest": {"files": {"family-inventory.jsonl": {"sha256": "digest"}}},
    }
    delta = {
        "release_id": "new",
        "manifest": {
            "build_identity": {
                "completed_preparation": {"request": {"inputs": capture([receipt])}}
            }
        },
    }
    assert verify_additive_anchor(base, delta)["prior"]["release_id"] == "old"
    receipt.write_text(receipt.read_text() + " ")
    with pytest.raises(ValueError, match="receipt fingerprint differs"):
        verify_additive_anchor(base, delta)
    delta["manifest"]["build_identity"]["completed_preparation"]["request"][
        "inputs"
    ] = capture([receipt])
    base["manifest"]["files"]["family-inventory.jsonl"]["sha256"] = "other"
    with pytest.raises(ValueError, match="prior split differs"):
        verify_additive_anchor(base, delta)


@pytest.mark.parametrize("quota,exposures", [(5, 1), (8, 1), (11, 2)])
def test_retained_offsets_match_existing_mixture_schedule(
    tmp_path, monkeypatch, quota, exposures
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    release = tmp_path / "release"
    (release / "lm").mkdir(parents=True)
    docs = [
        {
            "document_id": "a",
            "source_id": "code-source",
            "content_sha256": "x",
            "domains": ["code"],
            "rights": {"training_eligibility": "eligible"},
            "text": "abc",
        },
        {
            "document_id": "b",
            "source_id": "code-source",
            "content_sha256": "y",
            "domains": ["code"],
            "rights": {"training_eligibility": "eligible"},
            "text": "def",
        },
    ]
    prepared = tmp_path / "prepared.jsonl"
    raw = [(json.dumps(x) + "\n").encode() for x in docs]
    prepared.write_bytes(b"".join(raw))
    final_docs = [{**x, "split": "train", "drop_reason": None} for x in docs]
    write_rows(release / "documents.jsonl", final_docs)
    write_rows(
        release / "lm/train.lineage.jsonl",
        [{"record_id": x["document_id"]} for x in docs],
    )
    inventory = tmp_path / "inventory.jsonl"
    write_rows(
        inventory,
        [
            {
                "document_id": x["document_id"],
                "family_id": "family-" + x["document_id"],
                "split": "train",
                "stratum": "code",
                "content_sha256": x["content_sha256"],
            }
            for x in docs
        ],
    )
    references = tmp_path / "references.jsonl"
    write_rows(
        references,
        [
            [
                x["document_id"],
                0,
                0 if i == 0 else len(raw[0]),
                len(raw[i]),
                "train",
                "raw",
                x["content_sha256"],
                "rights",
                False,
                "source",
                ["code"],
                "family-" + x["document_id"],
                "code",
            ]
            for i, x in enumerate(docs)
        ],
    )
    parents = tmp_path / "parents.bin"
    parents.write_bytes(struct.pack("<QQ", 0, 1))
    origins = tmp_path / "origins.jsonl"
    write_rows(
        origins,
        [[1, "x", 0, "a", "train", "rights"], [1, "y", 1, "b", "train", "rights"]],
    )
    cache, proof = tmp_path / "cache.json", tmp_path / "proof.json"
    cache.write_text("cache")
    proof.write_text("proof")
    evidence = {
        "completion_path": str(cache),
        "cache_path": str(cache),
        "proof_path": str(proof),
    }
    cohort = {
        "release_id": "base",
        "manifest": {"corpus_id": "fixture"},
        "references": str(references),
        "parents": str(parents),
        "origins": str(origins),
        "documents": 2,
        "sources": [str(prepared)],
        "source_ids": ["code-source"],
        "evidence": {
            "references": evidence,
            "deduplication": evidence,
            "origins": evidence,
            "sources": {"code-source": evidence},
        },
    }
    spec = MixtureDeclaration.model_validate(
        {
            "schema_version": 1,
            "release_path": release,
            "tokenizer_config": tmp_path / "tokenizer",
            "family_inventory": inventory,
            "source_strata": {"code-source": "code"},
            "target_quotas": {"code": quota},
            "seed": 86,
            "max_exposures": exposures,
        }
    )
    tok = SimpleNamespace(
        encode=lambda text, **_: SimpleNamespace(ids=list(text.encode()))
    )
    manifest = {"release_id": "base"}
    (release / "manifest.json").write_text("{}")
    original_dir, reused_dir = tmp_path / "original", tmp_path / "reused"
    original_dir.mkdir()
    reused_dir.mkdir()
    scratch = tmp_path / "original-index"
    scratch.mkdir()
    with MixtureIndex(scratch, release / "documents.jsonl") as index:
        index.inventory(
            inventory,
            spec.source_strata,
            lambda row: _rank(
                spec.seed, row["stratum"], row["family_id"], row["document_id"]
            ),
        )
        index.select(release / "lm/train.lineage.jsonl")
        before = _schedule(
            spec,
            b"declaration",
            release,
            manifest,
            tok,
            "tokenizer",
            3,
            index,
            original_dir,
        )
    scratch = tmp_path / "reused-index"
    scratch.mkdir()
    with RetainedMixtureIndex(
        scratch, tmp_path, [cohort], spec.seed, spec.source_strata
    ) as index:
        after = _schedule(
            spec,
            b"declaration",
            release,
            manifest,
            tok,
            "tokenizer",
            3,
            index,
            reused_dir,
        )
        if exposures == 1:
            assert index.tokens.tell() == 0
        else:
            assert index.tokens.tell() > 0
    assert (original_dir / "train.tokens.jsonl").read_bytes() == (
        reused_dir / "train.tokens.jsonl"
    ).read_bytes()
    assert before == after


@pytest.mark.parametrize("floor,cap", [(3, None), (99, None), (3, 32)])
def test_direct_schedule_matches_duplicate_empty_and_floor_behavior(
    tmp_path, monkeypatch, floor, cap
):
    if cap is not None:
        monkeypatch.setattr(mixture, "MAX_ROW_BYTES", cap)
    release = tmp_path / "release"
    release.mkdir()
    (release / "manifest.json").write_text("{}")
    inventory = tmp_path / "inventory"
    inventory.write_text("inventory")
    spec = SimpleNamespace(
        target_quotas={"code": 3},
        unique_train_content_floors={"code": floor},
        max_exposures=1,
        additional_releases=[],
        family_inventory=inventory,
        seed=86,
        order_policy=None,
    )
    docs = [
        (
            {
                "document_id": "a",
                "source_id": "s",
                "content_sha256": "a",
                "text": "abc",
            },
            "f1",
        ),
        (
            {
                "document_id": "duplicate",
                "source_id": "s",
                "content_sha256": "a",
                "text": "abc",
            },
            "f1",
        ),
        (
            {
                "document_id": "empty",
                "source_id": "s",
                "content_sha256": "b",
                "text": "",
            },
            "f2",
        ),
        (
            {
                "document_id": "after-quota",
                "source_id": "s",
                "content_sha256": "c",
                "text": "de",
            },
            "f3",
        ),
    ]

    class Index:
        def __init__(self):
            self.content = set()
            self.rows = []
            self.families = set()

        def candidates(self, _):
            return iter(docs)

        def seen(self, digest):
            old = digest in self.content
            self.content.add(digest)
            return old

        def append(self, row):
            if cap is not None and len(mixture.canonical_json(row)) + 1 > cap:
                raise ValueError("mixture encoded record-byte cap exceeded")
            self.rows.append(row)
            self.families.add(row["family_id"])

        def encoded(self, _):
            return iter(self.rows)

        def family_count(self, _):
            return len(self.families)

    class DirectIndex(Index):
        def record_family(self, _, family):
            self.families.add(family)

        def append(self, _):
            raise AssertionError("direct path wrote a token spool")

    tok = SimpleNamespace(
        encode=lambda text, **_: SimpleNamespace(ids=list(text.encode()))
    )
    results = []
    for name, index in [("old", Index()), ("direct", DirectIndex())]:
        staging = tmp_path / name
        staging.mkdir()
        try:
            receipt = _schedule(
                spec,
                b"spec",
                release,
                {"release_id": "r"},
                tok,
                "tok",
                3,
                index,
                staging,
            )
        except ValueError as exc:
            receipt = str(exc)
        results.append((receipt, (staging / "train.tokens.jsonl").read_bytes()))
    if cap is not None:
        assert (
            results[0][0] == results[1][0] == "mixture encoded record-byte cap exceeded"
        )
    elif floor == 99:
        assert "unique TRAIN content supply below floor" in results[0][0]
        # Failed staging is discarded by the caller; direct scheduling may have
        # written it before discovering the shortfall.
        assert results[0][0] == results[1][0]
    else:
        assert results[0] == results[1]


def test_additive_mixture_keeps_origin_and_publishes_combined_identity(
    tmp_path, monkeypatch
):
    """Exercise public scheduling with tiny substituted execution receipts."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    base, delta = tmp_path / "base", tmp_path / "delta"
    base.mkdir()
    delta.mkdir()
    evidence_file = tmp_path / "proof.json"
    evidence_file.write_text("proof")
    evidence = {
        "completion_path": str(evidence_file),
        "cache_path": str(evidence_file),
        "proof_path": str(evidence_file),
    }
    receipt_path = tmp_path / "delta-splits.receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "prior_family_inventory": {
                    "release_id": "base-id",
                    "inventory_sha256": "family-digest",
                }
            }
        )
    )
    base_manifest = {
        "release_id": "base-id",
        "files": {"family-inventory.jsonl": {"sha256": "family-digest"}},
    }
    delta_manifest = {
        "release_id": "delta-id",
        "corpus_id": "delta-corpus",
        "build_identity": {
            "completed_preparation": {"request": {"inputs": capture([receipt_path])}}
        },
    }

    def cohort(path, source, stratum, text, manifest):
        (path / "manifest.json").write_text("{}")
        (path / "release-publication.json").write_text("{}")
        (path / "sources.json").write_text(
            json.dumps(
                [{"id": source, "rights_policy": {"training_eligibility": "eligible"}}]
            )
        )
        (path / "report.json").write_text(
            json.dumps(
                {
                    "requested_mixture": {"code": 0.5, "web": 0.5},
                    "view_counts": {"lm": {"train": 1}},
                    "split_counts": {"train": 1},
                }
            )
        )
        doc = {
            "document_id": source,
            "source_id": source,
            "content_sha256": source,
            "domains": [stratum],
            "rights": {"training_eligibility": "eligible"},
            "text": text,
        }
        docs = path / "prepared.jsonl"
        raw = (json.dumps(doc) + "\n").encode()
        docs.write_bytes(raw)
        refs = path / "refs.jsonl"
        write_rows(
            refs,
            [
                [
                    source,
                    0,
                    0,
                    len(raw),
                    "train",
                    source,
                    source,
                    "rights",
                    False,
                    "family",
                    [stratum],
                    "family-" + source,
                    stratum,
                ]
            ],
        )
        parents = path / "parents.bin"
        parents.write_bytes(struct.pack("<Q", 0))
        origins = path / "origins.jsonl"
        write_rows(origins, [[1, source, 0, source, "train", "rights"]])
        return {
            "release_id": manifest["release_id"],
            "release_path": str(path),
            "manifest": {"corpus_id": "fixture", **manifest},
            "references": str(refs),
            "parents": str(parents),
            "origins": str(origins),
            "documents": 1,
            "sources": [str(docs)],
            "source_ids": [source],
            "evidence": {
                "references": evidence,
                "deduplication": evidence,
                "origins": evidence,
                "sources": {source: evidence},
            },
        }

    left = cohort(base, "web-source", "web", "abcd", base_manifest)
    right = cohort(delta, "code-source", "code", "efgh", delta_manifest)
    family = base / "family-inventory.jsonl"
    family.write_text("unused signed fixture\n")
    monkeypatch.setattr(mixture, "verify_release", lambda _: base_manifest)
    monkeypatch.setattr(
        "sparselab.corpus.completed_preparation.corpus_work_root", lambda _: tmp_path
    )
    monkeypatch.setattr(
        "sparselab.corpus.retained_mixture.prepared_release",
        lambda _, path: left if path == base else right,
    )
    monkeypatch.setattr(
        mixture,
        "_tokenizer",
        lambda *_: (
            SimpleNamespace(
                encode=lambda text, **_: SimpleNamespace(ids=list(text.encode())),
                token_to_id=lambda _: 3,
            ),
            "tokenizer-digest",
        ),
    )
    declaration = tmp_path / "mixture.yaml"
    declaration.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "release_path": str(base),
                "additional_releases": [{"release_path": str(delta)}],
                "tokenizer_config": str(tmp_path / "tokenizer.yaml"),
                "family_inventory": str(family),
                "source_strata": {"code-source": "code", "web-source": "web"},
                "target_quotas": {"code": 4, "web": 4},
                "seed": 86,
                "max_exposures": 1,
            }
        )
    )
    output = tmp_path / "mixture"
    result = mixture.materialize_mixture(declaration, output)
    assert result["actual_target_tokens"] == {"code": 4, "web": 4}
    assert result["release_id"] not in {"base-id", "delta-id"}
    assert result["additive_release_binding"]["base_release_id"] == "base-id"
    assert result["additive_release_binding"]["additional_release_ids"] == ["delta-id"]
    assert {row["source_id"] for row in rows(output / "train.tokens.jsonl")} == {
        "code-source",
        "web-source",
    }
