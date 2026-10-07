"""Command-line entry points for immutable corpus workspaces."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sparselab.verification_proofs import verification_options
from sparselab.workdir import resolve_work_dir


def _release_path(value: str, work_root: Path) -> Path:
    if "@" in value and not Path(value).exists():
        corpus_id, digest = value.split("@", 1)
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", corpus_id) or not re.fullmatch(
            r"[0-9a-f]{64}", digest
        ):
            raise ValueError("release reference must be <corpus-id>@<full-sha256>")
        return work_root / "corpora" / corpus_id / "releases" / digest
    return Path(value).resolve()


def _handle(args: argparse.Namespace) -> None:
    from sparselab.corpus import release as publication
    from sparselab.corpus.acquisition import acquire
    from sparselab.corpus.export import export_release
    from sparselab.corpus.pipeline import build
    from sparselab.corpus.project import load_project

    root = resolve_work_dir(args.work_dir)
    command = args.corpus_command
    verification = (
        verification_options(root)
        if command in {"acquire", "build", "freeze", "export"}
        else {}
    )
    if command == "tokenizer-bakeoff":
        from sparselab.corpus.tokenizer_bakeoff import bakeoff

        print(bakeoff(Path(args.declaration), Path(args.output), work_root=root))
        return
    if command == "budget-init":
        from sparselab.corpus.transport_budget import TransportBudget

        project = load_project(Path(args.project))
        ledger = TransportBudget.initialize(
            root / "corpora" / project.config.id / "transport-budget.sqlite", project
        )
        print(json.dumps(ledger.receipt(), sort_keys=True))
        return
    if command in {"acquire", "build"}:
        project = load_project(Path(args.project))
        result = (
            acquire(project, root, offline=args.offline, **verification)
            if command == "acquire"
            else build(project, root, offline=args.offline, **verification)
        )
        print(
            json.dumps(result, sort_keys=True, default=str)
            if command == "acquire"
            else result
        )
        return
    if command == "freeze":
        print(publication.freeze(Path(args.build), root, **verification))
        return
    release = _release_path(args.release, root)
    if command == "measure-tokens":
        from sparselab.corpus.release import _verification_operation
        from sparselab.corpus.token_denominator import measure_source_tokens

        # Explicit evidence reuse accepts only the reviewed, Git-bound v5 cold
        # record. It is not authentication supplied by an arbitrary caller SHA.
        raw_release = Path(args.release)
        is_reference = "@" in args.release and not (
            raw_release.exists() or raw_release.is_symlink()
        )
        with _verification_operation():
            result = measure_source_tokens(
                release if is_reference else raw_release,
                Path(args.tokenizer),
                Path(args.policy),
                Path(args.output),
                evidence_commit=args.evidence_commit,
                release_evidence=Path(args.release_evidence)
                if args.release_evidence is not None
                else None,
                selection_evidence=Path(args.selection_evidence)
                if args.selection_evidence is not None
                else None,
                batch_documents=args.batch_documents,
                batch_source_bytes=args.batch_source_bytes,
            )
        print(json.dumps(result, sort_keys=True))
        return
    if command == "export":
        print(
            export_release(
                release,
                args.view,
                Path(args.base_run_config),
                args.vocab_size,
                root,
                **verification,
            )
        )
        return
    if command in {"sample", "review"}:
        result = getattr(publication, command)(
            release,
            domain=args.domain,
            kind=args.kind,
            status=args.status,
            limit=args.limit,
        )
    elif command == "lineage":
        result = publication.lineage(release, args.record_id)
    elif command == "describe":
        result = publication.describe(release, tokenizer=args.tokenizer)
    elif command == "compare":
        from sparselab.corpus.measurement import (
            capability_matrix,
            compare_capability_matrix,
        )

        if args.with_cases:
            pairs = [(release, args.run_dir)] + [
                (_release_path(reference, root), Path(run_dir))
                for reference, run_dir in args.with_cases
            ]
            result = compare_capability_matrix(pairs)
        else:
            result = capability_matrix(release, args.run_dir)
            if result is None:
                raise ValueError(
                    "verified checkpoint-bound capability results are required"
                )
    elif command == "publication":
        from sparselab.corpus.publication import publication_manifest

        result = publication_manifest(release)
    elif command == "sources":
        result = publication.sources(release)
    elif command == "audit":
        result = publication.audit(release)
    elif command == "near-duplicates":
        from sparselab.corpus.near_duplicates import audit_release

        result = audit_release(release)
    else:
        result = publication.consumers(release, args.runs_dir)
    print(json.dumps(result, sort_keys=True))


def add_commands(commands: argparse._SubParsersAction) -> None:
    corpus = commands.add_parser(
        "corpus", help="Acquire, derive, freeze and inspect corpus releases"
    )
    sub = corpus.add_subparsers(dest="corpus_command", required=True)
    for name in ("acquire", "build"):
        command = sub.add_parser(name)
        command.add_argument("project")
        command.add_argument("--offline", action="store_true")
        command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "budget-init", help="Start one durable bounded-HF transport attempt"
    )
    command.add_argument("project")
    command.set_defaults(handler=_handle)
    command = sub.add_parser("freeze")
    command.add_argument("build")
    command.set_defaults(handler=_handle)
    command = sub.add_parser("tokenizer-bakeoff")
    command.add_argument("declaration")
    command.add_argument("--output", required=True)
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "measure-tokens", help="Measure authenticated distinct source-domain tokens"
    )
    command.add_argument("release")
    command.add_argument("--tokenizer", required=True)
    command.add_argument("--policy", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--evidence-commit")
    command.add_argument("--release-evidence")
    command.add_argument("--selection-evidence")
    command.add_argument("--batch-documents", type=int, default=256)
    command.add_argument("--batch-source-bytes", type=int, default=1_048_576)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_handle)
    for name in (
        "describe",
        "sources",
        "audit",
        "near-duplicates",
        "sample",
        "review",
        "lineage",
        "consumers",
        "compare",
        "export",
        "publication",
    ):
        command = sub.add_parser(name)
        command.add_argument("release")
        if name in {"sample", "review"}:
            command.add_argument("--domain")
            command.add_argument("--kind")
            command.add_argument("--status")
            command.add_argument("--limit", type=int, default=10)
        if name == "lineage":
            command.add_argument("record_id")
        if name == "describe":
            command.add_argument("--tokenizer", type=Path)
        if name == "consumers":
            command.add_argument("--runs-dir", required=True, type=Path)
        if name == "compare":
            command.add_argument("--run-dir", required=True, type=Path)
            command.add_argument(
                "--with",
                dest="with_cases",
                nargs=2,
                action="append",
                metavar=("RELEASE", "RUN_DIR"),
            )
        if name == "export":
            command.add_argument("--view", choices=("lm", "chat"), required=True)
            command.add_argument("--base-run-config", required=True)
            command.add_argument("--vocab-size", required=True, type=int)
        command.set_defaults(handler=_handle)
