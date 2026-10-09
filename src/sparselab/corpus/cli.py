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
    if command == "admission-draft":
        from sparselab.corpus.admission_draft import draft_admission_manifest

        project = load_project(Path(args.project))
        result = draft_admission_manifest(
            project,
            root,
            Path(args.template),
            Path(args.policy_document),
            Path(args.output),
        )
        print(json.dumps(result, sort_keys=True))
        return
    if command == "split-inventory":
        from sparselab.corpus.split_inventory import write_split_inventory

        project = load_project(Path(args.project))
        print(
            json.dumps(
                write_split_inventory(project, root, Path(args.output)), sort_keys=True
            )
        )
        return
    if command == "freeze-splits":
        from sparselab.corpus.split_freeze import freeze_splits

        project = load_project(Path(args.project))
        result = freeze_splits(
            project,
            root,
            Path(args.inventory),
            Path(args.clusters),
            Path(args.output),
        )
        print(json.dumps(result, sort_keys=True))
        return
    if command == "finalize-family-inventory":
        from sparselab.corpus.split_freeze import finalize_family_inventory

        result = finalize_family_inventory(
            Path(args.release), Path(args.splits), Path(args.output)
        )
        print(json.dumps(result, sort_keys=True))
        return
    if command == "audit-protected-lineage":
        from sparselab.corpus.protected_lineage import audit_protected_lineage

        result = audit_protected_lineage(
            Path(args.prior_release),
            Path(args.candidate_release),
            Path(args.prior_inventory),
            Path(args.candidate_inventory),
            Path(args.profile),
            Path(args.suite),
            Path(args.output),
        )
        print(json.dumps(result, sort_keys=True))
        if result["status"] != "PASS":
            raise ValueError("protected evaluation lineage collision")
        return
    if command in {"materialize-mixture", "verify-mixture"}:
        from sparselab.corpus.mixture import materialize_mixture, verify_mixture

        result = (
            materialize_mixture(Path(args.declaration), Path(args.output))
            if command == "materialize-mixture"
            else verify_mixture(Path(args.declaration), Path(args.output))
        )
        print(json.dumps(result, sort_keys=True))
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
    if command == "verify-export":
        from sparselab.config.loading import load_config
        from sparselab.corpus.export import verify_release_export

        config = load_config(Path(args.config))
        print(json.dumps(verify_release_export(config.dataset), sort_keys=True))
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
                tokenizer_origin_release=(
                    _release_path(args.tokenizer_origin_release, root)
                    if args.tokenizer_origin_release is not None
                    else None
                ),
                family_inventory=(
                    Path(args.family_inventory)
                    if args.family_inventory is not None
                    else None
                ),
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
                min_frequency=args.min_frequency,
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

        caps = {
            name: value
            for name, value in (
                ("max_documents", args.max_documents),
                ("max_input_text_bytes", args.max_input_text_bytes),
                ("max_total_shingles", args.max_total_shingles),
                ("max_comparisons", args.max_comparisons),
                ("max_candidates", args.max_candidates),
            )
            if value is not None
        }
        result = audit_release(release, **caps)
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
    command = sub.add_parser(
        "admission-draft",
        help="Draft complete conservative decisions from verified acquired snapshots",
    )
    command.add_argument("project")
    command.add_argument("--template", required=True)
    command.add_argument("--policy-document", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "split-inventory", help="Record pre-build IDs from verified source snapshots"
    )
    command.add_argument("project")
    command.add_argument("--output", required=True)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "finalize-family-inventory",
        help="Bind kept release documents to frozen family assignments",
    )
    command.add_argument("release")
    command.add_argument("--splits", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "audit-protected-lineage",
        help="Cold-check protected evaluation documents against a candidate release",
    )
    for field in (
        "prior-release",
        "candidate-release",
        "prior-inventory",
        "candidate-inventory",
        "profile",
        "suite",
        "output",
    ):
        command.add_argument(f"--{field}", required=True)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "freeze-splits", help="Freeze reviewed admission-aware family assignments"
    )
    command.add_argument("project")
    command.add_argument("--inventory", required=True)
    command.add_argument("--clusters", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_handle)
    command = sub.add_parser("freeze")
    command.add_argument("build")
    command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "verify-export", help="Cold-verify one run configuration's shared corpus export"
    )
    command.add_argument("config")
    command.set_defaults(handler=_handle)
    command = sub.add_parser("tokenizer-bakeoff")
    command.add_argument("declaration")
    command.add_argument("--output", required=True)
    command.set_defaults(handler=_handle)
    for name in ("materialize-mixture", "verify-mixture"):
        command = sub.add_parser(
            name, help="Materialize or authenticate an exact token-ID source mixture"
        )
        command.add_argument("declaration")
        command.add_argument("--output", required=True)
        command.add_argument("--json", action="store_true")
        command.set_defaults(handler=_handle)
    command = sub.add_parser(
        "measure-tokens", help="Measure authenticated distinct source-domain tokens"
    )
    command.add_argument("release")
    command.add_argument("--tokenizer", required=True)
    command.add_argument("--tokenizer-origin-release")
    command.add_argument("--family-inventory")
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
            command.add_argument("--min-frequency", type=int, default=1)
        if name == "near-duplicates":
            for flag in (
                "max-documents",
                "max-input-text-bytes",
                "max-total-shingles",
                "max-comparisons",
                "max-candidates",
            ):
                command.add_argument("--" + flag, type=int)
        command.set_defaults(handler=_handle)
