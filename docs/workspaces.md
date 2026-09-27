# Experiment workspaces and retained evidence

One experiment has one local workspace. Seeds, architecture cells, budget
coordinates, and resumed children are scientific coordinates identified by run
IDs and immutable manifests; they are not peer repository-root directories.

```text
sparselab-work/experiments/dense-lm-v1/
  receipt.json
  runs/              # shared controller/store, run IDs, parent and child runs
  staging/
  exercises/
  captures/
  local-reports/
```

The source tree holds source, tests, configs, documentation, and declarative
research/lifecycle records. Ignored `sparselab-work/` holds mutable execution
state. `artifacts/acceptance/` and content-addressed `artifacts/research-reports/`
hold deliberately published compact evidence. Promoting a model does not promote
its full mutable workspace into version control. Reusable tokenizer and prepared
data caches keep their existing shared identities and locations; they need not
be duplicated into each experiment.

Use one variable for a study, including worker registration, submission,
controller operation, collection, and resume:

```sh
WORK=sparselab-work/experiments/dense-lm-v1
export SPARSELAB_WORK_DIR="$WORK"
mkdir -p "$WORK"
uv run --locked sparselab study submit configs/references/dense-lm-v1/study.yaml \
  --store "$WORK/runs" --receipt "$WORK/receipt.json"
uv run --locked sparselab controller run --store "$WORK/runs"
# After completion and verified ingestion:
uv run --locked sparselab study collect configs/references/dense-lm-v1/study.yaml \
  "$WORK/receipt.json" --runs-dir "$WORK/runs"
```

Register eligible workers in that same store before execution. One receipt
contains all submitted seed coordinates. Device leases serialize work where
required; separate seed stores are not required. Resume a controller run using
`experiment resume RUN_ID --store "$WORK/runs"`. The parent and child remain in
that store, linked by IDs and checkpoint identities. Keep the original submission
receipt as history; do not replace its submitted run IDs with resumed child IDs.

Choose `WORK=/mnt/nvme/sparselab/experiments/dense-lm-v1` for an external disk.
Explicit `--store`, `--runs-dir`, `--output`, and transcript destinations retain
their supplied meaning. The global `--work-dir "$WORK"` (before the subcommand)
or `SPARSELAB_WORK_DIR="$WORK"` also keeps implicit scratch inside the experiment.
This scratch setting alone does not override explicit destinations. Direct
training should pass `--runs-dir "$WORK/runs"` on every parent and child command;
legacy `logging.root_dir` values remain unchanged when no override is supplied.

A study without store or receipt overrides chooses
`<work-dir>/experiments/<study-name>/runs` and a sibling `receipt.json`. The generic
non-study CLI store/reader default is `sparselab-work/runs/` under the nearest
project root (or current directory outside a project). Existing root-level runs
remain readable with explicit `--runs-dir runs`; defaults do not search historical
locations or silently reinterpret configured paths.

Future experiments follow the same pattern:

```text
sparselab-work/experiments/
  dense-lm-v1/
  dense-lm-token-budget-v1/
  dense-lm-scale-v1/
  engram-ffn-substitution-v1/
  engram-mla-compression-v1/
```

## Historical paths and relocation

The original dense-lm-v1 seed YAMLs explicitly selected three root-level
`runs-dense-lm-v1-seed*` directories. The reference matrix repeated those paths,
and the manual instructions invoked direct training without overriding them.
This was a workflow convention, not controller isolation. New instructions use
one explicit shared store; frozen source configs and published evidence retain
their original values.

Before moving retained bytes, distinguish scientific identity (hashes, run IDs,
checkpoint and input identities), historical location (where execution or capture
occurred), and current local location (where readers should find retained bytes).
Preserve the first two. Update only supported current-location indexes or explicit
reader paths, and verify checkpoint integrity, lineage, evidence, observations,
static reports, and baseline descriptions afterward. Never edit old observations
or acceptance records to claim they were originally captured at a new path.
Do not move active runs or blindly merge controller databases. Prefer same-device
atomic renames when relocation is supported. If the evidence cannot be loaded
without rewriting scientific artifacts, retain the original locations and record
the limitation.

Some older instructional documents are themselves registered, byte-hashed
research evidence: [context study](context-engram-study.md),
[domain corpus study](path-domain-corpus.md), and
[portable adapter notes](portable-engram.md). Their original commands remain
historical records. Apply the workspace convention above to new execution rather
than copying their old root-level output paths. Updating those bytes would
invalidate the evidence; no registry hash is changed by this hygiene work.

The [dense-lm-v1 migration audit](dense-lm-v1-workspace-audit.md) records the actual
consolidation, preserved history, verification results, and remaining unrelated
lifecycle availability limits.
