# Experiment workspaces and retained evidence

One experiment has one local workspace. Seeds, architecture cells, budget
coordinates, and resumed children are scientific coordinates identified by run
IDs and immutable manifests; they are not peer repository-root directories.

Name every workspace for its task or experiment, independently of its execution
backend. CPU, ROCm, CUDA, and other backends are runtime parameters recorded in
configs and manifests. For example, runtime acceptance across devices shares
`$SPARSELAB_WORK_DIR/experiments/runtime-acceptance/runs/`; runtime smoke configs
share `$SPARSELAB_WORK_DIR/experiments/runtime-smoke/runs/`. Use distinct run IDs
for each execution, including backend labels when useful, rather than separate
`runs-rocm` or `runs-cpu` stores. Worker environments remain backend-specific.

The implicit persistent root is `${XDG_DATA_HOME}/sparselab` when
`XDG_DATA_HOME` is nonempty and absolute; otherwise it is
`~/.local/share/sparselab`. For large campaigns, explicitly choose an adequately
sized filesystem, for example `export SPARSELAB_WORK_DIR=/data/sparselab`.
The global `--work-dir PATH` overrides that environment variable, which overrides
the implicit root. An explicit relative path remains relative to the current
directory, including legacy `SPARSELAB_WORK_DIR=sparselab-work`. The root is not
derived from the repository, branch, or worktree. Merely resolving it does not
create it; read-only inspection does not need to initialize it.

```text
<persistent-root>/
  experiments/dense-lm-v1/
    receipt.json
    runs/              # shared controller/store, run IDs, parent and child runs
    staging/
    exercises/
    captures/
    local-reports/
  runs/                # implicit generic controller/store
  scratch/             # disposable temporary files
  cache/               # optional reconstructable caches, never scientific evidence
```

The source tree holds source, tests, configs, documentation, and declarative
research/lifecycle records. The persistent root holds downloads, prepared arrays,
checkpoints, immutable execution evidence, receipts, and logs. Its `scratch/`
directory contains disposable temporary files; `cache/` is reserved for optional
reconstructable caches. The old ignored in-checkout `sparselab-work/` remains
selectable explicitly, but is not moved or pruned automatically. A selected
persistent root or explicit long-lived output/cache destination inside any Git
checkout receives a `STORAGE_INSIDE_GIT_CHECKOUT` warning: execution is permitted,
but losing the checkout may lose the bytes. Published compact evidence may be
versioned separately; a digest or commit identifies content, not its root path.
Promoting a model does not promote its full mutable workspace into version
control. Reusable tokenizer and prepared-data caches keep their configured
identities and locations; they need not be duplicated into each experiment.

## Guarding one operational command

`sparselab monitor` launches one command once, records versioned launch,
per-sample and completion JSON under a new exclusive log directory, and returns
a nonzero exit code on any guard violation, monitor error or command failure.
For example:

```sh
uv run --locked --extra cpu sparselab monitor --policy monitor.yaml --workspace /data/sparselab \
  --log-dir /data/sparselab/logs/unique-launch \
  --reserve-bytes 1073741824 --reserve-inodes 100 --json -- \
  uv run --locked --extra cpu sparselab corpus measure-tokens ...
```

For a 12 GiB owned-tree RSS cap, 8 GiB host available-RAM floor, 1 GiB
owned-tree swap cap and independently selected free-swap floor:

```yaml
monitor_policy_version: 1
max_tree_rss_bytes: 12884901888
min_host_available_ram_bytes: 8589934592
max_tree_swap_bytes: 1073741824
min_host_free_swap_bytes: 1073741824
min_projected_disk_free_bytes: 10000000000
min_projected_disk_free_inodes: 10000
```

The strict YAML policy starts with `monitor_policy_version: 1`; optional guards
are `max_tree_rss_bytes`, `max_tree_swap_bytes`,
`min_host_available_ram_bytes`, `min_host_free_swap_bytes`,
`min_disk_free_bytes`, `min_disk_free_inodes`,
`min_projected_disk_free_bytes` and `min_projected_disk_free_inodes`.
`interval_seconds` defaults to 1 (maximum 2) and
`termination_grace_seconds` to 5. Reservations subtract from the current free
space/inodes on the workspace filesystem before projected checks. The policy is
operational: it is **not** a scientific config, and unlike
`ResourceEnvelope.min_swap_bytes` the tree swap guard applies only to the
command's PID/creation-time identified descendants. Host swap *used* is logged
but never a tree-swap threshold; free host swap has its own minimum.

Python callers can load `MonitorPolicy` with `load_monitor_policy(path)` and call
`monitor_command(command, policy, workspace=..., log_dir=...,
policy_path=..., reserved_bytes=..., reserved_inodes=..., cwd=...)`; the result
is a typed `MonitorCompletion`. The command working directory defaults to the
caller's directory and is recorded separately from the monitored storage
workspace; `--cwd` selects it explicitly. The event stream (`events.jsonl`)
includes the command, root PID and creation time, source/policy paths and SHA-256,
and log paths. Child stdout/stderr are retained separately as
`command.stdout.log` and `command.stderr.log`; the CLI emits the final completion
JSON. `completion.json` repeats that terminal record. `FAILED` preserves a nonzero
child return code and is not an accepted successful operation. Peaks are sampled
observations; unavailable readings remain null, not invented zeroes.
Monitored descendants retain their identity after reparenting if observed before
orphaning. Signals
use per-PID Linux pidfds after identity checks, never process-group signals.
Sampling cannot discover a process that was born and orphaned entirely between
samples; plan critical workloads accordingly. Missing requested metrics fail
closed; monitoring does not resume, retry or authenticate scientific inputs.

## Explicit campaign cleanup

Choose the campaign workspace under the selected persistent root when preparing
campaign-local data caches. New prepared-data caches inside that workspace receive
an ownership marker; existing caches without a marker and caches outside it are
never cleanup candidates. Shared data and worker dispatch caches are outside this
workflow. Keep `SPARSELAB_WORK_DIR` pointing at the persistent root, not a nested
experiment: explicit cache locations remain separate.

```sh
WORK="${SPARSELAB_WORK_DIR:-$HOME/.local/share/sparselab}/experiments/dense-lm-v1"
uv run --locked --extra cpu sparselab workspace cleanup plan "$WORK" \
  --output "$WORK/local-reports/cleanup-plan.json"
# Review the JSON paths, identities, and reclaimable bytes before applying.
uv run --locked --extra cpu sparselab workspace cleanup apply \
  "$WORK/local-reports/cleanup-plan.json"
```

The plan is read-only and defaults to retaining two additional eligible periodic
checkpoint generations per run and two campaign-owned prepared-data cache entries.
Change those limits with `--max-extra-periodic` and `--max-cache-entries` when
creating the plan. Latest, best, and the previous verified generation are always
protected. Every generation registered in the run database or bound by a child
run manifest is also protected,
even when this exceeds the count limit; most ordinary training checkpoints are
registered, so checkpoint proposals can be empty. Only verified, unregistered
generations in terminal runs can be proposed. Cache candidates require the
workspace ownership marker, a matching prepared-data manifest, and no active
campaign run or worker attempt. Applying a plan rechecks the database, candidate
inventory, and writer leases; changed plans must be created again. The command
never scans arbitrary sibling workspaces or deletes unmarked directories.

Use one variable for a study, including worker registration, submission,
controller operation, collection, and resume:

```sh
WORK="${SPARSELAB_WORK_DIR:-$HOME/.local/share/sparselab}/experiments/dense-lm-v1"
mkdir -p "$WORK"
uv run --locked --extra cpu sparselab study submit configs/references/dense-lm-v1/study.yaml \
  --store "$WORK/runs" --receipt "$WORK/receipt.json"
uv run --locked --extra cpu sparselab controller run --store "$WORK/runs"
# After completion and verified ingestion:
uv run --locked --extra cpu sparselab study collect configs/references/dense-lm-v1/study.yaml \
  "$WORK/receipt.json" --runs-dir "$WORK/runs"
```

Register eligible workers in that same store before execution. One receipt
contains all submitted seed coordinates. Device leases serialize work where
required; separate seed stores are not required. Resume a controller run using
`experiment resume RUN_ID --store "$WORK/runs"`. The parent and child remain in
that store, linked by IDs and checkpoint identities. Keep the original submission
receipt as history; do not replace its submitted run IDs with resumed child IDs.

Choose `export SPARSELAB_WORK_DIR=/data/sparselab` on a disk sized for the
campaign, then set `WORK="$SPARSELAB_WORK_DIR/experiments/dense-lm-v1"`.
Explicit `--store`, `--runs-dir`, `--output`, transcript destinations, old
`logging.root_dir`, and `dataset.cache_dir` retain their supplied meaning.
The global `--work-dir /data/sparselab` (before the subcommand) overrides the
environment for implicit stores and scratch but never relocates explicit paths.
Direct training should pass `--runs-dir "$WORK/runs"` on every parent and child
command when using a task-specific store.

A study without store or receipt overrides chooses
`<work-dir>/experiments/<study-name>/runs` and a sibling `receipt.json`. The generic
non-study CLI store/reader default is `<work-dir>/runs/`. Existing root-level
runs remain readable with explicit `--runs-dir runs`; defaults do not search
historical locations or silently reinterpret configured paths.

Future experiments follow the same pattern:

```text
<persistent-root>/experiments/
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

Intentional relocation to a genuinely different persistent root requires a new
location binding or manifest and re-verification of the referenced bytes. Do not
rewrite past receipts to imply that historical absolute paths moved.

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

On 2026-09-27, the remaining local `runs-rocm/` store was relocated to
`sparselab-work/experiments/runtime-acceptance/runs/`. Its sole completed run,
`rocm-wsl2-acceptance-20260926T035145Z`, retains its original ID, backend metadata,
and historical paths. All 34 run files retained their hashes, inodes, sizes, and
modification times; all five checkpoint generations verified before and after
the move, and the integrity-bound evaluation evidence was identical. The current
checkpoint locator was rebased using the same standalone-store helper. The
original database and sidecars, inventory, relocation script, and verification
record are retained under the workspace's `local-reports/relocation/`. No new
training or hardware acceptance was performed. Earlier accounts of leaving
`runs-rocm/` untouched describe those earlier operations and remain unchanged.
