# Research experiments

Personal research projects now have discovery homes in the
[Workshop portfolio](https://github.com/supernovae/kybern-workshop/blob/fd259e2cad4e9856d6d659e8df34728a98b32502/catalog/README.md):
[Kernel Memory Lab](https://github.com/supernovae/kybern-workshop/tree/fd259e2cad4e9856d6d659e8df34728a98b32502/experiments/kernel-memory-lab),
[DevMind — DevMind 340M Base](https://github.com/supernovae/kybern-workshop/tree/main/experiments/devmind),
and [TinyStories microlab](https://github.com/supernovae/kybern-workshop/tree/main/examples/tinystories-microlab).
KML has active research intent with preserved evidence awaiting current-runtime
qualification. DevMind 340M Base is authored/proposed, not trained or
execution-ready; it is distinct from historical DevMind v0–v5. The TinyStories
recipe has metadata validation but remains unexecuted, including its useful-story
proposal. Historical smoke results do not establish current story quality.

This directory still retains [KML declarations](kernel-memory-lab/README.md) and
[historical records](history/README.md) pending their dependency-preserving
retirement. The [history path map](history/path-map.json) resolves original
locations without rewriting evidence. These retained copies are not the home for
new personal experiments. EXP1 preparation moved to Workshop DevMind; see its
[file disposition](../../docs/exp1-retirement.json).

The [preserved navigation originals](https://github.com/supernovae/kybern-workshop/blob/a47389204a30da1bbfec5e0de4418b3e3e72d4f2/catalog/NAVIGATION-PRESERVATION.md)
record the three Lab README versions at `faf415e3754f4018674a734547c66df7ddca8333`.
Older README revisions remain in Lab Git history unless separately archived;
this is not a claim that every revision was copied to Workshop.

Do not add new experiments here. Author recipe DSL/configuration in any user-owned
directory or repository, with documentation and artifact references:


- question, competing explanations, and falsifiable acceptance gates;
- immutable or content-addressed config/protocol references;
- declared data ownership, splits, seeds, controls, and budgets;
- iteration log explaining why the next run changed;
- status and links to verified evidence or reports;
- explicit negative, failed, censored, and unresolved outcomes.

Use an external persistent workspace for execution (or the XDG/home default for
small work). Substantial campaigns need a filesystem with sufficient bytes/inodes:

```sh
export SPARSELAB_WORK_DIR=/data/sparselab
df -h "$SPARSELAB_WORK_DIR"
df -i "$SPARSELAB_WORK_DIR"
```

Pass explicit `--store "$SPARSELAB_WORK_DIR/runs"` and a receipt location beneath
the external root to study submission; use that same `runs/` for collection. Keep
all seed, architecture and budget coordinates in one named experiment namespace.
Configured absolute `logging.root_dir` and `dataset.cache_dir` do not move when the
global root changes. Legacy in-checkout `sparselab-work/` is not automatically moved.

Do not check in datasets, caches, checkpoints, run databases, raw logs, or
anonymous temporary paths. The durable record should reference verified evidence
without pretending that a hash establishes scientific validity.

Before acquisition, tokenizer fit, preparation or training, commit the exact
scientific declarations; use `research snapshot <plan-or-campaign> --json` to
see Git closure and fresh prerequisites. After verifying an output, publish its
small digest-only reference and commit that instead of payload bytes:

```sh
uv run --locked sparselab research evidence export --kind corpus_release <verified-release> \
  --declaration <recovery.yaml> --output <tracked-evidence.json> --json
git add <tracked-evidence.json> && git commit -m "Record verified release identity"
```

Source commit and declaration hashes matter; branch name, checkout path and
persistent-root path do not. A SHA proves identity, not quality or publication
rights. See the [lifecycle recovery procedure](../../docs/research/lifecycle-recovery.md)
before claiming a missing release was rebuilt or a lost checkpoint was restored.

Current long-form protocols and findings remain indexed from
[`docs/research/README.md`](../../docs/research/README.md). New campaigns belong outside Lab. Relocation preserves original scientific
identities and evidence; Git history retains removed source paths.

## Offline record lint

Before submitting research records, stage the intended files and run:

```sh
uv run --locked --extra cpu sparselab research lint --json
```

CI runs the same checker against tracked working-tree files (including staged
additions). Untracked local work is ignored. Missing tracked declarations,
symlinks, unsafe paths, malformed known schemas, and changed immutable bindings
fail the check. The checker reuses campaign, experiment, recovery, model-family,
evaluation-suite, readiness, run-config, and corpus declaration schemas and their
authored reference closure. Version-1 producer records also validate their
`protocol` and `corpus_project` declaration references. Local acquisition datasets and fractional-release
tokenizer bytes are excluded from this metadata-only closure.

Current immutable bindings include `frozen_sha256`, `inputs_sha256`, explicit
`bindings` path/digest objects, protocol/config/base-run path/digest objects, and
`scientific-evidence-reference-v1` declaration hashes. Active referenced declarations
must stay available at their declared paths with the same bytes; amendments need
new declarations, preserving earlier evidence. Archived records preserve bytes
and original embedded paths, resolved through the history path map; they do not
claim current executable reference closure. Historical implementation hashes
and external archive inventories are not assertions about today's checkout.
Evidence envelopes validate their recorded identity and verification scope;
passing lint does not reverify an external checkpoint, dataset, or scientific
conclusion, or authenticate a historical source commit.

Checked-in checkpoints, caches, datasets, log files, run databases, output/store
folders, CSV/TSV datasets, and binary payload formats are rejected. Supported
record suffixes are `.md`, `.py`, `.yaml`, `.yml`, `.json`, and validated `.jsonl`;
other formats require an explicit validator. JSON arrays are limited to the
existing blind-review panel schema and its summary-bound digest. Each file must
be at most 1 MiB. JSONL is limited to the existing frozen decoding evidence-panel
format under `evidence/{development,test}/`, bound by the summary's source-file
hash and the header's preregistration/prompt hashes. Arbitrary JSONL datasets or
logs are not accepted merely because they live under `evidence/`. New durable
record formats should extend the checker and its small offline fixtures; no
external artifact registry is needed.
