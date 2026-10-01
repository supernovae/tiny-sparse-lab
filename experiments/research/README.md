# Research experiments

Each subdirectory represents one real campaign or experiment family. Check in
only the material needed to understand and reproduce the design:

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
[`docs/research/README.md`](../../docs/research/README.md). New campaigns can
adopt this directory incrementally; moving an established identity-bound config
is not required merely to satisfy the layout.
