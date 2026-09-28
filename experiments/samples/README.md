# Sample experiments

Samples teach the workflow and are safe to copy. They are not claims that a
mechanism improves quality or performance.

Start with a packaged scaffold:

```sh
WORK="$PWD/sparselab-work/experiments/engram-ffn-substitution-v1"
uv run --locked sparselab --work-dir "$PWD/sparselab-work" research scaffold engram-ffn-substitution-v1 \
  --scale smoke --data offline --backend cpu \
  --output "$WORK/scaffold"
uv run --locked sparselab study plan "$WORK/scaffold/study.yaml"
```

The generated README identifies the explicit preparation and execution steps, sets one `WORK`, and keeps all seed coordinates in `$WORK/runs` with one `$WORK/receipt.json`. Its scratch setting is exported after scaffolding; setting the workspace itself as the global base before scaffolding would append another `experiments/<study-name>` level. Explicit generated store and receipt paths keep later commands inside the selected workspace.
For a checked-in reference layout, see
[`configs/references/dense-small-v1`](../../configs/references/dense-small-v1/).
Copy editable inputs; do not edit a known-good reference or treat smoke output as
research evidence.

A sample should say what it demonstrates, what it deliberately does not prove,
the expected cost class, and which paths can be deleted safely.

## Corpus shape and exact generated fraction

[`corpus-shape-fraction.yaml`](corpus-shape-fraction.yaml) is an offline CPU
example. Its four independently frozen releases compare chat training with and
without tool-trace shapes, and an unrestricted LM release with an exact
361/640-generated-token LM release. Both contrasts declare every changed
resolved field; neither is controlled, because the tokenizer and available
training records also change. The target optimizer exposure is 640 tokens per
arm, not proof that the same distinct source tokens were consumed. The pinned
fraction tokenizer is checked in under
[`corpora/devmind-sample-v0/pinned/`](../../corpora/devmind-sample-v0/pinned/).
If an exact subset becomes infeasible, preparation fails instead of rounding.

From the repository root, in the locked environment:

```sh
uv run --locked sparselab experiment validate experiments/samples/corpus-shape-fraction.yaml --json
uv run --locked sparselab experiment prepare experiments/samples/corpus-shape-fraction.yaml --json
uv run --locked sparselab experiment diff experiments/samples/corpus-shape-fraction.yaml --json
uv run --locked sparselab experiment lock experiments/samples/corpus-shape-fraction.yaml --json
```

Preparation materializes sources, releases, exports, tokenizers and packed
datasets but starts no training. Inspect the lock output and the effective
config before submitting any cell. `experiment run LOCK --cell
main:corpus=full --json` enqueues a worker run; run `sparselab controller run
--store sparselab-work/experiments/corpus-shape-fraction-sample-v5/controller`
in a second process to dispatch, verify and ingest it. `experiment collect LOCK
--json` writes a content-addressed evidence index; `experiment reconstruct
LOCK --index INDEX --json` checks that index and emits five retrospective views.
Use the paths returned by the commands for `LOCK` and `INDEX`. A single-cell
smoke leaves the remaining cells pending and comparisons unavailable. No
quality claim, review decision or baseline promotion follows from it.

The expected cost class is a small CPU smoke (four 20-update cells only if all
are explicitly submitted). Only task-owned directories under
`sparselab-work/experiments/corpus-shape-fraction-sample-v5/` may be removed
after confirming no worker is active; the checked-in declaration and pinned
tokenizer are durable inputs.
