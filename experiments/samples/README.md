# Sample experiments

Samples teach the workflow and are safe to copy. They are not claims that a
mechanism improves quality or performance.

Start with a packaged scaffold:

```sh
export SPARSELAB_WORK_DIR="$PWD/sparselab-work/experiments/ffn-memory-sample"
uv run --locked sparselab research scaffold engram-ffn-substitution-v1 \
  --scale smoke --data offline --backend cpu \
  --output "$SPARSELAB_WORK_DIR/scaffold"
uv run --locked sparselab study plan "$SPARSELAB_WORK_DIR/scaffold/study.yaml"
```

The generated README identifies the explicit preparation and execution steps.
For a checked-in reference layout, see
[`configs/references/dense-small-v1`](../../configs/references/dense-small-v1/).
Copy editable inputs; do not edit a known-good reference or treat smoke output as
research evidence.

A sample should say what it demonstrates, what it deliberately does not prove,
the expected cost class, and which paths can be deleted safely.
