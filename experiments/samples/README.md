# Sample experiments

Use externally authored current declarations with the native interfaces below.
Historical examples are evidence, not current execution guarantees or authority.

- [TinyStories microlab](../../docs/tinystories-microlab.md): an optional external
  Workshop recipe; its declarations are no longer copied into Lab.
- [Chained training program](../../docs/experiment-programs.md): the generic
  interface for explicit external
  plan/run/tokenizer inputs, immutable locking and checkpoint-bound child phases.

For an advanced research recipe, supply your own complete
[current resource closure](../../docs/external-research-resources.md). The entry
ID and resource path below are placeholders; no personal recipe is packaged.

```sh
sparselab research scaffold neutral-study --resource-root /path/to/resources \
  --scale smoke --data offline --backend cpu --output experiments/new-study
sparselab study plan experiments/new-study/study.yaml
```

Every seed must be explicit in the recipe. The generated README describes later
preparation and execution steps; scaffolding itself executes no experiment.
For a checked-in reference layout, see
[`configs/references/dense-small-v1`](../../configs/references/dense-small-v1/).
Copy editable inputs; do not edit a known-good reference or treat smoke output as
research evidence.

A sample should say what it demonstrates, what it deliberately does not prove,
the expected cost class, and which paths can be deleted safely.

## Historical sample evidence

The former training-program declarations, corpus-shape comparison and held-out
suite are preserved unchanged in [Workshop](https://github.com/supernovae/kybern-workshop/tree/df5a43aff682d14c3a3481413b80f12e7894a042/provenance/lab/experiments/samples/).
The [original sample guide](https://github.com/supernovae/kybern-workshop/blob/df5a43aff682d14c3a3481413b80f12e7894a042/provenance/lab/experiments/samples/README.md)
and [preservation manifest](https://github.com/supernovae/kybern-workshop/blob/df5a43aff682d14c3a3481413b80f12e7894a042/provenance/lab-sample-guides-manifest.json)
record the exact source bytes and identities. Their commands, budgets and host
paths are historical; they do not describe a current runnable recipe.

For a new comparison, author an external plan with explicit source, tokenizer,
held-out suite, runtime and output bindings. Follow the generic
[experiment interface](../../docs/experiments.md) and
[training-program interface](../../docs/experiment-programs.md). Reproduction is
a new experiment with a new identity and separately authorized resources. No
archived outcomes or approvals transfer to it.
