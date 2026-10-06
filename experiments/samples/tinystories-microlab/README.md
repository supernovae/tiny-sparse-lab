# TinyStories iteration sample

Follow [the complete walkthrough](../../../docs/tinystories-microlab.md) from
the checkout root. Copy the YAML declarations into a fresh external task workspace's `inputs/` directory before
running them. Their `../tokenizer`, `../data` and `../runs` paths are anchored to
the copied configs, so the payloads stay in that task workspace. Running templates
directly from this directory would put payloads in the checkout.

The sequence is one 40-update / 10,240-target baseline, native evaluation and
generation, then a verified full-state PyTorch AdamW child ending at 80 cumulative
updates / 20,480 targets. `continued.yaml` preserves the original effective
40-update decay horizon. Change backend selections in both copied run configs
together. Preserve parent bytes and negative outputs.

The optional matrix trains two separately labeled fresh FFN widths at the same
baseline budget. It does not reuse the preceding direct baseline automatically.
Its extra compute is explicit. A source download needs network access; the
models and runs are tiny teaching fixtures, not quality benchmarks. Native
checkpoint verification establishes integrity, not better stories.

Only task-owned mutable payloads may be removed after checking that no worker
is active; declarations and retained results stay available. The native commands
and required gates are described in [the iteration guide](../../../docs/iteration.md).
`source.yaml` pins the Hub repository and selection policy; native `data lock`
and `data snapshot` materialize it through generic dataset interfaces. This new
selection deduplicates whole documents and protects validation. Its identity is
different from historical direct-prefix TinyStories runs; those inputs remain intact.

The walkthrough also covers native `experiment bind-inputs`, a declared width
comparison with checkpoint continuations, and progression toward larger models
and full-source training. For a separate end-to-end exercise, copy `campaign.yaml`
to the workspace root; it references declarations in `inputs/` and performs
acquisition through evaluation and descriptive generation for three selected
cells. `plan.yaml` also declares an optional wider child which this Campaign does
not select. Neither route imports a prior baseline automatically.

Coverage and pass budgets use native `data coverage` / `data budget`. Full-data
execution needs separate resource admission; the sample is a bounded lesson.

The [engineering acceptance record](acceptance.md) retains the scope, identities,
results and limitations of a completed bounded Campaign using these interfaces.
