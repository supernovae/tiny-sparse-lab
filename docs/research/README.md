# Research workbench

SparseLab has two non-executing entry points: `sparselab learn list` for one initialized mechanism, and `sparselab research list` for declared controlled studies. Listing, describing, and scaffolding package metadata/configuration only: they do not train, fit a tokenizer, prepare data, open a run store, or access the network.

```sh
sparselab learn list
sparselab research list
sparselab research describe engram-ffn-substitution-v1
sparselab research scaffold engram-ffn-substitution-v1 \
  --scale smoke --data offline --backend cpu --output experiments/ffn-memory
sparselab study plan experiments/ffn-memory/study.yaml
```

The runnable scales are `smoke`, `nano`, `micro`, and `tiny`; `offline` and `tinystories` are separate data profiles. Offline uses the finite cyclic `chat_recall` fixture. TinyStories preparation is explicitly requested later and may download or use the Hugging Face cache at its pinned revision; it is not performed by browsing or scaffolding.

A scaffold publishes editable `base.yaml`, `tokenizer.yaml`, `matrix.yaml`, `study.yaml`, hashed standalone configs, and metadata. It does **not** execute them. Follow the generated README to train the tokenizer, prepare each required config's data, inspect/probe/train, or explicitly submit and collect a study. See [lesson paths](lesson-paths.md), [the catalog notes](engram-ffn-substitution.md), [evidence and reporting](evidence-and-reporting.md), and [dashboard browsing](dashboard.md).

Relative path values in matrix axis patches resolve from `matrix.yaml`; paths
in a base config resolve from that YAML file.

## Ownership-aware memory allocation

The allocation recipe requires the provenance-bound path-domain bundle before
scaffolding:

```sh
uv run sparselab research tasks build memory-allocation --output artifacts/allocation
uv run sparselab research scaffold memory-allocation-curve-v1 \
  --design iso-total --scale smoke --data offline --backend cpu \
  --output experiments/memory-allocation-iso-total
```

Run these commands from the repository root; the scaffolded configs bind the
existing `data/` and `artifacts/` source assets through project-relative paths.

Choose `default` for the iso-neural denominator, or `iso-total`, `iso-active`,
`iso-token`, or `iso-flop`. Each design crosses five ownership profiles with
neural-loss weights 1, 0.75, 0.5, 0.25, and 0; scaffolded studies additionally
match the standard three seeds. Resource labels select separate denominator
views on fixed architecture/token-budget runs, not independent replications or
measured compute regimes. The FLOP field is a rough parameter-based estimate
only. The semantic pack stores training answers and uses deterministic
structured `(path, operation, argument)` keys; it is not a natural-language
encoder or evidence of semantic generalization. See [the corpus and allocation
boundaries](../path-domain-corpus.md#ownership-aware-allocation).

Published worked examples: [FFN-substitution smoke report](../../artifacts/research-reports/fbdb00217f8e952e12bd07e053d97d85796f889748ee77d3bc81b21bb3c98c0b/index.html) and [nano/offline FFN follow-up report](../../artifacts/research-reports/a444e2869973568e28315aa2cac1a97454d7f9d7b8742fd69de8358e867e7daf/index.html), with their [outcomes and interpretation](sample-report.md).
