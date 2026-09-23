# Independent experiments and controlled comparisons

Run each concrete configuration with its declared tokenizer, data source, engine/backend, precision, optimizer, sequence length, effective batch, token budget, and seed. Inspect first; do not infer parameter counts or scientific equivalence from a filename.

```sh
uv run sparselab inspect configs/micro_dense.yaml --json
uv run sparselab inspect configs/dense_7m.yaml --json
uv run sparselab inspect configs/dense_10m.yaml --json
uv run sparselab inspect configs/dense_25m.yaml --json
uv run sparselab inspect configs/dense_50m.yaml --json
uv run sparselab inspect configs/smoke_sliding_cpu.yaml --json
uv run sparselab inspect configs/smoke_mla_cpu.yaml --json
uv run sparselab inspect configs/smoke_combined_cpu.yaml --json
```


The dense scale presets are inspected at 3,344,064, 6,917,376, 10,244,160, 29,893,120, and 50,274,752 parameters. They share the pinned TinyStories revision, 8192-token tokenizer, sequence length, token budget, optimizer, and seed. Parameter count alone is not a comparison result: report each completed run's observed validation loss, perplexity, throughput, device, and metric coordinates.

## Explicit matrices and worker execution

The checked-in matrix expands the CPU runtime smoke configuration over seeds 7, 17, and 41:

```sh
uv run sparselab experiment submit --matrix tests/fixtures/runtime-matrix.yaml \
  --dry-run --store /tmp/sparselab-controller
```

Dry-run prints all resolved coordinates and hashes without downloading/preparing data, creating the store, or enqueueing. For execution, prepare the tokenizer, register an eligible worker, remove `--dry-run`, and run `sparselab controller run --store /tmp/sparselab-controller`. Every coordinate gets distinct experiment/attempt/run IDs. Preparation must succeed for all coordinates before the enqueue transaction.

Matrix v1 uses an explicit base config and insertion-ordered axes with labeled dotted-path patches. Duplicate labels, conflicting patches, invalid configs, and excessive expansion are rejected. Labels such as “50M” or “MoE” never infer a model. Workers execute independent optimizers, not distributed gradients; unsupported requirements remain queued with a reason. See [worker contracts and matrix format](workers.md#explicit-matrices).

## Completed multi-seed studies

- [Context/Engram study](context-engram-study.md#execution-results--2026-09-22): 24 endpoints spanning seeds 17/41/73, two exact target budgets, dense/backbone and dense-total comparisons, and collision/address-order diagnostics. Every untouched override endpoint remained 0/8. Added memory capacity and observed bucket collisions are not evidence of a generalization advantage.
- [Domain adaptation](path-domain-corpus.md#2026-09-22-execution-record): three pretraining/adaptation pairs with frozen supervision, provenance, semantic leakage checks, per-case results, and static-retention measurements. Training-case acquisition improved, but held-out reliability remained poor and retention worsened sharply.

[Independent acceptance](../artifacts/acceptance/scientific_studies_2026_09_22.json) binds the input inventories, endpoint identities, stored responses, paired deltas, and retention observations. These studies retain all declared seeds/endpoints and their negative outcomes; they are not a general model-selection or significance framework.

## Historical controlled dense runs

These recorded runs used the pinned TinyStories revision, the same 8192-token tokenizer, seed, optimizer, 128-token sequence length, and 204,800 target-token budget on MPS. Validation was standalone evaluation over the retained validation split. They predate the current integrity-bound report format; retain them as historical loss observations, not newly verified capability evidence.

| Run | Parameters | Validation loss | Validation perplexity |
|---|---:|---:|---:|
| `scale-micro-3m` | 3,344,064 | 5.1614 | 174.42 |
| `scale-dense-7m` | 6,917,376 | 4.9942 | 147.55 |
| `scale-dense-10m` | 10,244,160 | 4.8790 | 131.49 |
| `scale-dense-25m` | 29,893,120 | 4.5751 | 97.04 |
| `scale-dense-50m` | 50,274,752 | 4.2770 | 72.03 |

These observations suggest lower validation loss at this fixed budget as dense parameter count increases. They do not establish an optimal scaling law, chat ability, or an Engram benefit. Previously published throughput values are withdrawn: the trainer divided one update's tokens by cumulative run time. New measurements use synchronized update duration; rerun controlled pairs before making a performance claim.
The smoke configurations prove CPU wiring only. For a meaningful comparison, train separate unique run IDs with matching source, tokenizer, device, token budget, sequence length, optimizer, and seed. Store the resulting run directory and SQLite metrics; compare observed loss or throughput only at matching recorded budgets. Do not convert unmatched runs into an aggregate quality score.

Dense attention, sliding-window attention, MLA, MoE, and byte memory alter different resource boundaries. Attribute an observed difference only after a controlled ablation; a combined run is a compatibility check, not evidence that its mechanisms compound beneficially.

## Evidence before comparison

Every serious local run should have verified checkpoint/held-out evidence before it enters a comparison:

```sh
uv run sparselab checkpoint verify runs/RUN_ID/checkpoints/latest.json --json
uv run sparselab evidence RUN_ID --json
```

Training records held-out validation at initial, configured periodic, and terminal boundaries. A validation metric is not by itself a saved model: checkpoint cadence, a new best loss, or termination triggers a verified generation. Use checkpoint-bound evaluation reports when claiming results for exact weights. See [experiment evidence](evidence.md) for evidence levels, controlled-comparison requirements, and the separately unimplemented hardware/reference-harness protocol.

For a named architectural hypothesis rather than aggregate loss alone, use a [capability card](capabilities.md). Cards preserve their own prompt set, scorer, baseline controls, and scope-limited conclusion.
