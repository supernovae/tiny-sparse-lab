# Sample report: FFN substitution smoke run

[Open the static report](../../artifacts/research-reports/fbdb00217f8e952e12bd07e053d97d85796f889748ee77d3bc81b21bb3c98c0b/index.html) · [Inspect its manifest](../../artifacts/research-reports/fbdb00217f8e952e12bd07e053d97d85796f889748ee77d3bc81b21bb3c98c0b/manifest.json)

This is a completed CPU/offline smoke campaign for `engram-ffn-substitution-v1`, not a pretrained model or a performance claim. The report bundles the original receipt and collected evidence, study inputs, per-run results, static charts, and validated local checkpoint evidence. Its verification scope is `report_plus_local_checkpoint_validation`. It contains no model weights, prepared arrays, dataset cache, or runnable checkpoints.

## What was compared

The study crossed three FFN widths with and without lexical lookup: `4x`, `2x`, and `1x` the smoke profile's hidden width. Each of the six conditions used seeds 17, 41, and 73: 18 runs total. It declared 21 matched comparisons. All runs reached step 32 and 4,096 tokens on the finite `chat_recall` offline curriculum. The lexical table had 257 entries, value width 8, order 3, and final injection.

The held-out validation-loss comparison is reported separately from the capability cards. For lexical-memory versus no-memory pairings at fixed width, the three-seed mean `variant - baseline` loss deltas were:

| FFN width | Mean paired validation-loss delta | Per-seed range | Direction |
|---|---:|---:|---|
| 4x | -0.025567 | -0.035811 to -0.010128 | Lower in all three observed pairs |
| 2x | -0.048185 | -0.073429 to -0.013180 | Lower in all three observed pairs |
| 1x | +0.040160 | -0.021868 to +0.105647 | Mixed; higher on average |

Negative deltas mean lower validation loss for the lexical-memory variant. These are short-run observations, not evidence that lookup generally improves language modeling or replaces FFN capacity. The 1x pairs varied in direction.

All 18 runs scored zero on each of `chat-alias-retention-v1`, `chat-alias-recall-v1`, and `chat-context-override-v1`. Thus this campaign found no capability-card evidence that lookup compensated for a narrower FFN at this endpoint. A zero is a measured score, not missing evidence; it also does not establish that longer training or another dataset would fail.

The bundle validates 18 local run/checkpoint records and includes 5,850 bounded telemetry rows. Telemetry is locally captured evidence, not a checkpoint-signed measurement. The short smoke endpoint, finite repeated training curriculum, three seeds, and narrow cards limit interpretation. The cards test alias acquisition/recall and context override, not general reasoning or story quality.

## Experiments to run next

The report is read-only evidence; it cannot resume these runs or evaluate another checkpoint. To test whether the zero card scores reflect insufficient training, first repeat the same question at the supported `nano` scale with the same offline data route and recipe controls:

```sh
sparselab research scaffold engram-ffn-substitution-v1 \
  --scale nano --data offline --backend cpu --output experiments/ffn-memory-nano
sparselab study plan experiments/ffn-memory-nano/study.yaml
```

Then follow the generated `README.md` to explicitly train the tokenizer, prepare data for the required configs, and run/collect the campaign. Keep all six conditions and the declared three seeds. Interpret the new report within nano; do not pool its deltas with smoke or select a favorable endpoint. Check whether the 1x loss pattern repeats and whether any card produces nonzero evidence. Even then, the result is limited to these tasks and endpoints.

A separate public-data question can use the micro profile and pinned TinyStories recipe:

```sh
sparselab research scaffold engram-ffn-substitution-v1 \
  --scale micro --data tinystories --backend cpu --output experiments/ffn-memory-tinystories
sparselab study plan experiments/ffn-memory-tinystories/study.yaml
```

This requires explicit tokenizer training and data preparation and may download/cache TinyStories. Treat it as a separate dataset experiment: the alias and context cards are out-of-domain stress checks for stories, not TinyStories quality measures. Use held-out TinyStories loss and fixed, user-visible generation examples; do not combine these scores with the offline results.

Other controlled questions already available through `sparselab research list` include:

- `engram-placement-v1`: compare embedding versus final memory injection.
- `engram-mla-compression-v1`: compare dense/MLA attention with and without lexical memory.
- `engram-sparse-budget-v1`: compare dense/block-sparse attention budgets with and without memory.
- `engram-moe-capacity-v1`: compare dense FFN width and local MoE capacity with and without memory.
- `lexical-memory-heavy-v1`: vary lexical-table capacity while holding the reduced neural backbone fixed.

These are new experiments, not analyses that can be run against the sample's existing checkpoints. Each has its own declared controls, recipe, and limitations; scaffold it and follow its generated README. To inspect the lookup mechanism independently before training, use the [lexical Engram lesson](lesson-paths.md) and its initialized-model probe. A probe explains shapes and diagnostics; it is not learned-behavior evidence.

## Sharing and reproducibility

The report directory is content-addressed by its manifest digest. Its original evidence copies include machine-local paths; the hashes protect byte identity, not privacy or authenticity. Review those paths before redistributing the bundle. Reproducing a new training result also requires the explicitly prepared tokenizer/data and run artifacts, which are intentionally excluded from this sample.
