# Sample report: FFN substitution smoke run

[Open the static report](../../artifacts/research-reports/fbdb00217f8e952e12bd07e053d97d85796f889748ee77d3bc81b21bb3c98c0b/index.html) · [Inspect its manifest](../../artifacts/research-reports/fbdb00217f8e952e12bd07e053d97d85796f889748ee77d3bc81b21bb3c98c0b/manifest.json)

This is a completed CPU/offline smoke campaign for `engram-ffn-substitution-v1`, not a pretrained model or a performance claim. The report bundles the original receipt and collected evidence, study inputs, per-run results, static charts, and validated local checkpoint evidence. Its verification scope is `report_plus_local_checkpoint_validation`. It contains no model weights, prepared arrays, dataset cache, or runnable checkpoints.

## Summary

Across the original FFN smoke and nano campaigns plus this fixed-seed nano rerun, lexical memory showed no consistent held-out loss benefit; the nano `1x` lexical-memory pairs had higher loss in all three seeds. All 54 run executions scored zero on each of the three capability cards (162 run/card scores), so these observations provide no evidence that lookup compensates for reduced FFN capacity. These short CPU/offline observations do not establish a general model-quality result.

The nano `4x`/`1x` × no-memory/lexical-memory interaction is complete and repeated with identical per-coordinate losses. The next engineering step is Phase C milestone/timing measurement, which needs a declared observation contract; see below.

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

## Nano/offline FFN follow-up

[Open the original static report](../../artifacts/research-reports/a444e2869973568e28315aa2cac1a97454d7f9d7b8742fd69de8358e867e7daf/index.html) · [Inspect its manifest](../../artifacts/research-reports/a444e2869973568e28315aa2cac1a97454d7f9d7b8742fd69de8358e867e7daf/manifest.json) · [Open the repeat report](../../artifacts/research-reports/9afaedb3e6f20769ef71f3211077d6d8f624b95a6e31805804f2d1fce79673e2/index.html) · [Inspect its manifest](../../artifacts/research-reports/9afaedb3e6f20769ef71f3211077d6d8f624b95a6e31805804f2d1fce79673e2/manifest.json)

This completed CPU/PyTorch FP32 follow-up used the finite `chat_recall` offline fixture. It crossed FFN widths `4x`, `2x`, and `1x` with and without a lexical table (4,095 entries, value width 16, order 3, final injection), using seeds 17, 41, and 73. All 18 runs reached step 128 and 32,640 tokens; the step cap came before the 32,768-token cap. All 21 declared comparisons completed.

At each fixed width, the paired held-out validation-loss delta is `lexical - no memory`:

| FFN width | Mean delta | Per-seed range | Lower loss with lexical memory |
|---|---:|---:|---:|
| 4x | +0.032894 | -0.011908 to +0.066751 | 1/3 pairs |
| 2x | +0.000376 | -0.032826 to +0.027593 | 1/3 pairs |
| 1x | +0.057911 | +0.033254 to +0.083721 | 0/3 pairs |

The 1x lexical-memory variant had higher validation loss than its no-memory match in all three seeds. The 2x mean is near zero and mixed. No width shows a consistent same-width validation-loss improvement from lexical memory.

Width contrasts (narrower width minus 4x at the same memory setting) were:

| Memory | 1x - 4x mean (range) | 2x - 4x mean (range) |
|---|---:|---:|
| None | +0.055161 (-0.048475 to +0.112366) | +0.021280 (-0.003388 to +0.050933) |
| Lexical | +0.080178 (+0.047154 to +0.125285) | -0.011238 (-0.024306 to +0.013456) |

At 1x, lexical memory did not close the width gap: 1x remained worse than 4x lexical in all three seed pairs. At 2x, the lexical width contrast was slightly lower on average but crossed zero; it is a descriptive result from three seeds, not evidence that memory replaces FFN capacity.

The matched four-cell `4x`/`1x` × no-memory/lexical held-out-loss interaction is `y11 - y10 - y01 + y00`, where `y00=4x/no-memory`, `y10=1x/no-memory`, `y01=4x/lexical`, and `y11=1x/lexical`. Per-seed deltas were -0.033497, +0.095629, and +0.012919 (mean +0.025017); signs were mixed.

All 18 runs scored zero on each capability card: `chat-alias-retention-v1` (24 cases), `chat-alias-recall-v1` (12), and `chat-context-override-v1` (8). This follow-up found no card evidence that lexical memory compensated for a narrower FFN. The bundle validates all 18 local run records and all five checkpoint generations per run (including the initial state; 90 verified checkpoint/quality-observation records); all comparisons completed with no missing or rejected evidence. These short synthetic-task observations are not significance tests or general language-model claims. The static bundle contains no weights, prepared arrays, dataset cache, or runnable checkpoints. Its evidence copies include machine-local paths; hashes protect byte identity, not privacy.

An independent execution with distinct run IDs and a new receipt reused the same study, matrix, data-profile, and scale hashes. It reproduced the prior held-out validation loss at all 18 coordinates exactly. This is a same-seed reproducibility rerun, not additional independent seed evidence.

## Phase B: research-analysis pipeline smoke

A CPU/offline `engram-mla-compression-v1` campaign exercised the factorial, nondominance, allocation, and boundary-sweep report paths. This is an engineering smoke, not an attention or memory-quality conclusion. All 12 runs (dense/MLA-half attention × no/lexical memory × seeds 17, 41, and 73) passed local evidence validation and reached step 32 / 4,096 tokens.

The raw validation-loss interaction is `y11 - y10 - y01 + y00`, with `y00=dense/no-memory`, `y10=MLA-half/no-memory`, `y01=dense/lexical`, and `y11=MLA-half/lexical`:

| Seed | Interaction delta |
|---|---:|
| 17 | -0.030430 |
| 41 | -0.033917 |
| 73 | +0.035364 |

The signs are mixed. All three capability-card scores were zero in each of the four cells for all three seeds (36 measured zeros); this smoke found no card evidence of benefit. The generated analysis contained four complete cells per seed, complete matched nondominance groups, configuration-derived allocation heatmaps, and observed-only sweeps. These outputs validate report behavior on this evidence; they do not establish quality, efficiency, or a useful interaction.

Verification: `uv run pytest -q tests/test_study_reporting.py tests/test_research_workbench.py` (13 passed); the regenerated static report and interaction SVG were opened in a browser. The nano FFN-width × lookup experiment above is complete; this MLA smoke remains a separate engineering validation, not a training-quality result.

## Next work

### Phase C: define milestone and timing contract

This nano campaign captured 128 optimizer-update timing samples per run (`performance/step_seconds` and `performance/tokens_per_second`), but it did not measure end-to-end wall time or time to a quality threshold. The recipe declares `milestones_supported: false` and no primary thresholds. Update telemetry excludes evaluation, checkpoint, setup, and idle time; no billing or energy evidence is present.

Freeze a versioned milestone contract first: outcome and direction, threshold, observation boundary, checkpoint/card capture, and censored or failed-run representation. Then independent subagents can work in parallel on milestone capture/evaluation, cost/telemetry aggregation, and report/chart presentation. Keep one integration owner for the shared trainer and report schemas.

Do not run timing jobs concurrently on the same physical CPU/device. Controller capacity is reserved per worker ID, not per host; separate worker IDs on this workstation would contend. Serialize timed runs per resource or use separate machines.

### Further training: map failure boundaries

Use separate, declared sweeps to locate where results change with FFN width/table size, MLA latent width, sparse selected-block budget, and MoE expert capacity. Use heatmaps only for points within the same evidence group (dataset, runtime, endpoint, and metric direction); keep architecture estimates separate from measured outcomes and preserve tied values. Increase scale only after the smaller comparison is interpretable.

### Separate data follow-up

For a public-data question, use the micro profile and pinned TinyStories recipe:

```sh
sparselab research scaffold engram-ffn-substitution-v1 \
  --scale micro --data tinystories --backend cpu --output experiments/ffn-memory-tinystories
sparselab study plan experiments/ffn-memory-tinystories/study.yaml
```

This requires explicit tokenizer training and data preparation and may download/cache TinyStories. Use held-out story loss and fixed, user-visible generation examples; the alias and context cards remain out-of-domain stress checks, not TinyStories quality measures. Do not pool these scores with offline results.

Other controlled questions already available through `sparselab research list` include:

- `engram-placement-v1`: compare embedding versus final memory injection.
- `engram-mla-compression-v1`: compare dense/MLA attention with and without lexical memory.
- `engram-sparse-budget-v1`: compare dense/block-sparse attention budgets with and without memory.
- `engram-moe-capacity-v1`: compare dense FFN width and local MoE capacity with and without memory.
- `lexical-memory-heavy-v1`: vary lexical-table capacity while holding the reduced neural backbone fixed.

These are new experiments, not analyses that can be run against the sample's existing checkpoints. Each has its own declared controls, recipe, and limitations; scaffold it and follow its generated README. To inspect the lookup mechanism independently before training, use the [lexical Engram lesson](lesson-paths.md) and its initialized-model probe. A probe explains shapes and diagnostics; it is not learned-behavior evidence.

## Sharing and reproducibility

The report directory is content-addressed by its manifest digest. Its original evidence copies include machine-local paths; the hashes protect byte identity, not privacy or authenticity. Review those paths before redistributing the bundle. Reproducing a new training result also requires the explicitly prepared tokenizer/data and run artifacts, which are intentionally excluded from this sample.
