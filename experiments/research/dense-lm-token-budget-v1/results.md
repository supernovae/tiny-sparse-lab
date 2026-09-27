# dense-lm-token-budget-v1: measured results

Protocol: [frozen protocol](protocol.md) and [pre-outcome registration](preregistration.md). Machine-readable raw outputs, case diagnostics, checkpoint digests, cursor/schedule, manifest and observation hashes, and interval timings: [acceptance evidence](../../../artifacts/acceptance/dense_lm_token_budget_v1.json). Mutable full-state generations, periodic evaluation records, staged bundles, exercise observations and SQLite metrics remain in `sparselab-work/experiments/dense-lm-token-budget-v1/`. No baseline or source run was modified.

## Controlled continuation and integrity

The three original 4,096-update AdamW/FP32 ROCm generations are the parents; `--extend-budget` restored model, optimizer, RNG, cursor and counters, preserving the 4,096-step decay horizon and applying floor LR 3e-5 thereafter. Effective batch 8 × 128 = 1,024 supervised targets/update. All three new 8,192 and 16,384 checkpoints passed full checkpoint verification; each exercise reported exact 8,388,608 or 16,777,216 targets, finite weights, a mechanically valid and deterministic fixed panel, and the same tokenizer, train array and validation array hashes. Epoch/cursor `(0,32768) → (1,31141) → (3,27887)` for all three seeds; 4,402,585 packed training tokens are repeatedly exposed across epochs, **not** new documents. ROCm was AMD Radeon RX 7900 XTX (24 GiB), PyTorch 2.13.0+rocm10.0.0, FP32 under WSL2. Source identity changed from `927dca9d…` to `fb979d47…`; original-to-extended runs explicitly allowed audited instrumentation/host-identity and epoch-boundary code drift, while the data, evaluator, optimizer state and scientific config stayed fixed. The second intervals used strict ordinary resume, no drift allowance.

An initial seed-42 attempt was **interrupted** at step 8,192 with 8,387,968 targets (640 short): a batch had been committed across an epoch boundary with only a partial update. `model exercise` rejected the coordinate before any accepted observation. It remains preserved as `dense-lm-token-budget-v1-seed42-step8192` (manifest `f788f15157459a69b48ce6399acadfc11d3a41765ccace28de6f0e27d558ce41`). The trainer was corrected to collect whole updates across boundaries, regression-tested, re-staged, and the original immutable parent was re-extended into the separate `…step8192-corrected` run. No results from the interrupted attempt enter the comparison.

## Frozen held-out measurement

Each observation measured exactly 64 validation batches / 65,408 targets using the unchanged validation array. Lower is better. Gains are nats per held-out target; per-million rates divide the gain by 4.194304M then 8.388608M additional supervised target exposures.

| Seed | 4,096 / 4.19M | 8,192 / 8.39M | 16,384 / 16.78M | First gain (per added M) | Second gain (per added M) | Step 16,000 → 16,384 decrease |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 2.482474 | 2.368982 | 2.243266 | 0.113492 (0.027059) | 0.125716 (0.014987) | 0.003474 |
| 17 | 2.484719 | 2.369684 | 2.243805 | 0.115034 (0.027426) | 0.125879 (0.015006) | 0.004283 |
| 73 | 2.470434 | 2.360011 | 2.234135 | 0.110423 (0.026327) | 0.125876 (0.015006) | 0.003203 |

Seed 42 crossed the preregistered 0.05 material-gain gate at 8M (0.113492); its 16M loss decreased a further 0.125716. This triggered both seeds 17 and 73 through **both** milestones before inspecting their outcomes. Their 8M/16M decreases are similarly material (0.115034/0.125879 and 0.110423/0.125876). The last periodic-to-end decrease is <0.01 for each seed, but the 8M→16M decrease exceeds one-quarter of the first decrease in every seed. The three-part **observable flattening criterion therefore fails**; do not call this convergence. Loss improvement does not establish better text.

## Fixed greedy panel: independent reading

Classification is relative to each seed’s original 4,096-update output: improved / unchanged / worse; OOD is descriptive, not a promotion gate. Diagnostic cells are repeated-token / bigram / trigram excess, distinct-token ratio, at baseline → 8M → 16M. All outputs are nonempty, boundary-valid and repeatable; full verbatim strings and per-case metrics are retained in the acceptance JSON.

| Seed | Case | 8M vs baseline | 16M vs baseline | Diagnostics: baseline → 8M → 16M |
|---|---|---|---|---|
| 42 | simple-continuation | improved | improved | 6/1/0, 0.812 → 10/4/3, 0.688 → 12/4/1, 0.625 |
| 42 | named-character-continuity | improved | improved | 12/7/6, 0.625 → 1/0/0, 0.933 → 1/0/0, 0.933 |
| 42 | color-object-continuity | improved | worse | 18/13/11, 0.438 → 7/1/0, 0.781 → 13/6/3, 0.594 |
| 42 | temporal-causal-continuation | unchanged | improved | 10/2/0, 0.688 → 12/3/1, 0.625 → 12/7/5, 0.613 |
| 42 | ordinary-sentence-completion | improved | improved | 5/0/0, 0.737 → 2/0/0, 0.778 → 1/0/0, 0.833 |
| 42 | odd-out-of-domain | descriptive | descriptive | 6/1/0, 0.812 → 9/2/0, 0.719 → 7/0/0, 0.781 |
| 17 | simple-continuation | improved | unchanged | 11/5/2, 0.656 → 9/1/0, 0.719 → 11/3/1, 0.656 |
| 17 | named-character-continuity | improved | improved | 6/1/0, 0.812 → 2/0/0, 0.889 → 7/1/0, 0.781 |
| 17 | color-object-continuity | worse | worse | 13/6/2, 0.581 → 21/17/15, 0.344 → 10/4/2, 0.688 |
| 17 | temporal-causal-continuation | improved | unchanged | 16/9/7, 0.500 → 12/8/6, 0.613 → 7/2/1, 0.781 |
| 17 | ordinary-sentence-completion | unchanged | unchanged | 1/0/0, 0.833 → 1/0/0, 0.833 → 1/0/0, 0.833 |
| 17 | odd-out-of-domain | descriptive | descriptive | 1/0/0, 0.941 → 11/6/4, 0.656 → 13/6/2, 0.594 |
| 73 | simple-continuation | worse | worse | 6/1/0, 0.806 → 14/9/7, 0.562 → 17/12/10, 0.469 |
| 73 | named-character-continuity | improved | improved | 8/2/1, 0.750 → 1/0/0, 0.938 → 3/2/1, 0.800 |
| 73 | color-object-continuity | worse | unchanged | 11/3/1, 0.656 → 7/1/0, 0.781 → 6/2/1, 0.760 |
| 73 | temporal-causal-continuation | improved | improved | 11/7/6, 0.645 → 8/3/2, 0.750 → 0/0/0, 1.000 |
| 73 | ordinary-sentence-completion | unchanged | unchanged | 2/0/0, 0.750 → 2/0/0, 0.750 → 2/0/0, 0.750 |
| 73 | odd-out-of-domain | descriptive | descriptive | 9/2/0, 0.719 → 13/9/7, 0.594 → 10/2/0, 0.688 |

Explicit continuity evidence and regressions: seed 42’s baseline named case introduces `Anna` and `Ben` after Mia and Leo launch the boat; both longer-budget outputs retain Mia/Leo and boat. Its baseline green meadow is `very happy`, corrected to `very big` at 8M; at 16M the meadow becomes `blue`, an explicit new contradiction of green despite lower loss. Seed 17’s baseline changes Mia/Leo into Mia/Tom; 8M retains Mia/Leo but `thanked the boat`, and 16M retains Mia and the boat. Seed 17’s baseline meadow is `very happy`; 8M repeats `the sun was shining` many times; at 16M the meadow becomes `blue`, contradicting green. Seed 73’s baseline names Mia/Leo but veers into a tree; both longer outputs keep Mia/Leo, while at 8M its green meadow `could fly` and at 16M the meadow is `very pretty`. For the rain/wet-path case, seed 42 first invents a dog at 8M, then Sam goes inside at 16M but talks to himself; seed 17’s 8M Sam returns home, whereas at 16M he is `happy to see the rain` after it stopped and thanks himself; seed 73’s 8M changes into unrelated ice-cream dialogue, then 16M says Sam is `happy to see the sun again`. Improvements are local to these fixed greedy outputs, not a semantic quality score. Seed 73’s simple rabbit output deteriorates into repetitive self-thanks (6 → 14 → 17 excess tokens). OOD chat capability cards were NOT_APPLICABLE and excluded.

## Costs and resource measurement

`/usr/bin/time` command wall includes process start and loading, while optimizer, inline validation, checkpoint and reporting seconds below are **interval-only** persisted `runtime_progress_snapshots` phases; command wall, cumulative optimizer counters and exercise durations are not interchangeable. Measurements in seconds; peak bytes from per-update SQLite metric samples.

| Seed | Interval | Command wall | Optimizer | Targets/s (optimizer) | Validation | Checkpoint | Reporting | Peak allocated / reserved (MiB) | Exercise wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4096→8192 | 513.10 | 337.49 | 12,428 | 32.11 | 23.86 | 100.00 | 1014.9 / 1066.0 | 17.57 |
| 42 | 8192→16384 | 1057.55 | 683.58 | 12,272 | 62.84 | 41.69 | 236.48 | 1014.9 / 1066.0 | 17.90 |
| 17 | 4096→8192 | 509.14 | 332.81 | 12,603 | 32.38 | 21.99 | 100.63 | 1014.9 / 1066.0 | 13.91 |
| 17 | 8192→16384 | 1013.25 | 674.41 | 12,439 | 62.90 | 40.66 | 203.33 | 1014.9 / 1066.0 | 19.00 |
| 73 | 4096→8192 | 512.31 | 336.24 | 12,474 | 32.17 | 20.94 | 100.63 | 1014.9 / 1066.0 | 16.06 |
| 73 | 8192→16384 | 1015.97 | 674.81 | 12,431 | 62.68 | 40.05 | 205.03 | 1014.9 / 1066.0 | 16.83 |

The historical seed-42 reference was 584.232 s training-command wall and 332.123 s optimizer total for 4,096 updates. Its forecast was about 332 optimizer seconds per additional 4.19M exposures: observed first interval 337.49 s (+5.37 s), second 683.58 s versus 664 forecast (+19.58 s); the second interval spans twice as many targets. Across successful intervals all sampled overflow/retry counts were zero, peak allocated 1,064,241,664 bytes and peak reserved 1,117,782,016 bytes. Seed 42 cumulative optimizer seconds were 332.12 → 669.61 → 1353.19; seed 17 335.79 → 668.60 → 1343.01; seed 73 335.71 → 671.95 → 1346.76. ROCm driver version was unavailable through safe PyTorch APIs. `generation` is a separate exercise rather than an inline training timing; hardware portability, long-run convergence and broader generation quality remain unproven.

## Decision

The study supports lower held-out loss under **additional exposures to the same prepared data at floor LR** on all three seeds, but the fixed panel has material counterexamples, notably blue/green contradictions and increased repetition. Do not promote the 16M checkpoint automatically or infer a new data regime; retain the frozen baseline and all negative/interrupted evidence. Any further budget level or changed criterion needs a newly registered research protocol.
