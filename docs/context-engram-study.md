# Context-local overrides and Engram study

## Status, scope, and frozen packet

This is a preregistration and input packet, not a result. It neither replaces the historical `chat-context-override-v1` control nor changes the previously recorded negative dense/Engram evidence in [project-review.md](project-review.md). Historical assets, runs, and outcomes remain outside this packet and untouched.

The corpus is original synthetic `local_chat` material, under the repository MIT license and Copyright (c) 2026 Byron Miller (MOBB). Its source provenance is `data/context_override_v2/provenance.json`. The immutable input record is `data/context_override_v2/context_engram_study_input_ledger_v1.json` (`format: context-engram-study-input-ledger`, `version: 1`). It fixes SHA-256 and byte size for training, validation, provenance, semantic audit, every concrete config, all three cards, the train-only tokenizer, and the prepared token stream. It contains no run result, checkpoint, model output, or mutable metric.

The question is narrow: after explicit training, can a model use a supplied conversation-local assignment instead of its taught permanent association, including a later conflicting assignment? This is not a test of external knowledge, retrieval augmentation, native sparse speed, or general chat quality.

## Frozen corpus, cards, and audit boundary

| Asset | Frozen role | Checkpoint-selection role |
|---|---|---:|
| `data/context_override_v2/train.jsonl` | 24 original all-token curriculum conversations | Never |
| `data/context_override_v2/validation.jsonl` | 8 disjoint development conversations | Reported diagnostic only |
| `context_override_acquisition_v1.card.json` | Eight training-seen local bindings; acquisition/sanity report | Never |
| `context_static_retention_v1.card.json` | Eight taught permanent associations with held-out query wording and no value in prompt | Never |
| `context_override_untouched_v1.card.json` | Eight frozen local-binding cases, including four last-assignment cases | Never |

Keep the existing `capability_card_v2` cards and their `normalized_full_answer_exact_v1` scorer exactly as frozen. This packet deliberately does **not** migrate to assistant-only loss or alter any card/scorer. The curriculum remains the v1 all-token token-loss mask: every packed train and validation token is supervised.

The semantic audit is a manual scope audit, not a lexical-disjointness claim. Static-retention entities overlap training by design; untouched override final entity-to-value pairs are absent from training and validation; common instruction words and some value tokens are intentionally shared. Audit version 2 also explicitly enumerates the discarded initial bindings `atlas->navy`, `beacon->gold`, `cipher->mint`, and `delta->flax`, alongside the four final last-assignment bindings. This coverage clarification was frozen before model execution; it changes neither prompts nor outcomes.

## Train-only tokenizer and actual packed stream

Train the tokenizer only from `train.jsonl`, through `configs/context_study_tokenizer.yaml`, into `artifacts/tokenizer_context_study/`. The tokenizer training manifest records 24 selected training documents and no validation/card input. The prepared stream is at `artifacts/tokenizer_context_study/prepared_data/700fe5e374dd8f13/`; all hashes and sizes are in the ledger.

Deterministically measured facts from that frozen tokenizer and `contiguous-eos-v5` packing are:

| Fact | Value |
|---|---:|
| Train packed tokens / supervised tokens | 1,792 / 1,792 |
| Validation packed tokens / supervised tokens | 761 / 761 |
| Sequence length | 64 |
| Usable train blocks | 27 |
| Usable supervised targets per epoch | 1,728 |
| Initial input-only token / discarded trailing tokens per epoch | 1 / 63 |
| Update-target pattern per epoch (`micro_batch_size=2`, accumulation 1) | 13 × 128, then 1 × 64 |

The count follows the trainer's full-block rule: `floor((1792 - 1) / 64) = 27` blocks, each with 64 targets. The first token supplies input only; 63 trailing tokens lie beyond the full blocks. This is not an assistant-only mask effect.

This also freezes generation *bounds*, without consulting model outputs. Across all 24 frozen card cases, the longest prompt is 111 tokenizer tokens; the longest expected answer is 5 tokens; the largest prompt-plus-answer bound is 116 tokens. The three unchanged cards each configure greedy deterministic generation with `max_new_tokens=8`; the strict largest prompt-plus-configured-generation bound is therefore 119 tokens, below model context 128. Five tokens are sufficient to express every known expected label, while the frozen eight-token ceiling remains authoritative. The run protocol must not choose a longer generation based on an observed output.

## Exact 24-arm run set and horizons

All configurations are schema-v2, CPU/PyTorch/fp32 deterministic runs with D=64, L=2, H=4, tied 512-token vocabulary, `seq_len=64`, `micro_batch_size=2`, accumulation 1, AdamW, and scalar diagnostics. They have no automatic memory proposal or hidden runtime override.

The requested committed-target budgets are 24,576 (`b24k`) and 49,152 (`b48k`). The trainer never combines a short final epoch window with the next epoch in one update. Therefore the old 192/384 full-window assumptions were incorrect. The concrete config horizons are now:

| Target budget | Full 1,728-target epochs | Remaining targets | Exact max_steps | Endpoint pattern after complete epochs |
|---:|---:|---:|---:|---|
| 24,576 | 14 | 384 | **199** | 196 updates, then 3 × 128 |
| 49,152 | 28 | 768 | **398** | 392 updates, then 6 × 128 |

A run is complete only when a verified checkpoint generation has both its configured exact committed-target budget and its configured `max_steps` (199 or 398). A failed, interrupted, OOM, nonfinite, missing-endpoint, or partial-budget run is an individual failure/partial observation. Preserve it with its reason; do not evaluate it as if it had an endpoint, do not rerun an unfavorable seed, and do not silently pool it with complete runs.

For every seed `S ∈ {17,41,73}` and budget `B ∈ {24k,48k}`, run exactly these three primary configurations:

| Arm | Files | Total / active-per-token parameters |
|---|---|---:|
| Dense backbone | `context_study_dense_sS_bB.yaml` | 139,584 / 139,584 |
| Engram | `context_study_engram_sS_bB.yaml` | 174,368 / 141,728 |
| Dense total control | `context_study_dense_total_sS_bB.yaml` | 174,528 / 174,528 |

At `b48k` only, add these two diagnostics for each seed, each compared only with that seed’s base Engram b48k run:

| Diagnostic | Files | Total / active-per-token parameters |
|---|---|---:|
| Small collision table | `context_study_engram_collision127_sS_b48k.yaml` | 145,760 / 141,728 |
| Two address orders | `context_study_engram_orders23_sS_b48k.yaml` | 207,040 / 141,760 |

This is exactly 18 primary runs plus 6 b48k diagnostics: **24 runs**. There is no matrix, worker, scheduler, distributed process, or b24k diagnostic curve implied by these assets.

## Parameter and addressing accounting

The counts above come from the model’s canonical parameter inventory, not a rounded scale label. All listed parameters are trainable and frozen count is zero.

* The dense backbone comprises embedding 32,768, attention 32,768, dense FFN 73,728, and norms 320 parameters.
* Base Engram adds a 1,021 × 32 table (32,672) and a 2,112-parameter adapter. Its active direct-use count includes one 32-parameter table row plus the adapter, while the other table rows remain resident but inactive for that token.
* The FFN=283 dense-total control has FFN 108,672. It is the nearest larger total control because this backbone changes in 384-parameter FFN increments. Its 174,528 total is **160 parameters larger** than Engram, a residual of **0.0918%** of Engram total. It is not exactly parameter matched.
* The collision-127 diagnostic replaces only the table with 127 × 32 = 4,064 parameters; its adapter and active direct-use count are unchanged. It is a capacity/address-reuse contrast, not a matched-total control.
* The orders `[2,3]` diagnostic has two 1,021 × 32 tables (65,344 total table parameters) and the same 2,112-parameter adapter. It directly uses one 32-parameter row from each stream, so active count rises by 32. It is an addressing/capacity alternative, not a matched comparison.

The following are deterministic address previews from the actual 27 usable 64-token train blocks, using the implementation’s causal zero-padded polynomial address function. They are mechanism expectations for one pass over the packed stream, **not measured trained-model results** and not a sparse-speed claim.

| Address stream | Lookups | Unique buckets | Repeated-bucket lookups | Reuse rate | Utilization | Largest bucket |
|---|---:|---:|---:|---:|---:|---:|
| 3-gram, table 1,021 | 1,728 | 524 | 1,204 | 69.676% | 51.322% | 62 |
| 3-gram, table 127 | 1,728 | 127 | 1,601 | 92.650% | 100.000% | 70 |
| 2-gram, table 1,021 (orders `[2,3]` second stream) | 1,728 | 425 | 1,303 | 75.405% | 41.626% | 106 |

Actual run reports separately retain lookup count, unique buckets, the existing `collisions` metric, reuse, utilization, largest-bucket fraction, and gate mean. Here `collisions = lookups - unique buckets` includes repeated identical n-grams; it is not a count of distinct n-grams that hash together. Report a separate distinct-n-gram collision audit before making a hash-aliasing claim. Per-stream previews must not substitute for measured training diagnostics or imply a causal explanation of card outcomes.

## Endpoint, validation, cards, and reporting

For each completed run, choose exactly one checkpoint: the verified generation at the declared exact target budget. This is an endpoint rule, **not** a minimum-validation-loss selection rule. Validation loss/perplexity remains a reported diagnostic for that exact generation; it never selects a checkpoint. Do not select an earlier low-loss generation, break ties, inspect a card result, or rank by a card result.

Evaluate every valid endpoint exactly once with all three cards, in this fixed order:

1. `context_override_acquisition_v1.card.json` — acquisition/sanity only;
2. `context_static_retention_v1.card.json` — retained-static outcome;
3. `context_override_untouched_v1.card.json` — untouched-context outcome.

That is 24 × 3 = **72 planned card evaluations**. Preserve each full generated answer, exact outcome, and failure text. Retention and override remain separate 0–8 counts; acquisition is never pooled as a held-out outcome. Show individual seed values before descriptive summaries. Primary deltas are Engram minus dense backbone and dense-total minus Engram at identical seed/budget. Diagnostic deltas are the b48k variant minus same-seed base Engram only. A resource comparison requires identical seed, target budget, tokenizer/data identity, backend, precision, sequence length, microbatch, optimizer, and this endpoint rule.

For every endpoint (or for every failure record where no endpoint exists), report actual updates/targets; validation loss/perplexity when evaluated; all three card outcomes/failure text; total/trainable/frozen/active inventory; step seconds and tokens/sec; memory measurements or explicit unavailability; and the Engram occupancy fields above. Do not infer native sparse speedups from counts or collision observations.

## Commands for Main

Run from repository root after Main’s runtime/checkpoint acceptance work. These are direct single-machine commands; they do not invoke a worker, scheduler, matrix engine, model comparison selector, or distributed launcher.

```sh
# Build or verify the frozen train-only tokenizer artifact.
uv run sparselab tokenizer train configs/context_study_tokenizer.yaml

# Exactly the 24 preregistered runs. The second field is the deterministic run ID;
# the third is its required exact committed-target endpoint.
while IFS='|' read -r config run_id target; do
  uv run sparselab train "configs/$config" --run-id "$run_id"
done <<'RUNS'
context_study_dense_s17_b24k.yaml|context-study-dense-s17-b24k|24576
context_study_engram_s17_b24k.yaml|context-study-engram-s17-b24k|24576
context_study_dense_total_s17_b24k.yaml|context-study-dense-total-s17-b24k|24576
context_study_dense_s17_b48k.yaml|context-study-dense-s17-b48k|49152
context_study_engram_s17_b48k.yaml|context-study-engram-s17-b48k|49152
context_study_dense_total_s17_b48k.yaml|context-study-dense-total-s17-b48k|49152
context_study_engram_collision127_s17_b48k.yaml|context-study-engram-collision127-s17-b48k|49152
context_study_engram_orders23_s17_b48k.yaml|context-study-engram-orders23-s17-b48k|49152
context_study_dense_s41_b24k.yaml|context-study-dense-s41-b24k|24576
context_study_engram_s41_b24k.yaml|context-study-engram-s41-b24k|24576
context_study_dense_total_s41_b24k.yaml|context-study-dense-total-s41-b24k|24576
context_study_dense_s41_b48k.yaml|context-study-dense-s41-b48k|49152
context_study_engram_s41_b48k.yaml|context-study-engram-s41-b48k|49152
context_study_dense_total_s41_b48k.yaml|context-study-dense-total-s41-b48k|49152
context_study_engram_collision127_s41_b48k.yaml|context-study-engram-collision127-s41-b48k|49152
context_study_engram_orders23_s41_b48k.yaml|context-study-engram-orders23-s41-b48k|49152
context_study_dense_s73_b24k.yaml|context-study-dense-s73-b24k|24576
context_study_engram_s73_b24k.yaml|context-study-engram-s73-b24k|24576
context_study_dense_total_s73_b24k.yaml|context-study-dense-total-s73-b24k|24576
context_study_dense_s73_b48k.yaml|context-study-dense-s73-b48k|49152
context_study_engram_s73_b48k.yaml|context-study-engram-s73-b48k|49152
context_study_dense_total_s73_b48k.yaml|context-study-dense-total-s73-b48k|49152
context_study_engram_collision127_s73_b48k.yaml|context-study-engram-collision127-s73-b48k|49152
context_study_engram_orders23_s73_b48k.yaml|context-study-engram-orders23-s73-b48k|49152
RUNS
```

For each row, identify one generation under `runs/$run_id/checkpoints/` whose verified manifest reports the row’s `target` committed targets and the config’s exact `max_steps` (199 for b24k; 398 for b48k). Verify that generation against its config before any card evaluation. If it is absent, invalid, or partial, record that failure and skip its three endpoint evaluations rather than substituting a different generation.

```sh
# Set ENDPOINT only to the exact-target verified generation selected by the rule above.
uv run sparselab checkpoint verify "$ENDPOINT" --config "configs/$CONFIG"

# Repeat the following fixed three commands for every valid endpoint: 72 planned evaluations.
uv run sparselab capability evaluate "$RUN_ID" data/context_override_v2/context_override_acquisition_v1.card.json --checkpoint "$ENDPOINT" --backend cpu
uv run sparselab capability evaluate "$RUN_ID" data/context_override_v2/context_static_retention_v1.card.json --checkpoint "$ENDPOINT" --backend cpu
uv run sparselab capability evaluate "$RUN_ID" data/context_override_v2/context_override_untouched_v1.card.json --checkpoint "$ENDPOINT" --backend cpu
```

The built-in `capability compare` command is not the reporting mechanism for this packet: it cannot encode the fixed exact-target endpoint rule, failures/partials, paired seed table, or collision occupancy analysis.

## Deliberate exclusions

No byte-addressed arm appears here because it changes the data/address representation as well as memory. No learned retrieval, assistant-only objective, automatic memory policy, native sparse-kernel speed claim, KV-cache claim, distributed run, worker scheduling, or significance conclusion is expressible by these assets. Existing metrics’ token-address occupancy is a mechanism observation, not proof that semantic registry bindings occupy distinct buckets.

## Execution results — 2026-09-22

All 39 frozen ledger identities matched (hash and byte size) before execution. The 24 direct CPU/PyTorch runs were attempted once, sequentially in the preregistered order; all reached their required exact endpoint (199/24,576 or 398/49,152), and every selected generation passed `checkpoint verify` against its concrete config. Each valid endpoint then received the three cards once in the preregistered acquisition, static-retention, untouched order: 72/72 evaluations completed. The auditable command transcripts, full generated responses, exact scorers, endpoint verification, run-owned artifact records, metrics, inventories, and paired deltas are in `artifacts/studies/context_engram_2026_09_22.json`.

Score tuples below are acquisition/static-retention/untouched, each out of 8; loss and perplexity are validation diagnostics from the fixed exact-target endpoint, not selection criteria.

| Seed | Budget | Arm | Endpoint loss / perplexity | Scores |
|---:|---:|---|---:|---|
| 17 | 24k | Dense | 2.9785 / 19.6579 | 1 / 0 / 0 |
| 17 | 24k | Engram | 2.9152 / 18.4525 | 0 / 0 / 0 |
| 17 | 24k | Dense total | 2.9888 / 19.8609 | 0 / 0 / 0 |
| 17 | 48k | Dense | 4.0624 / 58.1123 | 4 / 0 / 0 |
| 17 | 48k | Engram | 4.0838 / 59.3713 | 5 / 1 / 0 |
| 17 | 48k | Dense total | 3.8465 / 46.8308 | 5 / 0 / 0 |
| 17 | 48k | Collision-127 | 3.8986 / 49.3328 | 6 / 0 / 0 |
| 17 | 48k | Orders [2,3] | 3.8429 / 46.6608 | 4 / 0 / 0 |
| 41 | 24k | Dense | 3.0715 / 21.5742 | 0 / 0 / 0 |
| 41 | 24k | Engram | 2.9752 / 19.5934 | 0 / 0 / 0 |
| 41 | 24k | Dense total | 2.8549 / 17.3719 | 0 / 0 / 0 |
| 41 | 48k | Dense | 3.8564 / 47.2960 | 1 / 0 / 0 |
| 41 | 48k | Engram | 3.9511 / 51.9922 | 4 / 0 / 0 |
| 41 | 48k | Dense total | 3.8167 / 45.4551 | 1 / 0 / 0 |
| 41 | 48k | Collision-127 | 3.9222 / 50.5111 | 3 / 0 / 0 |
| 41 | 48k | Orders [2,3] | 3.8773 / 48.2925 | 3 / 0 / 0 |
| 73 | 24k | Dense | 3.0674 / 21.4868 | 0 / 0 / 0 |
| 73 | 24k | Engram | 2.9888 / 19.8612 | 2 / 0 / 0 |
| 73 | 24k | Dense total | 2.9102 / 18.3611 | 0 / 0 / 0 |
| 73 | 48k | Dense | 3.9247 / 50.6356 | 3 / 0 / 0 |
| 73 | 48k | Engram | 3.7616 / 43.0193 | 4 / 0 / 0 |
| 73 | 48k | Dense total | 3.6197 / 37.3245 | 3 / 1 / 0 |
| 73 | 48k | Collision-127 | 4.0227 / 55.8502 | 4 / 0 / 0 |
| 73 | 48k | Orders [2,3] | 3.9980 / 54.4899 | 1 / 0 / 0 |

No endpoint acquired held-out context-local overrides: all 24 untouched scores are 0/8. Training-seen acquisition improves at the larger budget, but that does not establish override generalization or an Engram advantage. These negative outcomes are retained without selecting seeds or checkpoints.

The result artifact provides individual same-seed/same-budget Engram−dense and dense-total−Engram deltas, plus each b48k diagnostic variant−base-Engram delta. Dense total remains 160 parameters (0.0918%) larger than base Engram. Collision-127 and orders `[2,3]` alter capacity and/or addressing, so their deltas are explicitly confounded rather than matched-control claims. These three-seed observations are descriptive only; they do not support significance or general-quality claims.

The independent actual-implementation address audit covers all 27 usable train blocks with the causal zero-padded, block-reset address rule. For order-3/table-1021 it found 1,728 lookups, 524 unique buckets, 1,204 existing reuse counts, 719 distinct n-grams, 1,009 repeated-identical-n-gram lookups, and 160 buckets containing distinct-key aliases (195 extra distinct-key-to-bucket aliases). Order-3/table-127 had 127 unique buckets, 1,601 reuse counts, and 123 distinct-key collision buckets (592 aliases); order-2/table-1021 had 425 unique buckets, 1,303 reuse counts, and 110 distinct-key collision buckets (127 aliases). Thus the existing `lookups - unique buckets` metric is reported as bucket reuse, not pure hashing aliasing. Endpoint Engram diagnostics are last-forward, update-boundary scalar diagnostics; CPU records process RSS and has no native accelerator allocation/reserved/peak measurement.

Astra independently reverified all endpoint files, rescored all archived responses, checked all 18 paired deltas, and recomputed all four address streams. Excess-key aliases above are not unordered colliding-key pairs: the corresponding pair counts are 234, 2,009, and 148. The result artifact names the former `distinct_key_collision_extra_aliases`; no observed count changed. [Acceptance record](../artifacts/acceptance/scientific_studies_2026_09_22.json).
