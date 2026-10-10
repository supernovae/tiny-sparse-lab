# Probe battery: fast-fail checks for any checkpoint

The probe battery is a small, versioned suite of cheap checks that tells you, in
seconds to minutes, whether an idea is worth a longer run. It runs on any
checkpoint, against a baseline checkpoint, and ends with a machine-readable
verdict and a recommended next action.

**Probes screen; they do not prove usefulness.** A pass means "nothing obvious
broke and loss moved the right way on a small fixed sample", nothing more.
Selecting ideas again and again on the held-out probe items slowly turns them
into development data, so keep a separate, untouched final evaluation (a fresh
data split, a longer run, more seeds) for any claim that an idea is better.

```sh
# Every lab try runs the fast tier automatically (disable with --probe-tier none).
uv run --locked --extra cpu sparselab try wider-ffn.yaml --vs BASELINE.yaml

# Any run or checkpoint, any tier.
uv run --locked --extra cpu sparselab probe RUN_OR_CHECKPOINT --vs BASELINE_RUN --tier standard
uv run --locked --extra cpu sparselab probe RUN --vs BASE --json      # stable schema
uv run --locked --extra cpu sparselab report probe-20261010T150839Z-ece5a739

# Standard lm-eval tasks (optional extra).
uv sync --locked --extra cpu --extra lmeval
uv run --locked --extra cpu --extra lmeval sparselab probe RUN --vs BASE --tier full
```

A target is a run directory, a checkpoint directory inside a run, or a run id
under `WORK_DIR/lab/runs` (or `--runs-dir`). Results land in
`WORK_DIR/lab/probes/<probe-id>/probe.json`, sealed with `record_sha256` by
the same lab-record writer as `try.json`. `report`, `probe` and the dashboard
read every record through one verified reader (`sparselab.lab_records`), so an
edited record is rejected everywhere. A try stores its result in `try.json`
under `probe`.

```text
PROBE BATTERY  sparselab-probe-battery v2 · suite ec0544e7 · tier full (ran fast → standard → full) · 24.5s
  candidate lab-try-20261010T154716Z-f24f162a  step 12 · 1.5k tokens · 43.2k params
  baseline  lab-base-7e2b3094a1c9b4b2-try-20261010T154716Z-f24f162a  step 12 · 1.5k tokens · 43.2k params

  ✔ PASS        Held-out loss             4.083 vs 5.359    ◀◀◀◀◀│·····  Δ -1.276 (-23.8%) ±0.006   ppl 59.3
  ✔ PASS        Calibration (ECE)         0.111 vs 0.143    ··◀◀◀│·····  Δ -0.032
  ✔ PASS        Top-1 agreement           0.042                                                     JS 0.009 · KL 0.035 (near-identical: agreement uninformative)
  ✔ PASS        Degeneration              0.841 vs 0.966    ··◀◀◀│·····  Δ -0.125                   distinct-2 0.02
  ✔ PASS        Fact recall (reworded)    0.250 vs 0.250    ·····│·····  Δ +0.000 ±0.183            chance 0.25
  ✔ PASS        Needle in context         0.167 vs 0.167    ·····│·····  Δ +0.000 ±0.000            by length ▂▂▂ (27/40/55 tok)
  ✔ PASS        Standard tasks (lm-eval)  0.225 vs 0.225    ·····│·····  Δ +0.000                   lambada 0.00 hellaswag 0.20 arc 0.14 piqa 0.56

  verdict ✔ PASS  next → LONGER RUN
  Every tier holds up: schedule a longer run (more tokens or seeds) to confirm the gain. Probes screen; they do not prove usefulness.
  guard   verdicts use the held-out split · ok · fact_recall dev +0.00 / held-out +0.00; needle dev +0.17 / held-out +0.00
  legend  meter ◀ better │ worse ▶ (full = fail threshold)
```

The meter is centred on the baseline; a full half-bar is the probe's fail
threshold. Colors follow `NO_COLOR`/`FORCE_COLOR` and are off when piped.

## Probes

Probes run cheapest first: by tier, then declared cost.

| Probe | Tier · cost | Metric | Judged on (warn / fail) | Hard | A failure suggests |
|---|---|---|---|---|---|
| Held-out loss | fast · 1 | mean token cross-entropy on the validation split, same eval protocol as the baseline | relative increase 0.5% / 3% | yes | the change hurts at this budget; revert or retune LR/warmup |
| Calibration (ECE) | fast · 1 | top-1 expected calibration error, 15 equal-width bins (Guo et al. 2017) | +0.02 / +0.05 | no | confidence no longer tracks accuracy (schedule end, output scale) |
| Top-1 agreement | fast · 1 | argmax agreement with the baseline on fixed held-out prompts, plus KL(base‖cand) and JS in nats | below 0.5 (warn only) | no | a large behavior shift; informative only together with loss |
| Degeneration | fast · 2 | seq-rep-4 of greedy continuations (Welleck et al. 2019), distinct-1/2 (Li et al. 2016) | +0.05 / +0.20 | yes | loops: duplicated data, too-high LR, positional change |
| Fact recall (reworded) | standard · 2 | ranks the stated answer among candidates from the same relation (mean token log-prob), asked in different words than the fact was stated; fractional tie credit | −0.05 / −0.15 | no | context handling or memory wiring |
| Needle in context | standard · 3 | retrieve a code word stated at the start of a filler context at ~50/75/95% of `max_seq_len`; fractional tie credit | −0.05 / −0.15 | no | attention span, positions or sequence-length changes |
| Standard tasks (lm-eval) | full · 4 | mean `acc` over lambada_openai, hellaswag, arc_easy, piqa at `limit=50` | −0.02 / −0.05 | no | treat small moves as noise at tiny scale |

**Credit.** Recall and needle rank candidates by mean token log-prob of
`" " + candidate` after the prompt. When k candidates tie at the top score the
item earns 1/k if the answer is among them, else 0, so a model that scores
everything alike lands at chance, never at 100% (ties used to break toward the
answer; a uniform-score regression test pins this).

**Noise.** Each probe declares its `uncertainty`. Held-out loss is a ratio
(sum of token NLL / tokens) over evaluation windows, so its delta SE is the
token-weighted, window-clustered delta-method SE of the difference
(`window_clustered_ratio`). Recall and needle compare the same items on both
arms, so their delta SE is the paired SE of the per-item credit differences
(`paired_items`). A change within 2 SE counts as `pass` with a "within noise"
note, never as a win or a regression.

**Tiny-model honesty.** Fact recall is *in-context* (the fact is stated, then
asked in other words), because a tiny model knows no facts; parametric recall
lives in the withheld-facts study. When both models are near-uniform (JS < 0.01)
top-1 agreement is meaningless and is reported as uninformative instead of
warning. lm-eval tasks are borrowed for their established metric definitions,
but tiny models sit at or near chance (≈0.25 hellaswag/arc_easy, 0.5 piqa, ≈0
lambada); at `limit=50` a difference of a few points is noise. The full tier
records each task, the chance level and this caveat in the result.

## Tiers and fast-fail

- `fast` (default for `try`): loss, calibration, agreement, degeneration; well
  under a second on the CPU smoke model.
- `standard`: adds reworded fact recall and needle retrieval.
- `full`: adds lm-eval. The `lmeval` extra stays optional: without it the probe
  is `unavailable` with the install hint and the battery is **incomplete**
  (see below), never a pass.

The lm-eval adapter scores through the same token-scoring path as the probes
(`sparselab.probes.scoring`). It follows the harness boundary rules (trailing
context whitespace moves to the continuation; context and continuation are
encoded together and split at the context's token count; an empty context is
the end-of-text token) and scores every continuation token, windowing with
maximal left context when a continuation is longer than `max_seq_len`. A task
that produces no accuracy is an error, i.e. missing evidence.

A **hard** probe that fails (held-out loss, degeneration) stops the battery at
once; the remaining probes are `skipped`. A NaN/inf loss raised by the native
evaluator is a **numerical failure**, a distinct outcome rather than a probe
error: the held-out loss row fails with `details.numerical_failure`, the battery
stops (`fast-fail: numerical failure in heldout_loss`) and the verdict is
`fail`/`abandon` with a numerical-failure reason and suggestion, even without a
baseline. If the *baseline* is the one that fails numerically, nothing can be
judged against it and the row is `error` (missing evidence). Any other failure finishes the current
tier but the battery does not escalate ("not promising"). `--no-fast-fail` runs
everything. A probe that crashes is recorded as `error`; it never sinks the try,
but it is missing evidence: the battery does not escalate past it and the
verdict is `incomplete`.

## Missing evidence, cancellation and resources

A check that should have produced evidence and did not is never a success.
`unavailable` (optional dependency missing, unsupported engine), `error` (the
probe raised) and `not_comparable` rows, and probes left unrun by a stop, are
listed in `verdict.missing` with the reason, printed as `missing ...` lines and
shown in the dashboard banner. Unless a real failure already decides the
verdict, the status is `incomplete` and the action is `rerun`.

Arms are loaded one at a time: per tier the baseline is loaded, measured and
released, then the candidate is loaded, measured and judged probe by probe, so a
battery never holds two models in memory. Inside `try` the battery runs under
the try's own context (the same `CANCEL` sentinel, `--resource-envelope` and
SIGINT/SIGTERM handling as training and scoring); standalone `probe` creates
`LAB/probes/<id>/CANCEL` as its sentinel and accepts `--resource-envelope` too.
The context is checked before each arm and before every probe. `stop.kind`
records why a battery stopped: `fast_fail` (hard failure, including a
numerical failure), `not_promising`
(a failure in an earlier tier), `missing_evidence` (an earlier tier is
incomplete), `cancelled` (sentinel), `resources` (envelope violated) or `oom`
(host `MemoryError` or a torch out-of-memory error, which stops further probe
work instead of becoming an error row) or `interrupted` (Ctrl-C or SIGTERM).
Everything measured before the stop is kept: on a signal the battery is
finalized (unrun probes listed as missing evidence, verdict `incomplete`), the
progress file is marked `stopped` and the sealed record is published before the
command exits (130 for standalone `probe`; `try` records `interrupted` at
`phase: probing`). Inside `try`, a stopped battery never undoes the training comparison: a
cancel during probing marks the try `interrupted` and keeps both scored arms and
the comparison; resources or OOM leave the try `completed` with an incomplete
battery.

Validation is scored once per arm. Inside `try` the scoring pass doubles as the
probe's validation pass (an observer on the native evaluator collects per-window
sums and top-1 confidences); standalone, held-out loss and calibration share one
pass that is computed only on a cache miss.

## Verdict and next action

`verdict` is `{status, action, next_tier, reasons, missing, suggestion}`;
actions are:

| Condition (first match) | status | action |
|---|---|---|
| numerical failure (NaN/inf held-out loss) | fail | `abandon` |
| no baseline | info | `compare` |
| a hard probe failed | fail | `abandon` |
| any probe failed | fail | `tweak` (that probe's hint) |
| missing evidence (unavailable, error, not comparable, stopped by cancel/resources/OOM) | incomplete | `rerun` (names each gap) |
| overfit guard fired | warn | `tweak` |
| held-out loss improved beyond noise, tiers left | pass/warn | `escalate` with `next_tier` |
| held-out loss improved beyond noise, all tiers run | pass/warn | `longer_run` |
| otherwise (no measurable gain) | pass/warn | `tweak` |

## Held-out guard

Every item set has a **dev** split and a **held-out** split (different facts,
templates, needles, filler and prompts). Verdicts only use the held-out split.
Agents may look at dev results while iterating; the guard flags
`overfit_suspected` when the dev gain is beyond its paired noise and exceeds
the held-out gain by more than `max(0.15, 2 SE)`. Never edit probe items to make
an idea pass: changing any item or threshold changes the suite digest, and a
digest change requires bumping `SUITE_VERSION` (a pinned test enforces this;
v2 introduced fractional tie credit and the declared uncertainty methods).
Compare results only within one suite digest; the dashboard warns when history
mixes suites.

## Result schema (`sparselab-probe-v1`)

Top level: `format`, `created_at`, `suite` (`name`, `version`, `sha256`, split
digests, `verdict_split`), `tier`, `tiers_run`, `stop` (`stopped`, `kind`, `at`,
`reason`), `target`/`baseline` (run id, checkpoint and its sha256, step, tokens
seen, parameters, parameter bytes, tokenizer and validation digests,
`max_seq_len`, `eval_group`), `comparable`, `protocol`, `probes`, `guard`,
`verdict`, `seconds`; standalone records add `probe_id` and `record_sha256`.
`eval_group` hashes the validation data, tokenizer, loss mask and eval
protocol: losses are only comparable within one group.

Each probe row: `id`, `title`, `tier`, `cost`, `metric`, `higher_is_better`,
`hard`, `thresholds` (`mode`, `warn`, `fail`), `uncertainty`, `suggests`,
`explains`, `reference`, `status`
(`pass|warn|fail|info|skipped|unavailable|not_comparable|error`),
`value`, `baseline_value`, `delta`, `regression` (positive = worse; relative for
loss), `delta_se`, `within_noise`, `improved`, `note`, `details`, `seconds`.
Keys are stable within a format version; `tests/test_probes.py` pins them.

## Dashboard

`sparselab dashboard` has a **Probes** page (reads `WORK_DIR/lab`, or
`--lab-dir`): live progress of a running battery, the verdict banner (with any
missing evidence), per-probe results against the baseline with meters, charts
and greedy samples, a "What the probes mean" explainer, history across tries and
probes, and a Pareto view of held-out loss against parameters, weight bytes,
training tokens or latency. The page reads records through the same verified
reader as `report` and lists rejected (edited or unreadable) records instead of
showing them. The Pareto view plots one point per (comparison group,
checkpoint) for candidates and baselines, newest result wins, so the same
checkpoint scored under two protocols keeps a point in each group. It plots
only within one comparison group
(`eval_group`), chosen with a selector that defaults to the selected result's
group.

![Probe dashboard](assets/probe-dashboard.png)
