# Known-good baselines

A known-good baseline is a reviewed, reproducible, explicitly bounded lifecycle and behavior reference. It is not a best model, evidence of general intelligence, or an automatic replacement when a loss is lower. Declarations live in the lifecycle sidecar; `sparselab research validate` separately reports what is currently verified, unavailable, or invalid.

## Dense synthetic alias reference

`dense-small-v1` remains an unpromoted candidate: **Dense synthetic alias reference lifecycle**. It uses `configs/chat_recall_dense_cpu.yaml` and `configs/tokenizer_chat_recall.yaml`, the `smoke/offline` profile, seed 42, CPU/PyTorch FP32, sequence 128, batch 2, accumulation 1, a 1,536-update ceiling, and a 393,216 target-token cap. Its profile note is: “existing 64×2 dense chat-recall configuration, FFN 192, extended fixed acquisition budget; not the generic smoke profile”. The complete RunConfig, rather than that label, defines the experiment.

It contains no MLA, MoE, sparse attention, or memory. It is deliberately a cheap offline lifecycle/acquisition reference, not useful chat or an architecture-quality claim. The single run completed at step 1536 and 391424 target tokens. Initial held-out loss was 6.176402; terminal loss was 7.056087, so the required validation gate failed. Acquisition retention passed at 24/24; held-out recall was 8/12 and context override 0/8. The report bundle `1d471241a17eca7b6d9680cfc3f595a921ab0aabe4f8b2e54a60835a6e95bb00` contains one complete run and zero comparisons. The candidate remains unpromoted; no retry was run. The smaller `capability_recall_dense_cpu.yaml` is not this baseline because its diagnostic prompts include answers.

## dense-lm-v1 replicated learning reference

The retained three-seed dense TinyStories runs completed 4,096 steps / 4,194,304 tokens each: seed 42 terminal loss 2.4824737093453306, seed 17 2.484718531778414, and seed 73 2.470433681211826. All frozen terminal gates passed; each endpoint was reached while learning, with no plateau observed. The immutable source package identity is `927dca9db1746ce2909013763980c7403b0b165ac59ef559a1f78b0eac4d6650`; Git commit and dirty-tree fields are absent from retained run manifests. OOD capability cards are descriptive and do not establish chat quality. See the [dense-lm-v1 lifecycle record](dense-lm-v1.md).

The canonical lifecycle promotes dense-lm-v1 as a **bounded learning reference**, not a quality or architecture-superiority claim. The three-seed acceptance is [`artifacts/acceptance/dense_lm_v1.json`](../../artifacts/acceptance/dense_lm_v1.json).
The final identity-bound static report is [`the content-addressed report bundle`](../../artifacts/research-reports/3f6deec6c11b70750f92219e23fb37c017922017c4730b775b245f27be82c3f0/index.html) (manifest SHA-256 `dbd7e98f5b294ef49fc43c3b766cf35bf6f6e8cf569b087f7a9d35b46a1428ad`).
`sparselab research validate --baseline dense-lm-v1` validates the candidate independently while leaving global registry errors visible. The custom lifecycle JSON remains archival, noncanonical provenance.

For future execution, use one `WORK=sparselab-work/experiments/dense-small-v1` workspace; do not overwrite retained runs or receipts. These commands illustrate the execution convention and do not change the archived failed candidate.

The tokenizer is written to `artifacts/tokenizer_chat_recall_dense_small_v1`, preserving the existing `artifacts/tokenizer_chat_recall` output. This dedicated path was checked absent before capture. `tokenizer train` reuses only an exact current training-contract match and refuses mismatched existing output; if this path has appeared with a mismatched manifest, choose a fresh output path and update the frozen config rather than deleting or replacing it.

The controller runs in another terminal/service:

```sh
WORK=sparselab-work/experiments/dense-small-v1
export SPARSELAB_WORK_DIR="$WORK"
mkdir -p "$WORK/captures"
uv sync --locked --dev
uv run --locked sparselab tokenizer train configs/tokenizer_chat_recall.yaml
uv run --locked sparselab data prepare configs/chat_recall_dense_cpu.yaml
uv run --locked sparselab inspect configs/chat_recall_dense_cpu.yaml --json
uv run --locked sparselab study plan configs/references/dense-small-v1/study.yaml
uv run --locked sparselab worker register reference-cpu --backend cpu --store "$WORK/runs"
uv run --locked sparselab study submit configs/references/dense-small-v1/study.yaml --receipt "$WORK/receipt.json" --worker reference-cpu --store "$WORK/runs"
uv run --locked sparselab controller run --store "$WORK/runs"
```

Observe `experiment list --json --store "$WORK/runs"` until the receipt's one run is COMPLETE and its artifacts are ingested. Stop only the controller service, never the worker attempt. Resolve `RUN_ID` from `receipt.json`, not a guessed UUID; downstream paths exist only after verified terminal ingestion publishes them to `$WORK/runs/RUN_ID`.

```sh
uv run --locked sparselab checkpoint verify "$WORK/runs/RUN_ID/checkpoints/latest.json" --json
uv run --locked sparselab eval RUN_ID --runs-dir "$WORK/runs" --checkpoint latest.json --backend cpu
uv run --locked sparselab study collect configs/references/dense-small-v1/study.yaml "$WORK/receipt.json" --runs-dir "$WORK/runs" --backend cpu
uv run --locked sparselab evidence RUN_ID --runs-dir "$WORK/runs" --json
uv run --locked sparselab study report configs/references/dense-small-v1/study.yaml "$WORK/receipt.json" --evidence COLLECTED_PATH --runs-dir "$WORK/runs" --output artifacts/research-reports
```

`COLLECTED_PATH` is the `output` actually returned by collect. Do not use `--research` for this non-scaffolded integration study. Bind every sample to the terminal immutable checkpoint. Obtain each prompt from `capability_card(...).cases[0]`, retain its exact shell-quoted value, command, and stdout, and run:

```sh
uv run --locked sparselab capability describe chat-alias-retention-v1
uv run --locked sparselab capability describe chat-alias-recall-v1
uv run --locked sparselab capability describe chat-context-override-v1
uv run --locked sparselab generate RUN_ID --prompt PROMPT --runs-dir "$WORK/runs" --checkpoint latest.json --temperature 0 --top-k 0 --seed 0 --max-new-tokens 8 --backend cpu
uv run --locked sparselab chat RUN_ID --message MESSAGE --transcript "$WORK/captures/acquisition-chat.json" --json --runs-dir "$WORK/runs" --checkpoint latest.json --temperature 0 --top-k 0 --seed 0 --max-new-tokens 8 --backend cpu
```

The transcript is one checkpoint-bound acquisition result. Record `git rev-parse HEAD`, `git status --porcelain`, and the run's package-source digest separately; HEAD alone does not identify a dirty tree.

## Gates and retention

Predeclare and capture: preparation/lifecycle success; one configured budget limit reached without exceeding either; verified initial and terminal checkpoints, finite held-out validation, and terminal loss below initial; acquisition at least 12/24 full-answer correct; complete held-out wording and override results with no minimum score; bound generation samples respecting prompt/token budgets; resource/backend observations; and a static report containing the single complete run and zero comparisons. Preserve zero, negative, failed, and untested outcomes.

The dated capture record is `artifacts/acceptance/dense_small_v1_2026_09_26.json`; its byte-copied RunManifest is `artifacts/acceptance/dense_small_v1_run_manifest_20260926.json`, and the captured chat is `artifacts/acceptance/dense_small_v1_2026_09_26/acquisition-chat.json`. The lifecycle registry binds these to the validated report bundle. Preserve initial and terminal checkpoints. Static report validation preserves archival evidence but does not make omitted checkpoint binaries locally runnable; unavailable local binaries must remain visible rather than demoting intact history.

## Findings, branches, and promotion

A Finding records conditions, evidence, bounded claims, reviewer attribution, disposition, rationale, next action, and reopen conditions. Dispositions are: **rejected** (unsupported under declared conditions), **inconclusive** (unanswered), **learning** (useful evidence without justified compute), **replicate** (independent repetition justified), **scale** (replicated evidence warrants larger compute), and **promote** (eligible for integrated baseline review). Stages are descriptive: mechanism, micro, replicate, scale, confirm, promote. Maturity axes are independent implementation, capability, efficiency, and portability declarations; no axis numerically ranks another.

Use `sparselab research status`, `research next`, and `research baseline describe ID` to inspect declared branches and evidence availability. `research next` is a declared next-test view, not optimal planning. A prior-design advisory never blocks planning; changing the sealed study/protocol identity can make equivalence unassessed.

Promotion is an explicit reviewed lifecycle JSON change, never a metric trigger. A known-good baseline needs an accepted promotion and complete preparation, training, checkpoint, validation, capability, generation, resources, and static-report evidence. A replacement also needs its parent’s required regressions, identity-bound passed outcomes, a resource tradeoff, and explicit `supersedes`; parallel promotion leaves its parent active. Run `sparselab research validate --json` before relying on a declaration. Keep independent mechanism branches interpretable; a combined candidate needs its own campaign and regressions.
