# dense-lm-v1 lifecycle contract

`configs/dense_lm_v1_lifecycle.json` binds the candidate inputs, data arrays, training budget, milestones, seeds, and promotion rule. Seeds 42, 17, and 73 each completed step 4,096 (4,194,304 targets) and passed the five frozen machine-required terminal groups without changing training configuration or gates. All three terminal Learning Observations report `promotion_eligible=true`. The candidate remains a replication result, not a formal known-good baseline: the formal lifecycle still lacks the accepted integration record and verified static report bundle required for promotion. Repetition remains descriptive; no threshold was preregistered.

## Frozen experiment

- Dense tied-embedding Transformer, no external memory; 29,893,120 parameters.
- TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; dedicated 8,192-token BPE tokenizer; 20,000 train documents and 310 selected validation documents.
- Prepared packed arrays, supervision masks, tokenizer, prompt panel, source tree, and three seed configs are SHA-256-bound in the lifecycle contract.
- ROCm / PyTorch, FP32. Fresh source-identity-bound FP32 and BF16 runtime probes passed; FP32 remains the frozen reference precision. Training uses microbatch 8, accumulation 1, sequence length 128, effective 1,024 target tokens/update. The lifecycle declaration now matches the hashed seed-42 YAML; its config SHA remains unchanged.
- Pilot selection: 25M (29,893,120 parameters), no activation checkpointing or offload, peak reserved VRAM 1,134,559,232 bytes, versus 50M (50,274,752 parameters) at 1,530,920,960 bytes. Both completed 32 warmed updates with 32,768 targets, ROCm validation, full-state checkpoint verification, and finite reload. The last 24 synchronized update timings averaged 0.07216s (14,191.57 targets/s, population SD 0.00447s) for 25M and 0.11220s (9,126.81 targets/s, SD 0.00645s) for 50M. Both used the frozen batch/sequence/optimizer settings.
- Both model sizes fit comfortably: the GPU reports 25,708,240,896 bytes total VRAM; measured peak reserved memory used about 4.4% (25M) and 6.0% (50M). 50M is 35.7% slower with no memory or rerun-affordability benefit. The predefined smaller-model default is selected.
The original 5.9–6.7-minute estimate targeted training-command wall only. Seed 42 took 584.232s (9.74min); seed 17 took 586.03s (9.77min); seed 73 took 586.30s (9.77min). Progress telemetry separately reported 517.015s, 519.493s, and 521.199s cumulative wall; optimizer updates totaled 332.123s, 335.788s, and 335.712s. Each six-run lineage recorded 30 validation and 30 checkpoint events, versus 22 forecast validation passes. The new runs averaged 12,552.9 and 12,553.6 targets/s, below seed 42's 12,686.2 and the 14,191.6-target/s short pilot. Per-update throughput ranged 3,349.9–14,313.3 targets/s across the three lineages. These scopes are not additive. The warm-cache stages took 36–37s; seed 73's seven milestone exercises totaled 120.32s. Seed-17 exercise wall time for steps 0 and 200 was not retained; its five later exercises totaled 78.98s. Verification/reporting time is not included. The estimate missed because sustained throughput was below pilot and the six-run lineage incurred resume, validation, and checkpoint overhead beyond the optimizer-only estimate.

The three validation trajectories decreased monotonically over all seven milestones, with 65,408 held-out targets (64 batches) per evaluation:

| Seed | Loss at steps 0 / 200 / 400 / 1,024 / 2,048 / 3,072 / 4,096 | Loss decrease | Terminal run | Terminal checkpoint SHA-256 | Learning Observation identity |
|---:|---|---:|---|---|---|
| 42 | 9.113787 / 4.195254 / 3.653844 / 3.099474 / 2.751929 / 2.568681 / 2.482474 | 6.631313 | `dense-lm-v1-seed42-step4096` | `b6e77e30cc78731a74dd713471106c4b10afca2901da501b95d30e45c1b1b524` | `3d85e447e29fb59f7a328bd5f6c41172239febc752b9251d4f741fc51817910f` |
| 17 | 9.109390 / 4.266431 / 3.702325 / 3.107223 / 2.752145 / 2.562564 / 2.484719 | 6.624671 | `dense-lm-v1-seed17-step4096` | `2f50048fb000ad9df616fa2cd82c0337da328d057b2fd867dd893e9ed681146c` | `de183c2337018ab74908226f675219515e96d90a5f9363db5698343a10405ade` |
| 73 | 9.105279 / 4.184871 / 3.620907 / 3.095393 / 2.741962 / 2.554597 / 2.470434 | 6.634845 | `dense-lm-v1-seed73-step4096` | `55bfbca54566d39fac7020d7cc6cdb46159ec207887740345fa25693cf63c26b` | `76fd31e15ec4a5c3776301a082db8b35360992011c647259ce5f94dd2619b46d` |

The launch-through-step-4,096 shells each completed; intermediate `--stop-after-step` run records are marked `interrupted` because each was deliberately stopped at its milestone, and each final child is `completed`. All three terminal checkpoints passed full-state verification. No tokenizer or packed-data rebuild occurred. The held-out improvements are descriptive seed replications, not a significance test or general language-quality claim.

Across each 4,096-update lineage, optimizer mean update / mean per-update throughput / min–max per-update throughput were: seed 42, 0.081085s / 12,686.2 / 3,638.3–14,305.6 targets/s; seed 17, 0.081979s / 12,552.9 / 3,424.1–14,313.3; seed 73, 0.081961s / 12,553.6 / 3,349.9–14,284.6. All three peaked at 1,064,241,664 allocated and 1,134,559,232 reserved bytes, with zero optimizer overflow retries. The broad per-update ranges are reported as observed; no significance test or new acceptance threshold was applied.

The terminal panel remains variable and sometimes repetitive across seeds. Seed 42's maximum repeated-token/bigram/trigram excess was 18/13/11; seed 17's was 16/9/7; seed 73's was 11/7/6. Seed 73's step-0 and step-200 panel exercises failed the mechanical fixed-panel/determinism/degeneration groups due to empty or repetitive outputs; step 400 and later passed. These are intermediate observations, not the frozen terminal gate. No validation-loss stopping condition fired, and the gate was not weakened.

The preparation-stage snapshots and input-manifest hashes are recorded in the lifecycle sidecar. Seed-17 stage/input hashes are `2dbefa952b73c2feb252427ca21883062b87c47d9879d5cb4265f2295b3f728d` / `7799debddabee80b19e0ae3ac8db1699cc306a4badb3a98f49e32896c44a7383`; seed 73: `dc39e3a85f003b0af4f43b9e75c1f367220e0f2c9afb307c296b8059ce354cad` / `91213187263c3723da29c9e9174918c0922c724ccab729e7cb8c0340d1099897`.

Device identification is confirmed independently by PyTorch (`AMD Radeon RX 7900 XTX`, `gfx1100`, 24 GiB) and by fresh explicit-backend stage reports (`rocm`, PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333). Pilot device peak allocations/reservations were measured directly; the generic static memory estimate remains `UNKNOWN` because its safe inspection API does not expose the device capacity.

## Three-seed terminal panel and bounded conclusion

All six prompts were nonempty and repeatable at step 4,096 for all three seeds; cache/full-prefix parity held, no special tokens appeared, and the frozen diagnostics group passed. Exact completions and per-case repetition counts remain in the terminal Learning Observations linked above. These are fixed-prompt descriptive examples, not a broad capability evaluation. One mid-training seed-73 exercise failure recovered by step 400; it is retained in its observation record.

The terminal gate summary is seed 42: integrity / held-out LM / deterministic generation / fixed prompt panel / degeneration diagnostics = PASS; seed 17: PASS / PASS / PASS / PASS / PASS; seed 73: PASS / PASS / PASS / PASS / PASS. Capability cards are NOT_APPLICABLE because they are out of domain; exercise-level resource capture is UNAVAILABLE, while training resource telemetry is present. All three records report `promotion_eligible=true` under the frozen machine contract. The repeated phrase patterns remain non-gating limitations.

The status is **replicated frozen-gate pass; formal baseline promotion pending**. The separate project baseline policy requires a reviewed accepted promotion, identity-bound integration acceptance, complete evidence roles, and a validated static report bundle. The project registry currently has no `dense-lm-v1` promotion/report entry, so this replication does not by itself satisfy or fabricate those integration records. `sparselab research status --json` listed only the existing `dense-small-v1` baseline; the custom `sparselab_dense_lm_v1_lifecycle_contract` sidecar is not the formal `sparselab-research-lifecycle` schema and `research status --lifecycle` rejects it. `research validate --json` remains invalid on the unrelated historical `maturity_measured_evidence_missing` error for `learned-engram-portability-v1`, with additional unavailable archived-evidence warnings; no unrelated registry records were changed. Dashboard visual verification remains outstanding because the managed Chromium broker did not launch in this environment.

## Seed-42 outcome

The run family is `dense-lm-v1-seed42`, then native full-state resume children `...-step400`, `...-step1024`, `...-step2048`, `...-step3072`, and `...-step4096`. The terminal run status is `completed`; its step-4,096 checkpoint verifies as a full-resume checkpoint with finite weights and the frozen source, tokenizer, and data identities. No tokenizer or data rebuild occurred.

| Step | Tokens seen | Held-out loss | Held-out targets | Prompt panel |
|---:|---:|---:|---:|---|
| 0 | 0 | 9.113787 | 65,408 | Six nonempty; severe uniform repetition |
| 200 | 204,800 | 4.195254 | 65,408 | Six nonempty; repetitions remain |
| 400 | 409,600 | 3.653844 | 65,408 | One of six outputs empty; fixed-panel/determinism/degeneration groups failed |
| 1,024 | 1,048,576 | 3.099474 | 65,408 | Six nonempty |
| 2,048 | 2,097,152 | 2.751929 | 65,408 | Six nonempty; max repeated token/bigram/trigram excess 10/4/1 |
| 3,072 | 3,145,728 | 2.568681 | 65,408 | Six nonempty; max excess 19/13/11 |
| 4,096 | 4,194,304 | 2.482474 | 65,408 | Six nonempty; max excess 18/13/11 |

At step 2,048, checkpoint `7f9441ec8234185bf8df472b6c3ff9995efdd0fbde37c246dba693558b2e6eaa` verified as full-state. Resumed child `...-step3072` restored equal step/token coordinates, cursor, schedule, scaler, optimizer and parameter metadata, RNG streams, config/source identities; resumed initial weights were byte-equal. The parent checkpoint's training-state (`952a000b736146cb0b2584253e11351273608274969225cbd5810e762284f4f8`) and weights (`8230138ce27f3eb6a6265aacb2a7a2ba8d8f6881aeed1973a9ba1a18c99575d8`) hashes remained unchanged.

Terminal integrity, held-out LM, deterministic generation, and fixed-panel groups passed. All six outputs were nonempty and repeatable, cache/full-prefix outputs matched, and no special tokens appeared. The machine degeneration-diagnostics group also passed under its frozen implementation, which requires nonempty panel completions and reports repetition metrics without thresholds. `color-object-continuity` repeats the “bird was happy to have a new friend” sentence pattern multiple times (14 distinct of 32 tokens; repeated-token/bigram/trigram excess 18/13/11); “Thank you, Mia!” also repeats. These remain descriptive generation-quality limitations, not retrospective gate failures. The CLI's `promotion_eligible=true` matches the frozen machine-readable gates. Capability cards are out-of-domain and not applicable; no general quality or capability claim is made.

Peak allocated/reserved VRAM across all seeds was 1,064,241,664 / 1,134,559,232 bytes; no optimizer overflow retries occurred. Exercise allocator/optimizer captures were unavailable, while these training metrics came from the run databases. The completed three-seed reference is the decision boundary for separately deferred preprocessing options A–E; those investigations are not part of this result.

## Preregistration audit

The immutable preparation stage records source identity `927dca9db1746ce2909013763980c7403b0b165ac59ef559a1f78b0eac4d6650` (stage snapshot SHA-256 `c52e0b56ea13729503497492239a0520ba78819580a429ff0f68329e8b6462cb`). The frozen exercise implementation is in that source identity; its file SHA-256 is `3a2b3ed4e1b83753b5b6d337e2111904a6b42d74d56b6cf7570c7da9d0251f37`. The candidate training YAML was frozen at `354a2c5d55d74ccf6c0b27c50bfe8c6d98ff0cfc53bfcc4ec2c26e64c564643f`; the six-prompt panel at `118839f291b8741d233ce19baff15f0dcb9cef52eb907208ff6ac9a81bc3408e`.

The exact pre-run raw hashes for `configs/dense_lm_v1_lifecycle.json` and this research document are unavailable: neither path is git-tracked, and neither file was included in the immutable stage bundle. The machine behavior is independently frozen by the stage source identity and code. `reference_exercise.py` requires the five named machine groups; its `degeneration_diagnostics` status is `PASS` iff all prompt completions are nonempty. It records repeated-token, bigram, and trigram excess, but defines no thresholds. `promotion_gate` requires each machine group to be `PASS`; the existing exercise regression confirms an eligible result for nonempty deterministic outputs. Thus the built-in flag matches the frozen machine contract.

The approved plan did include “no catastrophic degeneration across all panel cases,” but did not define “catastrophic,” prescribe manual review, set repeated-token/bigram/trigram thresholds, or make `color-object-continuity` individually gating. The human sentence-pattern judgment occurred after observing the output. It cannot retroactively reject seed 42. The quality observation remains preserved in the terminal Learning Observation and the candidate Finding.

| Dimension | Seed-42 result |
|---|---|
| Lifecycle / provenance / full-state resume | PASS |
| Held-out LM learning | PASS; loss fell monotonically at all seven measured milestones |
| Mechanical generation and fixed-panel checks | PASS |
| Generation-quality observation | Descriptive, non-gating phrase repetition on `color-object-continuity` |
| Task capability | NOT APPLICABLE; available cards are out of domain |

Future research question: **At this model scale and training budget, how stable is open-ended generation degeneration across seeds and training tokens?** Proposed separate study: `dense-lm-generation-degeneration-v1`, with token budget, model scale, sampling strategy, and seed as potential axes. Do not alter this candidate’s training budget, model size, learning rate, or sampling settings to address the observation.

## Preprocessing incident and boundary

The user reported approximately 25 minutes for tokenizer build and 24 minutes for data prep. Both jobs were originally started concurrently; later their Python children were blocked in `futex_do_wait`, with no observed I/O/network. Only the identified tokenizer/data job processes were stopped. Their emitted artifacts were retained and digest-verified. `/home` (repository, HF cache, and `artifacts/data`) is on `/dev/sdd` ext4 under WSL2, not `/mnt/*`.

The repaired path checks the complete tokenizer manifest and raw tokenizer digest against configured source/revision/vocabulary before initializing the data stream. Preparation now emits flushed JSON-line start/progress/finish events to stderr and split counts/timings. The final serialized, warm-cache preparation took 16.40s; its source identity is `927dca9db1746ce2909013763980c7403b0b165ac59ef559a1f78b0eac4d6650`. Train and validation source-content digests and prepared-array/supervision digests match earlier packed arrays; details are in [the preprocessing report](dense-lm-v1-preprocessing.md). This warm-cache measurement is not the first-download performance baseline.

Do not optimize preprocessing as part of this completed reference. All three seeds passed their frozen terminal gates; options A–E remain a separate, deferred profiling and change-evaluation task. Any future candidate requires explicit identity-equivalence evidence for tokenizer, selected source, packed data, supervision, byte addresses, and document statistics. No ordinary-CI wall-time assertions.

## Execution procedure

Run from the repository root on the validated ROCm environment. The YAMLs resolve tokenizer/data/run paths relative to their own location. Use a fresh relative capture directory; do not reuse a nonempty path. The prepared tokenizer and dataset are already identity-bound and warm-cached. `tokenizer train` / `data prepare` must resolve to the frozen hashes; stop if an identity differs. This sequence does not change any training setting.

```sh
CAPTURE="sparselab-work/captures/dense-lm-v1-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$CAPTURE" || { echo "capture path already exists: $CAPTURE" >&2; exit 1; }
mkdir -p "$CAPTURE"
RUN_TAG="${CAPTURE##*-}"
uv run --locked sparselab tokenizer train configs/tokenizer_dense_lm_v1.yaml

for SEED in 42 17 73; do
  case "$SEED" in
    42) CONFIG=configs/dense_lm_v1.yaml ;;
    17) CONFIG=configs/dense_lm_v1_seed17.yaml ;;
    73) CONFIG=configs/dense_lm_v1_seed73.yaml ;;
  esac
  RUN_ROOT="runs-dense-lm-v1-seed${SEED}"
  RUN_ID="dense-lm-v1-seed${SEED}-${RUN_TAG}"
  uv run --locked sparselab data prepare "$CONFIG"
  uv run --locked sparselab inspect "$CONFIG" --json
  uv run --locked sparselab stage "$CONFIG" --through warmup --output "$CAPTURE/seed${SEED}-stage"
  uv run --locked sparselab train "$CONFIG" --backend rocm \
    --run-id "$RUN_ID" \
    --stage-bundle "$CAPTURE/seed${SEED}-stage" --stop-after-step 200
done
```

Continue each seed in its configured run root. Resume only from a verified immutable checkpoint manifest; never use `latest.json`. Each milestone creates a new child run with `--run-id "$RUN_ID-step${TARGET}"`, `--resume <previous-run>/checkpoints/<step-generation>/manifest.json`, and `--stop-after-step "$TARGET"` for targets 400, 1,024, 2,048, 3,072, and 4,096. Use the checkpoint generation actually emitted by the previous run and keep every parent artifact. Review every 64-batch held-out evaluation before continuing; stop at the next durable boundary if two successive milestones worsen and the latter remains worse than initialization. Verify each selected generation with `uv run --locked sparselab checkpoint verify <manifest.json> --config "$CONFIG" --json`. Seed 42's separately retained resume proof is at step 2,048; the replication runs resumed across the same native full-state milestone boundaries.

For each seed, set `RUN_ROOT` and `RUN_ID` to its configured run root and run; set `STEP_GENERATION` and `STEP` to the exact checkpoint directory and output step. Exercise the exact generation using the prompt panel, saving under that run's relative `learning_observations/step-NNNNNNNN` directory:

```sh
uv run --locked sparselab model exercise "$RUN_ID" \
  --checkpoint "$RUN_ROOT/$RUN_ID/checkpoints/$STEP_GENERATION" \
  --runs-dir "$RUN_ROOT" --backend rocm \
  --prompt-panel data/dense_lm_v1_prompts.json \
  --output "$RUN_ROOT/$RUN_ID/learning_observations/$STEP"
```

Retain raw output, terminal Learning Observation identity, per-prompt repetition counts, held-out loss/target counts, verified checkpoint manifest, and run database metrics. Apply only the five `required_gate_groups` in `configs/dense_lm_v1_lifecycle.json`; do not set repetition thresholds. Report the three terminal records and any intermediate failures without overwriting earlier observations. Use the formal registry commands below; acceptance and a verified static report bundle remain separate promotion prerequisites.

```sh
uv run --locked sparselab research validate --json
uv run --locked sparselab research status --json
uv run --locked sparselab research baseline list
```

A pre-existing 20-step ROCm smoke checkpoint was exercised read-only through the new CLI on CPU, with output directed to `/tmp`. Integrity and held-out evaluation were captured, and short-context failures were reported as explicit generation failures; it was not used as candidate evidence. No run in `runs-rocm/` was modified.

## Gates and reporting

The frozen machine groups require finite model state and immutable identities, exact step/token coordinate, held-out LM evaluation, fixed-panel mechanical validity, repeatable greedy generation/cache parity where available, and degeneration diagnostics. The degeneration implementation computes descriptive repetition measures but gates only nonempty completions. No manual review procedure, numeric repetition threshold, or prompt-specific quality failure was frozen; “no catastrophic degeneration” was an unquantified plan phrase. Record unavailable hardware/optimizer/timing values as unavailable, not zero.

`promotion_eligible=true` on each terminal observation reflects only the frozen machine groups. All three seeds meet that gate, but no formal promotion was made: the project baseline policy additionally requires a reviewed accepted promotion, identity-bound integration acceptance, complete evidence roles, and a validated static report bundle. None is declared for `dense-lm-v1` in the formal registry. Keep it as a completed replicated candidate until those records are truthfully assembled and reviewed; do not alter the existing baseline or the five frozen groups. Any new quality threshold needs a separately versioned preregistration.

The Research dashboard renders content-addressed learning observations from its `--reports-dir`; browser visual verification was attempted but the managed Chromium broker did not launch in this environment. Tests and the CLI checkpoint smoke passed; a visual Research-view check remains outstanding.
