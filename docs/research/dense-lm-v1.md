# dense-lm-v1 lifecycle contract

The retained runs completed the frozen 4,096-step / 4,194,304-token budget on seeds 42, 17, and 73, with terminal losses 2.4824737093453306, 2.484718531778414, and 2.470433681211826, respectively. All fixed terminal gates passed. The canonical lifecycle now promotes this bounded learning reference; that status does not establish chat quality or architecture superiority, and OOD capability cards remain descriptive only. Each endpoint was reached while learning, with no plateau observed. The immutable source package identity is `927dca9db1746ce2909013763980c7403b0b165ac59ef559a1f78b0eac4d6650`. Git commit and dirty-tree fields are absent from retained run manifests.

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

The canonical record is **promoted as a bounded replicated learning reference**. Its three seed identities and eight evidence roles are retained in [`artifacts/acceptance/dense_lm_v1.json`](../../artifacts/acceptance/dense_lm_v1.json).
The final identity-bound static report is [`artifacts/research-reports/3f6deec6c11b70750f92219e23fb37c017922017c4730b775b245f27be82c3f0/index.html`](../../artifacts/research-reports/3f6deec6c11b70750f92219e23fb37c017922017c4730b775b245f27be82c3f0/index.html) (manifest SHA-256 `dbd7e98f5b294ef49fc43c3b766cf35bf6f6e8cf569b087f7a9d35b46a1428ad`).
`sparselab research validate --baseline dense-lm-v1` reports candidate validity independently of global registry errors. The unrelated Engram maturity evidence error and unavailable archived artifacts remain visible in global validation; neither is hidden or treated as a dense-lm-v1 failure. The custom `sparselab_dense_lm_v1_lifecycle_contract` sidecar remains archival provenance, not the canonical `sparselab-research-lifecycle` record.

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

## Historical execution and future workspace convention

The completed reference was executed manually from three frozen seed YAMLs. Their
explicit `logging.root_dir` values selected `runs-dense-lm-v1-seed42`,
`runs-dense-lm-v1-seed17`, and `runs-dense-lm-v1-seed73`; the reference matrix
repeated those path overrides. The old instructions mirrored them with a
seed-specific `RUN_ROOT` and invoked direct `train` without an execution-path
override. This was a manual/configuration convention, not a requirement of the
controller, device leases, study machinery, or resume semantics. Do not repeat it.
The frozen YAMLs, matrix, and archived evidence preserve those historical values.

The historical milestone procedure stopped at 200, 400, 1,024, 2,048, 3,072,
and 4,096 updates, creating children from verified immutable generation manifests.
Every parent was retained, and each 64-batch held-out evaluation was reviewed
before continuing. The stopping rule was two successive worsening milestones,
with the latter still worse than initialization. Seed 42's separately retained
resume proof is at step 2,048. These are descriptions of completed execution;
changing workspace organization does not rerun or revise that protocol.

For a future authorized replicated study, all coordinates belong to one workspace.
The following illustrates normal study submission and collection, not a command
to rerun this promoted reference or reproduce its manual milestone controls. On a
vendor-provisioned ROCm checkout, always use `--no-sync` to avoid replacing PyTorch.

```sh
WORK=sparselab-work/experiments/dense-lm-v1
export SPARSELAB_WORK_DIR="$WORK"
mkdir -p "$WORK"
uv run --locked --no-sync sparselab study plan configs/references/dense-lm-v1/study.yaml
uv run --locked --no-sync sparselab worker register reference-rocm --backend rocm --store "$WORK/runs"
uv run --locked --no-sync sparselab study submit configs/references/dense-lm-v1/study.yaml \
  --worker reference-rocm --store "$WORK/runs" --receipt "$WORK/receipt.json"
uv run --locked --no-sync sparselab controller run --store "$WORK/runs"
```

Inspect effective configurations, exact frozen input identities, disk headroom,
and disposable smoke/warmup evidence before authorizing execution. The controller
uses one store for all three seed coordinates, even though the historical YAMLs
record old paths. Resolve IDs from the one receipt and immutable run metadata.
Only collect after terminal ingestion; use the same store:

```sh
uv run --locked --no-sync sparselab study collect configs/references/dense-lm-v1/study.yaml \
  "$WORK/receipt.json" --runs-dir "$WORK/runs" --backend rocm
```

A controller-managed interrupted run resumes with `experiment resume RUN_ID
--store "$WORK/runs"`; its child belongs in the same store. For direct training,
pass `train --runs-dir "$WORK/runs"` on both parent and child invocations and select
an immutable verified parent checkpoint with `--resume`. Retain the original
receipt and all parent/child identities; never rewrite a submission receipt to
pretend a child was the originally submitted coordinate.

For an exact selected checkpoint, put new local observations under the experiment
root. Set `RUN_ID`, `STEP_GENERATION`, and `STEP` from verified run metadata:

```sh
uv run --locked --no-sync sparselab model exercise "$RUN_ID" \
  --checkpoint "$WORK/runs/$RUN_ID/checkpoints/$STEP_GENERATION" \
  --runs-dir "$WORK/runs" --backend rocm \
  --prompt-panel data/dense_lm_v1_prompts.json \
  --output "$WORK/exercises/$RUN_ID/step-$STEP"
```

Keep stage bundles under `$WORK/staging`, generation captures under `$WORK/captures`,
and temporary reports under `$WORK/local-reports`. Promote only deliberately
retained compact evidence into `artifacts/acceptance` and content-addressed
`artifacts/research-reports`; full mutable run trees remain local. A different
physical disk may be selected by assigning `WORK` an absolute external path.
See [workspace policy](../workspaces.md) for path identities and relocation rules,
and the [migration audit](../dense-lm-v1-workspace-audit.md) for the completed
local consolidation and verification.

The immutable retained observations record the completed three-seed result; no
further execution is part of this record. The original read-only CPU exercise of
a pre-existing 20-step ROCm smoke checkpoint wrote to `/tmp`, recorded short-context
generation failures, and was not candidate evidence. No run in `runs-rocm/` was
modified by that historical exercise.

```sh
uv run --locked sparselab research validate --json
uv run --locked sparselab research status --json
uv run --locked sparselab research baseline list
```

## Gates and reporting

The frozen machine groups require finite model state and immutable identities, exact step/token coordinate, held-out LM evaluation, fixed-panel mechanical validity, repeatable greedy generation/cache parity where available, and degeneration diagnostics. The degeneration implementation computes descriptive repetition measures but gates only nonempty completions. No manual review procedure, numeric repetition threshold, or prompt-specific quality failure was frozen; “no catastrophic degeneration” was an unquantified plan phrase. Record unavailable hardware/optimizer/timing values as unavailable, not zero.

`promotion_eligible=true` on each terminal observation reflects the frozen machine groups. The accepted lifecycle promotion additionally binds all three seed identities, checkpoint chains, fixed generation captures, per-seed runtime evidence, and the static report. This establishes only the bounded learning reference. Do not alter the existing baseline conditions or frozen gates; any new quality threshold requires a separately versioned preregistration.

The Research dashboard renders content-addressed learning observations from its `--reports-dir`; browser visual verification was attempted but the managed Chromium broker did not launch in this environment. The Research-view screenshot remains unavailable and is explicitly not a scientific acceptance gate.
