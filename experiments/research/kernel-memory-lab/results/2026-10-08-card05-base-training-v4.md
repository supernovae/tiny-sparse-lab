# C05-B8 — 25M-position fresh base pretraining and fixed-profile evaluation

**State: runtime COMPLETED; model-quality decision READY FOR REVIEW.** The owner approved one local attempt under [the B7 training declaration](../CARD05_BASE_TRAINING_FROM_B7_PROPOSAL.md) at clean `f80079b0dc1ab05cf8ecaab9b0d31a2fefc015c4`. This record is the single fresh `kml-card05-base-pretraining-v4` run in `/srv/sparselab/state/experiments/kernel-memory-lab/card05-base-pretraining-v4/`. No acquisition, corpus rebuild, tokenizer fit, retry, resume, cloud work or Card 06 work occurred. Earlier failures and the C05-N1 0/200 reader result remain unchanged. Operator and qualitative reviewer: Codex; owner review of this result is pending.

## Execution and identities

The unchanged B7 release `67d727a073e59283e76e81965ff7dce33ceba41f0d6a9ee2161705d725f8d798`, mixture receipt SHA-256 `28a10f3831be92d11ca3239828fcf576d5dfdec31e39179ab81ae2b85c5f895c`, bundle manifest SHA-256 `bc869e7548a9480e48a3e18adc99fdaf775008bdaf500853b77a3645a3b5d02a`, tokenizer SHA-256 `308b33a6edbed613f3caf5232b124a7c6105a7a3dc273c999725be51c3ffeaa9`, config SHA-256 `674d79c779ed893286945e2868dce925bc3c10ddb0f3a2c673943b4a2dedcfa5`, and frozen base profile SHA-256 `bb4c1b719b29fa96c4429516022a5038f0e3595b9d8ef63394bf0a17a5c6851d` were bound before model work. Native cold verification passed for release, mixture, prepared bundle and profile; [the cold-verification receipt](/srv/sparselab/state/experiments/kernel-memory-lab/card05-base-pretraining-v4/cold-verification.json) has SHA-256 `eb28996636295f17b97a3d5d5a9c91962777e3dcffc9c41caa6e5de5b1508b57`. Native inspect and workspace preflight passed. The ROCm runtime used the local RX 7900 XTX, PyTorch `2.13.0+rocm10.0.0`, BF16 and the declared 341,885,952-parameter dense model. Fresh seed-17 weights and empty optimizer/scheduler state were used; no C05-N1 or P6 checkpoint was resumed.

One persistent v2 [attempt ledger](/srv/sparselab/state/experiments/kernel-memory-lab/card05-base-pretraining-v4/attempt-ledger.sqlite) started at `2026-10-09T02:35:29.927091Z` with the 14,400-second deadline and contract SHA-256 `060ad24fbc24e22ac331e101087e23012d2f7ebba035c1e367615e066e06984e`. The common-root baseline SHA-256 `fc49cec6ef37b70322e6a9e2eea97b8bcecdd715115ad2449c7ef6707d99b815` remained unchanged and includes retained preparation storage. Native `stage --through validate --prepared-inputs --cold-verify` completed with **zero optimizer updates**. The sole native `train` invocation pre-reserved and reconciled **24,415 actual updates and exactly 25,000,000 supervised targets**. Its last update carried 64 loss-bearing targets and 960 masked padding positions. The native evidence API verified the weight manifests of all eleven checkpoints and eleven one-batch operational validation reports, with no missing or rejected report. The selected terminal checkpoint separately passed full-state verification. The native training receipt and reconciled ledger, not a remembered counter, establish completion.

The [selected-checkpoint receipt](/srv/sparselab/state/experiments/kernel-memory-lab/card05-base-pretraining-v4/selected-checkpoint.json) was written only after scoring the frozen validation slices on **all eleven** declared checkpoints and verifying each result and checkpoint identity. It applies the earliest minimum finite token-weighted **fixed validation** loss tie-break. It selected terminal `step_00024415_gen_000011` at **5.0779**; the one-batch operational validation series was recorded separately and was not used for selection. Test, utility and continuation results were evaluated only after that receipt existed. The exact historical C05-N1 selected checkpoint was evaluated separately on the same base profile. All 18 fixed-profile operations had `COMPLETE` common-root and output monitors, zero violations, zero exit codes and zero living owned descendants.

## Learning curve and fixed outcomes

All loss values are nats per supervised token. Operational validation is one held-out batch per checkpoint; fixed validation is twelve frozen 256-target windows per checkpoint. The run does not retain a dense per-update training-loss series, so these are validation curves rather than training-loss measurements.

| Update | Targets seen | Operational validation | Fixed validation | General | Explanatory | Incident |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 10.6085 | 10.6062 | 10.6041 | 10.6039 | 10.6107 |
| 2,500 | 2,560,000 | 6.3489 | 6.4642 | 5.5941 | 6.9520 | 6.8464 |
| 5,000 | 5,120,000 | 6.0917 | 6.1020 | 5.3780 | 6.4811 | 6.4468 |
| 7,500 | 7,680,000 | 5.8649 | 5.8709 | 5.2002 | 6.2652 | 6.1472 |
| 10,000 | 10,240,000 | 5.7754 | 5.7556 | 5.1947 | 6.1871 | 5.8849 |
| 12,500 | 12,800,000 | 5.6633 | 5.6183 | 5.0306 | 6.0399 | 5.7845 |
| 15,000 | 15,360,000 | 5.5212 | 5.4581 | 4.9261 | 5.9700 | 5.4781 |
| 17,500 | 17,920,000 | 5.4379 | 5.2548 | 4.8319 | 5.5086 | 5.4238 |
| 20,000 | 20,480,000 | 5.2751 | 5.1908 | 4.7483 | 5.4848 | 5.3394 |
| 22,500 | 23,040,000 | 5.2851 | 5.1085 | 4.7232 | 5.3863 | 5.2159 |
| 24,415 | 25,000,000 | 5.2398 | **5.0779** | 4.6987 | 5.3328 | 5.2023 |

| Frozen profile | New selected | Historical v2 | New minus v2 |
| --- | ---: | ---: | ---: |
| Validation, all strata | 5.0779 | 5.5758 | -0.4979 |
| Test, all strata | 5.3022 | 5.4516 | -0.1493 |
| Test, general prose | 4.9216 | 5.6441 | -0.7225 |
| Test, explanatory prose | 5.6412 | 6.1644 | -0.5232 |
| Test, incident-response documents | 5.3438 | 4.5461 | **+0.7977** |
| True continuation preferred over decoy | 15/24 | 13/24 | +2/24 |

The combined 25M data/recipe change improves fixed general and explanatory loss but worsens incident-response loss. The 24-pair preference difference is small and is not evidence of reliable comprehension. These results do not isolate which recipe component caused either change.

All eight new frozen 64-token greedy prose continuations were reviewed against their exact prompts, with the eight paired v2 outputs inspected. The [local qualitative review](/srv/sparselab/state/experiments/kernel-memory-lab/card05-base-pretraining-v4/base-language-review.json), SHA-256 `ab17971769ad50d111caef03eb259fe260a25de3da91d24d50dd8e10e0fa4654`, references the full raw prompt/completion artifacts. It is an agent-assisted review, not an independent reader-gate review; the review used under one aggregate agent-hour, below the two-hour cap. Examples: `val-general-01` begins as plausible dialogue but repeats “I don't know”; `val-explanatory-01` moves a British unit to the United States and loops on “first two years”; `tes-explanatory-01` collapses a documentary list into repeated alphanumeric labels. Other outputs repeat generic phrases, invent unrelated biographical claims or lose the source thought. **Zero of eight** sustain coherent, relevant prose across their full span, and all eight show conspicuous repetition. The v2 examples also repeat. This small descriptive panel does not establish causality or instruction-format capability.

## Bounds, receipts and review state

The [final native ledger status](/srv/sparselab/state/experiments/kernel-memory-lab/card05-base-pretraining-v4/final-attempt-status.json) records 25,000,000 targets, 24,415 updates, 11 operational validations/11,264 forward positions, 52,392 fixed-profile forward positions, **161,448 aggregate charged nontraining forward positions** (below 170,000), 16 generation calls and 1,024 requested new tokens. Both models used the declared cached greedy decoder for every continuation; each generated 512 tokens and used 1,016 *actual* generation forward-input positions, while the ledger conservatively reserved 48,896 per model. Fixed-profile scoring stayed below 55,000 positions. The two evaluation directories added only 101,112 and 36,742 bytes, below the combined 64 MiB output cap.

Ledger creation to final evaluation monitor completion was about **7,960 seconds**, below 14,400; checkpoint selection closed about 7,437 seconds after ledger creation, below the 10,800-second stage/train/selection cap; evaluation took about **472 seconds** from its declared start, below 2,400. The training monitor ran 6,348 seconds. The maximum sampled whole-device VRAM was **10,953,015,296 bytes** (below 20 GiB); maximum sampled process-tree RSS across nested monitors was **4,019,204,096 bytes** (below 24 GiB). The largest monitor-sampled cumulative added storage was **46,863,130,232 apparent bytes**; a later native common-root sample after the artifact index was **46,863,235,311 bytes and 1,879 entries** above baseline, below 72 GiB and 10,000 entries, with 900,542,156,800 free bytes. All resource monitors reported `COMPLETE` and no violations; the owner-tracked stage, training, selection and fixed-evaluation completions reported no living descendants. These are sampled operational high waters, not an exhaustive unsampled-peak claim. No cloud spend was incurred.

The [external artifact index](/srv/sparselab/state/experiments/kernel-memory-lab/card05-base-pretraining-v4/final-artifact-index.json), SHA-256 `bcafb2d37b3f7422cfa4c2d3a3d6322671b622369c9f73e4de8a30113b023cb3`, lists paths, byte sizes and SHA-256 hashes for 89 declaration, ledger, checkpoint-selection, monitor, raw evaluation and review artifacts. Large mutable outputs remain outside Git. The frozen 200-item Card 05 reader suite and 400 evidence items were not run; the existing C05-N1 score remains **0/200, not eligible**. This base result is **READY FOR OWNER REVIEW**, not accepted as a reader model. Card 06 remains **BLOCKED**. The next decision is whether the lower general/explanatory loss with still-degenerate prose and worse incident loss warrants a separately declared recipe/data diagnostic; no further model work is authorized by this record.
