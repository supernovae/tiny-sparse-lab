# Repository review: from mechanism demos to measured small models

## Project contract

A useful experiment trains an actual checkpoint, exercises it through chat, measures a declared task, compares controlled alternatives, and preserves the conditions needed to repeat or falsify the result. "Useful" starts with a narrow, reliable learned behavior; it does not require mimicking a commercial assistant. The same contract should survive increased model size, data, and training budgets within the supported single-host runtime.

## Gaps closed by this review

| Gap | Resolution and evidence boundary |
|---|---|
| Chat cropped arbitrary tokens and depended on external tokenizers | Completion-reserved whole-turn context management; verified checkpoint selection; run-owned tokenizer/package; interactive reset, seeded sampling, JSON responses and saved transcripts. |
| User chat data required new Python sources | Strict licensed local JSONL conversation ingestion using the same chat formatter; byte-content cache identity and exact train/validation conversation-overlap rejection. |
| First recall card confused record IDs with generalization | Historical card labeled diagnostic. Separate training-seen retention, test-only held-out wording, and context-override cards. |
| No task acquisition control | Step-zero checkpoints and a training-seen retention card distinguish inability to learn from failure under new wording. |
| Scores could outlive or overwrite their checkpoint identity | Frozen generation and content-addressed evidence, full prompts/responses, generation protocol, trained budget, source identities and parameter counts. |
| Configuration equality hid unfair comparisons | Compare actual steps/tokens, seed, content digests and runtimes; explicit memory/attention/FFN/scale axis; exact changed fields and paired gains/losses. |
| Validation/evidence paths accepted incomplete or altered observations | Count-weighted loss, restored evaluation mode/RNG, finite checks, artifact/report/checkpoint binding, explicit rejected and missing reports. |
| Resume/promotion could misrepresent scientific settings or data | Compatibility checks remain mandatory; recovery uses the same checks; promotion uses destination data and compatible architecture/tokenizer semantics. |
| Partial token budgets and performance metrics were misleading | Exact valid-target masking, skipping empty microbatches, synchronized update timing; historical throughput claims withdrawn. |
| Architecture accounting/diagnostics were incomplete | Corrected MLA and multi-table parameter estimates, actual Engram storage versus active rows, persisted last-training-microbatch diagnostics, and pre-renormalization router selection mass. |
| Byte-memory Unicode behavior differed between packing and generation | Shared reversible ByteLevel token-byte conversion with causal UTF-8 regression coverage. |
| Runtime capabilities exceeded implemented behavior | Explicit runtime/precision validation, staged pilots, and MLX/target-hardware limitations documented instead of silent substitution or feature-parity claims. |

## Actual learning experiment

Ran the supplied dense and n-gram configurations on CPU with seed 42, a train-only 512-token BPE, two D64 decoder blocks, 256-token maximum context, and 128-token training blocks. Dense has **139,584 parameters**; Engram has **174,368**. This is a matched-backbone comparison, not a matched-total-parameter efficiency experiment.

| Observation | Dense | Engram |
|---|---:|---:|
| Step-zero retention | 0/24 | 0/24 |
| Retention at 1,536 steps | 24/24 | 24/24 |
| Held-out wording at 1,536 steps | 8/12 | 6/12 |
| Context overrides at 1,536 steps | 0/8 | 0/8 |
| Actual target tokens at 1,536 steps | 391,424 | 391,424 |

Run IDs: `review-chat-dense-long` and `review-chat-engram-long`. The configured token ceiling was 393,216; epoch-end short batches made the step limit bind first. Per-case artifacts and comparisons are under each run's `evaluations/`; run artifacts are intentionally not committed as source.

The earlier 384-step exploratory pair is retained as `review-chat-dense` and `review-chat-engram`: held-out wording was 0/12 versus 1/12, and override was 1/8 versus 0/8. Engram's 1/12 matched the constant-answer baseline. We increased a fixed budget without altering the test cases. Because these cases were inspected during development, this is **exploratory engineering evidence**, not a preregistered untouched-test result.

The trained models genuinely acquired the associations and generalized to some new wording. **This experiment does not support an Engram advantage**: dense performed better on the held-out wording card, and both failed the context-control card. It also does not show broadly useful chat, unseen factual knowledge, parameter efficiency, or statistical significance.

### Chat surface proof

The real interactive CLI, not a mocked response path, produced these replies from `review-chat-engram-long` with the trained system prefix:

```text
User: What value belongs to the alias amber?
Assistant: lumen
User: Please recall the value for birch.
Assistant: orbit
/reset
User: What value belongs to the alias coral?
Assistant: quartz
```

A separate local JSONL fixture completed preparation, training, checkpointing, and `checkpointed_held_out` evidence at 128 targets. That is ingestion/lifecycle proof, not a learned domain-quality claim. Tests cover the harder boundaries: overflow, role stopping, RNG isolation, Unicode addressing, shifted token budgets, comparison mismatches, relocated artifacts, tampered checkpoints/caches/reports, and custom-card schema/response budgets.

The interactive transcript is preserved at `runs/review-chat-engram-long/evaluations/interactive-chat.json`. The local-corpus run is retained as `runs/review-local-corpus`; standalone evaluation still succeeds after removing its original corpus, configuration and cache directory, using 224 run-owned validation targets.

### Verification

```sh
uv run ruff check src tests
uv run pytest -m "not cuda and not rocm and not xpu and not network"
uv run sparselab data prepare configs/chat_recall_dense_cpu.yaml
uv run sparselab eval review-local-corpus
```

The local suite includes available MPS/MLX coverage; it is not CUDA, ROCm or XPU acceptance. All local file links in the 40 Markdown documents were checked. Temporary corpus/configuration/card fixtures were removed after retaining the run-owned evidence.

## Next experiments, in order

1. **Broaden the curriculum without erasing the failed control.** Teach varied local-context overrides, then freeze a new card with unseen assignments and independently held-out paraphrases. Keep the current static-lookup and failed-override results.
2. **Test Engram-specific hypotheses.** Vary address order, table size/collision pressure, and hash heads one at a time; include a dense baseline and matched-total-parameter design where efficiency is the question. Measure quality and real resource cost together.
3. **Repeat seeds and budgets.** Report per-seed paired deltas and acquisition/generalization curves. Do not select the favorable seed or best test-scoring checkpoint. Automation for statistical aggregation remains future work.
4. **Move to a licensed domain corpus.** Use local conversation JSONL and a frozen domain card with exact answers where appropriate. Audit semantic leakage, provenance, failure modes, and retention after further training. Whole-transcript next-token training is currently supported; assistant-only loss and tool-call schemas are not.
5. **Increase scale only after the smaller experiment learns.** Reuse cards, data and tokenizer contracts at larger explicit backbones. Fit within one host; memory estimates are estimates, and native sparse speedups, KV caching, broad mixed-precision hardware acceptance, and distributed execution require separate implementations and evidence.

See [capability workflow](capabilities.md), [chat-oriented data](instruction-training.md), [evidence](evidence.md), and the [remaining roadmap](../TODO.md).
