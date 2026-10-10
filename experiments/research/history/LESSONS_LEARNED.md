# Lessons retained from earlier research

These summaries describe the retained observations. They are not new experiments,
reassessments of raw evidence or approvals to resume archived studies. Historical
identities and full reports remain unchanged; [the path map](path-map.json)
resolves their original locations.

## Dense learning, scale and decoding

- **Additional exposure improved held-out loss, with mixed generation.** Three
  seeds continued from 4.19M to 16.78M supervised targets on the same prepared
  data and preserved decay horizon. All improved loss; the declared flattening
  gate failed. Color contradictions and repetition persisted or worsened in
  individual fixed-panel outputs. Repeated target exposures are not new source
  documents, and lower loss is not automatic promotion.
  [Token-budget results](dense-lm-token-budget-v1/results.md)
- **A larger model improved likelihood within this controlled comparison.**
  At equal 16.78M targets, the approximately 50M model beat the approximately 30M
  model's terminal loss across three seeds, at higher runtime/memory cost.
  Story continuity remained mixed. This supports the bounded loss observation,
  not general prose superiority or a novel architecture claim.
  [Scale results](dense-lm-scale-v1/results.md)
- **Less repetition can exchange one failure for another.** A decoder selected
  on 22 development prompts reduced repetition on 55 separate test prompts,
  while drift and an explicit color contradiction remained. The 990 test cells
  were repeated observations over three trained seeds, not 990 model replicas.
  No human votes were collected; subjective quality stayed unavailable. Preserve
  the original regression decoder and do not tune on the opened test set.
  [Decoding results](dense-lm-decoding-v1/results.md)
- **Operational correctness needs measured counters.** The token-budget study
  retained a rejected partial-update attempt rather than treating the nominal
  step count as the target count. The scale study also records a locked CPU
  environment overwriting a vendor environment before training, motivating
  explicit device checks and the documented vendor environment.
  [Counter failure](dense-lm-token-budget-v1/results.md#controlled-continuation-and-integrity),
  [environment failure](dense-lm-scale-v1/results.md#preflight-before-full-training)

## DevMind source supply and recovery

- **Source collection, rights review and usable evaluation are different gates.**
  The early source/card candidates did not establish a useful developer model;
  v0's tokenizer stage stopped at its source-kind gate. The v1 evidence retains
  limited cards and outstanding source/publication review rather than turning
  metadata availability into admission or training approval.
  [v0 source report](devmind-pretrain-v0/source-report.md),
  [v0 tokenizer stop](devmind-pretrain-v0/tokenizer-report.md),
  [v1 evidence](devmind-pretrain-v1/evidence.md)
- **Measure the denominator before choosing exposure.** The v3 and v4 protocols
  separately retain 510,860,750 and 735,281,099 developer train bytes, with
  explicit heldout and overlap limitations. Those byte proxies did not establish
  tokenizer target supply or quality. v4's `READY_FOR_TOKENIZER` decision was a
  source-stage decision, not model promotion or weight-publication permission.
  [v3 protocol](devmind-pretrain-v3/protocol.md),
  [v4 protocol](devmind-pretrain-v4/protocol.md)
- **A hash without its authenticated production inputs does not guarantee
  recovery.** v4 retained replay blockers; v5 names the earlier v2/v3/v4 releases
  nonreconstructable with the available authenticated closure and creates a
  separately identified successor. Do not relabel new output as an old release.
  [v4 replay report](devmind-pretrain-v4/pinned-replay-report.md),
  [v5 historical identities](devmind-pretrain-v5/historical-releases.json),
  [v5 protocol](devmind-pretrain-v5/protocol.md)
- **A completed model remains bounded evidence.** v5 MODEL-0 completed 5,525
  updates and 45,260,800 supervised targets, with an accepted full-state endpoint.
  Its result explicitly retains repetitive, language-mixed outputs and remains
  unpromoted. Finite heldout loss does not prove coding usefulness, superiority,
  causality or portability. A proposed MODEL-1 continuation is not a completed
  experiment.
  [MODEL-0 result](devmind-pretrain-v5/model0-result.json),
  [MODEL-1 protocol](devmind-pretrain-v5/model1-protocol.md)

## TinyStories data-rich 30M

One seed-42 run reached 100,663,296 supervised targets with a verified full-state
endpoint. Its complete final 8,000-story loss was 1.610636 nats/native target.
On the same 256 raw stories, bits per UTF-8 byte were 0.668729 for the new run,
0.975226 for the old 30M reference and 0.955759 for the old 50M reference.
The same-text byte measure avoids equating different tokenizer target units,
but does not isolate a causal factor: source diversity, tokenizer, context,
schedule and precision changed together.
[Results and denominator definitions](tinystories-dense-30m-data-rich-v1/results.md)

Explicit color contradictions survived, and long-test completions were short.
Prompt train-disjointness, independently blinded behavior preference,
seed reliability and cross-device reproducibility were not established.
The recorded decision is an **unpromoted learning finding** with an independent
behavior protocol proposed before further training. Preserve every negative
output and distinguish optimizer throughput from end-to-end cost.
[Observed failures and decision](tinystories-dense-30m-data-rich-v1/results.md#sealed-observations)

## Implications for current work

Use one current native implementation and preserve historical evidence separately.
Bind sources, review samples, counters and outputs before execution; make cold
verification and refusal paths observable. Measure unique supply, repeated targets
and runtime phases separately. Frozen criteria, independent review and truthful
unavailable states matter more than a larger pile of successful commands.
