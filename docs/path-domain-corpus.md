# Path domain corpus v1

`data/path_domain_v1/` is an original, MIT-licensed narrow conversation corpus for lexical `pathlib.PurePosixPath` questions. It is not a filesystem corpus: it has no machine paths, personal data, repository inventory, filesystem reads, resolution, symlinks, permissions, globbing, or Windows semantics.

## Provenance, oracle, and answer contract

The authoritative machine-readable records are `data/path_domain_v1/provenance.json`, `oracle_audit.json`, `leakage_audit.json`, and `freeze_ledger_v1.json`. The corpus retains its original MIT notice and synthetic authorship. Every one of its 24 training, 8 development, and 12 frozen records was checked with CPython 3.14.7's `pathlib.PurePosixPath`; `oracle_audit.json` records the exact interpreter, operation, path, argument, oracle value, rendered label, and correction audit for every line. The audit found zero factual label corrections.

Astra independently parsed all 44 source prompts and recomputed their labels
with CPython 3.14.7. Normalized `(path, operation, argument)` identities have
zero cross-split overlap. All card prompts and labels match their source chat
envelopes: acquisition uses a preregistered **six-example subset** of training,
development covers all eight records, and frozen evaluation covers all twelve.
All 28 frozen domain file identities were checked again after the pre-training
audit clarifications. [Independent acceptance evidence](../artifacts/acceptance/domain_curation_2026_09_22.json)
is a corpus check, not a model-quality result.

The records are `format_version: 2` with `loss_mode: assistant_only`: only the assistant answer span is supervised. Empty suffixes are the visible two-character sentinel `''`, not an empty chat reply; the oracle value itself is the empty string. The `draft.` records intentionally use Python 3.14 single-dot suffix behavior.

All three cards use `literal_full_answer_exact_v1`. It compares the full answer after trimming **only outer reply whitespace**. Case, Unicode composition, and internal whitespace remain significant. Accordingly, values for which meaningful leading or trailing whitespace must survive chat-reply extraction are excluded.

## Split boundary and overlap disclosure

| Split | File | Records | Permitted use |
|---|---|---:|---|
| Training | `data/path_domain_v1/train.jsonl` | 24 | Immutable tokenizer input and assistant-only adaptation input |
| Development | `data/path_domain_v1/development.jsonl` | 8 | Fixed-endpoint diagnostics only; never tokenizer, adaptation, checkpoint, or seed selection |
| Frozen evaluation | `data/path_domain_v1/frozen_evaluation.jsonl` | 12 | Frozen card confirmation only; never tokenizer, adaptation, checkpoint selection, or rerun tuning input |

Exact prompts and normalized full prompts do not overlap across splits. That does **not** make the sets semantically independent: the single-turn chat envelope, answer grammar, all seven operations, and many lexical path families overlap by design. The frozen split supplies unseen literals and paraphrased templates within this finite lexical domain, not unseen operation semantics. The packet supports only training-seen acquisition, development performance on unseen literals/wording, and frozen performance on unseen literals/paraphrases. It does not support claims of independent semantics, general `pathlib` mastery, filesystem correctness, cross-platform behavior, version invariance, real-world path safety, or an architecture/memory advantage.

The cards have `uniform_chance` and `majority_answer` controls of `0.0`. These are explicit non-claims, not open-vocabulary chance estimates: a finite authored oracle-answer pool cannot establish such a baseline.

## Immutable six-run protocol

This is an execution protocol, not a result. Do not train, select, or rerun any endpoint based on loss or card outcomes. The immutable tokenizer at `artifacts/tokenizer_path_domain_v1/` is fitted from the 24 training records only and is used by every endpoint.

For each seed, run the synthetic endpoint first, then promote its verified final checkpoint into the matching domain endpoint. The three pretraining runs are dense tied D64/L2/H4/FFN192 CPU/fp32 runs over the same synthetic corpus (`synthetic_seed=20260922`, 128 train documents, 32 validation documents, train cap 65,536 tokens, validation cap 4,096 tokens). Each commits exactly 30,720 targets. Promotion starts a fresh optimizer, scheduler/counters, and RNG for a further exactly 30,720 assistant-only domain targets; it is not a resume.

| Seed | Synthetic pretraining config / run ID | Domain adaptation config / run ID |
|---:|---|---|
| 17 | `configs/path_domain_pretrain_seed17.yaml` / `path-domain-pretrain-seed17` | `configs/path_domain_adapt_seed17.yaml` / `path-domain-adapt-seed17` |
| 41 | `configs/path_domain_pretrain_seed41.yaml` / `path-domain-pretrain-seed41` | `configs/path_domain_adapt_seed41.yaml` / `path-domain-adapt-seed41` |
| 73 | `configs/path_domain_pretrain_seed73.yaml` / `path-domain-pretrain-seed73` | `configs/path_domain_adapt_seed73.yaml` / `path-domain-adapt-seed73` |

The window calculation in the freeze ledger uses actual packed labels and short-epoch accumulation windows, not a nominal full-window assumption. Each synthetic endpoint has 115 train blocks, 11,040 usable targets per epoch, one initial input-only token, and 19 discarded trailing tokens. It reaches 30,720 targets in 161 updates (159 192-target windows plus two short windows). Each assistant-only domain endpoint has 103 supervised tokens in the raw stream, of which 96 are usable across 11 packed blocks and seven are discarded at the tail. It reaches 30,720 targets in 1,920 updates for each listed seed; every domain update is short (4–27 targets). `warmup_steps: 12` is valid for all six horizons.

The pretraining evaluation covers all 30 packed synthetic-validation blocks. The adaptation evaluation covers all 4 packed development blocks. No checkpoint is selected by validation loss or card score: use the verified generation at the committed-target budget.

## Commands for Main

Run these only after the runtime acceptance gate. They are endpoint commands, not results.

```sh
uv run sparselab tokenizer train configs/path_domain_tokenizer_cpu.yaml

uv run sparselab train configs/path_domain_pretrain_seed17.yaml --run-id path-domain-pretrain-seed17
uv run sparselab train configs/path_domain_pretrain_seed41.yaml --run-id path-domain-pretrain-seed41
uv run sparselab train configs/path_domain_pretrain_seed73.yaml --run-id path-domain-pretrain-seed73

uv run sparselab capability evaluate path-domain-pretrain-seed17 data/path_domain_v1/cards/path-domain-acquisition-v1.json --runs-dir runs --backend cpu
uv run sparselab capability evaluate path-domain-pretrain-seed17 data/path_domain_v1/cards/path-domain-development-v1.json --runs-dir runs --backend cpu
uv run sparselab capability evaluate path-domain-pretrain-seed17 data/path_domain_v1/cards/path-domain-frozen-v1.json --runs-dir runs --backend cpu
# Repeat the three fixed card commands for pretrain seeds 41 and 73.

uv run sparselab train configs/path_domain_adapt_seed17.yaml --run-id path-domain-adapt-seed17 --promote runs/path-domain-pretrain-seed17/checkpoints/latest.json
uv run sparselab train configs/path_domain_adapt_seed41.yaml --run-id path-domain-adapt-seed41 --promote runs/path-domain-pretrain-seed41/checkpoints/latest.json
uv run sparselab train configs/path_domain_adapt_seed73.yaml --run-id path-domain-adapt-seed73 --promote runs/path-domain-pretrain-seed73/checkpoints/latest.json

uv run sparselab capability evaluate path-domain-adapt-seed17 data/path_domain_v1/cards/path-domain-acquisition-v1.json --runs-dir runs --backend cpu
uv run sparselab capability evaluate path-domain-adapt-seed17 data/path_domain_v1/cards/path-domain-development-v1.json --runs-dir runs --backend cpu
uv run sparselab capability evaluate path-domain-adapt-seed17 data/path_domain_v1/cards/path-domain-frozen-v1.json --runs-dir runs --backend cpu
# Repeat the three fixed card commands for adaptation seeds 41 and 73.
```

Evaluate the cards for every seed at both endpoints. Do not choose a checkpoint, a seed, or a rerun from any card result.

Retention is distinct from domain development loss. Record CE/perplexity on every endpoint's run-owned synthetic validation array (all 30 blocks) before adaptation, then run the same fixed synthetic array against the promoted adaptation checkpoint after adaptation. The source is the corresponding pretraining run's `data/validation.npy` and `data/validation_supervision.npy`; retain their ledger hashes in both reports. Adaptation's normal four-block validation is the separately reported domain-development loss. The ledger fixes this comparison and its input identity; it must not be replaced with a fresh synthetic sample or the development array.

## Generation bounds

Bounds use the immutable tokenizer's actual answer encodings plus four tokens of explicit headroom, never model output: acquisition 6+4 = 10, development 19+4 = 23, frozen 16+4 = 20. The largest prompt-plus-bound is 103 tokens for the frozen card, below the fixed 256-token context. Per-case counts and every asset digest are in `freeze_ledger_v1.json`.

## 2026-09-22 execution record

All 18 ledger-listed frozen identities verified before execution. The three exact
pretraining endpoints and their promoted adaptations each completed and passed
checkpoint verification at their registered budgets: 30,720 targets / 161 updates
for pretraining and 30,720 assistant-only targets / 1,920 updates for adaptation.
Every promotion bound the corresponding verified same-seed endpoint and started
fresh adaptation optimizer state, counters, and RNG.

The initial concurrent preparation attempts for pretraining seeds 17 and 73 failed
before creating a run or taking any optimizer update because
`artifacts/path_domain_synthetic_data_v1/502642703395d1a6.tmp` existed. After that
sibling finalized, Main authorized sequential infrastructure retries because no
scientific endpoint had begun. The two original failures and the authorization are
preserved in the result ledger; no frozen input, configuration, seed, bound, or
outcome-based selection changed.

| Seed | Pretrain cards (acq/dev/frozen) | Adapt cards (acq/dev/frozen) | Synthetic retention CE before → after | Adapt development CE |
|---:|---|---|---|---:|
| 17 | 0/6, 0/8, 0/12 | 3/6, 0/8, 0/12 | 0.48270738257302176 → 14.720103285047744 | 14.663577488490514 |
| 41 | 0/6, 0/8, 0/12 | 2/6, 0/8, 0/12 | 0.46399771240022447 → 14.141955354478624 | 15.165186745779854 |
| 73 | 0/6, 0/8, 0/12 | 4/6, 0/8, 1/12 | 0.6024650639957851 → 13.33100251091851 | 11.477950202094185 |

Each retention measure used the same seed-owned synthetic `validation.npy` and
`validation_supervision.npy`, all 30 blocks, and 2,880 valid targets before and
after adaptation. The ordinary adaptation development measures instead use four
blocks and 63 valid targets. Across the three paired endpoints, descriptive mean
deltas (adapt minus pretrain) are +3.0 acquisition-correct cases, +0 development,
+0.3333333333333333 frozen, and +13.54796366382528 retention CE. These are results
for finite lexical PurePosixPath 3.14 semantics only, not evidence of general chat
competence or safety.

At these fixed budgets and objectives, the models do not provide reliable unseen-path answers: adapted development remains 0/24 across seeds and adapted frozen accuracy is 1/36. Astra independently reverified all six checkpoints and promotion links, rescored all archived answers, and recomputed all six same-input retention measurements. The severe retention degradation is a negative result, not a successful general-purpose adaptation claim. [Acceptance record](../artifacts/acceptance/scientific_studies_2026_09_22.json).

The auditable six-endpoint ledger, original infrastructure failures, verified
lineage, exact retention bindings, descriptive aggregates, and all 18 full
literal-response card reports are in
[`artifacts/studies/path_domain_2026_09_22.json`](../artifacts/studies/path_domain_2026_09_22.json).
