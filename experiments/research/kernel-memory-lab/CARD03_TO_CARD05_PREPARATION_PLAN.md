# Consolidated Card 03 preparation plan for Card 05 readiness

Status: **PROPOSED ONLY** after source-admission policy v1 and the small
two-shard admission screen. No acquisition, tokenizer fitting, GPU work,
optimizer updates or Card 05 runtime is authorized by this plan.

The metadata-only sizing step is recorded in the [scaled preparation
proposal](CARD03_SCALE_PREPARATION_PROPOSAL.md). It recommends a bounded
5-million-position first candidate using the same two pinned complete shards
plus two incident-document repositories; the earlier 10-million-position
capacity target remains an aspiration, not an authorization or measured supply.

## Output target and sequencing

Prepare one versioned, rights-admitted data and evaluation package. The
original **10-million-position** candidate remains an eventual capacity
aspiration; the [sized first proposal](CARD03_SCALE_PREPARATION_PROPOSAL.md)
uses **5 million** model target positions because incident-source capacity is
uncertain. Any Card 05 tranche remains subject to a real-data forecast and
separate approval. Keep the accepted 341,885,952-parameter architecture and the
[Card 03 contract](CARD03_DATA_CONTRACT.md) unchanged: 65% general, 25%
explanatory and 10% incident-response prose by *realized training target tokens*;
train-only 32,768-entry byte-level BPE; 80/10/10 eligible family-count split;
frozen 200-item closed-book language and 400-item open-book evidence suites.

For the eventual 10-million-position candidate with at most two train exposures per unique
target position, require measured train-family capacity of at least **3.25
million unique general**, **1.25 million unique explanatory** and **0.5 million
unique incident-response** target positions after tokenization, with independent
validation/test families. The realized train schedule must log repeats and may
not exceed two exposures of any training position under this proposal. If the
measured supply misses a stratum, reduce the separately proposed tranche or
obtain another rights-reviewed source; never silently change the mixture or
fill the shortfall from held-out material. These are preparation sufficiency
targets, not an approved model-token budget.

1. **Inventory and admit sources.** Reuse verified existing snapshots and the
   versioned admission manifest. For broader language coverage, choose complete
   pinned shards from the reviewed Project Gutenberg filtered and Wikimedia
   filtered components, and if necessary a separately reviewed prose component.
   Record exact shard sizes, source license policy, expected retained rows,
   response-body retry allowance, decompression/retained-output/disk/inode/time
   caps before requesting acquisition. Do not reuse the tiny pilot's cap as a
   budget for new files. Separately request the already pinned bounded
   PagerDuty 38-file pilot; measure its eligible train-family tokens. Its 36
   Markdown blobs total only 302,785 raw bytes, so the 0.5-million unique
   incident target cannot be presumed. If short, propose a pinned,
   rights-reviewed additional incident-response documentation source and its
   own acquisition bounds before expanding. Complete automated exceptions,
   spot audits and manual flag review under policy v1 for each new source/pin.
2. **Freeze families and splits before chunking.** Inventory document/work,
   mirror, revision and topic-cluster families; hash with the accepted split
   seed and freeze actual 80/10/10 eligible family lists and denominators by
   stratum. Keep all sibling versions/paraphrases together, and independently
   hold out incident-response scenarios and entities. Perform exact content
   deduplication, native near-duplicate screening and manual review of flagged
   cross-split pairs. Freeze the raw-text and provenance manifest first; no
   evaluation answer may enter train or tokenizer-fit data.
3. **Fit and verify the main tokenizer.** Use only admitted train-family text,
   with an approved input-byte/document/time/disk ceiling. Produce a *new*
   32,768-entry byte-level BPE, verify its manifest and special IDs, round trips,
   held-out `<unk>` rate and efficiency by source family. Card 04's synthetic
   tokenizer and checkpoint remain separate. Freeze tokenizer identity before
   measuring the exact unique target-position capacities above.
4. **Materialize the token mixture.** Implement the small native deterministic
   multi-source operation already identified in [TODO.md](../../../TODO.md)
   if still missing. Bind the verified rights release, tokenizer, split and
   selected documents; record per-document order, encoded target positions,
   repeats, realized 65/25/10 totals and shortfalls. The existing
   `requested_mixture` field reports intent but does not enact sampling. Test
   fail-closed behavior on insufficient quotas, inadmissible rows and leakage
   before publishing a prepared-input bundle.
5. **Author and freeze evaluation.** Create the 200 language items across ten
   axes and 400 evidence items across ten categories from held-out families or
   original fictional cases. Freeze item/rubric/reviewer IDs, gold chunk/source
   versions and hashes, answerability and wrong/missing/shuffled evidence
   controls before inspecting model outputs. Review prompt/answer leakage
   against both train and tokenizer-fit inventories; preserve unresolved and
   unavailable categories.
6. **Forecast and seek Card 05 approval.** Only after the actual prepared inputs
   exist, measure real-data preparation/step/checkpoint/evaluation timing or
   validate a conservative forecast on the chosen local backend. Combine that
   with storage and memory high-water, exact effective batch, target positions
   per update, checkpoint schedule and a single attempt-wide time/update cap.
   Perform the named Card 05 readiness review and Gate 0 criterion review, then
   request the exact run approval. Synthetic P6 throughput is not this forecast.

## Decision boundary

The current pilot's 19 qualifying records are an admission-method
demonstration, not a sufficient language corpus: native `corpus describe`
reports 5,918,614 general-prose UTF-8 bytes, only 7,308 explanatory-prose
UTF-8 bytes and zero incident-response bytes, with model-token counts
unavailable until the main tokenizer exists. The inventory release has no
selected LM or chat view and does not freeze the full Card 03 inputs.
PagerDuty content is unacquired. The metadata-only inventory now fixes the
proposed transport, retention and preparation ceilings in the [scaled
preparation proposal](CARD03_SCALE_PREPARATION_PROPOSAL.md). Actual unique
train tokens, independent families and any eventual Card 05 update/time
allocation still require admitted content, frozen splits, a main tokenizer
and a real-data forecast. No training authority is inherited from source
admission or this preparation proposal.
