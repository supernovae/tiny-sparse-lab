# C05-B9 — shared-corpus and structure-cleaning candidate (review pending)

The accepted C05-B7 release and C05-B8 25M run remain unchanged. This is an
offline **candidate**, not a new admitted corpus, model result or Card 06 gate.
Starting checkout was clean at `ca0ce581c2aeea33ad3340179b1e68a0531f91fc`;
no newer branch work was present. No acquisition, fitting, model forward,
generation, optimizer update, GPU, hosted job or CI run occurred.

## Native change and shared reuse

The optional `release.normalizer: normalizer-structure-v3` creates a distinct
build/document/release identity. Its source-anchored structural record retains
readable text, heading/list/fence/code/equation annotations, raw line spans,
removed-span hashes and reasons, ambiguity flags, snapshot/file hashes, rights,
source family and split. Recognized YAML front matter is removed only when it
has reviewed metadata keys; a front-matter-only row remains in the canonical
document inventory with `metadata_only_front_matter` and is excluded from LM
views. Gutenberg production-credit paragraphs and explicit marker-delimited
wrappers are separated conservatively. Body prose, transcriber notes, meaningful
YAML configuration, code fences and contents, headings, lists, links,
indentation and equations are retained. Unrecognized openings stay in the text
with a reason flag. Ordinary and large builders share the same parser; cold
release replay checks the selected normalizer, raw span and output. The original
normalizer remains the absent-field default and the intermediate v2 identity
remains replayable.

`corpus verify-export RUN.yaml` is a read-only native binding inspection. Two
lightweight run fixtures with different seeds and output roots cold-verified
the **same** existing Card 03 LM export and retained its exact dataset,
tokenizer and cache declaration. The [reuse receipt](/srv/sparselab/state/experiments/kernel-memory-lab/card05-structure-candidate-v2/reuse-fixture.json)
has SHA-256 `a11fee561954b8a7f3337a458821e602af1162f9e117cacadc7ad7214890ca19`.
No release, export or prepared dataset was copied. C05-B8's token-mixture
bundle uses its separate authenticated mixture binding, not this LM export.

## Baseline impact and candidate

The [machine-readable audit](/srv/sparselab/state/experiments/kernel-memory-lab/card05-structure-candidate-v2/audit-summary.json)
has SHA-256 `c2bafbce4f2adeee913f86debe0cac0e94fa8e1ded322dc5dc9ef928922391b8`.
The [consolidated candidate record](/srv/sparselab/state/experiments/kernel-memory-lab/card05-structure-candidate-v2/candidate-review.json)
has SHA-256 `ec0b53aa3f5d3bc776242ee5a00ae53243f8272357b9ff15fa7e509296184daa`.
It joins the accepted release to the retained per-document 25M mixture rows;
eligible train positions and **actual supervised exposures** are separate. Its
old-tokenizer character-span overlap method excludes each scheduled row's EOS
from content counts and respects the final truncated row. Categories can
overlap, so the signal rows below must not be summed as safe removals.

| Retained train signal | Source | Eligible docs | Unique content positions | Scheduled content targets | Scheduled EOS |
| --- | --- | ---: | ---: | ---: | ---: |
| Metadata-only front matter | Scoutflo | 142 | 3,659 | 2,645 | 104 |
| Metadata-only front matter | PagerDuty | 2 | 98 | 56 | 1 |
| Mixed front matter | PagerDuty | 18 | 1,306 | 782 | 0 |
| Explicit production credit | Gutenberg | 144 | 2,513 | 2,444 | 0 |
| Repeated-line signal, **retained** | Scoutflo | 48 | 2,499 | 1,686 | 0 |
| Repeated-line signal, **retained** | Wikimedia | 147 | 3,522 | 4,854 | 0 |
| Markup residue, **retained** | Wikimedia | 37 | 2,319 | 2,981 | 0 |
| Label residue, **retained** | PagerDuty | 85 | 255 | 210 | 0 |

The four conservative removal categories account for **6,032 historical
supervised positions including 105 EOS**, about 0.024% of the 25M run. The
repeated-line rule is a whitespace-collapsed 25–240-character line found in at
least ten distinct train documents; it includes useful headings or instructions
and is only a review signal. Wikimedia table/image syntax and PagerDuty
`</label>` are possible extraction residue, not automatically damaged content.
No accepted train text contained U+FFFD or NUL under this audit. For example,
Scoutflo `KubeContainerOOMKilled-pod.md#lines=1-6` contains only title/weight/
categories and becomes an LM exclusion; Gutenberg shard row 226 loses only its
explicit production-credit paragraph, retaining the transcriber's note and
Hamlet body. Original source bytes, attribution and rights evidence remain in
the snapshots and release provenance.

The final [candidate release](/srv/sparselab/state/experiments/kernel-memory-lab/corpora/kernel-memory-lab-card05-base-v1/releases/6e63d39579864671916e06ba4f007d9f1a9bbb27f2e44396e11175262410b198/manifest.json)
(`6e63d395…`, manifest SHA-256 `d9e2170d6cdbf9841ac5f16ad35bc705bfebbd39c02f6562e8fc7fbaa76ee3d3`)
was built and cold-verified from the four retained snapshots with no network.
The first, narrower v2 release `ebb0f253…` is retained and cold-verifies under
its own normalizer. Candidate LM retention by source, train/validation/test:
Gutenberg **201/25/25**, Wikimedia **6,713/839/839**, PagerDuty **266/71/38**
(baseline 268/72/39), Scoutflo **1,326/160/136** (baseline 1,468/177/151).
There are 10,639 kept candidate documents. The [candidate family inventory](/srv/sparselab/state/experiments/kernel-memory-lab/card05-structure-candidate-v2/candidate-family-inventory.jsonl)
(SHA-256 `9e4ccc40b1101b78c5cb1f4b6a7d50dc3f3fdb36af7950d172fb75bf747ee505`)
uses the accepted family of each old record or its old exact-content
representative when a duplicate representative changed. Native mixture inventory
validation accepted all 10,639 kept rows; no accepted family was lost, no family
crossed splits and no new cross-split exact-content group appeared. This is not
a full renewed near-duplicate or rights review.

The [native token audit](/srv/sparselab/state/experiments/kernel-memory-lab/card05-structure-candidate-v2/unique-tokens.json)
(SHA-256 `0b7922aa6ad9ac2405b7c265c03fe89d64cf95464cd931d6527cc8120d6b7e47`)
uses the unchanged, verified train-only tokenizer and counts distinct kept train
content without EOS: **20,800,419 general**, **3,155,976 explanatory**, and
**338,641 incident** positions. All unchanged 20M/2.5M/200k floors pass. With
one EOS per kept train document, the available positions are 20,800,620 /
3,162,689 / 340,233. The unchanged 20.25M/4.5M/0.25M allocation is
arithmetically feasible with at most two exposures; a candidate mixture was
**not** materialized or accepted. The native public `corpus measure-tokens`
command still requires a same-release tokenizer origin, so this audit used its
existing `_measure_source_domains` API after native cold release and tokenizer
verification. The existing TODO entry tracks that CLI binding gap.

## Validation and next decision

The explicitly selected zero-model suite passed **18/18** tests. It covers front matter/configuration distinction,
Gutenberg credit and marker bounds, fenced headings, useful syntax,
determinism, versioned IDs, source-span replay, large-shard preparation,
metadata-only exclusion, shared fixture declarations and legacy payload
compatibility. Ruff format/check and `git diff --check` passed. The accepted release and both candidate versions passed cold
verification. The candidate's family inventory passed native validation. No
model-bearing integration or live hardware check was attempted.

The small measured removal cannot explain the run's global repetitive prose
on its own. The smallest next evidence task is a **stratified offline review of
the retained repeated-line and markup-residue signals**, with source spans and
rights intact, before deciding whether any additional rule is justified. Do
not tune a new mixture to the four incident test windows or infer model quality
from this candidate. C05-B8 stays READY FOR REVIEW, C05-N1 stays 0/200, and
Card 06 stays blocked.
