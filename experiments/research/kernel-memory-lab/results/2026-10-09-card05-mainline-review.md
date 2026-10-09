# C05-B10 — bounded source-span and mainline delta review

Status: READY FOR REVIEW. This is an offline code/evidence checkpoint, not
admission of C05-B9's cleaning candidate or a new model-quality result. The
reviewed range is merged `main` `c32381d434dc1c6f0b3bbf1754208f2c38c15725`
through `codex/shared-corpus-cleaning` `b5a784f78f1009da0a99fc0bd12f52fe30a967e4`,
plus the focused corrections in this review. Fetched refs still had those exact
heads; the checkout was clean before edits and no model run was active.

## Whole-delta review and corrections

The inherited physical-LF JSONL reader is used by admission, inventory, build,
release and family-freeze paths. Its retained physical row and source hash
semantics cover embedded U+0085/U+2028/U+2029, CRLF and the final non-LF row.
Reviewed admission and family aliases preserve old source/family/split bindings;
their retained C05-B7 result is the specific preparation input for C05-B8.
The fixed-slice profile binds held-out document/family/content/tokenizer offsets,
uses next-token targets and selects on validation before test. Generation reserves
worst-case full-prefix inputs under an active attempt allocation and reports
actual cache behavior; operational validation and scoring share the durable
forward-input ledger. The C05-B8 runtime result is bound to its earlier `f80079b`
checkout and pinned inputs. It does not qualify the later normalizer or export
convenience code.

Three concrete compatibility/correctness issues were corrected without changing
historical releases or scientific settings:

1. Native split inventory now passes the declared release normalizer to the
   ordinary parser. The previous v1 default yielded incompatible document IDs
   for a prospective v3 build. An absent field still gives v1 IDs and hashes.
2. Ordinary and large deduplication now fail closed when one connected duplicate
   component has both metadata-only and meaningful-text cleaning decisions. It
   cannot silently choose a metadata-only representative and drop useful
   configuration. A decision on such a component requires source review; the
   existing C05-B9 build audit has 14 raw/normalized duplicate groups over 14
   distinct document IDs and no discordant cleaning decision.
3. A fixed-continuation accounting label can be present without a ledger for
   standalone use. If the optional forward allocation is active, an unlabelled
   request still fails and labelled requests still precharge the durable caps.
   A labelled request under a legacy ledger without that allocation fails before
   decoding rather than silently bypassing accounting.

The optional v2/v3 normalizer retains versioned document/release IDs, raw spans,
removal reasons and cold replay. The absent normalizer field preserves legacy
declaration bytes; `corpus verify-export` only verifies an existing bound export.
No cache path is independently overridden. The shared export fixture has two
different run configurations against one authenticated export and no data copy.

## Bounded source-span review

The review ranked flagged spans by historical **scheduled** target overlap from
the retained 25M mixture, then inspected 24 deterministic train documents:
three source locations for each of six high-exposure signature groups and six
additional high-exposure contexts. This is a bounded sample, not a source-wide
quality certification. Retained raw Wikimedia rows and v3 structural spans show
the flags preceded the candidate normalizer.

| Flag and raw source location | Finding |
| --- | --- |
| Wikimedia `wikimedia-0027.json.gz.sample.jsonl#row=10903`, lines 10–16; also rows 12832 and 42369 | Orphan `* (for Chronology of Bishops) $[self-published]$` citation placeholders repeat 108 times in 40 train documents, about 1,772 historical supervised targets. The original URL is absent. This is a concrete upstream extraction defect, about 0.0071% of 25M targets. |
| Scoutflo `Sentry Playbooks/01-Error-Tracking/ConsumerError-ConnectionError-Kafka-Error-application.md#lines=11-30` | Repeated event-frequency and comparison steps are useful incident procedure, not safe boilerplate removal. |
| Wikimedia rows 33141, 48202 and 36398 | Wildcard/qualifier lists, `!colspan` season labels and `{{legend}}` map percentages contain meaningful structure even where markup is rough. |
| PagerDuty `docs/training/courses/incident_response.md#lines=1028-1044` | `</label>` and image markup surround a caption and useful response prose; deleting the whole span would lose content. |

No broad stripping is justified. A future source-specific rule could target the
orphan citation under the recognized external-links section, but the small
exposure and other variants do not justify a fourth normalizer and fresh
candidate audit in this checkpoint. The orphan remains flagged; the C05-B9 v3
candidate stays NOT ADMITTED and its release/manifest hashes stay unchanged.
The accepted C05-B7 release, C05-B8 mixed result, C05-N1 0/200 reader result,
failed attempts and Card 06 block are unchanged.

## Verification boundary

An explicitly selected 64-node offline suite covered the affected corpus,
family, policy, cleaning and forward-accounting paths. The two synthetic
forward-bearing `tests/test_fixed_slices.py` nodes were excluded; its remaining
selected nodes use file fixtures and a mocked generation callable. The new
standalone request regression stops before decoder/model access. Repository Ruff
rule/format checks, research lint, `git diff --check` and the inspected ordinary
PR-safe 20-node suite passed. The accepted release `67d727a0...`, narrower v2
candidate `ebb0f253...` and v3 candidate `6e63d395...` each passed native cold
verification under the corrected code. No broad pytest
collection, model forward, generation, optimizer update, tokenizer fit, data
acquisition, GPU job or CI dispatch was performed locally.

This review does not qualify live ROCm, long training, real panel parity,
reviewer independence or the prospective candidate's scientific usefulness.
The next bounded scientific decision remains a separately approved data and
training proposal; neither this review nor mainline integration grants runtime.
