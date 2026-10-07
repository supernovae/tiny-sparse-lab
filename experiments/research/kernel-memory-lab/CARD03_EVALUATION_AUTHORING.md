# Card 03 evaluation authoring workflow

Status: **tooling only; zero authored or frozen project items**. The owner-approved
preparation attempt includes item authoring and review within an aggregate
20-agent-reviewer-hour ceiling. A complete 600-item suite is conditional on
actual, rights-admitted, family-separated held-out content and review. At 600
items, that ceiling averages two minutes per item before chunk/rubric checks;
stop and report an incomplete suite if review cannot be completed honestly.

Use `sparselab.evaluation.kml_card03_items.freeze_card03_items` after a cold
verified corpus release and frozen `family_inventory` exist. Its input is a
JSON draft with `schema_version: 1`, `protocol:
kml-card03-evaluation-v1`, the verified `release_id`,
`family_inventory_sha256`, one shared `decoder`, a positive shared
`evidence_token_budget`, `chunks`, and `items`. Every chunk names a release
document, Unicode character offsets into its exact normalized text, the
SHA-256 of that text slice, and an ID equal to the canonical SHA-256 of its
remaining fields. Only test-family chunks are accepted. This binds evidence
text to authenticated release documents; it never invents a source or chunk ID.

Each item declares an ID, `closed_book` or `open_book` suite, exact Card 03
category, `split: test`, parent family and document IDs, a source-version map
from those parents, question, answerability, required and prohibited atomic
claims, acceptable paraphrases, optional numeric tolerance, support chunk IDs,
optional precedence rule, clarification/abstention target, maximum answer
words, reviewer, `review_status: reviewed`, all four evidence conditions, and
a canonical content SHA-256 of all other item fields. Closed-book items mark
all evidence controls inapplicable (`null`). Answerable open-book items use an
empty no-evidence condition, exact gold support IDs, and distinct verified
held-out chunks for plausible-wrong and shuffled/absent controls. Unanswerable
open-book items mark all four conditions inapplicable and declare a target.

The default freeze requires **20 items in each of ten closed-book categories**
and **40 in each of ten open-book categories**. It records separate missing and
clarification denominators in the final category. `require_complete=False` is
for validating an explicitly incomplete draft; its output records
`complete_denominators: false` and cannot count as the Card 03 frozen suite.
Publication is exclusive and content-addressed. Keep candidate, source and
review receipts in the persistent work root; check in only small immutable
summaries and artifact references.

The validator checks structure, identity, split and evidence binding. It
cannot judge whether a question is useful, an answer is semantically correct,
a plausible-wrong passage is plausible, or a near paraphrase leaks into train.
Review those properties, compare prompts/answers against train and tokenizer
inputs, and freeze the item denominator **before** inspecting model outputs.
Bind later raw-text oracle, lexical retrieval and integrated reader conditions
to this same chunk set, query, decoder and evidence-token budget. No model
score or Card 03 acceptance is implied by this tooling.
