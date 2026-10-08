# Card 03 independent item review receipts

This schema supplements the bound evaluation continuation protocol. It records
review decisions; a `reviewed` field alone is never evidence of semantic
approval.

Each author owns 200 candidate IDs and writes a draft item file, any new chunk
file, and a one-row-per-ID decision log identifying keep/rewrite/replace, the
source checked, candidate/final hashes and reason. Draft items retain
`review_status: draft` and `reviewer: PENDING-INDEPENDENT-REVIEW`.

The root creates a gold-blinded worksheet from the exact authored draft. A
different agent writes `blind_solutions.jsonl` **before opening that author's
gold/controls**. Each row has exactly `id`, `draft_item_sha256`,
`independent_answer`, `support_used` (chunk IDs, possibly empty for closed-book
or abstention), and `reasoning_note`. A nonempty answer or explicit
abstention/clarification is required. Preserve this file and its hash.

After blind solutions are sealed, the reviewer checks exact source text,
category, atomic gold and prohibited claims, alternatives, answerability,
controls, held-out lineage, leakage flags and tokenizer context fit. Each
`review_decisions.jsonl` row has `id`, `author`, `reviewer`,
`draft_item_sha256`, `blind_row_sha256`, `verdict` (`pass`, `revise`, `reject`),
`checks`, `notes`, and `reviewed_item_sha256` (null unless passed). `checks`
has Boolean keys `source_support`, `category_fit`, `answer_rubric`,
`alternatives`, `controls`, `lineage`, `leakage`, `context_fit`, and
`question_clarity`; all must be true for `pass`. A two-source pass also has
`both_sources_required: true`; a missing/ambiguous-evidence pass also has
`whole_parent_absence_checked: true`. `notes` names the facts checked, not
merely “looks good.” The reviewer may reject or request a revision; they may
not silently edit semantic fields.

For each pass, the reviewer publishes a copy of the authored item with only
`reviewer`, `review_status: reviewed`, and the recomputed native
`content_sha256` changed. `reviewed_item_sha256` binds this exact final copy.
The root verifies all 200 draft/final semantic fields match, author and
reviewer differ, the blind solution matches the draft hash, every check passes,
and the final item hash matches the decision. Any author revision changes the
draft hash and needs a new blind solution and review. Preserve old negative
decisions and disagreements; a later pass does not erase them.

Review rotation: language author reviews evidence-b; evidence-a author reviews
language; evidence-b author reviews evidence-a. Only 600 independently passed
final items can enter the native freeze.
