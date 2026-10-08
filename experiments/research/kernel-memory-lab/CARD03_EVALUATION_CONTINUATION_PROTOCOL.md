# Card 03 evaluation continuation protocol

Status: execution protocol for the owner's one offline evaluation-only
continuation from `2366df7`; it does not alter the accepted 200/400 item counts,
category denominators, source policy, model architecture or training authority.
The unreviewed C03-C1 candidate bank is an input for item-by-item decisions, not
a frozen suite.

Use the cold-verified release
`e893cb2e3c65f7f6360f73e4daa159dcec5da1b4c2c6f911ff5d9d549b815818`,
family inventory `f3b506be3dcd9b99401c5dae72f6ef16b51441394e9a485913200e7eeaad1af5`,
train-only tokenizer `308b33a6edbed613f3caf5232b124a7c6105a7a3dc273c999725be51c3ffeaa9`,
and sealed prepared inputs `aaa260d41ead37c6b347e4ec49cf57b16b6fc8c532820cc9a588f5d11bd47852`.
Do not refit, repartition, reacquire, run an optimizer or inspect model outputs.

The model context is 1,024 tokens. The C03-C1 candidate's provisional 2,048
evidence-token allowance cannot fit and is replaced before any model result by
a shared **512-token evidence allowance** and **128-token generation cap**. Use
greedy decoding at temperature zero. Render closed-book prompts as
`Question: {question}\nAnswer:`. Render each open-book condition as
`Question: {question}\nEvidence:\n[1] {first exact chunk}\n...\nAnswer:`;
render the no-evidence condition with `Evidence:\n(none)`. The ordinal labels
map to frozen chunk IDs in the item manifest. Count tokens on the *complete*
rendered prompt using the frozen tokenizer, add one BOS token and reserve all
128 generation tokens. Every condition must total at most 1,024, its selected
evidence text at most 512, and its question at most 64 tokenizer tokens. Record
the measured counts per final item and condition. A failing item is rewritten,
not truncated or waived.

Three disjoint author partitions cover all 600 existing candidate IDs. For
each ID, preserve a decision that names keep/rewrite/replace, the source
passage checked, the author, candidate hash and final item hash. A different
agent blind-solves the question from its exact final visible context *before*
seeing the author's gold claims, then compares its answer with every required
and prohibited atomic claim, acceptable paraphrase, category, answerability,
parent/version binding and controls. Record the blind answer, verdict and
specific correction or disagreement per item. If the author revises an item,
the independent reviewer must solve and check the new hash; an earlier review
does not transfer. Unresolved disagreements keep the item draft.

For answerable items, the cited gold passage must actually entail the answer.
Conditional, exception, negation, version and contradiction questions must
name the operative condition or incompatible claims. Two-source items must
need facts from both independently held-out families. For missing/ambiguous
items, inspect the complete parent source for the purported absence and write
an explicit abstention or clarification target. Plausible-wrong and
shuffled/absent evidence must fit the shared budget, not entail the gold answer,
and serve their declared controls. Keep questions short and meaningful for the
341M model. Do not manufacture a category merely to meet its denominator.

Before native freeze, cold-check release lineage, all 600 final item hashes,
author/reviewer separation and decision receipts, exact 20-by-10 and 40-by-10
counts, held-out parent/chunk identities, exact/fuzzy train and tokenizer-input
overlap flags, and the tokenizer context counts. Review every flagged leakage
case; ordinary answer words in training prose alone are not a leak. Use
`freeze_card03_items(..., require_complete=True)` only after all 600 final
items pass independent semantic review and mechanical checks. An unreviewed
or structurally passing draft cannot be substituted for that gate.
