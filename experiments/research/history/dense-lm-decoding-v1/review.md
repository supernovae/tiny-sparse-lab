# Blinded independent-text review

This protocol consumes the sealed **test** outputs only after the preregistered decoder selection. It does not select decoders or alter weights. The independent [test prompts](test.json) have 55 cases across eleven categories; the deterministic subset for manual review is the two lexicographically first prompt IDs per category (22 prompts), selected before examining outcomes. The runner prepares three matched pair types per prompt and training seed: 30M versus 50M with the frozen selected decoder and generation seed 11; and greedy versus the selected decoder within each model using generation seed 11. This is 22 × 3 × 3 = 198 pairs, all retained, with no favorable-generation selection. The second sampled generation seed 29 is preserved in full evidence but not in this bounded manual bundle.

The ignored workspace `sparselab-work/experiments/dense-lm-decoding-v1/review/blind.json` contains prompt, anonymized A and B text, opaque pair ID and dimensions. `key.json` is separate and must not be shown to reviewers until all judgments are sealed; it maps each pair to width, training seed, decoder, checkpoint and sampling seed. `test_sha256.json` binds the six complete test files. Reviewers should not be told which condition is expected to improve. They must see both completions with the *same* prompt, preserve complete text, and decide each dimension independently:

| Dimension | Pairwise question |
|---|---|
| Prompt adherence | Which completion follows the provided setup rather than switching tasks/story? |
| Entities | Which keeps established characters and their roles without unexplained substitutions? |
| Stated attributes | Which retains explicit colors and object properties? |
| Temporal/causal consistency | Which respects sequence, location, and simple cause/effect? |
| Repetition | Which avoids obvious repeated sentences/loops without penalizing legitimate referents? |
| Local readability | Which is easier to follow sentence by sentence? |

For each dimension record `A`, `B`, `tie`, or `uncertain`, with optional quoted rationale. `tie` means no discernible difference; `uncertain` means neither can be assessed confidently. Keep every independent reviewer row (pair ID, reviewer ID, six votes, quotations) even if reviewers disagree; do not force consensus, conflate dimensions, infer absent judgments, or replace review with loss or mechanical counters. Adjudication, if ever done, must be declared as a separate review decision after preserving original votes. The bundled `judgments: []` is an empty review form, **not** a completed evaluation. Until independent humans submit sealed votes, subjective model-versus-decoder text quality is `UNAVAILABLE` and no preference rate may be reported.
