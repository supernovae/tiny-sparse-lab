# Continuation/source overlap diagnostic

`src/sparselab/corpus/memorization.py` exposes a **read-only, descriptive** comparison of one generated continuation to known source passages. It does not decide whether text is memorized, licensed, infringing, eligible for training, or releasable; there is no copyright threshold and no corpus-construction gate.

There is currently no native CLI or DSL operation for this diagnostic. It is
an implementation capability, not a runnable lab workflow. The
[native diagnostic backlog](../TODO.md#native-diagnostic-interfaces) specifies
the required adapter; do not substitute a custom experiment script.

Each `SourceMatch` contains the caller's source ID, SHA-256 of the **original UTF-8 passage bytes**, longest exact contiguous normalized substring (text and character count), and overlap fractions between 0 and 1. `MemorizationDiagnostic.sources` includes every source, sorted by ID. Best-source selection is deterministic: longest exact substring, then n-gram, word, character and available edit similarity, with lexicographically smallest ID breaking ties. Duplicate or empty IDs and an empty passage list are rejected.

Normalization is `NFC+casefold+whitespace-collapse-v1`: Unicode NFC, Unicode casefold, then Unicode whitespace splitting and joining with single spaces (trims edges). Punctuation remains; the returned exact excerpt is **normalized**, not a byte-offset into the source. Character and word overlap are multiset recall relative to continuation characters and Unicode `\w+` words. Configurable `ngram_size` (1–10; default 3) computes the fraction of **distinct continuation word n-grams** appearing in the passage; if no n-grams exist the fraction is zero. Empty continuation character/word denominators yield zero. An empty/short continuation can therefore produce zero overlap without implying non-memorization. Common words, repeated characters and generic phrasing may yield overlap without copying; paraphrases can show low exact overlap despite semantic similarity.

Edit similarity is `1 - Levenshtein_distance / max(normalized_character_lengths)` for the whole continuation and whole passage, not the best matching window. It is `None` when **either** normalized string exceeds `max_edit_chars` (default 512, allowed 1–2048) to bound quadratic work. Longest exact substring and other overlap metrics remain available for longer passages. Scores from passages of different lengths or sampling policies are not a calibrated probability; retain passage IDs, raw digests, model/checkpoint identity and sampling context when reporting observations.
