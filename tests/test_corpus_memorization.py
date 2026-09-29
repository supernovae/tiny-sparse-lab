"""Descriptive continuation/source comparisons do not make policy decisions."""

from __future__ import annotations

import hashlib

import pytest

from sparselab.corpus.memorization import SourcePassage, diagnose_memorization


def test_copied_continuation_identifies_source_and_raw_passage_digest() -> None:
    passage = "A valve opens slowly; stop when pressure reaches 2 bar."
    continuation = "Stop when pressure reaches 2 bar."
    result = diagnose_memorization(
        continuation,
        [SourcePassage("manual", passage), SourcePassage("other", "A bird sings.")],
    )
    assert result.best_source.source_id == "manual"
    assert (
        result.best_source.passage_sha256
        == hashlib.sha256(passage.encode()).hexdigest()
    )
    assert result.best_source.longest_exact_match == continuation.casefold()
    assert result.best_source.longest_exact_match_chars == len(continuation)
    assert result.best_source.word_overlap == 1.0
    assert result.best_source.ngram_overlap == 1.0
    assert result.best_source.edit_similarity is not None
    assert result.best_source.edit_similarity < 1.0  # passage has a preceding clause


def test_paraphrase_has_word_overlap_without_a_long_exact_match() -> None:
    result = diagnose_memorization(
        "The small cat quickly moved indoors.",
        [SourcePassage("source", "A cat small and quick entered the house.")],
        ngram_size=2,
    ).best_source
    assert 0 < result.word_overlap < 1
    assert result.ngram_overlap == 0
    assert result.longest_exact_match_chars < len("small cat quickly moved indoors")


def test_unrelated_continuation_has_zero_word_and_ngram_overlap() -> None:
    match = diagnose_memorization(
        "galaxies orbit",
        [SourcePassage("unrelated", "plants bloom")],
        ngram_size=2,
    ).best_source
    assert match.word_overlap == match.ngram_overlap == 0
    assert match.longest_exact_match_chars < len("galaxies orbit")


def test_normalization_preserves_punctuation_and_bounds_edit_work() -> None:
    original = "CAFÉ!\r\n Next"
    result = diagnose_memorization(
        "  cafe\u0301!   next ",
        [SourcePassage("normalized", original)],
        ngram_size=2,
        max_edit_chars=10,
    ).best_source
    assert result.longest_exact_match == "café! next"
    assert result.ngram_overlap == result.character_overlap == 1
    assert result.edit_similarity == 1.0  # exactly 10 normalized characters
    assert result.passage_sha256 == hashlib.sha256(original.encode()).hexdigest()
    assert (
        diagnose_memorization(
            "  cafe\u0301!   next ",
            [SourcePassage("normalized", original)],
            max_edit_chars=9,
        ).best_source.edit_similarity
        is None
    )
    assert (
        diagnose_memorization(
            "a" * 3000, [SourcePassage("long", "a" * 3000)]
        ).best_source.longest_exact_match_chars
        == 3000
    )
    assert (
        diagnose_memorization(
            "a" * 3000, [SourcePassage("long", "a" * 3000)]
        ).best_source.edit_similarity
        is None
    )


def test_best_source_order_is_deterministic_on_ties() -> None:
    sources = [SourcePassage("z", "the same text"), SourcePassage("a", "the same text")]
    first = diagnose_memorization("same text", sources)
    second = diagnose_memorization("same text", list(reversed(sources)))
    assert first == second
    assert first.best_source.source_id == "a"
    assert tuple(item.source_id for item in first.sources) == ("a", "z")


def test_invalid_bounds_and_source_identifiers_fail() -> None:
    source = SourcePassage("a", "some text")
    with pytest.raises(ValueError, match="source passage"):
        diagnose_memorization("text", [])
    with pytest.raises(ValueError, match="unique"):
        diagnose_memorization("text", [source, source])
    with pytest.raises(ValueError, match="ngram_size"):
        diagnose_memorization("text", [source], ngram_size=0)
    with pytest.raises(ValueError, match="max_edit_chars"):
        diagnose_memorization("text", [source], max_edit_chars=2049)
