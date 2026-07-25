from __future__ import annotations

import pytest

from title_safety import (
    finalize_inferred_title,
    safe_title_text,
    title_rejection_reasons,
)


@pytest.mark.parametrize(
    "title",
    [
        "Patterns in Mathematics",
        "How I Taught My Grandmother to Read",
        "जल ही जीवन है",
        "सुभाषितानि",
        "ہماری دنیا",
    ],
)
def test_valid_multilingual_titles_are_safe(
    title: str,
) -> None:
    assert safe_title_text(
        title,
        book_title="Different Book",
        grade=6,
    )


def test_repeated_book_header_is_rejected() -> None:
    reasons = title_rejection_reasons(
        "Poorvi—Grade 6",
        book_title="Poorvi",
        grade=6,
    )

    assert "repeated_book_header" in reasons


def test_long_body_sentence_is_rejected() -> None:
    reasons = title_rejection_reasons(
        (
            "Neem is a common tree in our country. "
            "Discuss in groups of four and fill "
            "the following table."
        ),
        book_title="Poorvi",
        grade=6,
    )

    assert "too_many_words" in reasons
    assert "multiple_sentence_markers" in reasons


def test_weak_source_sentence_fragment_uses_fallback() -> None:
    result = finalize_inferred_title(
        {
            "title": (
                "II Neem is a common tree in our "
                "country. Discuss in groups of four "
                "and fill"
            ),
            "confidence": "high",
            "source": "line-after-heading",
            "candidates": [
                "Neem Baba",
                "Nurturing Nature",
                (
                    "II Neem is a common tree in our "
                    "country. Discuss in groups of "
                    "four and fill"
                ),
            ],
        },
        fallback="Unit 3",
        book_title="Poorvi",
        grade=6,
    )

    assert result["title"] == "Unit 3"
    assert result["confidence"] == "low"
    assert result["used_fallback"] is True
    assert result["rejected_title"].startswith(
        "II Neem"
    )
    assert (
        "weak_source_sentence_fragment"
        in result["rejection_reasons"]
    )
    assert "Neem Baba" in result["candidates"]
    assert "Nurturing Nature" in result["candidates"]


def test_incomplete_title_is_rejected() -> None:
    reasons = title_rejection_reasons(
        "Hamara Bharat —",
        book_title="Poorvi",
        grade=6,
    )

    assert (
        "incomplete_trailing_punctuation"
        in reasons
    )


def test_weak_source_cannot_claim_high_confidence() -> None:
    result = finalize_inferred_title(
        {
            "title": "A Bottle of Dew",
            "confidence": "high",
            "source": "line-after-heading",
            "candidates": ["A Bottle of Dew"],
        },
        fallback="Unit 1",
        book_title="Poorvi",
        grade=6,
    )

    assert result["title"] == "A Bottle of Dew"
    assert result["confidence"] == "medium"
    assert result["used_fallback"] is False


def test_layout_title_can_remain_high_when_safe() -> None:
    result = finalize_inferred_title(
        {
            "title": "Patterns in Mathematics",
            "confidence": "high",
            "source": "first-page-layout-title",
            "candidates": [
                "Patterns in Mathematics"
            ],
        },
        fallback="Unit 1",
        book_title="Ganita Prakash",
        grade=6,
    )

    assert result["confidence"] == "high"
    assert result["used_fallback"] is False


def test_unsafe_title_uses_stable_fallback() -> None:
    result = finalize_inferred_title(
        {
            "title": "Poorvi—Grade 6",
            "confidence": "high",
            "source": "first-page-layout-title",
            "candidates": [
                "Poorvi—Grade 6",
                "A Bottle of Dew",
            ],
        },
        fallback="Unit 1",
        book_title="Poorvi",
        grade=6,
    )

    assert result["title"] == "Unit 1"
    assert result["confidence"] == "low"
    assert result["used_fallback"] is True
    assert result["rejected_title"] == (
        "Poorvi—Grade 6"
    )
    assert (
        "repeated_book_header"
        in result["rejection_reasons"]
    )
    assert result["candidates"] == [
        "Poorvi—Grade 6",
        "A Bottle of Dew",
    ]


def test_missing_title_uses_fallback() -> None:
    result = finalize_inferred_title(
        None,
        fallback="اکائی 2",
        book_title="Shahnai",
        grade=1,
    )

    assert result["title"] == "اکائی 2"
    assert result["used_fallback"] is True
    assert result["rejection_reasons"] == [
        "missing_title"
    ]
