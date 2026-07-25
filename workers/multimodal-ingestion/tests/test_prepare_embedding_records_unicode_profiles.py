from __future__ import annotations

import pytest

from scripts.prepare_embedding_records import (
    build_text_embedding_text,
    is_low_information_text,
    normalize_whitespace,
)


@pytest.mark.parametrize(
    "text",
    [
        "Water conservation",
        "जल संरक्षण",
        "संस्कृतम्",
        "ہماری دنیا",
        "x² + y² = z²",
        "२x + ३y = १२",
    ],
)
def test_meaningful_multilingual_text_is_not_low_information(
    text: str,
) -> None:
    assert is_low_information_text(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "123",
        "१२३",
        "۱۲۳",
        "  123  ",
        "--- १२३ ---",
    ],
)
def test_numeric_page_markers_remain_low_information(
    text: str,
) -> None:
    assert is_low_information_text(text) is True


def test_unicode_whitespace_normalization_preserves_text() -> None:
    text = (
        "जल   संरक्षण  संस्कृतम्   "
        "ہماری دنیا"
    )

    assert normalize_whitespace(text) == (
        "जल संरक्षण संस्कृतम् ہماری دنیا"
    )


def test_embedding_text_preserves_unicode_and_math_notation() -> None:
    content_unit = {
        "search_text": (
            "क्षेत्रफल = πr²; "
            "२x + ३y = १२; "
            "ہماری دنیا"
        ),
    }

    result = build_text_embedding_text(
        content_unit
    )

    assert "क्षेत्रफल" in result
    assert "πr²" in result
    assert "२x + ३y = १२" in result
    assert "ہماری دنیا" in result


def test_raw_text_is_preserved_when_search_text_is_missing() -> None:
    content_unit = {
        "raw_text": (
            "संस्कृतम् तथा اردو عبارت"
        ),
    }

    assert build_text_embedding_text(
        content_unit
    ) == "संस्कृतम् तथा اردو عبارت"
