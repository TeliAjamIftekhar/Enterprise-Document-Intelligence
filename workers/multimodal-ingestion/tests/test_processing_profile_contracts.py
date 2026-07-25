from __future__ import annotations

import json

import pytest

from processing_adapters import (
    adapter_runtime_metadata,
    resolve_processing_adapter_for_book,
)
from scripts.prepare_embedding_records import (
    build_text_embedding_text,
    is_low_information_text,
)


PROFILE_CASES = [
    (
        {
            "book_id": "english-test",
            "subject": "Science",
            "language": "English",
            "script": "latin",
            "processing_profile": (
                "multilingual-language"
            ),
        },
        {
            "adapter_key": "english-general",
            "expected_language": "English",
            "expected_script": "latin",
            "reading_direction": "ltr",
            "processing_profile": (
                "multilingual-language"
            ),
            "title_fallback_label": "Unit",
            "preserve_unicode": True,
            "preserve_mathematical_notation": False,
        },
        "Water conservation",
    ),
    (
        {
            "book_id": "mathematics-test",
            "subject": "Mathematics",
            "language": "English",
            "script": "mixed",
            "processing_profile": (
                "mathematics-multimodal"
            ),
        },
        {
            "adapter_key": "mathematics",
            "expected_language": "Mathematics",
            "expected_script": "mixed",
            "reading_direction": "ltr",
            "processing_profile": (
                "mathematics-multimodal"
            ),
            "title_fallback_label": "Unit",
            "preserve_unicode": True,
            "preserve_mathematical_notation": True,
        },
        "२x + ३y = १२",
    ),
    (
        {
            "book_id": "hindi-test",
            "subject": "Hindi",
            "language": "Hindi",
            "script": "devanagari",
            "processing_profile": (
                "multilingual-language"
            ),
        },
        {
            "adapter_key": "hindi",
            "expected_language": "Hindi",
            "expected_script": "devanagari",
            "reading_direction": "ltr",
            "processing_profile": (
                "multilingual-language"
            ),
            "title_fallback_label": "इकाई",
            "preserve_unicode": True,
            "preserve_mathematical_notation": False,
        },
        "जल संरक्षण",
    ),
    (
        {
            "book_id": "sanskrit-test",
            "subject": "Sanskrit",
            "language": "Sanskrit",
            "script": "devanagari",
            "processing_profile": (
                "multilingual-language"
            ),
        },
        {
            "adapter_key": "sanskrit",
            "expected_language": "Sanskrit",
            "expected_script": "devanagari",
            "reading_direction": "ltr",
            "processing_profile": (
                "multilingual-language"
            ),
            "title_fallback_label": "इकाई",
            "preserve_unicode": True,
            "preserve_mathematical_notation": False,
        },
        "संस्कृतम्",
    ),
    (
        {
            "book_id": "urdu-test",
            "subject": "Urdu",
            "language": "Urdu",
            "script": "arabic",
            "reading_direction": "rtl",
            "processing_profile": (
                "multilingual-language"
            ),
        },
        {
            "adapter_key": "urdu",
            "expected_language": "Urdu",
            "expected_script": "arabic",
            "reading_direction": "rtl",
            "processing_profile": (
                "multilingual-language"
            ),
            "title_fallback_label": "اکائی",
            "preserve_unicode": True,
            "preserve_mathematical_notation": False,
        },
        "ہماری دنیا",
    ),
]


@pytest.mark.parametrize(
    (
        "book",
        "expected",
        "meaningful_text",
    ),
    PROFILE_CASES,
)
def test_processing_profile_contract(
    book: dict,
    expected: dict,
    meaningful_text: str,
) -> None:
    adapter = (
        resolve_processing_adapter_for_book(
            book
        )
    )

    metadata = adapter_runtime_metadata(
        book
    )

    assert adapter.key == (
        expected["adapter_key"]
    )

    for field, expected_value in (
        expected.items()
    ):
        assert metadata[field] == expected_value

    assert (
        is_low_information_text(
            meaningful_text
        )
        is False
    )

    json.dumps(
        metadata,
        ensure_ascii=False,
    )


@pytest.mark.parametrize(
    "page_marker",
    [
        "123",
        "१२३",
        "۱۲۳",
        "--- १२३ ---",
    ],
)
def test_numeric_page_markers_are_excluded(
    page_marker: str,
) -> None:
    assert (
        is_low_information_text(
            page_marker
        )
        is True
    )


def test_mixed_profile_embedding_text_is_lossless() -> None:
    original = (
        "Water conservation | "
        "जल संरक्षण | "
        "संस्कृतम् | "
        "ہماری دنیا | "
        "क्षेत्रफल = πr² | "
        "२x + ३y = १२"
    )

    result = build_text_embedding_text({
        "search_text": original,
    })

    assert result == original
