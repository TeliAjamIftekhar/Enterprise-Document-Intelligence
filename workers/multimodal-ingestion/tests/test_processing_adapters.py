from __future__ import annotations

import pytest

from processing_adapters import (
    UnsupportedProcessingAdapter,
    adapter_runtime_metadata,
    resolve_processing_adapter,
)


@pytest.mark.parametrize(
    (
        "subject",
        "language",
        "script",
        "expected_key",
    ),
    [
        (
            "english",
            "english",
            "latin",
            "english-general",
        ),
        (
            "mathematics",
            "english",
            "latin",
            "mathematics",
        ),
        (
            "hindi",
            "hindi",
            "devanagari",
            "hindi",
        ),
        (
            "sanskrit",
            "sanskrit",
            "devanagari",
            "sanskrit",
        ),
        (
            "urdu",
            "urdu",
            "arabic",
            "urdu",
        ),
    ],
)
def test_representative_adapter_routing(
    subject: str,
    language: str,
    script: str,
    expected_key: str,
) -> None:
    adapter = resolve_processing_adapter(
        subject=subject,
        language=language,
        script=script,
    )

    assert adapter.key == expected_key


def test_mathematics_profile_has_priority() -> None:
    adapter = resolve_processing_adapter(
        subject="general",
        language="english",
        script="latin",
        processing_profile=(
            "mathematics-multimodal"
        ),
    )

    assert adapter.key == "mathematics"
    assert adapter.expected_language == "Mathematics"
    assert (
        adapter.preserve_mathematical_notation
        is True
    )


def test_arabic_script_routes_to_urdu() -> None:
    adapter = resolve_processing_adapter(
        subject="language",
        language="",
        script="arabic",
    )

    assert adapter.key == "urdu"
    assert adapter.reading_direction == "rtl"


def test_runtime_metadata_is_dashboard_ready() -> None:
    metadata = adapter_runtime_metadata({
        "subject": "sanskrit",
        "language": "sanskrit",
        "script": "devanagari",
        "processing_profile": (
            "multilingual-language"
        ),
    })

    assert metadata["adapter_key"] == "sanskrit"
    assert (
        metadata["representative_book_id"]
        == "grade-6-sanskrit-deepakam"
    )
    assert metadata["expected_script"] == "devanagari"
    assert metadata["reading_direction"] == "ltr"


def test_unsupported_metadata_fails_safely() -> None:
    with pytest.raises(
        UnsupportedProcessingAdapter
    ):
        resolve_processing_adapter(
            subject="unknown",
            language="unknown",
            script="unknown",
        )
