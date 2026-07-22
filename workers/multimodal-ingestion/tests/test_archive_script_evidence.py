from collections import Counter

from inspect_textbook_source_archive import (
    assess_script_evidence,
)


def test_weak_conflicting_text_routes_to_visual_bda():
    result = assess_script_evidence(
        expected_script="arabic",
        script_counts=Counter({
            "latin": 39,
        }),
        total_pages=120,
    )

    assert (
        result["raw_detected_script"]
        == "latin"
    )

    assert result[
        "detected_script"
    ] == "unknown"

    assert result[
        "script_verification"
    ] == "unverified"

    assert result[
        "extraction_route"
    ] == "visual-bda"

    assert result[
        "text_evidence_status"
    ] == "insufficient"


def test_strong_conflicting_text_requires_review():
    result = assess_script_evidence(
        expected_script="arabic",
        script_counts=Counter({
            "latin": 5000,
        }),
        total_pages=120,
    )

    assert result[
        "detected_script"
    ] == "latin"

    assert result[
        "script_verification"
    ] == "mismatch"

    assert result[
        "extraction_route"
    ] == "manual-review"


def test_expected_script_is_verified():
    result = assess_script_evidence(
        expected_script="arabic",
        script_counts=Counter({
            "arabic": 900,
            "latin": 10,
        }),
        total_pages=20,
    )

    assert result[
        "detected_script"
    ] == "arabic"

    assert result[
        "script_verification"
    ] == "verified"

    assert result[
        "extraction_route"
    ] == "text-layout"


def test_empty_text_routes_to_visual_bda():
    result = assess_script_evidence(
        expected_script="devanagari",
        script_counts=Counter(),
        total_pages=50,
    )

    assert result[
        "detected_script"
    ] == "unknown"

    assert result[
        "script_verification"
    ] == "unverified"

    assert result[
        "extraction_route"
    ] == "visual-bda"
