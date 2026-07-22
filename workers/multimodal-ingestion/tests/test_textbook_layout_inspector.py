from inspect_textbook_layout import (
    count_scripts,
    detect_script_profile,
    normalize_text,
)


def test_detects_devanagari_ltr() -> None:
    counts = count_scripts(
        "मीना का परिवार"
    )

    profile = detect_script_profile(
        counts,
        book_id="grade-1-hindi-sarangi",
    )

    assert profile["dominant_script"] == (
        "devanagari"
    )
    assert profile["direction"] == "ltr"
    assert profile["language_hint"] == (
        "hindi"
    )


def test_detects_urdu_rtl() -> None:
    counts = count_scripts(
        "اردو کی کتاب"
    )

    profile = detect_script_profile(
        counts,
        book_id="grade-1-urdu-rimjhim",
    )

    assert profile["dominant_script"] == (
        "arabic"
    )
    assert profile["direction"] == "rtl"
    assert profile["language_hint"] == (
        "urdu"
    )


def test_detects_latin_ltr() -> None:
    counts = count_scripts(
        "English Textbook"
    )

    profile = detect_script_profile(
        counts,
        book_id="grade-1-english-mridang",
    )

    assert profile["dominant_script"] == (
        "latin"
    )
    assert profile["direction"] == "ltr"


def test_normalizes_zero_width_text() -> None:
    assert normalize_text(
        "पाठ्यपुस्\u200dतक"
    ) == "पाठ्यपुस्तक"
