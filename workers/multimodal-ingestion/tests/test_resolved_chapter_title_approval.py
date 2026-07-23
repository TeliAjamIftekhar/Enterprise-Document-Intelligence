from generate_book_config_from_inspection import (
    apply_resolved_chapter_title,
)


def inferred_title():
    return {
        "title": "Wrong inferred title",
        "confidence": "medium",
        "source": "first-title-candidate",
        "candidates": ["Wrong inferred title"],
    }


def test_single_approved_chapter_overrides_title():
    result = apply_resolved_chapter_title(
        inferred_title(),
        [{
            "chapter_title": (
                "Data Handling and Presentation"
            ),
        }],
    )

    assert result == {
        "title": (
            "Data Handling and Presentation"
        ),
        "confidence": "high",
        "source": (
            "resolved-chapter-structure"
        ),
        "candidates": [
            "Data Handling and Presentation"
        ],
    }


def test_no_approved_chapter_preserves_inference():
    inferred = inferred_title()

    result = apply_resolved_chapter_title(
        inferred,
        [],
    )

    assert result is inferred


def test_multiple_chapters_preserve_inference():
    inferred = inferred_title()

    result = apply_resolved_chapter_title(
        inferred,
        [
            {"chapter_title": "Chapter One"},
            {"chapter_title": "Chapter Two"},
        ],
    )

    assert result is inferred
