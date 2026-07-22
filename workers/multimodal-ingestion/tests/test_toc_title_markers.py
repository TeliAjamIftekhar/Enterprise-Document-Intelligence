from resolve_chapter_structure import (
    strip_toc_decorative_prefix,
)


def test_removes_supplementary_star() -> None:
    assert strip_toc_decorative_prefix(
        "* चंदा मामा दूर के"
    ) == "चंदा मामा दूर के"


def test_removes_common_toc_bullets() -> None:
    assert strip_toc_decorative_prefix(
        "• Supplementary Reading"
    ) == "Supplementary Reading"

    assert strip_toc_decorative_prefix(
        "★ اضافی سبق"
    ) == "اضافی سبق"


def test_preserves_internal_punctuation() -> None:
    assert strip_toc_decorative_prefix(
        "Stars * and Stories"
    ) == "Stars * and Stories"

    assert strip_toc_decorative_prefix(
        "चाँद का बच्चा"
    ) == "चाँद का बच्चा"
