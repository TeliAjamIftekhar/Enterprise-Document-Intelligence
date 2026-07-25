from __future__ import annotations

import re
from typing import Any, Mapping


HIGH_TRUST_SOURCES = {
    "resolved-chapter-structure",
    "front-matter-numbered-toc",
    "first-page-layout-title",
}

SENTENCE_ENDINGS = {
    ".",
    "?",
    "!",
    "।",
    "॥",
    "؟",
}

DANGLING_ENDINGS = {
    ":",
    ";",
    ",",
    "-",
    "–",
    "—",
}

WEAK_EXTRACTION_SOURCES = {
    "line-after-heading",
    "first-title-candidate",
}


def normalize_title_text(
    value: object,
) -> str:
    """Normalize whitespace without changing Unicode text."""

    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def title_words(
    value: str,
) -> list[str]:
    """Return language-neutral Unicode word-like tokens."""

    return re.findall(
        r"[^\W_]+",
        value,
        flags=re.UNICODE,
    )


def normalized_identity(
    value: object,
) -> str:
    """Create a comparison-only identity for titles."""

    return "".join(
        character.casefold()
        for character in normalize_title_text(value)
        if character.isalnum()
    )


def repeated_book_header(
    value: str,
    *,
    book_title: object,
    grade: object,
) -> bool:
    """Detect a book name/class header being mistaken for a unit."""

    candidate_identity = normalized_identity(
        value
    )

    book_identity = normalized_identity(
        book_title
    )

    if (
        not candidate_identity
        or not book_identity
    ):
        return False

    if candidate_identity == book_identity:
        return True

    grade_identity = normalized_identity(
        grade
    )

    if (
        book_identity in candidate_identity
        and grade_identity
        and grade_identity in candidate_identity
        and len(title_words(value)) <= 10
    ):
        return True

    return False


def title_rejection_reasons(
    value: object,
    *,
    book_title: object = None,
    grade: object = None,
) -> list[str]:
    """Explain why extracted text is unsafe as a title."""

    title = normalize_title_text(
        value
    )

    reasons: list[str] = []

    if len(title) < 2:
        reasons.append("too_short")

    if len(title) > 120:
        reasons.append("too_long")

    if title.isdigit():
        reasons.append("numeric_only")

    words = title_words(title)

    if len(words) > 16:
        reasons.append("too_many_words")

    if (
        title
        and title[-1] in DANGLING_ENDINGS
    ):
        reasons.append(
            "incomplete_trailing_punctuation"
        )

    sentence_mark_count = sum(
        title.count(mark)
        for mark in SENTENCE_ENDINGS
    )

    if sentence_mark_count >= 2:
        reasons.append(
            "multiple_sentence_markers"
        )

    if (
        len(words) >= 7
        and title
        and title[-1] in SENTENCE_ENDINGS
    ):
        reasons.append(
            "sentence_like_text"
        )

    if repeated_book_header(
        title,
        book_title=book_title,
        grade=grade,
    ):
        reasons.append(
            "repeated_book_header"
        )

    alphanumeric_count = sum(
        character.isalnum()
        for character in title
    )

    if alphanumeric_count < 2:
        reasons.append(
            "insufficient_text"
        )

    return list(
        dict.fromkeys(reasons)
    )


def safe_title_text(
    value: object,
    *,
    book_title: object = None,
    grade: object = None,
) -> bool:
    """Return True only when extracted text is safe to expose."""

    return not title_rejection_reasons(
        value,
        book_title=book_title,
        grade=grade,
    )


def weak_source_sentence_fragment(
    value: object,
    *,
    source: object,
) -> bool:
    """Detect body text selected by a weak extraction source."""

    title = normalize_title_text(
        value
    )

    normalized_source = normalize_title_text(
        source
    )

    if (
        normalized_source
        not in WEAK_EXTRACTION_SOURCES
    ):
        return False

    words = title_words(title)

    if len(words) < 10:
        return False

    return any(
        marker in title[:-1]
        for marker in SENTENCE_ENDINGS
    )


def ordered_unique_titles(
    values: list[object],
) -> list[str]:
    """Preserve candidates for later metadata enrichment."""

    output: list[str] = []

    for value in values:
        normalized = normalize_title_text(
            value
        )

        if (
            normalized
            and normalized not in output
        ):
            output.append(normalized)

    return output


def finalize_inferred_title(
    inferred: Mapping[str, Any] | None,
    *,
    fallback: str,
    book_title: object,
    grade: object,
) -> dict[str, Any]:
    """Accept safe titles or use a deterministic fallback.

    Weak extraction sources can never assign high confidence.
    Rejected source text remains available in the audit metadata.
    """

    fallback_title = normalize_title_text(
        fallback
    )

    if not fallback_title:
        raise ValueError(
            "A non-empty deterministic fallback is required."
        )

    inferred_payload = dict(
        inferred or {}
    )

    selected = normalize_title_text(
        inferred_payload.get(
            "title"
        )
    )

    source = normalize_title_text(
        inferred_payload.get(
            "source"
        )
    ) or "unknown"

    candidates = ordered_unique_titles([
        *list(
            inferred_payload.get(
                "candidates",
                [],
            )
            or []
        ),
        selected,
    ])

    if selected:
        reasons = title_rejection_reasons(
            selected,
            book_title=book_title,
            grade=grade,
        )

        if weak_source_sentence_fragment(
            selected,
            source=source,
        ):
            reasons.append(
                "weak_source_sentence_fragment"
            )

        reasons = list(
            dict.fromkeys(reasons)
        )
    else:
        reasons = ["missing_title"]

    if selected and not reasons:
        confidence = (
            "high"
            if source in HIGH_TRUST_SOURCES
            else "medium"
        )

        return {
            "title": selected,
            "confidence": confidence,
            "source": source,
            "candidates": candidates,
            "used_fallback": False,
            "rejected_title": None,
            "rejection_reasons": [],
        }

    return {
        "title": fallback_title,
        "confidence": "low",
        "source": "generated-safe-fallback",
        "candidates": candidates,
        "used_fallback": True,
        "rejected_title": (
            selected or None
        ),
        "rejection_reasons": reasons,
    }
