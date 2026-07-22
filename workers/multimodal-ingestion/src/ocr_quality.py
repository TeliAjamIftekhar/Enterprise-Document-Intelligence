"""Language-aware OCR quality validation.

This module evaluates text extracted by BDA or an OCR fallback before the
text is accepted for embeddings and OpenSearch indexing.

It intentionally separates:

1. Expected-script ratio against every non-space character.
2. Expected-script ratio against recognised script letters only.

The second metric prevents sparse worksheets containing blanks, punctuation,
numbers and symbols from being incorrectly rejected.
"""

from __future__ import annotations

import collections
import html
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Any, Literal


QualityClassification = Literal["PASS", "REVIEW", "FAIL"]
ExpectedScript = Literal[
    "arabic",
    "devanagari",
    "latin",
    "mixed",
]


LANGUAGE_SCRIPT_MAP: dict[str, ExpectedScript] = {
    "urdu": "arabic",
    "ur": "arabic",
    "hindi": "devanagari",
    "hi": "devanagari",
    "marathi": "devanagari",
    "mr": "devanagari",
    "sanskrit": "devanagari",
    "sa": "devanagari",
    "english": "latin",
    "en": "latin",
    "mathematics": "mixed",
    "mathematics/mixed": "mixed",
    "math": "mixed",
    "mixed": "mixed",
    "default": "mixed",
}


SCRIPT_PATTERNS: dict[str, re.Pattern[str]] = {
    "arabic": re.compile(
        r"[\u0600-\u06FF"
        r"\u0750-\u077F"
        r"\u08A0-\u08FF"
        r"\uFB50-\uFDFF"
        r"\uFE70-\uFEFF]"
    ),
    "devanagari": re.compile(
        r"[\u0900-\u097F"
        r"\uA8E0-\uA8FF]"
    ),
    "latin": re.compile(r"[A-Za-z]"),
}


WORD_PATTERN = re.compile(
    r"[\u0600-\u06FF"
    r"\u0750-\u077F"
    r"\u08A0-\u08FF"
    r"\u0900-\u097F"
    r"A-Za-z0-9]+"
)


@dataclass(frozen=True)
class OCRQualityThresholds:
    """Configurable quality thresholds."""

    minimum_nonspace_characters: int = 10
    minimum_expected_script_characters: int = 40
    sparse_page_minimum_script_characters: int = 20

    minimum_expected_ratio: float = 0.70
    minimum_script_only_ratio: float = 0.80
    sparse_page_script_only_ratio: float = 0.90

    repeated_phrase_minimum_count: int = 5
    duplicate_line_minimum_count: int = 5

    repeated_phrase_minimum_words: int = 5
    repeated_phrase_maximum_words: int = 8


@dataclass(frozen=True)
class OCRQualityMetrics:
    """Measured properties of one page or text block."""

    original_characters: int
    clean_characters: int
    nonspace_characters: int

    arabic_characters: int
    devanagari_characters: int
    latin_characters: int
    digit_characters: int

    expected_script: ExpectedScript
    expected_script_characters: int

    expected_script_ratio: float
    script_only_expected_ratio: float

    line_count: int
    max_duplicate_line_count: int
    max_repeated_phrase_count: int

    confidence: float | None


@dataclass(frozen=True)
class OCRQualityDecision:
    """Final quality decision used by the pipeline."""

    classification: QualityClassification
    accepted: bool
    fallback_recommended: bool
    sparse_page: bool

    expected_language: str
    expected_script: ExpectedScript
    source: str

    reasons: tuple[str, ...]
    metrics: OCRQualityMetrics
    clean_text: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


class _MarkupTextExtractor(HTMLParser):
    """Convert basic OCR HTML into clean plain text."""

    BLOCK_TAGS = {
        "p",
        "div",
        "br",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "ul",
        "ol",
        "table",
        "tr",
        "td",
        "th",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs

        normalized_tag = tag.lower()

        if normalized_tag == "li":
            self.parts.append("\n• ")
        elif normalized_tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def normalize_language(language: str | None) -> str:
    """Normalize language labels used in textbook configs."""

    normalized = (language or "default").strip().casefold()

    aliases = {
        "eng": "english",
        "hin": "hindi",
        "mar": "marathi",
        "san": "sanskrit",
        "urd": "urdu",
        "maths": "mathematics",
        "mathematics and statistics": "mathematics",
    }

    return aliases.get(normalized, normalized)


def expected_script_for_language(
    language: str | None,
) -> ExpectedScript:
    """Return the expected writing system for a language."""

    normalized_language = normalize_language(language)

    return LANGUAGE_SCRIPT_MAP.get(
        normalized_language,
        "mixed",
    )


def clean_ocr_text(text: str | None) -> str:
    """Remove markup and normalize whitespace."""

    if not text:
        return ""

    parser = _MarkupTextExtractor()
    parser.feed(str(text))

    clean_text = html.unescape(
        "".join(parser.parts)
    )

    clean_text = clean_text.replace("\u200c", "")
    clean_text = clean_text.replace("\u200d", "")
    clean_text = clean_text.replace("\ufeff", "")

    clean_text = re.sub(
        r"[ \t]+",
        " ",
        clean_text,
    )

    clean_text = re.sub(
        r" *\n *",
        "\n",
        clean_text,
    )

    clean_text = re.sub(
        r"\n{3,}",
        "\n\n",
        clean_text,
    )

    return clean_text.strip()


def _count_scripts(text: str) -> dict[str, int]:
    return {
        script: len(pattern.findall(text))
        for script, pattern in SCRIPT_PATTERNS.items()
    }


def _line_has_meaningful_repetition_content(
    line: str,
) -> bool:
    """Return whether a line contains meaningful text for duplicate checks."""

    meaningful_characters = sum(
        len(token)
        for token in WORD_PATTERN.findall(line)
    )

    return meaningful_characters >= 5


def _is_coherent_bilingual_surya_page(
    text: str,
    *,
    expected_script: ExpectedScript,
    expected_script_characters: int,
    expected_script_ratio: float,
    script_only_expected_ratio: float,
    confidence: float | None,
    source: str,
    thresholds: OCRQualityThresholds,
) -> bool:
    """Accept coherent high-confidence bilingual Surya pages."""

    if source.strip().casefold() != "surya":
        return False

    if expected_script == "mixed":
        return False

    if confidence is None or confidence < 0.85:
        return False

    if (
        expected_script_characters
        < thresholds.minimum_expected_script_characters
    ):
        return False

    if expected_script_ratio < 0.40:
        return False

    if script_only_expected_ratio < 0.50:
        return False

    expected_dominant_lines = 0
    secondary_dominant_lines = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        line_counts = _count_scripts(line)
        script_total = sum(line_counts.values())

        if script_total < 5:
            continue

        expected_count = line_counts[expected_script]
        other_count = script_total - expected_count

        if (
            expected_count >= 5
            and expected_count >= other_count
        ):
            expected_dominant_lines += 1
        elif (
            other_count >= 5
            and other_count > expected_count
        ):
            secondary_dominant_lines += 1

    return (
        expected_dominant_lines >= 2
        and secondary_dominant_lines >= 1
    )


def _repetition_metrics(
    text: str,
    thresholds: OCRQualityThresholds,
) -> tuple[int, int]:
    normalized_lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    line_counts = collections.Counter(
        line
        for line in normalized_lines
        if (
            len(line) >= 15
            and _line_has_meaningful_repetition_content(
                line
            )
        )
    )

    max_duplicate_line_count = max(
        line_counts.values(),
        default=0,
    )

    words = WORD_PATTERN.findall(text)

    max_repeated_phrase_count = 0

    for phrase_size in range(
        thresholds.repeated_phrase_minimum_words,
        thresholds.repeated_phrase_maximum_words + 1,
    ):
        if len(words) < phrase_size:
            continue

        phrase_counts = collections.Counter(
            tuple(
                words[index:index + phrase_size]
            )
            for index in range(
                len(words) - phrase_size + 1
            )
        )

        max_repeated_phrase_count = max(
            max_repeated_phrase_count,
            max(
                phrase_counts.values(),
                default=0,
            ),
        )

    return (
        max_duplicate_line_count,
        max_repeated_phrase_count,
    )


def evaluate_ocr_text(
    text: str | None,
    *,
    expected_language: str,
    source: str = "unknown",
    confidence: float | None = None,
    thresholds: OCRQualityThresholds | None = None,
) -> OCRQualityDecision:
    """Evaluate OCR/BDA text for one page.

    REVIEW and FAIL both recommend OCR fallback when the source is BDA.
    A Surya result should be accepted only when the classification is PASS.
    """

    active_thresholds = (
        thresholds
        if thresholds is not None
        else OCRQualityThresholds()
    )

    original_text = text or ""
    clean_text = clean_ocr_text(original_text)

    normalized_language = normalize_language(
        expected_language
    )

    expected_script = expected_script_for_language(
        normalized_language
    )

    script_counts = _count_scripts(clean_text)

    arabic_characters = script_counts["arabic"]
    devanagari_characters = script_counts["devanagari"]
    latin_characters = script_counts["latin"]

    script_character_total = (
        arabic_characters
        + devanagari_characters
        + latin_characters
    )

    nonspace_characters = len(
        re.findall(r"\S", clean_text)
    )

    digit_characters = len(
        re.findall(
            r"[0-9\u0660-\u0669\u06F0-\u06F9]",
            clean_text,
        )
    )

    if expected_script == "mixed":
        expected_script_characters = (
            script_character_total
        )
    else:
        expected_script_characters = (
            script_counts[expected_script]
        )

    expected_script_ratio = (
        expected_script_characters
        / nonspace_characters
        if nonspace_characters
        else 0.0
    )

    script_only_expected_ratio = (
        expected_script_characters
        / script_character_total
        if script_character_total
        else 0.0
    )

    (
        max_duplicate_line_count,
        max_repeated_phrase_count,
    ) = _repetition_metrics(
        clean_text,
        active_thresholds,
    )

    metrics = OCRQualityMetrics(
        original_characters=len(original_text),
        clean_characters=len(clean_text),
        nonspace_characters=nonspace_characters,
        arabic_characters=arabic_characters,
        devanagari_characters=devanagari_characters,
        latin_characters=latin_characters,
        digit_characters=digit_characters,
        expected_script=expected_script,
        expected_script_characters=(
            expected_script_characters
        ),
        expected_script_ratio=(
            expected_script_ratio
        ),
        script_only_expected_ratio=(
            script_only_expected_ratio
        ),
        line_count=len(
            [
                line
                for line in clean_text.splitlines()
                if line.strip()
            ]
        ),
        max_duplicate_line_count=(
            max_duplicate_line_count
        ),
        max_repeated_phrase_count=(
            max_repeated_phrase_count
        ),
        confidence=confidence,
    )

    classification: QualityClassification
    reasons: list[str] = []
    sparse_page = False

    if (
        nonspace_characters
        < active_thresholds.minimum_nonspace_characters
    ):
        classification = "FAIL"
        reasons.append(
            "empty_or_nearly_empty_text"
        )

    elif (
        max_duplicate_line_count
        >= active_thresholds.duplicate_line_minimum_count
    ):
        classification = "FAIL"
        reasons.append(
            "runaway_duplicate_lines"
        )

    elif (
        max_repeated_phrase_count
        >= active_thresholds.repeated_phrase_minimum_count
    ):
        classification = "FAIL"
        reasons.append(
            "runaway_phrase_repetition"
        )

    elif expected_script == "mixed":
        classification = "PASS"
        reasons.append(
            "mixed_script_content_accepted"
        )

    elif expected_script_characters == 0:
        classification = "FAIL"
        reasons.append(
            "expected_script_not_detected"
        )

    elif _is_coherent_bilingual_surya_page(
        clean_text,
        expected_script=expected_script,
        expected_script_characters=(
            expected_script_characters
        ),
        expected_script_ratio=expected_script_ratio,
        script_only_expected_ratio=(
            script_only_expected_ratio
        ),
        confidence=confidence,
        source=source,
        thresholds=active_thresholds,
    ):
        classification = "PASS"
        reasons.extend(
            [
                "coherent_bilingual_page_accepted",
                "substantial_expected_script_content",
            ]
        )

    elif (
        script_only_expected_ratio
        < 0.60
    ):
        classification = "FAIL"
        reasons.append(
            "dominant_wrong_script"
        )

    elif (
        expected_script_characters
        < active_thresholds.minimum_expected_script_characters
    ):
        if (
            expected_script_characters
            >= active_thresholds.sparse_page_minimum_script_characters
            and script_only_expected_ratio
            >= active_thresholds.sparse_page_script_only_ratio
        ):
            classification = "PASS"
            sparse_page = True
            reasons.extend(
                [
                    "sparse_page_accepted",
                    "expected_script_dominates_detected_letters",
                ]
            )
        else:
            classification = "REVIEW"
            reasons.append(
                "insufficient_expected_script_characters"
            )

    elif (
        script_only_expected_ratio
        < active_thresholds.minimum_script_only_ratio
    ):
        classification = "REVIEW"
        reasons.append(
            "moderate_wrong_script_contamination"
        )

    elif (
        expected_script_ratio
        < active_thresholds.minimum_expected_ratio
    ):
        if (
            script_only_expected_ratio
            >= active_thresholds.sparse_page_script_only_ratio
            and expected_script_characters
            >= active_thresholds.sparse_page_minimum_script_characters
        ):
            classification = "PASS"
            sparse_page = True
            reasons.extend(
                [
                    "sparse_or_symbol_heavy_page_accepted",
                    "low_ratio_caused_by_non_script_content",
                ]
            )
        else:
            classification = "REVIEW"
            reasons.append(
                "low_expected_script_ratio"
            )

    else:
        classification = "PASS"
        reasons.append(
            "expected_script_quality_gate_passed"
        )

    return OCRQualityDecision(
        classification=classification,
        accepted=classification == "PASS",
        fallback_recommended=(
            classification != "PASS"
        ),
        sparse_page=sparse_page,
        expected_language=normalized_language,
        expected_script=expected_script,
        source=source,
        reasons=tuple(reasons),
        metrics=metrics,
        clean_text=clean_text,
    )


def requires_ocr_fallback(
    text: str | None,
    *,
    expected_language: str,
    confidence: float | None = None,
    thresholds: OCRQualityThresholds | None = None,
) -> bool:
    """Return whether BDA output should use OCR fallback."""

    decision = evaluate_ocr_text(
        text,
        expected_language=expected_language,
        source="bda",
        confidence=confidence,
        thresholds=thresholds,
    )

    return decision.fallback_recommended
