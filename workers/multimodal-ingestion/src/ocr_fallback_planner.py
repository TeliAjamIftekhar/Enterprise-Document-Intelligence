"""Plan page-level OCR fallback from normalized BDA output."""

from __future__ import annotations

import difflib
import json
import re
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from src.ocr_quality import (
    OCRQualityDecision,
    OCRQualityThresholds,
    clean_ocr_text,
    evaluate_ocr_text,
)


PlanClassification = Literal[
    "BDA_ACCEPTED",
    "OCR_FALLBACK_REQUIRED",
    "INVALID_INPUT",
]


PAGE_KEYS = (
    "canonical_page",
    "canonical_page_number",
    "page_number",
    "page",
    "source_page",
    "document_page",
)

TEXT_KEYS = (
    "clean_text",
    "normalized_text",
    "text",
    "content",
    "markdown",
    "html",
    "ocr_text",
    "recognized_text",
)

CONFIDENCE_KEYS = (
    "confidence",
    "ocr_confidence",
    "average_confidence",
    "mean_confidence",
    "probability",
    "conf",
)


@dataclass(frozen=True)
class BDARecordCandidate:
    canonical_page: int
    clean_text: str
    confidence: float | None
    source_record_count: int


@dataclass(frozen=True)
class PageQualityAssessment:
    canonical_page: int
    source_record_count: int
    text_characters: int
    confidence: float | None
    decision: OCRQualityDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_page": self.canonical_page,
            "source_record_count": self.source_record_count,
            "text_characters": self.text_characters,
            "confidence": self.confidence,
            "decision": self.decision.to_dict(),
        }


@dataclass(frozen=True)
class OCRFallbackPlan:
    expected_language: str
    classification: PlanClassification

    expected_pages: tuple[int, ...]
    discovered_pages: tuple[int, ...]
    accepted_bda_pages: tuple[int, ...]
    fallback_pages: tuple[int, ...]
    review_pages: tuple[int, ...]
    failed_pages: tuple[int, ...]
    missing_pages: tuple[int, ...]
    canonical_recovered_pages: tuple[int, ...]

    assessments: tuple[PageQualityAssessment, ...]

    @property
    def requires_fallback(self) -> bool:
        return bool(self.fallback_pages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_language": self.expected_language,
            "classification": self.classification,
            "requires_fallback": self.requires_fallback,
            "expected_pages": list(self.expected_pages),
            "discovered_pages": list(self.discovered_pages),
            "accepted_bda_pages": list(self.accepted_bda_pages),
            "fallback_pages": list(self.fallback_pages),
            "review_pages": list(self.review_pages),
            "failed_pages": list(self.failed_pages),
            "missing_pages": list(self.missing_pages),
            "canonical_recovered_pages": list(
                self.canonical_recovered_pages
            ),
            "summary": {
                "expected": len(self.expected_pages),
                "discovered": len(self.discovered_pages),
                "accepted_bda": len(self.accepted_bda_pages),
                "fallback": len(self.fallback_pages),
                "review": len(self.review_pages),
                "failed": len(self.failed_pages),
                "missing": len(self.missing_pages),
                "canonical_recovered": len(
                    self.canonical_recovered_pages
                ),
            },
            "assessments": [
                assessment.to_dict()
                for assessment in self.assessments
            ],
        }


def _coerce_positive_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value > 0 else None

    if isinstance(value, float):
        if value.is_integer() and value > 0:
            return int(value)

        return None

    if isinstance(value, str):
        stripped = value.strip()

        if stripped.isdigit():
            number = int(stripped)
            return number if number > 0 else None

    return None


def extract_page_numbers(
    record: dict[str, Any],
) -> tuple[int, ...]:
    """Extract all one-based canonical source pages."""

    page_numbers: set[int] = set()

    source_pages = record.get(
        "source_page_numbers"
    )

    if isinstance(source_pages, (list, tuple)):
        for value in source_pages:
            page_number = (
                _coerce_positive_integer(value)
            )

            if page_number is not None:
                page_numbers.add(page_number)

    locations = record.get("locations")

    if isinstance(locations, list):
        for location in locations:
            if not isinstance(location, dict):
                continue

            page_number = (
                _coerce_positive_integer(
                    location.get(
                        "source_page_number"
                    )
                )
            )

            if page_number is not None:
                page_numbers.add(page_number)

    if page_numbers:
        return tuple(sorted(page_numbers))

    containers: list[dict[str, Any]] = [record]

    for metadata_key in (
        "metadata",
        "source_metadata",
        "page_metadata",
        "document_metadata",
    ):
        metadata = record.get(metadata_key)

        if isinstance(metadata, dict):
            containers.append(metadata)

    for container in containers:
        for key in PAGE_KEYS:
            if key not in container:
                continue

            value = container[key]

            if isinstance(value, (list, tuple)):
                values = value
            else:
                values = (value,)

            for item in values:
                page_number = (
                    _coerce_positive_integer(item)
                )

                if page_number is not None:
                    page_numbers.add(page_number)

    return tuple(sorted(page_numbers))


def extract_page_number(
    record: dict[str, Any],
) -> int | None:
    """Extract the first canonical source page."""

    page_numbers = extract_page_numbers(
        record
    )

    if not page_numbers:
        return None

    return page_numbers[0]


def _collect_text_values(
    value: Any,
    output: list[str],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).casefold()

            if (
                normalized_key in TEXT_KEYS
                or normalized_key.endswith("_text")
            ):
                if isinstance(child, str) and child.strip():
                    output.append(child.strip())

                elif isinstance(child, list):
                    for item in child:
                        if (
                            isinstance(item, str)
                            and item.strip()
                        ):
                            output.append(item.strip())

            if normalized_key in {
                "blocks",
                "elements",
                "paragraphs",
                "sections",
                "children",
                "items",
            }:
                _collect_text_values(child, output)

    elif isinstance(value, list):
        for child in value:
            _collect_text_values(child, output)


def extract_record_text(record: dict[str, Any]) -> str:
    """Extract and deduplicate usable text from a normalized record."""

    values: list[str] = []

    _collect_text_values(record, values)

    unique_values: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = clean_ocr_text(value)

        if not cleaned or cleaned in seen:
            continue

        seen.add(cleaned)
        unique_values.append(cleaned)

    return "\n".join(unique_values).strip()


def _collect_confidences(
    value: Any,
    output: list[float],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).casefold()

            if (
                normalized_key in CONFIDENCE_KEYS
                or "confidence" in normalized_key
            ):
                if (
                    isinstance(child, (int, float))
                    and not isinstance(child, bool)
                ):
                    number = float(child)

                    if 0 <= number <= 100:
                        if number > 1:
                            number /= 100

                        output.append(number)

            if isinstance(child, (dict, list)):
                _collect_confidences(child, output)

    elif isinstance(value, list):
        for child in value:
            _collect_confidences(child, output)


def extract_record_confidence(
    record: dict[str, Any],
) -> float | None:
    values: list[float] = []

    _collect_confidences(record, values)

    if not values:
        return None

    return statistics.mean(values)


def records_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Normalize common JSON/JSONL roots into page records."""

    if isinstance(payload, list):
        return [
            item
            for item in payload
            if isinstance(item, dict)
        ]

    if not isinstance(payload, dict):
        return []

    # NORMALIZED_RECORD_DETECTION_GUARD
    #
    # A content-units.jsonl row is already one
    # normalized retrieval record. Recognize it
    # before applying the legacy page-key mapping
    # heuristic. Otherwise fields such as
    # asset_s3_uris are incorrectly interpreted as
    # page 3 merely because their key contains a
    # digit.
    if (
        extract_page_numbers(payload)
        and extract_record_text(payload)
    ):
        return [payload]

    for key in (
        "records",
        "pages",
        "documents",
        "items",
        "elements",
        "results",
    ):
        value = payload.get(key)

        if isinstance(value, list):
            records = [
                item
                for item in value
                if isinstance(item, dict)
            ]

            if records:
                return records

    page_key_records: list[dict[str, Any]] = []

    for key, value in payload.items():
        if not isinstance(value, (dict, list)):
            continue

        key_text = str(key).strip().casefold()

        # Legacy page maps may use keys such as
        # "12", "page_12", "page-12" or "page 12".
        # Arbitrary keys containing digits, such as
        # asset_s3_uris, are not page identifiers.
        is_page_key = (
            key_text.isdigit()
            or key_text.startswith("page_")
            or key_text.startswith("page-")
            or key_text.startswith("page ")
        )

        if not is_page_key:
            continue

        digits = "".join(
            character
            for character in key_text
            if character.isdigit()
        )

        if not digits:
            continue

        page_number = int(digits)

        if page_number <= 0:
            continue

        if isinstance(value, dict):
            record = dict(value)
        else:
            record = {
                "elements": value,
            }

        record.setdefault(
            "canonical_page",
            page_number,
        )

        page_key_records.append(record)

    if page_key_records:
        return page_key_records

    if (
        extract_page_number(payload) is not None
        and extract_record_text(payload)
    ):
        return [payload]

    return []


def load_normalized_records(path: Path) -> list[dict[str, Any]]:
    """Load records from JSON, JSONL or a directory."""

    if not path.exists():
        raise FileNotFoundError(
            f"Normalized BDA input not found: {path}"
        )

    if path.is_dir():
        records: list[dict[str, Any]] = []

        # Normalized directories contain reports,
        # validation markers and modality-specific
        # files. content-units.jsonl is the canonical
        # complete retrieval-record source.
        content_unit_files = sorted(
            path.rglob("content-units.jsonl")
        )

        if content_unit_files:
            files = content_unit_files
        else:
            files = sorted(
                list(path.rglob("*.json"))
                + list(path.rglob("*.jsonl"))
                + list(path.rglob("*.ndjson"))
            )

        for file_path in files:
            records.extend(
                load_normalized_records(file_path)
            )

        return records

    suffix = path.suffix.casefold()

    if suffix in {".jsonl", ".ndjson"}:
        records = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, line in enumerate(
                handle,
                start=1,
            ):
                stripped = line.strip()

                if not stripped:
                    continue

                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL at {path}:"
                        f"{line_number}"
                    ) from error

                records.extend(
                    records_from_payload(payload)
                )

        return records

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    return records_from_payload(payload)


def normalize_expected_pages(
    expected_pages: Iterable[int] | None,
) -> tuple[int, ...]:
    if expected_pages is None:
        return ()

    pages: set[int] = set()

    for page in expected_pages:
        if isinstance(page, bool) or not isinstance(page, int):
            raise TypeError(
                "Expected pages must be integers"
            )

        if page <= 0:
            raise ValueError(
                "Expected pages must be positive"
            )

        pages.add(page)

    return tuple(sorted(pages))


def normalized_element_type(
    record: dict[str, Any],
) -> str:
    """Return the normalized BDA element type when available."""

    value = record.get("element_type")

    if value is None:
        metadata = record.get("metadata")

        if isinstance(metadata, dict):
            value = metadata.get("element_type")

    return str(value or "").strip().upper()


def extract_quality_record_text(
    record: dict[str, Any],
) -> str:
    """Select one primary text representation for quality checks.

    Normalized BDA rows usually contain raw_text, markdown and
    search_text versions of the same element. Combining all of them
    creates artificial repetition. Use only the first usable primary
    field. Legacy records without an element type retain the existing
    recursive extraction behaviour.
    """

    for key in (
        "raw_text",
        "markdown",
        "search_text",
        "clean_text",
        "normalized_text",
        "text",
    ):
        value = record.get(key)

        if isinstance(value, str):
            cleaned = clean_ocr_text(value)

            if cleaned:
                return cleaned

    if not normalized_element_type(record):
        return extract_record_text(record)

    return ""


def select_page_candidates(
    records: Sequence[dict[str, Any]],
) -> tuple[BDARecordCandidate, ...]:
    """Build page text from primary TEXT records.

    FIGURE descriptions are intentionally excluded because visual
    summaries can repeat large portions of a page. TABLE records are
    used only when a page has no usable TEXT record. Legacy untyped
    records are treated as TEXT records for backward compatibility.
    """

    grouped: dict[int, list[dict[str, Any]]] = {}

    for record in records:
        element_type = normalized_element_type(
            record
        )

        if element_type and element_type not in {
            "TEXT",
            "TABLE",
        }:
            continue

        page_numbers = extract_page_numbers(
            record
        )

        for page_number in page_numbers:
            grouped.setdefault(
                page_number,
                [],
            ).append(record)

    candidates: list[BDARecordCandidate] = []

    for page_number, page_records in grouped.items():
        text_entries: list[
            tuple[dict[str, Any], str]
        ] = []
        table_entries: list[
            tuple[dict[str, Any], str]
        ] = []

        for record in page_records:
            text = extract_quality_record_text(
                record
            )

            if not text:
                continue

            entry = (record, text)

            if normalized_element_type(record) == "TABLE":
                table_entries.append(entry)
            else:
                text_entries.append(entry)

        selected_entries = (
            text_entries
            if text_entries
            else table_entries
        )

        if not selected_entries:
            continue

        unique_texts: list[str] = []
        seen: set[str] = set()
        confidences: list[float] = []

        for record, text in selected_entries:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

            confidence = extract_record_confidence(
                record
            )

            if confidence is not None:
                confidences.append(confidence)

        clean_text = "\n".join(
            unique_texts
        ).strip()

        confidence = (
            statistics.mean(confidences)
            if confidences
            else None
        )

        candidates.append(
            BDARecordCandidate(
                canonical_page=page_number,
                clean_text=clean_text,
                confidence=confidence,
                source_record_count=len(
                    selected_entries
                ),
            )
        )

    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.canonical_page
            ),
        )
    )


def _comparison_text(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        clean_ocr_text(text),
    ).strip().casefold()


def _agreement_metrics(
    left: str,
    right: str,
) -> tuple[float, float, float]:
    normalized_left = _comparison_text(left)
    normalized_right = _comparison_text(right)

    if not normalized_left or not normalized_right:
        return 0.0, 0.0, 0.0

    sequence_similarity = (
        difflib.SequenceMatcher(
            None,
            normalized_left,
            normalized_right,
            autojunk=False,
        ).ratio()
    )

    left_tokens = set(
        re.findall(
            r"[\w'-]+",
            normalized_left,
            flags=re.UNICODE,
        )
    )

    right_tokens = set(
        re.findall(
            r"[\w'-]+",
            normalized_right,
            flags=re.UNICODE,
        )
    )

    token_union = left_tokens | right_tokens

    token_jaccard = (
        len(left_tokens & right_tokens)
        / len(token_union)
        if token_union
        else 0.0
    )

    length_ratio = (
        min(
            len(normalized_left),
            len(normalized_right),
        )
        / max(
            len(normalized_left),
            len(normalized_right),
        )
    )

    return (
        sequence_similarity,
        token_jaccard,
        length_ratio,
    )


def _load_canonical_page_texts(
    canonical_pdf_path: Path,
    page_numbers: Iterable[int],
) -> dict[int, str]:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "PyMuPDF is required for native PDF "
            "text recovery."
        ) from error

    if not canonical_pdf_path.is_file():
        raise FileNotFoundError(
            "Canonical PDF not found: "
            f"{canonical_pdf_path}"
        )

    texts: dict[int, str] = {}

    with fitz.open(
        str(canonical_pdf_path)
    ) as document:
        for page_number in sorted(
            set(page_numbers)
        ):
            if not 1 <= page_number <= len(document):
                continue

            text = clean_ocr_text(
                document.load_page(
                    page_number - 1
                ).get_text("text")
            )

            if text:
                texts[page_number] = text

    return texts


def _recover_with_canonical_text(
    *,
    bda_text: str,
    bda_decision: OCRQualityDecision,
    canonical_text: str,
    expected_language: str,
    thresholds: OCRQualityThresholds | None,
) -> tuple[OCRQualityDecision, bool]:
    canonical_decision = evaluate_ocr_text(
        canonical_text,
        expected_language=expected_language,
        source="canonical_pdf",
        thresholds=thresholds,
    )

    if canonical_decision.classification == "PASS":
        return (
            replace(
                canonical_decision,
                reasons=(
                    "canonical_pdf_text_recovered",
                    *canonical_decision.reasons,
                ),
            ),
            True,
        )

    repetition_reasons = {
        "runaway_phrase_repetition",
        "runaway_duplicate_lines",
    }

    bda_reasons = set(
        bda_decision.reasons
    )
    canonical_reasons = set(
        canonical_decision.reasons
    )

    if (
        not bda_text
        or not bda_reasons
        or not canonical_reasons
        or not bda_reasons.issubset(
            repetition_reasons
        )
        or not canonical_reasons.issubset(
            repetition_reasons
        )
    ):
        return bda_decision, False

    (
        sequence_similarity,
        token_jaccard,
        length_ratio,
    ) = _agreement_metrics(
        bda_text,
        canonical_text,
    )

    metrics = canonical_decision.metrics

    independently_verified = (
        metrics.expected_script_characters >= 40
        and metrics.script_only_expected_ratio >= 0.90
        and sequence_similarity >= 0.80
        and token_jaccard >= 0.60
        and length_ratio >= 0.70
    )

    if not independently_verified:
        return bda_decision, False

    return (
        replace(
            canonical_decision,
            classification="PASS",
            accepted=True,
            fallback_recommended=False,
            reasons=(
                "canonical_pdf_repetition_verified",
                "independent_bda_canonical_agreement",
            ),
        ),
        True,
    )


def plan_ocr_fallback(
    records: Sequence[dict[str, Any]],
    *,
    expected_language: str,
    expected_pages: Iterable[int] | None = None,
    thresholds: OCRQualityThresholds | None = None,
    canonical_pdf_path: Path | None = None,
    allow_native_text_recovery: bool = False,
) -> OCRFallbackPlan:
    """Evaluate BDA pages and select pages requiring Surya.

    Native PDF recovery is opt-in and should only be enabled for a
    previously verified text-layout textbook.
    """

    if (
        allow_native_text_recovery
        and canonical_pdf_path is None
    ):
        raise ValueError(
            "canonical_pdf_path is required when "
            "native text recovery is enabled."
        )

    candidates = select_page_candidates(records)

    candidate_by_page = {
        candidate.canonical_page: candidate
        for candidate in candidates
    }

    normalized_expected_pages = (
        normalize_expected_pages(expected_pages)
    )

    discovered_pages = tuple(
        candidate.canonical_page
        for candidate in candidates
    )

    if not normalized_expected_pages:
        normalized_expected_pages = discovered_pages

    pages_to_assess = tuple(
        sorted(
            set(discovered_pages)
            | set(normalized_expected_pages)
        )
    )

    canonical_texts: dict[int, str] = {}

    if allow_native_text_recovery:
        assert canonical_pdf_path is not None

        canonical_texts = (
            _load_canonical_page_texts(
                canonical_pdf_path,
                pages_to_assess,
            )
        )

    assessments: list[
        PageQualityAssessment
    ] = []

    canonical_recovered_pages: list[int] = []

    for page_number in pages_to_assess:
        candidate = candidate_by_page.get(
            page_number
        )

        bda_text = (
            candidate.clean_text
            if candidate is not None
            else ""
        )

        bda_confidence = (
            candidate.confidence
            if candidate is not None
            else None
        )

        bda_decision = evaluate_ocr_text(
            bda_text,
            expected_language=expected_language,
            source="bda",
            confidence=bda_confidence,
            thresholds=thresholds,
        )

        decision = bda_decision
        recovered = False

        canonical_text = canonical_texts.get(
            page_number
        )

        if (
            canonical_text
            and bda_decision.classification
            != "PASS"
        ):
            decision, recovered = (
                _recover_with_canonical_text(
                    bda_text=bda_text,
                    bda_decision=bda_decision,
                    canonical_text=canonical_text,
                    expected_language=(
                        expected_language
                    ),
                    thresholds=thresholds,
                )
            )

        if recovered:
            canonical_recovered_pages.append(
                page_number
            )

        if candidate is None and not recovered:
            continue

        assessments.append(
            PageQualityAssessment(
                canonical_page=page_number,
                source_record_count=(
                    candidate.source_record_count
                    if candidate is not None
                    else 0
                ),
                text_characters=len(
                    decision.clean_text
                ),
                confidence=(
                    bda_confidence
                ),
                decision=decision,
            )
        )

    recovered_page_set = set(
        canonical_recovered_pages
    )

    missing_pages = tuple(
        page
        for page in normalized_expected_pages
        if (
            page not in set(discovered_pages)
            and page not in recovered_page_set
        )
    )

    accepted_bda_pages = tuple(
        assessment.canonical_page
        for assessment in assessments
        if assessment.decision.classification
        == "PASS"
    )

    review_pages = tuple(
        assessment.canonical_page
        for assessment in assessments
        if assessment.decision.classification
        == "REVIEW"
    )

    failed_pages = tuple(
        assessment.canonical_page
        for assessment in assessments
        if assessment.decision.classification
        == "FAIL"
    )

    fallback_pages = tuple(
        sorted(
            set(review_pages)
            | set(failed_pages)
            | set(missing_pages)
        )
    )

    if not candidates and not normalized_expected_pages:
        classification: PlanClassification = (
            "INVALID_INPUT"
        )

    elif fallback_pages:
        classification = (
            "OCR_FALLBACK_REQUIRED"
        )

    else:
        classification = "BDA_ACCEPTED"

    return OCRFallbackPlan(
        expected_language=expected_language,
        classification=classification,
        expected_pages=normalized_expected_pages,
        discovered_pages=discovered_pages,
        accepted_bda_pages=accepted_bda_pages,
        fallback_pages=fallback_pages,
        review_pages=review_pages,
        failed_pages=failed_pages,
        missing_pages=missing_pages,
        canonical_recovered_pages=tuple(
            canonical_recovered_pages
        ),
        assessments=tuple(assessments),
    )


def write_fallback_plan(
    plan: OCRFallbackPlan,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            plan.to_dict(),
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
