"""Plan page-level OCR fallback from normalized BDA output."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
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
            "summary": {
                "expected": len(self.expected_pages),
                "discovered": len(self.discovered_pages),
                "accepted_bda": len(self.accepted_bda_pages),
                "fallback": len(self.fallback_pages),
                "review": len(self.review_pages),
                "failed": len(self.failed_pages),
                "missing": len(self.missing_pages),
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


def select_page_candidates(
    records: Sequence[dict[str, Any]],
) -> tuple[BDARecordCandidate, ...]:
    """Combine duplicate normalized records for each canonical page."""

    grouped: dict[int, list[dict[str, Any]]] = {}

    for record in records:
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
        text_values: list[str] = []
        confidences: list[float] = []

        for record in page_records:
            text = extract_record_text(record)

            if text:
                text_values.append(text)

            confidence = extract_record_confidence(
                record
            )

            if confidence is not None:
                confidences.append(confidence)

        unique_texts: list[str] = []
        seen: set[str] = set()

        for text in text_values:
            if text in seen:
                continue

            seen.add(text)
            unique_texts.append(text)

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
                    page_records
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


def plan_ocr_fallback(
    records: Sequence[dict[str, Any]],
    *,
    expected_language: str,
    expected_pages: Iterable[int] | None = None,
    thresholds: OCRQualityThresholds | None = None,
) -> OCRFallbackPlan:
    """Evaluate BDA pages and select pages requiring Surya."""

    candidates = select_page_candidates(records)

    normalized_expected_pages = (
        normalize_expected_pages(expected_pages)
    )

    discovered_pages = tuple(
        candidate.canonical_page
        for candidate in candidates
    )

    if not normalized_expected_pages:
        normalized_expected_pages = discovered_pages

    missing_pages = tuple(
        page
        for page in normalized_expected_pages
        if page not in set(discovered_pages)
    )

    assessments: list[PageQualityAssessment] = []

    for candidate in candidates:
        decision = evaluate_ocr_text(
            candidate.clean_text,
            expected_language=expected_language,
            source="bda",
            confidence=candidate.confidence,
            thresholds=thresholds,
        )

        assessments.append(
            PageQualityAssessment(
                canonical_page=(
                    candidate.canonical_page
                ),
                source_record_count=(
                    candidate.source_record_count
                ),
                text_characters=len(
                    candidate.clean_text
                ),
                confidence=candidate.confidence,
                decision=decision,
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
