"""Merge verified Surya OCR text with normalized BDA records.

Merge policy:

- BDA text is retained for pages that pass the quality gate.
- BDA TEXT units are removed only for approved OCR fallback pages.
- One deterministic Surya text unit is created per fallback page.
- BDA figures, tables, visual assets and metadata remain unchanged.
- Chapter and canonical-page metadata comes from chapter-page-map.json.
- Existing normalized BDA files are never modified.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class UnifiedMergeReport:
    book_id: str
    book_version: str

    fallback_pages: tuple[int, ...]
    accepted_bda_pages: tuple[int, ...]
    canonical_recovered_pages: tuple[int, ...]

    input_content_units: int
    output_content_units: int

    removed_bda_text_units: int
    created_surya_text_units: int
    created_native_pdf_text_units: int
    preserved_bda_content_units: int

    preserved_figures: int
    preserved_tables: int

    output_content_units_path: Path
    output_figures_path: Path
    output_tables_path: Path

    status: str = "VALID"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)

        for key in (
            "output_content_units_path",
            "output_figures_path",
            "output_tables_path",
        ):
            payload[key] = str(payload[key])

        payload["fallback_pages"] = list(
            self.fallback_pages
        )

        payload["accepted_bda_pages"] = list(
            self.accepted_bda_pages
        )

        payload[
            "canonical_recovered_pages"
        ] = list(
            self.canonical_recovered_pages
        )

        return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dictionaries."""

    if not path.is_file():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
        )

    records: list[dict[str, Any]] = []

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
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL in {path} "
                    f"at line {line_number}"
                ) from error

            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object in {path} "
                    f"at line {line_number}"
                )

            records.append(value)

    return records


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )

        handle.write("\n")
        temporary_path = Path(handle.name)

    temporary_path.replace(path)


def atomic_write_jsonl(
    path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

            handle.write("\n")

        temporary_path = Path(handle.name)

    temporary_path.replace(path)


def normalize_positive_pages(
    values: Iterable[Any],
    *,
    field_name: str,
) -> tuple[int, ...]:
    pages: set[int] = set()

    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{field_name} must contain positive "
                f"integers: {value!r}"
            )

        pages.add(value)

    return tuple(sorted(pages))


def load_ocr_plan(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "OCR fallback plan must be a JSON object"
        )

    classification = payload.get(
        "classification"
    )

    if classification not in {
        "BDA_ACCEPTED",
        "OCR_FALLBACK_REQUIRED",
    }:
        raise ValueError(
            "Unsupported OCR plan classification: "
            f"{classification!r}"
        )

    fallback_pages = normalize_positive_pages(
        payload.get(
            "fallback_pages",
            [],
        ),
        field_name="fallback_pages",
    )

    accepted_bda_pages = normalize_positive_pages(
        payload.get(
            "accepted_bda_pages",
            [],
        ),
        field_name="accepted_bda_pages",
    )

    canonical_recovered_pages = (
        normalize_positive_pages(
            payload.get(
                "canonical_recovered_pages",
                [],
            ),
            field_name=(
                "canonical_recovered_pages"
            ),
        )
    )

    if (
        classification == "OCR_FALLBACK_REQUIRED"
        and not fallback_pages
    ):
        raise ValueError(
            "OCR_FALLBACK_REQUIRED plan has no "
            "fallback pages"
        )

    recovered_set = set(
        canonical_recovered_pages
    )

    if recovered_set & set(fallback_pages):
        raise ValueError(
            "Canonical recovered pages cannot also "
            "be OCR fallback pages"
        )

    if not recovered_set.issubset(
        set(accepted_bda_pages)
    ):
        raise ValueError(
            "Canonical recovered pages must be "
            "included in accepted_bda_pages"
        )

    payload = dict(payload)
    payload["fallback_pages"] = fallback_pages
    payload["accepted_bda_pages"] = (
        accepted_bda_pages
    )
    payload[
        "canonical_recovered_pages"
    ] = canonical_recovered_pages

    return payload


def load_page_map(
    path: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Chapter page map must be a JSON object"
        )

    pages = payload.get("pages")

    if not isinstance(pages, list):
        raise ValueError(
            "Chapter page map is missing pages list"
        )

    page_lookup: dict[int, dict[str, Any]] = {}

    for item in pages:
        if not isinstance(item, dict):
            raise ValueError(
                "Page-map entries must be objects"
            )

        page = item.get("canonical_page")

        if (
            isinstance(page, bool)
            or not isinstance(page, int)
            or page <= 0
        ):
            raise ValueError(
                f"Invalid canonical page: {page!r}"
            )

        if page in page_lookup:
            raise ValueError(
                f"Duplicate canonical page: {page}"
            )

        page_lookup[page] = copy.deepcopy(item)

    return payload, page_lookup


def load_verified_surya_pages(
    path: Path,
    *,
    fallback_pages: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    """Load only pipeline-verified Surya pages."""

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Surya report must be a JSON object"
        )

    if payload.get("classification") != "PASS":
        raise ValueError(
            "Surya fallback report is not PASS"
        )

    if payload.get(
        "accepted_for_pipeline"
    ) is not True:
        raise ValueError(
            "Surya fallback report is not approved "
            "for pipeline use"
        )

    pages = payload.get("pages")

    if not isinstance(pages, list):
        raise ValueError(
            "Surya report is missing pages list"
        )

    page_lookup: dict[int, dict[str, Any]] = {}

    for page_record in pages:
        if not isinstance(page_record, dict):
            raise ValueError(
                "Surya page entries must be objects"
            )

        page = page_record.get(
            "canonical_page"
        )

        if (
            isinstance(page, bool)
            or not isinstance(page, int)
            or page <= 0
        ):
            raise ValueError(
                f"Invalid Surya canonical page: {page!r}"
            )

        decision = page_record.get(
            "decision"
        )

        if not isinstance(decision, dict):
            raise ValueError(
                f"Surya page {page} has no decision"
            )

        if decision.get(
            "classification"
        ) != "PASS":
            raise ValueError(
                f"Surya page {page} did not pass "
                "quality validation"
            )

        if decision.get("accepted") is not True:
            raise ValueError(
                f"Surya page {page} is not accepted"
            )

        clean_text = page_record.get(
            "clean_text"
        )

        if (
            not isinstance(clean_text, str)
            or not clean_text.strip()
        ):
            raise ValueError(
                f"Surya page {page} has empty text"
            )

        if page in page_lookup:
            raise ValueError(
                f"Duplicate Surya page: {page}"
            )

        page_lookup[page] = copy.deepcopy(
            page_record
        )

    expected = set(fallback_pages)
    discovered = set(page_lookup)

    missing = sorted(expected - discovered)
    unexpected = sorted(discovered - expected)

    if missing:
        raise ValueError(
            "Missing verified Surya pages: "
            + ", ".join(
                str(page)
                for page in missing
            )
        )

    if unexpected:
        raise ValueError(
            "Surya report contains pages outside "
            "the fallback plan: "
            + ", ".join(
                str(page)
                for page in unexpected
            )
        )

    return page_lookup


def record_page_numbers(
    record: dict[str, Any],
) -> tuple[int, ...]:
    """Resolve canonical pages for a normalized record."""

    raw_pages = record.get(
        "source_page_numbers"
    )

    if isinstance(raw_pages, list):
        valid_pages = [
            page
            for page in raw_pages
            if (
                isinstance(page, int)
                and not isinstance(page, bool)
                and page > 0
            )
        ]

        if valid_pages:
            return tuple(
                sorted(set(valid_pages))
            )

    contexts = record.get(
        "resolved_page_contexts"
    )

    if isinstance(contexts, list):
        valid_pages = []

        for context in contexts:
            if not isinstance(context, dict):
                continue

            page = context.get(
                "canonical_page"
            )

            if (
                isinstance(page, int)
                and not isinstance(page, bool)
                and page > 0
            ):
                valid_pages.append(page)

        if valid_pages:
            return tuple(
                sorted(set(valid_pages))
            )

    return ()


def is_bda_text_unit(
    record: dict[str, Any],
) -> bool:
    element_type = str(
        record.get(
            "element_type",
            "",
        )
    ).strip().upper()

    return element_type == "TEXT"


def _non_null_list(value: Any) -> list[Any]:
    if value is None:
        return []

    return [value]


def _page_context_fields(
    page_context: dict[str, Any],
) -> dict[str, Any]:
    chapter_id = page_context.get(
        "chapter_id"
    )

    chapter_title = page_context.get(
        "chapter_title"
    )

    return {
        "page_context_status": "resolved",
        "chapter_context_status": (
            "resolved"
            if chapter_id is not None
            or chapter_title is not None
            else "none"
        ),
        "resolved_page_contexts": [
            copy.deepcopy(page_context)
        ],
        "unresolved_source_page_numbers": [],
        "page_types": _non_null_list(
            page_context.get("page_type")
        ),
        "document_ids": _non_null_list(
            page_context.get("document_id")
        ),
        "document_titles": _non_null_list(
            page_context.get(
                "document_title"
            )
        ),
        "document_types": _non_null_list(
            page_context.get(
                "document_type"
            )
        ),
        "unit_numbers": _non_null_list(
            page_context.get("unit_number")
        ),
        "chapter_ids": _non_null_list(
            chapter_id
        ),
        "chapter_titles": _non_null_list(
            chapter_title
        ),
        "source_filenames": _non_null_list(
            page_context.get(
                "source_filename"
            )
        ),
        "document_id": page_context.get(
            "document_id"
        ),
        "document_title": page_context.get(
            "document_title"
        ),
        "document_type": page_context.get(
            "document_type"
        ),
        "unit_number": page_context.get(
            "unit_number"
        ),
        "chapter_id": chapter_id,
        "chapter_title": chapter_title,
    }


def deterministic_surya_unit_id(
    *,
    book_id: str,
    book_version: str,
    canonical_page: int,
    clean_text: str,
) -> str:
    digest = hashlib.sha256(
        (
            f"{book_id}\n"
            f"{book_version}\n"
            f"{canonical_page}\n"
            f"{clean_text}"
        ).encode("utf-8")
    ).hexdigest()[:24]

    return (
        f"{book_id}:{book_version}:"
        f"surya:page-{canonical_page:04d}:"
        f"{digest}"
    )


def build_surya_content_unit(
    *,
    book_id: str,
    book_version: str,
    source_pdf: str,
    canonical_page: int,
    page_context: dict[str, Any],
    surya_page: dict[str, Any],
) -> dict[str, Any]:
    clean_text = str(
        surya_page["clean_text"]
    ).strip()

    unit_id = deterministic_surya_unit_id(
        book_id=book_id,
        book_version=book_version,
        canonical_page=canonical_page,
        clean_text=clean_text,
    )

    decision = copy.deepcopy(
        surya_page["decision"]
    )

    unit: dict[str, Any] = {
        "schema_version": "1.1",
        "unit_id": unit_id,
        "book_id": book_id,
        "book_version": book_version,
        "source_kind": (
            "surya_ocr_fallback"
        ),
        "source_pdf": source_pdf,
        "source_sample_s3_uri": "",
        "bda_element_id": None,
        "element_index": (
            1_000_000 + canonical_page
        ),
        "element_type": "TEXT",
        "element_sub_type": (
            "OCR_FALLBACK_PAGE"
        ),
        "modality": "paragraph",
        "reading_order": 0,
        "sample_page_indices": [],
        "source_page_numbers": [
            canonical_page
        ],
        "locations": [],
        "raw_text": clean_text,
        "markdown": clean_text,
        "generated_title": "",
        "generated_summary": "",
        "search_text": clean_text,
        "asset_s3_uris": [],
        "asset_local_paths": [],
        "retrieval_priority": "normal",
        "text_source": "surya",
        "ocr_quality": decision,
        "quality_flags": [
            "surya_ocr_fallback_verified",
            "bda_text_replaced",
        ],
    }

    unit.update(
        _page_context_fields(
            page_context
        )
    )

    return unit



def deterministic_native_pdf_unit_id(
    *,
    book_id: str,
    book_version: str,
    canonical_page: int,
    clean_text: str,
) -> str:
    """Create a stable identifier for recovered native PDF text."""

    digest = hashlib.sha256(
        (
            f"{book_id}\n"
            f"{book_version}\n"
            f"{canonical_page}\n"
            f"{clean_text}"
        ).encode("utf-8")
    ).hexdigest()[:24]

    return (
        f"{book_id}:{book_version}:"
        f"canonical-pdf:page-{canonical_page:04d}:"
        f"{digest}"
    )


def build_native_pdf_content_unit(
    *,
    book_id: str,
    book_version: str,
    source_pdf: str,
    canonical_page: int,
    page_context: dict[str, Any],
    clean_text: str,
    decision: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one embedding-compatible native PDF recovery unit."""

    clean_text = clean_text.strip()

    if not clean_text:
        raise ValueError(
            "Native PDF recovery text is empty for "
            f"page {canonical_page}"
        )

    unit: dict[str, Any] = {
        "schema_version": "1.1",
        "unit_id": deterministic_native_pdf_unit_id(
            book_id=book_id,
            book_version=book_version,
            canonical_page=canonical_page,
            clean_text=clean_text,
        ),
        "book_id": book_id,
        "book_version": book_version,
        "source_kind": (
            "canonical_pdf_text_recovery"
        ),
        "source_pdf": source_pdf,
        "source_sample_s3_uri": "",
        "bda_element_id": None,
        "element_index": (
            2_000_000 + canonical_page
        ),
        "element_type": "TEXT",
        "element_sub_type": (
            "NATIVE_PDF_TEXT_RECOVERY"
        ),
        "modality": "paragraph",
        "reading_order": 0,
        "sample_page_indices": [],
        "source_page_numbers": [
            canonical_page
        ],
        "locations": [],
        "raw_text": clean_text,
        "markdown": clean_text,
        "generated_title": "",
        "generated_summary": "",
        "search_text": clean_text,
        "asset_s3_uris": [],
        "asset_local_paths": [],
        "retrieval_priority": "normal",
        "text_source": "canonical_pdf",
        "ocr_quality": copy.deepcopy(
            decision or {}
        ),
        "quality_flags": [
            "canonical_pdf_text_recovered",
            "bda_text_unit_missing",
        ],
    }

    unit.update(
        _page_context_fields(
            page_context
        )
    )

    return unit


def load_native_pdf_page_texts(
    source_pdf: str,
    page_numbers: Iterable[int],
) -> dict[int, str]:
    """Extract selected native text pages from a local canonical PDF."""

    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "PyMuPDF is required for canonical "
            "PDF text recovery."
        ) from error

    pdf_path = Path(source_pdf)

    if not pdf_path.is_file():
        raise FileNotFoundError(
            "Canonical PDF is not available locally: "
            f"{pdf_path}"
        )

    requested_pages = tuple(
        sorted(
            set(page_numbers)
        )
    )

    recovered: dict[int, str] = {}

    with fitz.open(
        str(pdf_path)
    ) as document:
        for page_number in requested_pages:
            if not 1 <= page_number <= len(document):
                raise ValueError(
                    "Canonical recovery page is outside "
                    f"the PDF range: {page_number}"
                )

            raw_text = document.load_page(
                page_number - 1
            ).get_text("text")

            clean_text = "\n".join(
                line.rstrip()
                for line in str(
                    raw_text
                ).splitlines()
            ).strip()

            if clean_text:
                recovered[page_number] = clean_text

    return recovered


def plan_assessments_by_page(
    plan: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    assessments = plan.get(
        "assessments",
        []
    )

    if not isinstance(assessments, list):
        return {}

    page_lookup: dict[int, dict[str, Any]] = {}

    for assessment in assessments:
        if not isinstance(assessment, dict):
            continue

        page = assessment.get(
            "canonical_page"
        )

        decision = assessment.get(
            "decision"
        )

        if (
            isinstance(page, int)
            and not isinstance(page, bool)
            and page > 0
            and isinstance(decision, dict)
        ):
            page_lookup[page] = copy.deepcopy(
                decision
            )

    return page_lookup


def validate_unique_ids(
    records: Sequence[dict[str, Any]],
    *,
    identifier_field: str,
    record_type: str,
) -> None:
    seen: set[str] = set()

    for record in records:
        identifier = record.get(
            identifier_field
        )

        if not isinstance(identifier, str):
            continue

        if identifier in seen:
            raise ValueError(
                f"Duplicate {record_type} identifier: "
                f"{identifier}"
            )

        seen.add(identifier)


def merge_records(
    *,
    content_units: Sequence[dict[str, Any]],
    figures: Sequence[dict[str, Any]],
    tables: Sequence[dict[str, Any]],
    plan: dict[str, Any],
    surya_pages: dict[int, dict[str, Any]],
    page_lookup: dict[int, dict[str, Any]],
    book_id: str,
    book_version: str,
    source_pdf: str,
    native_pdf_pages: dict[int, str] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
]:
    fallback_pages = set(
        plan["fallback_pages"]
    )

    canonical_recovered_pages = set(
        plan.get(
            "canonical_recovered_pages",
            (),
        )
    )

    native_pdf_pages = dict(
        native_pdf_pages or {}
    )

    assessments = (
        plan_assessments_by_page(plan)
    )

    output_units: list[
        dict[str, Any]
    ] = []

    removed_text_units = 0
    created_native_pdf_text_units = 0

    bda_text_pages: set[int] = set()

    for record in content_units:
        if is_bda_text_unit(record):
            bda_text_pages.update(
                record_page_numbers(record)
            )

    for original in content_units:
        record = copy.deepcopy(original)
        pages = record_page_numbers(record)
        page_set = set(pages)

        if (
            page_set & fallback_pages
            and is_bda_text_unit(record)
        ):
            if not page_set:
                raise ValueError(
                    "Fallback BDA text unit has no "
                    "canonical page metadata: "
                    f"{record.get('unit_id')}"
                )

            if not page_set.issubset(
                fallback_pages
            ):
                raise ValueError(
                    "BDA text unit spans fallback and "
                    "non-fallback pages: "
                    f"{record.get('unit_id')}"
                )

            removed_text_units += 1
            continue

        record["text_source"] = "bda"

        if len(pages) == 1:
            quality = assessments.get(
                pages[0]
            )

            if quality is not None:
                record["ocr_quality"] = quality

        output_units.append(record)

    native_only_pages = (
        canonical_recovered_pages
        - bda_text_pages
    )

    for page in sorted(native_only_pages):
        page_context = page_lookup.get(page)

        if page_context is None:
            raise ValueError(
                "Canonical recovered page is missing "
                f"from chapter page map: {page}"
            )

        clean_text = str(
            native_pdf_pages.get(
                page,
                "",
            )
        ).strip()

        if not clean_text:
            raise ValueError(
                "Canonical recovered page has no "
                f"native PDF text: {page}"
            )

        output_units.append(
            build_native_pdf_content_unit(
                book_id=book_id,
                book_version=book_version,
                source_pdf=source_pdf,
                canonical_page=page,
                page_context=page_context,
                clean_text=clean_text,
                decision=assessments.get(page),
            )
        )

        created_native_pdf_text_units += 1

    for page in sorted(fallback_pages):
        page_context = page_lookup.get(page)

        if page_context is None:
            raise ValueError(
                "Fallback page is missing from chapter "
                f"page map: {page}"
            )

        surya_page = surya_pages.get(page)

        if surya_page is None:
            raise ValueError(
                f"Verified Surya page missing: {page}"
            )

        output_units.append(
            build_surya_content_unit(
                book_id=book_id,
                book_version=book_version,
                source_pdf=source_pdf,
                canonical_page=page,
                page_context=page_context,
                surya_page=surya_page,
            )
        )

    output_units.sort(
        key=lambda record: (
            record_page_numbers(record)[0]
            if record_page_numbers(record)
            else 10**9,
            int(
                record.get(
                    "reading_order",
                    10**9,
                )
            )
            if isinstance(
                record.get("reading_order"),
                int,
            )
            else 10**9,
            str(record.get("unit_id", "")),
        )
    )

    output_figures = [
        copy.deepcopy(record)
        for record in figures
    ]

    output_tables = [
        copy.deepcopy(record)
        for record in tables
    ]

    validate_unique_ids(
        output_units,
        identifier_field="unit_id",
        record_type="content unit",
    )

    validate_unique_ids(
        output_figures,
        identifier_field="figure_id",
        record_type="figure",
    )

    validate_unique_ids(
        output_tables,
        identifier_field="table_id",
        record_type="table",
    )

    statistics = {
        "input_content_units": len(
            content_units
        ),
        "output_content_units": len(
            output_units
        ),
        "removed_bda_text_units": (
            removed_text_units
        ),
        "created_surya_text_units": len(
            fallback_pages
        ),
        "created_native_pdf_text_units": (
            created_native_pdf_text_units
        ),
        "preserved_bda_content_units": (
            len(content_units)
            - removed_text_units
        ),
        "preserved_figures": len(
            output_figures
        ),
        "preserved_tables": len(
            output_tables
        ),
    }

    return (
        output_units,
        output_figures,
        output_tables,
        statistics,
    )


def discover_jsonl_files(
    roots: Sequence[Path],
    filename: str,
) -> tuple[Path, ...]:
    discovered: set[Path] = set()

    for root in roots:
        if not root.exists():
            raise FileNotFoundError(
                f"Normalized root not found: {root}"
            )

        if root.is_file():
            if root.name == filename:
                discovered.add(
                    root.resolve()
                )

            continue

        direct = root / filename

        if direct.is_file():
            discovered.add(
                direct.resolve()
            )

        for candidate in root.rglob(
            filename
        ):
            if "embedding-ready" in {
                part.casefold()
                for part in candidate.parts
            }:
                continue

            discovered.add(
                candidate.resolve()
            )

    return tuple(
        sorted(
            discovered,
            key=str,
        )
    )


def load_records_from_roots(
    roots: Sequence[Path],
    filename: str,
    *,
    required: bool,
) -> list[dict[str, Any]]:
    files = discover_jsonl_files(
        roots,
        filename,
    )

    if required and not files:
        raise FileNotFoundError(
            f"No {filename} files found under: "
            + ", ".join(
                str(root)
                for root in roots
            )
        )

    records: list[dict[str, Any]] = []

    for path in files:
        records.extend(
            read_jsonl(path)
        )

    return records


def merge_bda_surya_outputs(
    *,
    normalized_roots: Sequence[Path],
    ocr_plan_path: Path,
    surya_report_path: Path,
    page_map_path: Path,
    output_dir: Path,
    source_pdf: str | None = None,
    replace: bool = False,
) -> UnifiedMergeReport:
    """Create embedding-compatible unified normalized records."""

    if output_dir.exists() and any(
        output_dir.iterdir()
    ):
        if not replace:
            raise FileExistsError(
                "Unified output directory already "
                f"contains files: {output_dir}"
            )

    plan = load_ocr_plan(
        ocr_plan_path
    )

    page_map, page_lookup = load_page_map(
        page_map_path
    )

    fallback_pages: tuple[int, ...] = (
        plan["fallback_pages"]
    )

    if fallback_pages:
        surya_pages = (
            load_verified_surya_pages(
                surya_report_path,
                fallback_pages=fallback_pages,
            )
        )
    else:
        surya_pages = {}

    content_units = load_records_from_roots(
        normalized_roots,
        "content-units.jsonl",
        required=True,
    )

    figures = load_records_from_roots(
        normalized_roots,
        "figures.jsonl",
        required=False,
    )

    tables = load_records_from_roots(
        normalized_roots,
        "tables.jsonl",
        required=False,
    )

    book_id = str(
        page_map.get(
            "book_id",
            content_units[0].get(
                "book_id",
                "",
            )
            if content_units
            else "",
        )
    )

    book_version = str(
        page_map.get(
            "book_version",
            content_units[0].get(
                "book_version",
                "",
            )
            if content_units
            else "",
        )
    )

    if not book_id or not book_version:
        raise ValueError(
            "Unable to resolve book ID/version"
        )

    resolved_source_pdf = source_pdf

    if not resolved_source_pdf:
        for record in content_units:
            candidate = record.get(
                "source_pdf"
            )

            if (
                isinstance(candidate, str)
                and candidate.strip()
            ):
                resolved_source_pdf = (
                    candidate.strip()
                )
                break

    if not resolved_source_pdf:
        raise ValueError(
            "Unable to resolve canonical source PDF"
        )

    canonical_recovered_pages: tuple[int, ...] = (
        plan["canonical_recovered_pages"]
    )

    bda_text_pages: set[int] = set()

    for record in content_units:
        if is_bda_text_unit(record):
            bda_text_pages.update(
                record_page_numbers(record)
            )

    native_only_pages = tuple(
        sorted(
            set(canonical_recovered_pages)
            - bda_text_pages
        )
    )

    native_pdf_pages = (
        load_native_pdf_page_texts(
            resolved_source_pdf,
            native_only_pages,
        )
        if native_only_pages
        else {}
    )

    (
        output_units,
        output_figures,
        output_tables,
        statistics,
    ) = merge_records(
        content_units=content_units,
        figures=figures,
        tables=tables,
        plan=plan,
        surya_pages=surya_pages,
        page_lookup=page_lookup,
        book_id=book_id,
        book_version=book_version,
        source_pdf=resolved_source_pdf,
        native_pdf_pages=native_pdf_pages,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    content_path = (
        output_dir
        / "content-units.jsonl"
    )

    figures_path = (
        output_dir
        / "figures.jsonl"
    )

    tables_path = (
        output_dir
        / "tables.jsonl"
    )

    atomic_write_jsonl(
        content_path,
        output_units,
    )

    atomic_write_jsonl(
        figures_path,
        output_figures,
    )

    atomic_write_jsonl(
        tables_path,
        output_tables,
    )

    report = UnifiedMergeReport(
        book_id=book_id,
        book_version=book_version,
        fallback_pages=fallback_pages,
        accepted_bda_pages=(
            plan["accepted_bda_pages"]
        ),
        canonical_recovered_pages=(
            plan["canonical_recovered_pages"]
        ),
        input_content_units=statistics[
            "input_content_units"
        ],
        output_content_units=statistics[
            "output_content_units"
        ],
        removed_bda_text_units=statistics[
            "removed_bda_text_units"
        ],
        created_surya_text_units=statistics[
            "created_surya_text_units"
        ],
        created_native_pdf_text_units=statistics[
            "created_native_pdf_text_units"
        ],
        preserved_bda_content_units=statistics[
            "preserved_bda_content_units"
        ],
        preserved_figures=statistics[
            "preserved_figures"
        ],
        preserved_tables=statistics[
            "preserved_tables"
        ],
        output_content_units_path=(
            content_path
        ),
        output_figures_path=figures_path,
        output_tables_path=tables_path,
    )

    atomic_write_json(
        output_dir
        / "bda-surya-merge-report.json",
        report.to_dict(),
    )

    (
        output_dir
        / "BDA_SURYA_MERGE_VALID"
    ).write_text(
        "VALID\n",
        encoding="utf-8",
    )

    return report
