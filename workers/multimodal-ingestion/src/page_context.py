from __future__ import annotations

from collections.abc import Iterable
from typing import Any


PAGE_CONTEXT_FIELDS = (
    "canonical_page",
    "batch_page",
    "page_type",
    "document_order",
    "document_id",
    "document_type",
    "document_title",
    "source_filename",
    "source_page",
    "unit_number",
    "chapter_id",
    "chapter_title",
    "chapter_page",
)


def normalize_page_numbers(
    values: Any,
) -> list[int]:
    if not isinstance(values, list):
        return []

    pages: list[int] = []
    seen: set[int] = set()

    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            continue

        if value in seen:
            continue

        seen.add(value)
        pages.append(value)

    return pages


def ordered_unique(
    values: Iterable[Any],
) -> list[Any]:
    results: list[Any] = []
    seen: set[Any] = set()

    for value in values:
        if value is None:
            continue

        if value in seen:
            continue

        seen.add(value)
        results.append(value)

    return results


def single_value_or_none(
    values: list[Any],
) -> Any:
    if len(values) == 1:
        return values[0]

    return None


def validate_context_record(
    context: dict[str, Any],
) -> dict[str, Any]:
    canonical_page = context.get(
        "canonical_page"
    )

    if (
        isinstance(canonical_page, bool)
        or not isinstance(canonical_page, int)
        or canonical_page < 1
    ):
        raise ValueError(
            "Page context has an invalid "
            "canonical_page."
        )

    batch_page = context.get(
        "batch_page"
    )

    if batch_page is not None and (
        isinstance(batch_page, bool)
        or not isinstance(batch_page, int)
        or batch_page < 1
    ):
        raise ValueError(
            "Page context has an invalid "
            "batch_page."
        )

    source_page = context.get(
        "source_page"
    )

    if source_page is not None and (
        isinstance(source_page, bool)
        or not isinstance(source_page, int)
        or source_page < 1
    ):
        raise ValueError(
            "Page context has an invalid "
            "source_page."
        )

    chapter_page = context.get(
        "chapter_page"
    )

    if chapter_page is not None and (
        isinstance(chapter_page, bool)
        or not isinstance(chapter_page, int)
        or chapter_page < 1
    ):
        raise ValueError(
            "Page context has an invalid "
            "chapter_page."
        )

    unit_number = context.get(
        "unit_number"
    )

    if unit_number is not None and (
        isinstance(unit_number, bool)
        or not isinstance(unit_number, int)
        or unit_number < 1
    ):
        raise ValueError(
            "Page context has an invalid "
            "unit_number."
        )

    chapter_id = context.get(
        "chapter_id"
    )
    chapter_title = context.get(
        "chapter_title"
    )

    if (
        chapter_id is None
        and chapter_title is not None
    ):
        raise ValueError(
            "chapter_title cannot exist "
            "without chapter_id."
        )

    if (
        chapter_id is not None
        and (
            not isinstance(chapter_id, str)
            or not chapter_id.strip()
        )
    ):
        raise ValueError(
            "chapter_id must be a non-empty "
            "string or null."
        )

    if (
        chapter_title is not None
        and (
            not isinstance(
                chapter_title,
                str,
            )
            or not chapter_title.strip()
        )
    ):
        raise ValueError(
            "chapter_title must be a "
            "non-empty string or null."
        )

    if (
        chapter_id is not None
        and context.get("document_id")
        is None
    ):
        raise ValueError(
            "A named chapter must belong to "
            "a document."
        )

    return {
        field_name: context.get(
            field_name
        )
        for field_name in PAGE_CONTEXT_FIELDS
    }


def build_page_context_index(
    metadata: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    raw_contexts = metadata.get(
        "page_contexts"
    )

    # Backward compatibility for old sample
    # metadata that predates chapter support.
    if raw_contexts is None:
        return {}

    if not isinstance(raw_contexts, list):
        raise ValueError(
            "sample metadata page_contexts "
            "must be a list."
        )

    contexts_by_page: dict[
        int,
        dict[str, Any],
    ] = {}

    for raw_context in raw_contexts:
        if not isinstance(
            raw_context,
            dict,
        ):
            raise ValueError(
                "Each page_context must be "
                "an object."
            )

        context = validate_context_record(
            raw_context
        )

        canonical_page = int(
            context["canonical_page"]
        )

        if canonical_page in contexts_by_page:
            raise ValueError(
                "Duplicate canonical page in "
                "page_contexts: "
                f"{canonical_page}"
            )

        contexts_by_page[
            canonical_page
        ] = context

    source_start_page = metadata.get(
        "source_start_page"
    )
    source_end_page = metadata.get(
        "source_end_page"
    )
    batch_page_count = metadata.get(
        "batch_page_count"
    )

    if contexts_by_page:
        if not isinstance(
            source_start_page,
            int,
        ) or not isinstance(
            source_end_page,
            int,
        ):
            raise ValueError(
                "Metadata with page_contexts "
                "must define integer "
                "source_start_page and "
                "source_end_page."
            )

        expected_pages = list(
            range(
                source_start_page,
                source_end_page + 1,
            )
        )

        actual_pages = sorted(
            contexts_by_page
        )

        if actual_pages != expected_pages:
            raise ValueError(
                "page_contexts do not provide "
                "exact contiguous coverage for "
                "the configured source range."
            )

        if (
            isinstance(
                batch_page_count,
                int,
            )
            and batch_page_count
            != len(expected_pages)
        ):
            raise ValueError(
                "batch_page_count does not "
                "match the source page range."
            )

        for canonical_page in expected_pages:
            context = contexts_by_page[
                canonical_page
            ]

            batch_page = context.get(
                "batch_page"
            )

            expected_batch_page = (
                canonical_page
                - source_start_page
                + 1
            )

            if (
                batch_page is not None
                and batch_page
                != expected_batch_page
            ):
                raise ValueError(
                    "Page-context batch_page "
                    "does not match its "
                    "canonical-page offset."
                )

    return contexts_by_page


def determine_chapter_status(
    requested_pages: list[int],
    matched_contexts: list[
        dict[str, Any]
    ],
    missing_pages: list[int],
    index_available: bool,
) -> str:
    if not requested_pages:
        return "no_pages"

    if not index_available:
        return "metadata_unavailable"

    if not matched_contexts:
        return "unresolved"

    chapter_ids = ordered_unique(
        context.get("chapter_id")
        for context in matched_contexts
    )

    has_unnamed_pages = any(
        context.get("chapter_id") is None
        for context in matched_contexts
    )

    if missing_pages:
        return "partial"

    if len(chapter_ids) > 1:
        return "multiple"

    if len(chapter_ids) == 1:
        if has_unnamed_pages:
            return "multiple"

        return "single"

    return "none"


def resolve_page_context(
    source_page_numbers: Any,
    contexts_by_page: dict[
        int,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    requested_pages = normalize_page_numbers(
        source_page_numbers
    )

    matched_contexts = [
        contexts_by_page[page_number]
        for page_number in requested_pages
        if page_number in contexts_by_page
    ]

    missing_pages = [
        page_number
        for page_number in requested_pages
        if page_number not in contexts_by_page
    ]

    document_ids = ordered_unique(
        context.get("document_id")
        for context in matched_contexts
    )

    document_titles = ordered_unique(
        context.get("document_title")
        for context in matched_contexts
    )

    document_types = ordered_unique(
        context.get("document_type")
        for context in matched_contexts
    )

    chapter_ids = ordered_unique(
        context.get("chapter_id")
        for context in matched_contexts
    )

    chapter_titles = ordered_unique(
        context.get("chapter_title")
        for context in matched_contexts
    )

    unit_numbers = ordered_unique(
        context.get("unit_number")
        for context in matched_contexts
    )

    page_types = ordered_unique(
        context.get("page_type")
        for context in matched_contexts
    )

    source_filenames = ordered_unique(
        context.get("source_filename")
        for context in matched_contexts
    )

    chapter_status = (
        determine_chapter_status(
            requested_pages,
            matched_contexts,
            missing_pages,
            bool(contexts_by_page),
        )
    )

    if not requested_pages:
        page_context_status = "no_pages"

    elif not contexts_by_page:
        page_context_status = (
            "metadata_unavailable"
        )

    elif not matched_contexts:
        page_context_status = "unresolved"

    elif missing_pages:
        page_context_status = "partial"

    else:
        page_context_status = "resolved"

    return {
        "page_context_status": (
            page_context_status
        ),
        "chapter_context_status": (
            chapter_status
        ),
        "resolved_page_contexts": (
            matched_contexts
        ),
        "unresolved_source_page_numbers": (
            missing_pages
        ),
        "page_types": page_types,
        "document_ids": document_ids,
        "document_titles": (
            document_titles
        ),
        "document_types": document_types,
        "unit_numbers": unit_numbers,
        "chapter_ids": chapter_ids,
        "chapter_titles": chapter_titles,
        "source_filenames": (
            source_filenames
        ),
        # Singular fields are populated only
        # when the element belongs to exactly
        # one unambiguous document/chapter.
        "document_id": (
            single_value_or_none(
                document_ids
            )
        ),
        "document_title": (
            single_value_or_none(
                document_titles
            )
        ),
        "document_type": (
            single_value_or_none(
                document_types
            )
        ),
        "unit_number": (
            single_value_or_none(
                unit_numbers
            )
        ),
        "chapter_id": (
            single_value_or_none(
                chapter_ids
            )
            if chapter_status == "single"
            else None
        ),
        "chapter_title": (
            single_value_or_none(
                chapter_titles
            )
            if chapter_status == "single"
            else None
        ),
    }


def enrich_record_with_page_context(
    record: dict[str, Any],
    contexts_by_page: dict[
        int,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    enriched = dict(record)

    enriched.update(
        resolve_page_context(
            record.get(
                "source_page_numbers",
                [],
            ),
            contexts_by_page,
        )
    )

    return enriched
