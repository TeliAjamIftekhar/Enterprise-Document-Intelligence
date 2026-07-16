from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


PAGE_CONTEXT_FIELDS = (
    "canonical_page",
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


def calculate_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while block := file_handle.read(
            1024 * 1024
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON file: {path}"
        ) from error

    if not isinstance(value, dict):
        raise ValueError(
            f"JSON root must be an object: {path}"
        )

    return value


def write_json_atomically(
    path: Path,
    value: dict[str, Any],
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
    ) as temporary_file:
        temporary_path = Path(
            temporary_file.name
        )

        json.dump(
            value,
            temporary_file,
            indent=2,
            ensure_ascii=False,
        )
        temporary_file.write("\n")

    temporary_path.replace(path)


def validate_page_map(
    page_map: dict[str, Any],
    batch_manifest: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    if (
        page_map.get("book_id")
        != batch_manifest.get("book_id")
    ):
        raise ValueError(
            "Page-map and batch-manifest "
            "book IDs do not match."
        )

    if (
        page_map.get("book_version")
        != batch_manifest.get(
            "book_version"
        )
    ):
        raise ValueError(
            "Page-map and batch-manifest "
            "book versions do not match."
        )

    source = batch_manifest.get(
        "source",
        {},
    )

    if not isinstance(source, dict):
        raise ValueError(
            "Batch manifest has no valid "
            "source object."
        )

    expected_page_count = source.get(
        "page_count"
    )

    if not isinstance(
        expected_page_count,
        int,
    ):
        raise ValueError(
            "Batch manifest source page count "
            "is invalid."
        )

    if (
        page_map.get(
            "canonical_page_count"
        )
        != expected_page_count
    ):
        raise ValueError(
            "Page-map canonical page count "
            "does not match the batch source."
        )

    pages = page_map.get("pages")

    if not isinstance(pages, list):
        raise ValueError(
            "Page map has no pages list."
        )

    if len(pages) != expected_page_count:
        raise ValueError(
            "Page-map page count does not "
            "match its canonical page count."
        )

    pages_by_number: dict[
        int,
        dict[str, Any],
    ] = {}

    for page in pages:
        if not isinstance(page, dict):
            raise ValueError(
                "Page-map entries must be "
                "objects."
            )

        page_number = page.get(
            "canonical_page"
        )

        if not isinstance(page_number, int):
            raise ValueError(
                "Page-map entry has an invalid "
                "canonical page number."
            )

        if page_number in pages_by_number:
            raise ValueError(
                "Duplicate canonical page in "
                f"page map: {page_number}"
            )

        pages_by_number[
            page_number
        ] = page

    expected_numbers = set(
        range(1, expected_page_count + 1)
    )

    actual_numbers = set(
        pages_by_number
    )

    missing_pages = sorted(
        expected_numbers - actual_numbers
    )

    unexpected_pages = sorted(
        actual_numbers - expected_numbers
    )

    if missing_pages:
        raise ValueError(
            "Page map is missing canonical "
            f"pages: {missing_pages}"
        )

    if unexpected_pages:
        raise ValueError(
            "Page map contains unexpected "
            f"pages: {unexpected_pages}"
        )

    return pages_by_number


def span_identity(
    page: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        page.get("page_type"),
        page.get("document_order"),
        page.get("document_id"),
        page.get("document_type"),
        page.get("document_title"),
        page.get("source_filename"),
        page.get("unit_number"),
        page.get("chapter_id"),
        page.get("chapter_title"),
    )


def build_page_span(
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    first = pages[0]
    last = pages[-1]

    return {
        "batch_page_start": (
            first["batch_page"]
        ),
        "batch_page_end": (
            last["batch_page"]
        ),
        "canonical_page_start": (
            first["canonical_page"]
        ),
        "canonical_page_end": (
            last["canonical_page"]
        ),
        "page_count": len(pages),
        "page_type": first.get(
            "page_type"
        ),
        "document_order": first.get(
            "document_order"
        ),
        "document_id": first.get(
            "document_id"
        ),
        "document_type": first.get(
            "document_type"
        ),
        "document_title": first.get(
            "document_title"
        ),
        "source_filename": first.get(
            "source_filename"
        ),
        "source_page_start": first.get(
            "source_page"
        ),
        "source_page_end": last.get(
            "source_page"
        ),
        "unit_number": first.get(
            "unit_number"
        ),
        "chapter_id": first.get(
            "chapter_id"
        ),
        "chapter_title": first.get(
            "chapter_title"
        ),
        "chapter_page_start": first.get(
            "chapter_page"
        ),
        "chapter_page_end": last.get(
            "chapter_page"
        ),
    }


def build_spans(
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not pages:
        return []

    spans: list[dict[str, Any]] = []
    current_pages = [pages[0]]
    current_identity = span_identity(
        pages[0]
    )

    for page in pages[1:]:
        identity = span_identity(page)

        if identity == current_identity:
            current_pages.append(page)
            continue

        spans.append(
            build_page_span(
                current_pages
            )
        )

        current_pages = [page]
        current_identity = identity

    spans.append(
        build_page_span(
            current_pages
        )
    )

    return spans


def ordered_unique_values(
    pages: list[dict[str, Any]],
    field_name: str,
) -> list[Any]:
    values = []
    seen = set()

    for page in pages:
        value = page.get(field_name)

        if value is None:
            continue

        if value in seen:
            continue

        seen.add(value)
        values.append(value)

    return values


def build_batch_page_contexts(
    batch: dict[str, Any],
    pages_by_number: dict[
        int,
        dict[str, Any],
    ],
) -> list[dict[str, Any]]:
    source_page_start = batch.get(
        "source_page_start"
    )

    source_page_end = batch.get(
        "source_page_end"
    )

    page_count = batch.get("page_count")

    if not isinstance(
        source_page_start,
        int,
    ):
        raise ValueError(
            "Batch has an invalid source "
            "start page."
        )

    if not isinstance(
        source_page_end,
        int,
    ):
        raise ValueError(
            "Batch has an invalid source "
            "end page."
        )

    if not isinstance(page_count, int):
        raise ValueError(
            "Batch has an invalid page count."
        )

    expected_page_count = (
        source_page_end
        - source_page_start
        + 1
    )

    if page_count != expected_page_count:
        raise ValueError(
            f"Batch {batch.get('batch_id')} "
            "page count does not match its "
            "source range."
        )

    contexts = []

    for canonical_page in range(
        source_page_start,
        source_page_end + 1,
    ):
        page = pages_by_number.get(
            canonical_page
        )

        if page is None:
            raise ValueError(
                "No page-map context for "
                f"canonical page {canonical_page}."
            )

        context = {
            field_name: page.get(
                field_name
            )
            for field_name
            in PAGE_CONTEXT_FIELDS
        }

        context["batch_page"] = (
            canonical_page
            - source_page_start
            + 1
        )

        contexts.append(context)

    return contexts


def validate_batch_coverage(
    batches: list[dict[str, Any]],
    expected_page_count: int,
) -> None:
    covered_pages = []

    for batch in batches:
        start = batch.get(
            "source_page_start"
        )
        end = batch.get(
            "source_page_end"
        )

        if not isinstance(
            start,
            int,
        ) or not isinstance(
            end,
            int,
        ):
            raise ValueError(
                "Batch source ranges are "
                "invalid."
            )

        covered_pages.extend(
            range(start, end + 1)
        )

    expected_pages = list(
        range(1, expected_page_count + 1)
    )

    if covered_pages != expected_pages:
        raise ValueError(
            "Batch page ranges do not provide "
            "exact contiguous textbook coverage."
        )


def build_normalizer_metadata(
    batch_manifest: dict[str, Any],
    batch: dict[str, Any],
    page_contexts: list[
        dict[str, Any]
    ],
    chapter_spans: list[
        dict[str, Any]
    ],
    page_map_path: Path,
) -> dict[str, Any]:
    source = batch_manifest["source"]

    return {
        "schema_version": "1.0",
        "metadata_type": (
            "full_book_batch"
        ),
        "book_id": batch_manifest[
            "book_id"
        ],
        "book_version": batch_manifest[
            "book_version"
        ],
        "batch_id": batch["batch_id"],
        "batch_number": batch[
            "batch_number"
        ],
        "source_start_page": batch[
            "source_page_start"
        ],
        "source_end_page": batch[
            "source_page_end"
        ],
        "source_page_offset": batch[
            "source_page_offset"
        ],
        "batch_page_count": batch[
            "page_count"
        ],
        "source_pdf": source[
            "local_path"
        ],
        "source_pdf_uri": source.get(
            "configured_s3_uri"
        ),
        "sample_s3_uri": batch[
            "s3_uri"
        ],
        "batch_s3_uri": batch[
            "s3_uri"
        ],
        "batch_local_path": batch[
            "local_path"
        ],
        "chapter_page_map_path": str(
            page_map_path
        ),
        "document_ids": (
            ordered_unique_values(
                page_contexts,
                "document_id",
            )
        ),
        "chapter_ids": (
            ordered_unique_values(
                page_contexts,
                "chapter_id",
            )
        ),
        "page_contexts": page_contexts,
        "chapter_spans": chapter_spans,
        "aws_calls": 0,
    }


def enrich_batch_manifest(
    batch_manifest_path: Path,
    page_map_path: Path,
    metadata_directory: Path,
    *,
    output_manifest_path: (
        Path | None
    ) = None,
) -> dict[str, Any]:
    batch_manifest = load_json_object(
        batch_manifest_path
    )

    page_map = load_json_object(
        page_map_path
    )

    pages_by_number = validate_page_map(
        page_map,
        batch_manifest,
    )

    batches = batch_manifest.get(
        "batches"
    )

    if not isinstance(batches, list):
        raise ValueError(
            "Batch manifest has no batches list."
        )

    expected_page_count = (
        batch_manifest["source"][
            "page_count"
        ]
    )

    validate_batch_coverage(
        batches,
        expected_page_count,
    )

    metadata_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_context_count = 0
    total_span_count = 0
    chapter_page_count = 0
    document_page_count = 0
    blank_page_count = 0
    all_chapter_ids = []
    seen_chapter_ids = set()

    for batch in batches:
        page_contexts = (
            build_batch_page_contexts(
                batch,
                pages_by_number,
            )
        )

        chapter_spans = build_spans(
            page_contexts
        )

        for page in page_contexts:
            total_context_count += 1

            if (
                page.get("document_id")
                is not None
            ):
                document_page_count += 1
            else:
                blank_page_count += 1

            chapter_id = page.get(
                "chapter_id"
            )

            if chapter_id is not None:
                chapter_page_count += 1

                if (
                    chapter_id
                    not in seen_chapter_ids
                ):
                    seen_chapter_ids.add(
                        chapter_id
                    )
                    all_chapter_ids.append(
                        chapter_id
                    )

        total_span_count += len(
            chapter_spans
        )

        metadata = (
            build_normalizer_metadata(
                batch_manifest,
                batch,
                page_contexts,
                chapter_spans,
                page_map_path,
            )
        )

        metadata_path = (
            metadata_directory
            / f"{batch['batch_id']}.json"
        )

        write_json_atomically(
            metadata_path,
            metadata,
        )

        batch["chapter_metadata"] = {
            "status": "ATTACHED",
            "page_context_count": len(
                page_contexts
            ),
            "span_count": len(
                chapter_spans
            ),
            "document_ids": metadata[
                "document_ids"
            ],
            "chapter_ids": metadata[
                "chapter_ids"
            ],
            "chapter_spans": (
                chapter_spans
            ),
        }

        batch[
            "normalizer_metadata_path"
        ] = str(metadata_path)

        batch[
            "normalizer_metadata_sha256"
        ] = calculate_sha256(
            metadata_path
        )

    batch_manifest[
        "chapter_metadata"
    ] = {
        "status": "ATTACHED",
        "page_map_path": str(
            page_map_path
        ),
        "page_map_sha256": (
            calculate_sha256(
                page_map_path
            )
        ),
        "batch_metadata_directory": str(
            metadata_directory
        ),
        "batch_metadata_count": len(
            batches
        ),
        "page_context_count": (
            total_context_count
        ),
        "span_count": (
            total_span_count
        ),
        "document_page_count": (
            document_page_count
        ),
        "chapter_page_count": (
            chapter_page_count
        ),
        "blank_page_count": (
            blank_page_count
        ),
        "unique_chapter_count": len(
            all_chapter_ids
        ),
        "chapter_ids": (
            all_chapter_ids
        ),
        "aws_calls": 0,
    }

    batch_manifest["aws_calls"] = 0

    destination = (
        output_manifest_path
        if output_manifest_path
        is not None
        else batch_manifest_path
    )

    write_json_atomically(
        destination,
        batch_manifest,
    )

    return batch_manifest
