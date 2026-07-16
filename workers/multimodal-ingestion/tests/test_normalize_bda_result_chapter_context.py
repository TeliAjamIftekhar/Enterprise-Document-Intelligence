from pathlib import Path

from scripts.normalize_bda_result import (
    normalize,
)


def build_page_context(
    canonical_page: int,
    batch_page: int,
    *,
    chapter_id: str,
    chapter_title: str,
    chapter_page: int,
) -> dict:
    return {
        "canonical_page": canonical_page,
        "batch_page": batch_page,
        "page_type": "unit",
        "document_order": 1,
        "document_id": "unit-1",
        "document_type": "unit",
        "document_title": "Unit 1",
        "source_filename": "unit-1.pdf",
        "source_page": canonical_page,
        "unit_number": 1,
        "chapter_id": chapter_id,
        "chapter_title": chapter_title,
        "chapter_page": chapter_page,
    }


def build_chapter_metadata() -> dict:
    return {
        "book_id": "test-book",
        "book_version": "v1-test",
        "source_pdf": "source.pdf",
        "sample_s3_uri": (
            "s3://example/batch.pdf"
        ),
        "source_start_page": 41,
        "source_end_page": 43,
        "batch_page_count": 3,
        "page_contexts": [
            build_page_context(
                41,
                1,
                chapter_id="chapter-a",
                chapter_title="Chapter A",
                chapter_page=1,
            ),
            build_page_context(
                42,
                2,
                chapter_id="chapter-a",
                chapter_title="Chapter A",
                chapter_page=2,
            ),
            build_page_context(
                43,
                3,
                chapter_id="chapter-b",
                chapter_title="Chapter B",
                chapter_page=1,
            ),
        ],
    }


def build_result() -> dict:
    return {
        "metadata": {
            "number_of_pages": 3,
        },
        "document": {
            "statistics": {
                "table_count": 1,
                "figure_count": 1,
            }
        },
        "elements": [
            {
                "type": "TEXT",
                "id": "text-1",
                "sub_type": "PARAGRAPH",
                "reading_order": 0,
                "page_indices": [0],
                "representation": {
                    "text": "Chapter A text.",
                    "markdown": (
                        "Chapter A text."
                    ),
                },
                "locations": [
                    {
                        "page_index": 0,
                        "bounding_box": {
                            "left": 0.1,
                            "top": 0.1,
                            "width": 0.5,
                            "height": 0.1,
                        },
                    }
                ],
            },
            {
                "type": "FIGURE",
                "id": "figure-1",
                "sub_type": "IMAGE",
                "reading_order": 1,
                "page_indices": [1, 2],
                "representation": {
                    "text": "",
                    "markdown": "",
                },
                "title": "Cross-chapter figure",
                "summary": (
                    "A figure spanning two "
                    "chapters."
                ),
                "locations": [
                    {
                        "page_index": 1,
                    },
                    {
                        "page_index": 2,
                    },
                ],
                "crop_images": [],
            },
            {
                "type": "TABLE",
                "id": "table-1",
                "sub_type": "TABLE",
                "reading_order": 2,
                "page_indices": [2],
                "representation": {
                    "text": "Column A Column B",
                    "markdown": (
                        "| Column A | Column B |"
                    ),
                },
                "title": "Chapter B table",
                "summary": "A test table.",
                "locations": [
                    {
                        "page_index": 2,
                    }
                ],
                "crop_images": [],
            },
        ],
    }


def record_by_element_id(
    records: list[dict],
    element_id: str,
) -> dict:
    return next(
        record
        for record in records
        if record["bda_element_id"]
        == element_id
    )


def test_normalizer_attaches_chapter_context(
    tmp_path: Path,
):
    (
        content_units,
        figures,
        tables,
        report,
    ) = normalize(
        result=build_result(),
        sample_metadata=(
            build_chapter_metadata()
        ),
        result_json_path=(
            tmp_path / "result.json"
        ),
    )

    text_unit = record_by_element_id(
        content_units,
        "text-1",
    )

    figure_unit = record_by_element_id(
        content_units,
        "figure-1",
    )

    table_unit = record_by_element_id(
        content_units,
        "table-1",
    )

    assert text_unit["source_page_numbers"] == [
        41
    ]
    assert text_unit["document_id"] == (
        "unit-1"
    )
    assert text_unit["chapter_id"] == (
        "chapter-a"
    )
    assert text_unit["chapter_title"] == (
        "Chapter A"
    )
    assert (
        text_unit["chapter_context_status"]
        == "single"
    )

    assert figure_unit[
        "source_page_numbers"
    ] == [42, 43]
    assert (
        figure_unit[
            "chapter_context_status"
        ]
        == "multiple"
    )
    assert figure_unit["chapter_id"] is None
    assert figure_unit["chapter_ids"] == [
        "chapter-a",
        "chapter-b",
    ]

    assert table_unit["chapter_id"] == (
        "chapter-b"
    )
    assert table_unit["chapter_title"] == (
        "Chapter B"
    )

    figure = record_by_element_id(
        figures,
        "figure-1",
    )

    table = record_by_element_id(
        tables,
        "table-1",
    )

    assert (
        figure["chapter_ids"]
        == figure_unit["chapter_ids"]
    )
    assert (
        figure["chapter_context_status"]
        == "multiple"
    )

    assert table["chapter_id"] == (
        table_unit["chapter_id"]
    )
    assert table["document_id"] == (
        "unit-1"
    )

    assert (
        report[
            "page_context_metadata_available"
        ]
        is True
    )
    assert report[
        "page_context_record_count"
    ] == 3
    assert report[
        "page_context_status_counts"
    ] == {
        "resolved": 3,
    }
    assert report[
        "chapter_context_status_counts"
    ] == {
        "single": 2,
        "multiple": 1,
    }
    assert (
        report["normalization_policy"]
        [
            "chapter_context_from_page_metadata"
        ]
        is True
    )


def test_normalizer_preserves_legacy_records(
    tmp_path: Path,
):
    result = {
        "metadata": {
            "number_of_pages": 1,
        },
        "document": {
            "statistics": {
                "table_count": 0,
                "figure_count": 0,
            }
        },
        "elements": [
            {
                "type": "TEXT",
                "id": "legacy-text",
                "sub_type": "PARAGRAPH",
                "page_indices": [0],
                "representation": {
                    "text": "Legacy text.",
                    "markdown": "Legacy text.",
                },
                "locations": [
                    {
                        "page_index": 0,
                    }
                ],
            }
        ],
    }

    legacy_metadata = {
        "book_id": "test-book",
        "book_version": "v1",
        "source_pdf": "source.pdf",
        "sample_s3_uri": (
            "s3://example/sample.pdf"
        ),
        "source_start_page": 1,
        "source_end_page": 1,
        "sample_page_count": 1,
    }

    (
        content_units,
        figures,
        tables,
        report,
    ) = normalize(
        result=result,
        sample_metadata=legacy_metadata,
        result_json_path=(
            tmp_path / "result.json"
        ),
    )

    assert len(content_units) == 1
    assert figures == []
    assert tables == []

    unit = content_units[0]

    # Legacy records retain their previous
    # schema when no page_contexts exist.
    assert "chapter_id" not in unit
    assert "document_id" not in unit
    assert (
        "page_context_status"
        not in unit
    )

    assert (
        report[
            "page_context_metadata_available"
        ]
        is False
    )
    assert report[
        "page_context_record_count"
    ] == 0
    assert report[
        "page_context_status_counts"
    ] == {}
    assert report[
        "chapter_context_status_counts"
    ] == {}
    assert (
        report["normalization_policy"]
        [
            "chapter_context_from_page_metadata"
        ]
        is False
    )
