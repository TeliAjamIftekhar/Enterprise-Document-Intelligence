import pytest

from src.page_context import (
    build_page_context_index,
    enrich_record_with_page_context,
    resolve_page_context,
)


def context(
    canonical_page: int,
    *,
    batch_page: int,
    document_id: str | None,
    document_title: str | None,
    document_type: str | None,
    unit_number: int | None,
    chapter_id: str | None,
    chapter_title: str | None,
    chapter_page: int | None,
    page_type: str = "unit",
) -> dict:
    return {
        "canonical_page": canonical_page,
        "batch_page": batch_page,
        "page_type": page_type,
        "document_order": (
            unit_number
            if unit_number
            else None
        ),
        "document_id": document_id,
        "document_type": document_type,
        "document_title": document_title,
        "source_filename": (
            f"{document_id}.pdf"
            if document_id
            else None
        ),
        "source_page": (
            batch_page
            if document_id
            else None
        ),
        "unit_number": unit_number,
        "chapter_id": chapter_id,
        "chapter_title": chapter_title,
        "chapter_page": chapter_page,
    }


def build_metadata() -> dict:
    return {
        "source_start_page": 41,
        "source_end_page": 45,
        "batch_page_count": 5,
        "page_contexts": [
            context(
                41,
                batch_page=1,
                document_id="unit-1",
                document_title="Unit 1",
                document_type="unit",
                unit_number=1,
                chapter_id="chapter-a",
                chapter_title="Chapter A",
                chapter_page=21,
            ),
            context(
                42,
                batch_page=2,
                document_id="unit-1",
                document_title="Unit 1",
                document_type="unit",
                unit_number=1,
                chapter_id="chapter-a",
                chapter_title="Chapter A",
                chapter_page=22,
            ),
            context(
                43,
                batch_page=3,
                document_id="unit-1",
                document_title="Unit 1",
                document_type="unit",
                unit_number=1,
                chapter_id="chapter-b",
                chapter_title="Chapter B",
                chapter_page=1,
            ),
            context(
                44,
                batch_page=4,
                document_id="appendix",
                document_title="Appendix",
                document_type="appendix",
                unit_number=None,
                chapter_id=None,
                chapter_title=None,
                chapter_page=None,
                page_type="appendix",
            ),
            context(
                45,
                batch_page=5,
                document_id=None,
                document_title=None,
                document_type=None,
                unit_number=None,
                chapter_id=None,
                chapter_title=None,
                chapter_page=None,
                page_type="trailing_blank",
            ),
        ],
    }


def test_resolves_single_chapter():
    index = build_page_context_index(
        build_metadata()
    )

    result = resolve_page_context(
        [41, 42],
        index,
    )

    assert (
        result["page_context_status"]
        == "resolved"
    )
    assert (
        result["chapter_context_status"]
        == "single"
    )
    assert result["document_id"] == (
        "unit-1"
    )
    assert result["chapter_id"] == (
        "chapter-a"
    )
    assert result["chapter_title"] == (
        "Chapter A"
    )
    assert result["unit_number"] == 1
    assert len(
        result["resolved_page_contexts"]
    ) == 2


def test_marks_cross_chapter_element():
    index = build_page_context_index(
        build_metadata()
    )

    result = resolve_page_context(
        [42, 43],
        index,
    )

    assert (
        result["chapter_context_status"]
        == "multiple"
    )
    assert result["chapter_id"] is None
    assert result["chapter_title"] is None
    assert result["chapter_ids"] == [
        "chapter-a",
        "chapter-b",
    ]
    assert result["document_id"] == (
        "unit-1"
    )


def test_resolves_appendix_without_chapter():
    index = build_page_context_index(
        build_metadata()
    )

    result = resolve_page_context(
        [44],
        index,
    )

    assert (
        result["chapter_context_status"]
        == "none"
    )
    assert result["document_id"] == (
        "appendix"
    )
    assert result["chapter_id"] is None
    assert result["page_types"] == [
        "appendix"
    ]


def test_resolves_blank_page():
    index = build_page_context_index(
        build_metadata()
    )

    result = resolve_page_context(
        [45],
        index,
    )

    assert (
        result["page_context_status"]
        == "resolved"
    )
    assert (
        result["chapter_context_status"]
        == "none"
    )
    assert result["document_id"] is None
    assert result["chapter_id"] is None
    assert result["page_types"] == [
        "trailing_blank"
    ]


def test_reports_partial_resolution():
    index = build_page_context_index(
        build_metadata()
    )

    result = resolve_page_context(
        [41, 99],
        index,
    )

    assert (
        result["page_context_status"]
        == "partial"
    )
    assert (
        result["chapter_context_status"]
        == "partial"
    )
    assert (
        result[
            "unresolved_source_page_numbers"
        ]
        == [99]
    )


def test_supports_legacy_metadata():
    index = build_page_context_index({
        "source_start_page": 1,
        "source_end_page": 20,
    })

    result = resolve_page_context(
        [1],
        index,
    )

    assert index == {}
    assert (
        result["page_context_status"]
        == "metadata_unavailable"
    )
    assert (
        result["chapter_context_status"]
        == "metadata_unavailable"
    )


def test_enriches_record_without_mutation():
    index = build_page_context_index(
        build_metadata()
    )

    original = {
        "unit_id": "unit-123",
        "source_page_numbers": [43],
        "raw_text": "Example",
    }

    enriched = (
        enrich_record_with_page_context(
            original,
            index,
        )
    )

    assert "chapter_id" not in original
    assert enriched["chapter_id"] == (
        "chapter-b"
    )
    assert enriched["document_id"] == (
        "unit-1"
    )


def test_rejects_noncontiguous_contexts():
    metadata = build_metadata()

    metadata["page_contexts"] = (
        metadata["page_contexts"][:-1]
    )

    with pytest.raises(
        ValueError,
        match="contiguous coverage",
    ):
        build_page_context_index(
            metadata
        )
