import pytest

from scripts.prepare_opensearch_bulk import (
    DOCUMENT_FIELDS,
    OPTIONAL_CONTEXT_FIELDS,
    REQUIRED_DOCUMENT_FIELDS,
    validate_record,
)


VECTOR_DIMENSIONS = 1024


def normalized_vector() -> list[float]:
    return [1.0] + [0.0] * (
        VECTOR_DIMENSIONS - 1
    )


def base_embedding_record() -> dict:
    text = (
        "A detailed educational paragraph "
        "used for OpenSearch bulk testing."
    )

    return {
        "schema_version": "1.0",
        "record_id": (
            "unit-1:chunk-0001"
        ),
        "book_id": "test-book",
        "book_version": "v1-test",
        "source_unit_id": "unit-1",
        "element_index": 1,
        "element_type": "TEXT",
        "element_sub_type": "PARAGRAPH",
        "modality": "paragraph",
        "source_page_numbers": [43],
        "citation_label": (
            "test-book, page 43"
        ),
        "embedding_text": text,
        "asset_s3_uris": [],
        "quality_flags": [],
        "retrieval_priority": "normal",
        "chunk_index": 1,
        "chunk_count": 1,
        "character_count": len(text),
        "input_token_count": 12,
        "input_text_sha256": "a" * 64,
        "embedding_model_id": (
            "amazon.titan-embed-text-v2:0"
        ),
        "embedding_dimensions": (
            VECTOR_DIMENSIONS
        ),
        "embedding_normalized": True,
        "vector_length": VECTOR_DIMENSIONS,
        "vector_l2_norm": 1.0,
        "locations": [],
        "embedding": normalized_vector(),
        "asset_local_paths": [
            "/tmp/local-only.png"
        ],
    }


def chapter_context() -> dict:
    return {
        "page_context_status": "resolved",
        "chapter_context_status": "single",
        "unresolved_source_page_numbers": [],
        "page_types": ["unit"],
        "document_ids": ["unit-1"],
        "document_titles": ["Unit 1"],
        "document_types": ["unit"],
        "unit_numbers": [1],
        "chapter_ids": ["chapter-a"],
        "chapter_titles": ["Chapter A"],
        "source_filenames": ["unit-1.pdf"],
        "document_id": "unit-1",
        "document_title": "Unit 1",
        "document_type": "unit",
        "unit_number": 1,
        "chapter_id": "chapter-a",
        "chapter_title": "Chapter A",
        "section_id": None,
        "context_citation_label": (
            "Chapter A "
            "(test-book, page 43)"
        ),
    }


def test_field_groups_are_disjoint():
    required = set(
        REQUIRED_DOCUMENT_FIELDS
    )

    optional = set(
        OPTIONAL_CONTEXT_FIELDS
    )

    assert required
    assert optional
    assert required.isdisjoint(optional)

    assert set(DOCUMENT_FIELDS) == (
        required | optional
    )


def test_bulk_document_preserves_chapter_context():
    record = base_embedding_record()
    record.update(
        chapter_context()
    )

    document, norm = validate_record(
        record=record,
        line_number=1,
    )

    assert norm == pytest.approx(1.0)

    assert document[
        "document_id"
    ] == "unit-1"

    assert document[
        "chapter_id"
    ] == "chapter-a"

    assert document[
        "chapter_title"
    ] == "Chapter A"

    assert document[
        "chapter_ids"
    ] == ["chapter-a"]

    assert document[
        "context_citation_label"
    ] == (
        "Chapter A "
        "(test-book, page 43)"
    )

    assert document[
        "section_id"
    ] is None

    assert (
        "asset_local_paths"
        not in document
    )

    assert len(
        document["embedding"]
    ) == VECTOR_DIMENSIONS


def test_legacy_bulk_document_schema_remains_valid():
    record = base_embedding_record()

    document, norm = validate_record(
        record=record,
        line_number=1,
    )

    assert norm == pytest.approx(1.0)

    for field_name in (
        "document_id",
        "chapter_id",
        "chapter_title",
        "context_citation_label",
    ):
        assert field_name not in document

    assert (
        set(REQUIRED_DOCUMENT_FIELDS)
        <= set(document)
    )

    assert (
        "asset_local_paths"
        not in document
    )


def test_rejects_unknown_field():
    record = base_embedding_record()

    record["unexpected_field"] = (
        "not mapped"
    )

    with pytest.raises(
        ValueError,
        match="contains unmapped fields",
    ):
        validate_record(
            record=record,
            line_number=1,
        )
