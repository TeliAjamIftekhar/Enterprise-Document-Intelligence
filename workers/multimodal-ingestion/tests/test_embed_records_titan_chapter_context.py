from scripts.embed_records_titan_v2 import (
    build_consolidated_record,
    build_item,
    copy_context_metadata,
    validate_saved_item,
)


MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 256


def normalized_vector() -> list[float]:
    return [1.0] + [0.0] * (
        DIMENSIONS - 1
    )


def base_source_record() -> dict:
    return {
        "schema_version": "1.0",
        "record_id": (
            "unit-1:chunk-0001"
        ),
        "source_unit_id": "unit-1",
        "book_id": "test-book",
        "book_version": "v1-test",
        "element_index": 1,
        "element_type": "TEXT",
        "element_sub_type": "PARAGRAPH",
        "modality": "paragraph",
        "retrieval_priority": "normal",
        "chunk_index": 1,
        "chunk_count": 1,
        "source_page_numbers": [43],
        "locations": [],
        "citation_label": (
            "test-book, page 43"
        ),
        "embedding_text": (
            "A detailed educational paragraph "
            "from Chapter A."
        ),
        "character_count": 48,
        "asset_s3_uris": [],
        "asset_local_paths": [],
        "quality_flags": [],
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
        "context_citation_label": (
            "Chapter A "
            "(test-book, page 43)"
        ),
    }


def build_test_item(
    record: dict,
) -> dict:
    return build_item(
        record=record,
        vector=normalized_vector(),
        input_token_count=12,
        model_id=MODEL_ID,
        dimensions=DIMENSIONS,
        normalize=True,
        source="unit_test",
    )


def test_copy_context_metadata_only_copies_present_fields():
    record = base_source_record()

    record.update({
        "document_id": "unit-1",
        "chapter_id": "chapter-a",
        "chapter_title": "Chapter A",
    })

    copied = copy_context_metadata(
        record
    )

    assert copied == {
        "document_id": "unit-1",
        "chapter_id": "chapter-a",
        "chapter_title": "Chapter A",
    }


def test_consolidated_record_preserves_chapter_context():
    source_record = base_source_record()
    source_record.update(
        chapter_context()
    )

    item = build_test_item(
        source_record
    )

    consolidated = (
        build_consolidated_record(
            source_record=source_record,
            item=item,
        )
    )

    assert consolidated[
        "document_id"
    ] == "unit-1"

    assert consolidated[
        "chapter_id"
    ] == "chapter-a"

    assert consolidated[
        "chapter_title"
    ] == "Chapter A"

    assert consolidated[
        "chapter_ids"
    ] == ["chapter-a"]

    assert consolidated[
        "context_citation_label"
    ] == (
        "Chapter A "
        "(test-book, page 43)"
    )

    assert len(
        consolidated["embedding"]
    ) == DIMENSIONS

    assert consolidated[
        "embedding_dimensions"
    ] == DIMENSIONS


def test_consolidated_legacy_schema_is_unchanged():
    source_record = base_source_record()

    item = build_test_item(
        source_record
    )

    consolidated = (
        build_consolidated_record(
            source_record=source_record,
            item=item,
        )
    )

    context_fields = {
        "page_context_status",
        "chapter_context_status",
        "document_id",
        "document_ids",
        "chapter_id",
        "chapter_ids",
        "chapter_title",
        "chapter_titles",
        "context_citation_label",
    }

    unexpected_fields = (
        context_fields
        & set(consolidated)
    )

    assert unexpected_fields == set()


def test_existing_checkpoint_accepts_new_context_metadata():
    legacy_record = base_source_record()

    saved_item = build_test_item(
        legacy_record
    )

    enriched_record = dict(
        legacy_record
    )
    enriched_record.update(
        chapter_context()
    )

    # Chapter metadata does not change the
    # embedded text or checkpoint identity.
    # Therefore an existing valid checkpoint
    # remains reusable without a model call.
    validate_saved_item(
        item=saved_item,
        record=enriched_record,
        model_id=MODEL_ID,
        dimensions=DIMENSIONS,
        normalize=True,
    )
