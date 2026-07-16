from pathlib import Path

import pytest

from scripts.prepare_opensearch_bulk import (
    resolve_runtime_identity,
    validate_record,
    validate_records_match_runtime,
)


CONFIG_PATH = Path(
    "workers/multimodal-ingestion/"
    "config/books/"
    "grade-9-english-kaveri-"
    "v1-chapter-test.json"
)


def build_record(
    *,
    dimensions: int = 1024,
    book_id: str = (
        "grade-9-english-kaveri"
    ),
    book_version: str = (
        "v1-chapter-test"
    ),
) -> dict:
    text = (
        "A detailed educational paragraph "
        "for config-driven bulk testing."
    )

    return {
        "schema_version": "1.0",
        "record_id": "record-1",
        "book_id": book_id,
        "book_version": book_version,
        "source_unit_id": "unit-1",
        "element_index": 1,
        "element_type": "TEXT",
        "element_sub_type": "PARAGRAPH",
        "modality": "paragraph",
        "source_page_numbers": [43],
        "citation_label": (
            f"{book_id}, page 43"
        ),
        "embedding_text": text,
        "asset_s3_uris": [],
        "quality_flags": [],
        "retrieval_priority": "normal",
        "chunk_index": 1,
        "chunk_count": 1,
        "character_count": len(text),
        "input_token_count": 10,
        "input_text_sha256": "a" * 64,
        "embedding_model_id": (
            "amazon.titan-embed-text-v2:0"
        ),
        "embedding_dimensions": dimensions,
        "embedding_normalized": True,
        "vector_length": dimensions,
        "vector_l2_norm": 1.0,
        "locations": [],
        "embedding": (
            [1.0]
            + [0.0] * (dimensions - 1)
        ),
    }


def test_resolves_chapter_test_config():
    runtime = resolve_runtime_identity(
        CONFIG_PATH
    )

    assert runtime["mode"] == (
        "book_config"
    )

    assert runtime["book_id"] == (
        "grade-9-english-kaveri"
    )

    assert runtime["book_version"] == (
        "v1-chapter-test"
    )

    assert runtime["index_name"] == (
        "grade-9-english-kaveri-"
        "v1-chapter-test"
    )

    assert runtime[
        "vector_dimensions"
    ] == 1024


def test_legacy_identity_is_preserved():
    runtime = resolve_runtime_identity(
        None
    )

    assert runtime["mode"] == "legacy"

    assert runtime["index_name"] == (
        "grade-9-english-kaveri-v1"
    )

    assert runtime[
        "vector_dimensions"
    ] == 1024


def test_rejects_record_identity_mismatch():
    runtime = resolve_runtime_identity(
        CONFIG_PATH
    )

    wrong_record = build_record(
        book_version="v1"
    )

    with pytest.raises(
        ValueError,
        match="book_version",
    ):
        validate_records_match_runtime(
            [wrong_record],
            runtime,
        )


def test_validate_record_accepts_runtime_dimensions():
    record = build_record(
        dimensions=256
    )

    document, norm = validate_record(
        record=record,
        line_number=1,
        vector_dimensions=256,
    )

    assert len(
        document["embedding"]
    ) == 256

    assert norm == pytest.approx(1.0)
