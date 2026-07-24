from copy import deepcopy
from pathlib import Path

from src.opensearch_index_schema import (
    BASE_PROPERTIES,
    CONTEXT_PROPERTIES,
    build_index_plan,
    build_index_schema,
    resolve_index_runtime,
    validate_index_schema,
)


CONFIG_PATH = Path(
    "workers/multimodal-ingestion/"
    "config/books/"
    "grade-9-english-kaveri-"
    "v1-chapter-test.json"
)


def test_resolves_chapter_test_runtime():
    runtime = resolve_index_runtime(
        CONFIG_PATH
    )

    assert runtime["mode"] == (
        "book_config"
    )

    assert runtime["region"] == (
        "us-east-1"
    )

    assert runtime[
        "collection_name"
    ] == "edi-textbook-vector"

    assert runtime[
        "collection_id"
    ] == "kqjqddn0b5gmcfvgsd2e"

    assert runtime["index_name"] == (
        "grade-9-english-kaveri-"
        "v1-chapter-test"
    )

    assert runtime[
        "vector_dimensions"
    ] == 1024

    assert runtime[
        "include_context_fields"
    ] is True

    assert runtime["output_path"] == (
        "data/multimodal-output/"
        "grade-9-english-kaveri/"
        "v1-chapter-test/"
        "unified-normalized/"
        "opensearch-serverless/"
        "index-provisioning-report.json"
    )


def test_preserves_legacy_runtime():
    runtime = resolve_index_runtime(
        None
    )

    assert runtime["mode"] == "legacy"

    assert runtime["index_name"] == (
        "grade-9-english-kaveri-v1"
    )

    assert runtime[
        "vector_dimensions"
    ] == 1024

    assert runtime[
        "include_context_fields"
    ] is False

    assert runtime["output_path"] == (
        "data/multimodal-output/"
        "grade-9-english-kaveri/v1/"
        "opensearch-serverless/"
        "index-provisioning-report.json"
    )


def test_builds_strict_chapter_schema():
    plan = build_index_plan(
        CONFIG_PATH
    )

    schema = plan["schema"]

    properties = schema[
        "mappings"
    ][
        "properties"
    ]

    assert schema["mappings"][
        "dynamic"
    ] == "strict"

    assert plan["field_count"] == (
        len(BASE_PROPERTIES)
        + len(CONTEXT_PROPERTIES)
        + 1
    )

    assert plan["field_count"] == 46

    assert properties[
        "chapter_id"
    ]["type"] == "keyword"

    assert properties[
        "chapter_title"
    ]["type"] == "text"

    assert properties[
        "context_citation_label"
    ]["type"] == "text"

    assert properties[
        "unit_number"
    ]["type"] == "integer"

    assert properties[
        "embedding"
    ]["dimension"] == 1024


def test_generated_legacy_and_chapter_schemas_validate():
    legacy = build_index_plan(None)

    chapter = build_index_plan(
        CONFIG_PATH
    )

    assert legacy[
        "schema_validation_errors"
    ] == []

    assert chapter[
        "schema_validation_errors"
    ] == []

    assert legacy["field_count"] == 27
    assert chapter["field_count"] == 46


def test_validator_rejects_missing_context_mapping():
    schema = build_index_schema(
        vector_dimensions=1024,
        include_context_fields=True,
    )

    broken = deepcopy(schema)

    del broken[
        "mappings"
    ][
        "properties"
    ][
        "chapter_id"
    ]

    errors = validate_index_schema(
        broken,
        vector_dimensions=1024,
        include_context_fields=True,
    )

    assert (
        "Required field missing: "
        "chapter_id"
    ) in errors


def test_validator_rejects_wrong_dimension():
    schema = build_index_schema(
        vector_dimensions=256,
        include_context_fields=True,
    )

    errors = validate_index_schema(
        schema,
        vector_dimensions=1024,
        include_context_fields=True,
    )

    assert (
        "embedding dimension mismatch: "
        "256"
    ) in errors
