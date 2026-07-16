from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.book_config import (
    load_book_config,
)


LEGACY_REGION = "us-east-1"

LEGACY_COLLECTION_NAME = (
    "edi-textbook-vector"
)

LEGACY_COLLECTION_ID = (
    "kqjqddn0b5gmcfvgsd2e"
)

LEGACY_INDEX_NAME = (
    "grade-9-english-kaveri-v1"
)

LEGACY_VECTOR_DIMENSIONS = 1024

LEGACY_OUTPUT_PATH = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/"
    "opensearch-serverless/"
    "index-provisioning-report.json"
)


def text_with_keyword_mapping(
    *,
    ignore_above: int = 512,
) -> dict[str, Any]:
    return {
        "type": "text",
        "fields": {
            "keyword": {
                "type": "keyword",
                "ignore_above": ignore_above,
            },
        },
    }


BASE_PROPERTIES: dict[str, Any] = {
    "schema_version": {
        "type": "keyword",
    },
    "record_id": {
        "type": "keyword",
    },
    "book_id": {
        "type": "keyword",
    },
    "book_version": {
        "type": "keyword",
    },
    "source_unit_id": {
        "type": "keyword",
    },
    "element_index": {
        "type": "integer",
    },
    "element_type": {
        "type": "keyword",
    },
    "element_sub_type": {
        "type": "keyword",
    },
    "modality": {
        "type": "keyword",
    },
    "source_page_numbers": {
        "type": "integer",
    },
    "citation_label": (
        text_with_keyword_mapping()
    ),
    "embedding_text": {
        "type": "text",
    },
    "asset_s3_uris": {
        "type": "keyword",
        "ignore_above": 2048,
    },
    "quality_flags": {
        "type": "keyword",
    },
    "retrieval_priority": {
        "type": "keyword",
    },
    "chunk_index": {
        "type": "integer",
    },
    "chunk_count": {
        "type": "integer",
    },
    "character_count": {
        "type": "integer",
    },
    "input_token_count": {
        "type": "integer",
    },
    "input_text_sha256": {
        "type": "keyword",
    },
    "embedding_model_id": {
        "type": "keyword",
    },
    "embedding_dimensions": {
        "type": "integer",
    },
    "embedding_normalized": {
        "type": "boolean",
    },
    "vector_length": {
        "type": "integer",
    },
    "vector_l2_norm": {
        "type": "float",
    },
    "locations": {
        "properties": {
            "page_index": {
                "type": "integer",
            },
            "bounding_box": {
                "properties": {
                    "left": {
                        "type": "float",
                    },
                    "top": {
                        "type": "float",
                    },
                    "width": {
                        "type": "float",
                    },
                    "height": {
                        "type": "float",
                    },
                },
            },
        },
    },
}


CONTEXT_PROPERTIES: dict[str, Any] = {
    "page_context_status": {
        "type": "keyword",
    },
    "chapter_context_status": {
        "type": "keyword",
    },
    "unresolved_source_page_numbers": {
        "type": "integer",
    },
    "page_types": {
        "type": "keyword",
    },
    "document_ids": {
        "type": "keyword",
    },
    "document_titles": (
        text_with_keyword_mapping()
    ),
    "document_types": {
        "type": "keyword",
    },
    "unit_numbers": {
        "type": "integer",
    },
    "chapter_ids": {
        "type": "keyword",
    },
    "chapter_titles": (
        text_with_keyword_mapping()
    ),
    "source_filenames": {
        "type": "keyword",
        "ignore_above": 1024,
    },
    "document_id": {
        "type": "keyword",
    },
    "document_title": (
        text_with_keyword_mapping()
    ),
    "document_type": {
        "type": "keyword",
    },
    "unit_number": {
        "type": "integer",
    },
    "chapter_id": {
        "type": "keyword",
    },
    "chapter_title": (
        text_with_keyword_mapping()
    ),
    "section_id": {
        "type": "keyword",
    },
    "context_citation_label": (
        text_with_keyword_mapping(
            ignore_above=1024
        )
    ),
}


BASE_FIELD_TYPES = {
    field_name: str(
        mapping["type"]
    )
    for field_name, mapping
    in BASE_PROPERTIES.items()
    if "type" in mapping
}


CONTEXT_FIELD_TYPES = {
    field_name: str(
        mapping["type"]
    )
    for field_name, mapping
    in CONTEXT_PROPERTIES.items()
}


def build_expected_field_types(
    *,
    include_context_fields: bool,
) -> dict[str, str]:
    field_types = dict(
        BASE_FIELD_TYPES
    )

    if include_context_fields:
        field_types.update(
            CONTEXT_FIELD_TYPES
        )

    return field_types


def build_index_schema(
    *,
    vector_dimensions: int,
    include_context_fields: bool,
) -> dict[str, Any]:
    if vector_dimensions not in {
        256,
        512,
        1024,
    }:
        raise ValueError(
            "Vector dimensions must be "
            "256, 512, or 1024."
        )

    properties = deepcopy(
        BASE_PROPERTIES
    )

    if include_context_fields:
        properties.update(
            deepcopy(
                CONTEXT_PROPERTIES
            )
        )

    properties["embedding"] = {
        "type": "knn_vector",
        "dimension": vector_dimensions,
        "compression_level": "1x",
        "space_type": "l2",
    }

    return {
        "settings": {
            "index.knn": True,
        },
        "mappings": {
            "dynamic": "strict",
            "properties": properties,
        },
    }


def extract_collection_id(
    endpoint: str,
) -> str:
    parsed = urlparse(endpoint)
    hostname = parsed.hostname

    if (
        parsed.scheme != "https"
        or not hostname
    ):
        raise ValueError(
            "OpenSearch collection endpoint "
            "must be a valid HTTPS URL."
        )

    labels = hostname.split(".")

    if (
        len(labels) < 5
        or labels[1] != "aoss"
        or labels[-2:] != [
            "on",
            "aws",
        ]
    ):
        raise ValueError(
            "OpenSearch endpoint does not "
            "match the expected AOSS format."
        )

    collection_id = labels[0].strip()

    if not collection_id:
        raise ValueError(
            "OpenSearch endpoint contains "
            "no collection ID."
        )

    return collection_id


def resolve_index_runtime(
    config_path: Path | None,
) -> dict[str, Any]:
    if config_path is None:
        return {
            "mode": "legacy",
            "config_path": None,
            "region": LEGACY_REGION,
            "collection_name": (
                LEGACY_COLLECTION_NAME
            ),
            "collection_id": (
                LEGACY_COLLECTION_ID
            ),
            "collection_endpoint": None,
            "index_name": (
                LEGACY_INDEX_NAME
            ),
            "vector_field": "embedding",
            "vector_dimensions": (
                LEGACY_VECTOR_DIMENSIONS
            ),
            "include_context_fields": False,
            "output_path": str(
                LEGACY_OUTPUT_PATH
            ),
        }

    config = load_book_config(
        config_path
    )

    vector_field = (
        config.opensearch.vector_field
    )

    if vector_field != "embedding":
        raise ValueError(
            "Current textbook index pipeline "
            "requires vector_field='embedding'."
        )

    endpoint = (
        config.opensearch
        .collection_endpoint
    )

    output_path = (
        Path(config.storage.local_root)
        / "opensearch-serverless"
        / "index-provisioning-report.json"
    )

    return {
        "mode": "book_config",
        "config_path": str(
            config_path
        ),
        "region": config.aws.region,
        "collection_name": (
            LEGACY_COLLECTION_NAME
        ),
        "collection_id": (
            extract_collection_id(
                endpoint
            )
        ),
        "collection_endpoint": endpoint,
        "index_name": (
            config.opensearch.index_name
        ),
        "vector_field": vector_field,
        "vector_dimensions": (
            config.models.embedding
            .dimensions
        ),
        "include_context_fields": True,
        "output_path": str(
            output_path
        ),
    }


def validate_index_schema(
    schema: dict[str, Any],
    *,
    vector_dimensions: int,
    include_context_fields: bool,
) -> list[str]:
    errors: list[str] = []

    settings = schema.get(
        "settings",
        {},
    )

    if not isinstance(settings, dict):
        errors.append(
            "settings is not an object."
        )
        settings = {}

    knn_enabled = settings.get(
        "index.knn"
    )

    nested_index = settings.get(
        "index"
    )

    if (
        knn_enabled is None
        and isinstance(
            nested_index,
            dict,
        )
    ):
        knn_enabled = nested_index.get(
            "knn"
        )

    if not (
        knn_enabled is True
        or knn_enabled == 1
        or (
            isinstance(
                knn_enabled,
                str,
            )
            and knn_enabled.lower()
            == "true"
        )
    ):
        errors.append(
            "index.knn is not enabled."
        )

    mappings = schema.get(
        "mappings",
        {},
    )

    if not isinstance(mappings, dict):
        errors.append(
            "mappings is not an object."
        )
        mappings = {}

    if mappings.get("dynamic") != "strict":
        errors.append(
            "mappings.dynamic is not strict."
        )

    properties = mappings.get(
        "properties",
        {},
    )

    if not isinstance(properties, dict):
        errors.append(
            "mappings.properties is not "
            "an object."
        )
        properties = {}

    expected_types = (
        build_expected_field_types(
            include_context_fields=(
                include_context_fields
            )
        )
    )

    for (
        field_name,
        expected_type,
    ) in expected_types.items():
        field = properties.get(
            field_name
        )

        if not isinstance(field, dict):
            errors.append(
                "Required field missing: "
                f"{field_name}"
            )
            continue

        actual_type = field.get("type")

        if actual_type != expected_type:
            errors.append(
                f"Field {field_name} "
                "type mismatch: "
                f"expected={expected_type}, "
                f"actual={actual_type}"
            )

    locations = properties.get(
        "locations"
    )

    if not isinstance(locations, dict):
        errors.append(
            "locations mapping is missing."
        )

    vector = properties.get(
        "embedding"
    )

    if not isinstance(vector, dict):
        errors.append(
            "Vector field embedding "
            "is missing."
        )
        return errors

    if vector.get("type") != "knn_vector":
        errors.append(
            "embedding is not a "
            "knn_vector."
        )

    if (
        vector.get("dimension")
        != vector_dimensions
    ):
        errors.append(
            "embedding dimension mismatch: "
            f"{vector.get('dimension')}"
        )

    if vector.get("space_type") != "l2":
        errors.append(
            "embedding space_type is not "
            f"l2: {vector.get('space_type')}"
        )

    if (
        vector.get("compression_level")
        != "1x"
    ):
        errors.append(
            "embedding compression is not "
            "1x: "
            f"{vector.get('compression_level')}"
        )

    return errors


def build_index_plan(
    config_path: Path | None,
) -> dict[str, Any]:
    runtime = resolve_index_runtime(
        config_path
    )

    schema = build_index_schema(
        vector_dimensions=int(
            runtime[
                "vector_dimensions"
            ]
        ),
        include_context_fields=bool(
            runtime[
                "include_context_fields"
            ]
        ),
    )

    errors = validate_index_schema(
        schema,
        vector_dimensions=int(
            runtime[
                "vector_dimensions"
            ]
        ),
        include_context_fields=bool(
            runtime[
                "include_context_fields"
            ]
        ),
    )

    if errors:
        raise RuntimeError(
            "Generated index schema is "
            "invalid:\n- "
            + "\n- ".join(errors)
        )

    return {
        "runtime": runtime,
        "schema": schema,
        "field_count": len(
            schema[
                "mappings"
            ][
                "properties"
            ]
        ),
        "schema_validation_errors": (
            errors
        ),
    }
