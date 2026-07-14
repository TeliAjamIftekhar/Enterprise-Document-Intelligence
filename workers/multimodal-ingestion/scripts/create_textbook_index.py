from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
)


REGION = "us-east-1"

COLLECTION_NAME = "edi-textbook-vector"
EXPECTED_COLLECTION_ID = "kqjqddn0b5gmcfvgsd2e"

INDEX_NAME = "grade-9-english-kaveri-v1"
VECTOR_DIMENSIONS = 1024

OUTPUT_PATH = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/"
    "opensearch-serverless/"
    "index-provisioning-report.json"
)


INDEX_SCHEMA: dict[str, Any] = {
    "settings": {
        "index.knn": True,
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
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
            "citation_label": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 512,
                    },
                },
            },
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
            "embedding": {
                "type": "knn_vector",
                "dimension": VECTOR_DIMENSIONS,
                "compression_level": "1x",
                "space_type": "l2",
            },
        },
    },
}


REQUIRED_FIELD_TYPES = {
    "schema_version": "keyword",
    "record_id": "keyword",
    "book_id": "keyword",
    "book_version": "keyword",
    "source_unit_id": "keyword",
    "element_index": "integer",
    "element_type": "keyword",
    "element_sub_type": "keyword",
    "modality": "keyword",
    "source_page_numbers": "integer",
    "citation_label": "text",
    "embedding_text": "text",
    "asset_s3_uris": "keyword",
    "quality_flags": "keyword",
    "retrieval_priority": "keyword",
    "chunk_index": "integer",
    "chunk_count": "integer",
    "character_count": "integer",
    "input_token_count": "integer",
    "input_text_sha256": "keyword",
    "embedding_model_id": "keyword",
    "embedding_dimensions": "integer",
    "embedding_normalized": "boolean",
    "vector_length": "integer",
    "vector_l2_norm": "float",
}


RETRYABLE_CODES = {
    "AccessDeniedException",
    "ConflictException",
    "InternalServerException",
    "ThrottlingException",
    "TooManyRequestsException",
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_response(
    response: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in response.items()
        if key != "ResponseMetadata"
    }


def write_report(
    value: dict[str, Any],
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def get_collection(
    client: Any,
) -> dict[str, Any]:
    response = client.batch_get_collection(
        names=[COLLECTION_NAME]
    )

    details = response.get(
        "collectionDetails",
        [],
    )

    if not isinstance(
        details,
        list,
    ) or len(details) != 1:
        raise RuntimeError(
            "Expected exactly one collection named "
            f"{COLLECTION_NAME}."
        )

    collection = details[0]

    if collection.get("id") != EXPECTED_COLLECTION_ID:
        raise RuntimeError(
            "Collection ID mismatch. "
            f"Expected={EXPECTED_COLLECTION_ID}, "
            f"actual={collection.get('id')}"
        )

    if collection.get("status") != "ACTIVE":
        raise RuntimeError(
            "Collection is not ACTIVE: "
            f"{collection.get('status')}"
        )

    if collection.get("type") != "VECTORSEARCH":
        raise RuntimeError(
            "Collection is not VECTORSEARCH."
        )

    return collection


def get_index_schema(
    client: Any,
    collection_id: str,
) -> dict[str, Any] | None:
    try:
        response = client.get_index(
            id=collection_id,
            indexName=INDEX_NAME,
        )

    except client.exceptions.ResourceNotFoundException:
        return None

    except ClientError as exc:
        code = str(
            exc.response.get(
                "Error",
                {},
            ).get(
                "Code",
                "",
            )
        )

        if code in {
            "ResourceNotFoundException",
            "NotFoundException",
        }:
            return None

        raise

    schema_wrapper = response.get(
        "indexSchema"
    )

    if not isinstance(schema_wrapper, dict):
        raise RuntimeError(
            "GetIndex returned no index schema."
        )

    schema = schema_wrapper.get(
        INDEX_NAME
    )

    if not isinstance(schema, dict):
        raise RuntimeError(
            "GetIndex schema does not contain the "
            f"expected index key: {INDEX_NAME}"
        )

    return schema


def value_is_true(
    value: Any,
) -> bool:
    if value is True or value == 1:
        return True

    if isinstance(value, str):
        return value.lower() == "true"

    return False


def validate_index_schema(
    schema: dict[str, Any],
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
        and isinstance(nested_index, dict)
    ):
        knn_enabled = nested_index.get(
            "knn"
        )

    if not value_is_true(knn_enabled):
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

    dynamic = mappings.get(
        "dynamic"
    )

    if dynamic != "strict":
        errors.append(
            "mappings.dynamic is not strict."
        )

    properties = mappings.get(
        "properties",
        {},
    )

    if not isinstance(properties, dict):
        errors.append(
            "mappings.properties is not an object."
        )
        properties = {}

    for field_name, expected_type in (
        REQUIRED_FIELD_TYPES.items()
    ):
        field = properties.get(
            field_name
        )

        if not isinstance(field, dict):
            errors.append(
                f"Required field missing: {field_name}"
            )
            continue

        actual_type = field.get("type")

        if actual_type != expected_type:
            errors.append(
                f"Field {field_name} type mismatch: "
                f"expected={expected_type}, "
                f"actual={actual_type}"
            )

    vector_field = properties.get(
        "embedding"
    )

    if not isinstance(vector_field, dict):
        errors.append(
            "Vector field embedding is missing."
        )
        return errors

    if vector_field.get("type") != "knn_vector":
        errors.append(
            "embedding is not a knn_vector."
        )

    if vector_field.get(
        "dimension"
    ) != VECTOR_DIMENSIONS:
        errors.append(
            "embedding dimension mismatch: "
            f"{vector_field.get('dimension')}"
        )

    space_type = vector_field.get(
        "space_type"
    )

    if space_type != "l2":
        errors.append(
            "embedding space_type is not l2: "
            f"{space_type}"
        )

    compression = vector_field.get(
        "compression_level"
    )

    if compression != "1x":
        errors.append(
            "embedding compression is not 1x: "
            f"{compression}"
        )

    locations_field = properties.get(
        "locations"
    )

    if not isinstance(
        locations_field,
        dict,
    ):
        errors.append(
            "locations mapping is missing."
        )

    return errors


def create_index_with_retries(
    client: Any,
    collection_id: str,
    maximum_attempts: int = 8,
) -> None:
    for attempt in range(
        1,
        maximum_attempts + 1,
    ):
        existing = get_index_schema(
            client=client,
            collection_id=collection_id,
        )

        if existing is not None:
            return

        try:
            client.create_index(
                id=collection_id,
                indexName=INDEX_NAME,
                indexSchema=INDEX_SCHEMA,
            )

            print(
                f"CreateIndex request accepted "
                f"on attempt {attempt}."
            )

            return

        except ClientError as exc:
            code = str(
                exc.response.get(
                    "Error",
                    {},
                ).get(
                    "Code",
                    "",
                )
            )

            if code not in RETRYABLE_CODES:
                raise

            print(
                f"CreateIndex attempt {attempt} "
                f"returned {code}."
            )

            if attempt == maximum_attempts:
                raise

            time.sleep(
                min(5 * attempt, 20)
            )


def wait_for_index(
    client: Any,
    collection_id: str,
    maximum_checks: int = 60,
) -> dict[str, Any]:
    for check in range(
        1,
        maximum_checks + 1,
    ):
        schema = get_index_schema(
            client=client,
            collection_id=collection_id,
        )

        status = (
            "AVAILABLE"
            if schema is not None
            else "NOT_FOUND"
        )

        print(
            f"Index check {check}: {status}"
        )

        if schema is not None:
            return schema

        time.sleep(5)

    raise TimeoutError(
        "Index did not become available."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create and validate the hybrid textbook "
            "retrieval index."
        )
    )

    parser.add_argument(
        "--create",
        action="store_true",
        help=(
            "Create the index when it is missing. "
            "Without this option, only validation and "
            "planning are performed."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    client = boto3.client(
        "opensearchserverless",
        region_name=REGION,
    )

    collection = get_collection(client)

    collection_id = str(
        collection["id"]
    )

    existing_schema = get_index_schema(
        client=client,
        collection_id=collection_id,
    )

    print("============================================")
    print("TEXTBOOK HYBRID INDEX")
    print("============================================")
    print(f"Region:       {REGION}")
    print(f"Collection:   {COLLECTION_NAME}")
    print(f"Collection ID:{collection_id}")
    print(f"Index:        {INDEX_NAME}")
    print(f"Dimensions:   {VECTOR_DIMENSIONS}")
    print("Space type:   l2")
    print("Method:       serverless-managed")
    print("Compression:  1x")
    print(f"Create:       {args.create}")
    print()

    action = "planned"
    final_schema = existing_schema

    if existing_schema is not None:
        errors = validate_index_schema(
            existing_schema
        )

        if errors:
            raise RuntimeError(
                "Existing index schema is incompatible:\n- "
                + "\n- ".join(errors)
            )

        action = "matching"

        print(
            f"MATCHING index: {INDEX_NAME}"
        )

    elif not args.create:
        print(
            f"PLANNED  index: {INDEX_NAME}"
        )

    else:
        create_index_with_retries(
            client=client,
            collection_id=collection_id,
        )

        final_schema = wait_for_index(
            client=client,
            collection_id=collection_id,
        )

        errors = validate_index_schema(
            final_schema
        )

        if errors:
            raise RuntimeError(
                "Created index schema validation failed:\n- "
                + "\n- ".join(errors)
            )

        action = "created"

        print(
            f"CREATED  index: {INDEX_NAME}"
        )

    status = (
        "PROVISIONED"
        if action in {
            "matching",
            "created",
        }
        else "PLANNED"
    )

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": status,
        "action": action,
        "region": REGION,
        "collection": {
            "name": COLLECTION_NAME,
            "id": collection_id,
            "endpoint": collection.get(
                "collectionEndpoint"
            ),
            "status": collection.get(
                "status"
            ),
            "type": collection.get(
                "type"
            ),
        },
        "index": {
            "name": INDEX_NAME,
            "vector_dimensions": (
                VECTOR_DIMENSIONS
            ),
            "space_type": "l2",
            "method": "serverless-managed",
            "compression_level": "1x",
        },
        "requested_schema": INDEX_SCHEMA,
        "actual_schema": final_schema,
        "resources_created": (
            action == "created"
        ),
    }

    write_report(report)

    print()
    print("============================================")
    print("INDEX RESULT")
    print("============================================")
    print(f"Status:      {status}")
    print(f"Action:      {action}")
    print(f"Collection:  {COLLECTION_NAME}")
    print(f"Index:       {INDEX_NAME}")
    print(
        f"Fields:      "
        f"{len(INDEX_SCHEMA['mappings']['properties'])}"
    )
    print(
        f"Vector:      "
        f"{VECTOR_DIMENSIONS} dimensions"
    )
    print(f"Report:      {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (ClientError, BotoCoreError) as exc:
        print(
            f"AWS index error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"Index provisioning failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
