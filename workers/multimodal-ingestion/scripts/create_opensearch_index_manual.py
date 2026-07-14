from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
)


REGION = "us-east-1"
SERVICE = "aoss"

COLLECTION_ID = "kqjqddn0b5gmcfvgsd2e"

COLLECTION_ENDPOINT = (
    "https://"
    "kqjqddn0b5gmcfvgsd2e"
    ".aoss.us-east-1.on.aws"
)

INDEX_NAME = "grade-9-english-kaveri-v1"
VECTOR_DIMENSIONS = 1024

OUTPUT_DIRECTORY = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/"
    "opensearch-serverless"
)

SCHEMA_PATH = (
    OUTPUT_DIRECTORY
    / "grade-9-english-kaveri-v1-index-schema.json"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "index-creation-report.json"
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
            "source_unit_id": {
                "type": "keyword",
            },
            "book_id": {
                "type": "keyword",
            },
            "book_version": {
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
            "retrieval_priority": {
                "type": "keyword",
            },
            "chunk_index": {
                "type": "integer",
            },
            "chunk_count": {
                "type": "integer",
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
                    }
                },
            },
            "embedding_text": {
                "type": "text",
            },
            "character_count": {
                "type": "integer",
            },
            "asset_s3_uris": {
                "type": "keyword",
                "ignore_above": 2048,
            },
            "quality_flags": {
                "type": "keyword",
            },
            "locations": {
                "type": "nested",
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
                        }
                    },
                },
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
            "input_text_sha256": {
                "type": "keyword",
            },
            "input_token_count": {
                "type": "integer",
            },
            "vector_length": {
                "type": "integer",
            },
            "vector_l2_norm": {
                "type": "float",
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": VECTOR_DIMENSIONS,
                "space_type": "l2",
                "compression_level": "1x",
            },
        },
    },
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def signed_request(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> requests.Response:
    session = boto3.Session()

    credentials = (
        session.get_credentials()
    )

    if credentials is None:
        raise RuntimeError(
            "AWS credentials could not be resolved."
        )

    frozen_credentials = (
        credentials.get_frozen_credentials()
    )

    url = (
        COLLECTION_ENDPOINT.rstrip("/")
        + "/"
        + path.lstrip("/")
    )

    payload = (
        json.dumps(
            body,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        if body is not None
        else ""
    )

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    aws_request = AWSRequest(
        method=method,
        url=url,
        data=payload,
        headers=headers,
    )

    SigV4Auth(
        frozen_credentials,
        SERVICE,
        REGION,
    ).add_auth(aws_request)

    response = requests.request(
        method=method,
        url=url,
        data=payload or None,
        headers=dict(
            aws_request.headers.items()
        ),
        timeout=120,
    )

    return response


def parse_response(
    response: requests.Response,
) -> Any:
    if not response.text.strip():
        return None

    try:
        return response.json()

    except ValueError:
        return response.text


def require_success(
    response: requests.Response,
    operation: str,
    allowed_statuses: set[int],
) -> Any:
    result = parse_response(response)

    if response.status_code not in allowed_statuses:
        raise RuntimeError(
            f"{operation} failed.\n"
            f"HTTP status: {response.status_code}\n"
            f"Response: "
            f"{json.dumps(result, indent=2, default=str)}"
        )

    return result


def get_existing_index() -> dict[str, Any] | None:
    response = signed_request(
        method="GET",
        path=INDEX_NAME,
    )

    if response.status_code == 404:
        return None

    result = require_success(
        response=response,
        operation="Get index",
        allowed_statuses={200},
    )

    if not isinstance(result, dict):
        raise RuntimeError(
            "Get-index response is not an object."
        )

    return result


def validate_existing_index(
    index_response: dict[str, Any],
) -> None:
    index_detail = index_response.get(
        INDEX_NAME
    )

    if not isinstance(index_detail, dict):
        raise RuntimeError(
            "Existing index response does not contain "
            f"{INDEX_NAME}."
        )

    mappings = index_detail.get(
        "mappings",
        {},
    )

    properties = mappings.get(
        "properties",
        {},
    )

    vector_mapping = properties.get(
        "embedding",
        {},
    )

    checks = {
        "type": "knn_vector",
        "dimension": VECTOR_DIMENSIONS,
        "space_type": "l2",
    }

    mismatches = []

    for field, expected in checks.items():
        actual = vector_mapping.get(field)

        if actual != expected:
            mismatches.append(
                {
                    "field": field,
                    "expected": expected,
                    "actual": actual,
                }
            )

    embedding_text_mapping = properties.get(
        "embedding_text",
        {},
    )

    if (
        embedding_text_mapping.get("type")
        != "text"
    ):
        mismatches.append(
            {
                "field": "embedding_text.type",
                "expected": "text",
                "actual": (
                    embedding_text_mapping.get(
                        "type"
                    )
                ),
            }
        )

    if mismatches:
        raise RuntimeError(
            "Existing index has an incompatible schema:\n"
            + json.dumps(
                mismatches,
                indent=2,
            )
        )


def wait_for_index(
    timeout_seconds: int = 300,
    interval_seconds: int = 5,
) -> dict[str, Any]:
    deadline = (
        time.monotonic()
        + timeout_seconds
    )

    attempt = 0

    while True:
        attempt += 1

        response = signed_request(
            method="GET",
            path=INDEX_NAME,
        )

        print(
            f"Index status check {attempt}: "
            f"HTTP {response.status_code}"
        )

        if response.status_code == 200:
            result = parse_response(
                response
            )

            if not isinstance(result, dict):
                raise RuntimeError(
                    "Index response is not a JSON object."
                )

            validate_existing_index(result)

            return result

        if response.status_code not in {
            404,
            429,
            503,
        }:
            require_success(
                response=response,
                operation="Check index",
                allowed_statuses={200},
            )

        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for index."
            )

        time.sleep(interval_seconds)


def main() -> int:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    SCHEMA_PATH.write_text(
        json.dumps(
            INDEX_SCHEMA,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("============================================")
    print("OPENSEARCH INDEX CREATION")
    print("============================================")
    print(f"Region:       {REGION}")
    print(f"Collection:   {COLLECTION_ID}")
    print(f"Endpoint:     {COLLECTION_ENDPOINT}")
    print(f"Index:        {INDEX_NAME}")
    print(f"Dimensions:   {VECTOR_DIMENSIONS}")
    print(f"Space type:   l2")
    print(f"Compression:  1x")
    print(f"Schema:       {SCHEMA_PATH}")
    print()

    existing = get_existing_index()

    action: str

    if existing is not None:
        validate_existing_index(existing)

        action = "reused"

        print(
            "Existing compatible index found."
        )

        final_index = existing

    else:
        print(
            "Creating hybrid vector index..."
        )

        response = signed_request(
            method="PUT",
            path=INDEX_NAME,
            body=INDEX_SCHEMA,
        )

        create_result = require_success(
            response=response,
            operation="Create index",
            allowed_statuses={
                200,
                201,
            },
        )

        print(
            "Create response:",
            json.dumps(
                create_result,
                indent=2,
                default=str,
            ),
        )

        action = "created"

        final_index = wait_for_index()

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": "ACTIVE",
        "action": action,
        "region": REGION,
        "collection_id": COLLECTION_ID,
        "collection_endpoint": (
            COLLECTION_ENDPOINT
        ),
        "index_name": INDEX_NAME,
        "vector_field": "embedding",
        "vector_dimensions": (
            VECTOR_DIMENSIONS
        ),
        "vector_space_type": "l2",
        "vector_compression": "1x",
        "text_field": "embedding_text",
        "schema_path": str(
            SCHEMA_PATH
        ),
        "index_detail": final_index,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print("============================================")
    print("INDEX CREATION COMPLETED")
    print("============================================")
    print(f"Status:      ACTIVE")
    print(f"Action:      {action}")
    print(f"Index:       {INDEX_NAME}")
    print(f"Dimensions:  {VECTOR_DIMENSIONS}")
    print(f"Report:      {REPORT_PATH}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (
        ClientError,
        BotoCoreError,
    ) as exc:
        print(
            f"AWS index error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"Index creation failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
