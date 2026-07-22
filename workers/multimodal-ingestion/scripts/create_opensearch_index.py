from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from opensearchpy import (
    AWSV4SignerAuth,
    OpenSearch,
    RequestsHttpConnection,
)
from opensearchpy.exceptions import (
    AuthorizationException,
    ConnectionError as OpenSearchConnectionError,
    NotFoundError,
    RequestError,
    TransportError,
)


REGION = "us-east-1"
SERVICE = "aoss"

COLLECTION_ID = "kqjqddn0b5gmcfvgsd2e"

COLLECTION_HOST = (
    "kqjqddn0b5gmcfvgsd2e"
    ".aoss.us-east-1.on.aws"
)

COLLECTION_ENDPOINT = (
    f"https://{COLLECTION_HOST}"
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
                "compression_level": "1x",
                "space_type": "l2",
            },
        },
    },
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def create_client() -> OpenSearch:
    session = boto3.Session()

    credentials = session.get_credentials()

    if credentials is None:
        raise RuntimeError(
            "AWS credentials could not be resolved."
        )

    auth = AWSV4SignerAuth(
        credentials,
        REGION,
        SERVICE,
    )

    return OpenSearch(
        hosts=[
            {
                "host": COLLECTION_HOST,
                "port": 443,
            }
        ],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
        timeout=120,
        max_retries=3,
        retry_on_timeout=True,
    )


def get_existing_index(
    client: OpenSearch,
) -> dict[str, Any] | None:
    try:
        exists = client.indices.exists(
            index=INDEX_NAME
        )

    except NotFoundError:
        return None

    if not exists:
        return None

    response = client.indices.get(
        index=INDEX_NAME
    )

    if not isinstance(response, dict):
        raise RuntimeError(
            "Existing index response is not a JSON object."
        )

    return response


def validate_existing_index(
    index_response: dict[str, Any],
) -> None:
    index_detail = index_response.get(
        INDEX_NAME
    )

    if not isinstance(index_detail, dict):
        raise RuntimeError(
            "Index response does not contain "
            f"{INDEX_NAME}."
        )

    mappings = index_detail.get(
        "mappings",
        {}
    )

    properties = mappings.get(
        "properties",
        {}
    )

    vector_mapping = properties.get(
        "embedding",
        {}
    )

    expected_values = {
        "type": "knn_vector",
        "dimension": VECTOR_DIMENSIONS,
        "space_type": "l2",
    }

    mismatches: list[dict[str, Any]] = []

    for field, expected in expected_values.items():
        actual = vector_mapping.get(field)

        if actual != expected:
            mismatches.append(
                {
                    "field": (
                        f"embedding.{field}"
                    ),
                    "expected": expected,
                    "actual": actual,
                }
            )

    text_mapping = properties.get(
        "embedding_text",
        {}
    )

    if text_mapping.get("type") != "text":
        mismatches.append(
            {
                "field": "embedding_text.type",
                "expected": "text",
                "actual": text_mapping.get("type"),
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
    client: OpenSearch,
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

        try:
            exists = client.indices.exists(
                index=INDEX_NAME
            )

        except NotFoundError:
            exists = False

        print(
            f"Index status check {attempt}: "
            + (
                "FOUND"
                if exists
                else "NOT FOUND"
            )
        )

        if exists:
            response = client.indices.get(
                index=INDEX_NAME
            )

            if not isinstance(response, dict):
                raise RuntimeError(
                    "Index response is not a JSON object."
                )

            validate_existing_index(response)

            return response

        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for the index."
            )

        time.sleep(interval_seconds)


def main() -> int:
    # SAFE_HELP_GUARD
    if (
        "-h" in sys.argv[1:]
        or "--help" in sys.argv[1:]
    ):
        print(
            "usage: create_opensearch_index.py [-h]"
        )
        print()
        print(
            "Create or reuse the configured "
            "OpenSearch Serverless textbook index."
        )
        print()
        print("options:")
        print(
            "  -h, --help  show this help "
            "message and exit"
        )
        return 0

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

    client = create_client()

    print("============================================")
    print("OPENSEARCH INDEX CREATION")
    print("============================================")
    print(f"Region:       {REGION}")
    print(f"Service:      {SERVICE}")
    print(f"Collection:   {COLLECTION_ID}")
    print(f"Endpoint:     {COLLECTION_ENDPOINT}")
    print(f"Index:        {INDEX_NAME}")
    print(f"Dimensions:   {VECTOR_DIMENSIONS}")
    print("Space type:   l2")
    print("Compression:  1x")
    print("Signer:       AWSV4SignerAuth")
    print(f"Schema:       {SCHEMA_PATH}")
    print()

    existing = get_existing_index(
        client
    )

    if existing is not None:
        validate_existing_index(
            existing
        )

        action = "reused"
        final_index = existing

        print(
            "Existing compatible index found."
        )

    else:
        print(
            "Creating hybrid vector index..."
        )

        create_response = (
            client.indices.create(
                index=INDEX_NAME,
                body=INDEX_SCHEMA,
            )
        )

        print(
            "Create response:"
        )
        print(
            json.dumps(
                create_response,
                indent=2,
                default=str,
            )
        )

        action = "created"

        final_index = wait_for_index(
            client
        )

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": "ACTIVE",
        "action": action,
        "region": REGION,
        "service": SERVICE,
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
    print("Status:      ACTIVE")
    print(f"Action:      {action}")
    print(f"Index:       {INDEX_NAME}")
    print(
        f"Dimensions:  {VECTOR_DIMENSIONS}"
    )
    print(f"Report:      {REPORT_PATH}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except AuthorizationException as exc:
        print(
            "OpenSearch authorization failed:",
            exc,
            file=sys.stderr,
        )
        raise SystemExit(1)

    except RequestError as exc:
        print(
            "OpenSearch index request was rejected:",
            exc,
            file=sys.stderr,
        )
        raise SystemExit(1)

    except (
        OpenSearchConnectionError,
        TransportError,
    ) as exc:
        print(
            "OpenSearch transport error:",
            exc,
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"Index creation failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
