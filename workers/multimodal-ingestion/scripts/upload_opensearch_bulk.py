from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import urllib3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
)

from src.book_config import load_book_config


REGION = "us-east-1"
SERVICE = "aoss"

COLLECTION_ENDPOINT = (
    "https://"
    "kqjqddn0b5gmcfvgsd2e."
    "aoss.us-east-1.on.aws"
)

INDEX_NAME = "grade-9-english-kaveri-v1"
BOOK_ID = "grade-9-english-kaveri"
BOOK_VERSION = "v1"


RETRYABLE_HTTP_STATUS = {
    403,
    408,
    429,
    500,
    502,
    503,
    504,
}




DEFAULT_MAX_BATCH_BYTES = (
    5 * 1024 * 1024
)

CHECKPOINT_SCHEMA_VERSION = "1.0"


def resolve_runtime_identity(
    config_path: Path | None,
) -> dict[str, Any]:
    if config_path is None:
        return {
            "mode": "legacy",
            "config_path": None,
            "region": REGION,
            "collection_endpoint": (
                COLLECTION_ENDPOINT
            ),
            "index_name": INDEX_NAME,
            "book_id": BOOK_ID,
            "book_version": BOOK_VERSION,
            "vector_dimensions": 1024,
        }

    config = load_book_config(
        config_path
    )

    return {
        "mode": "book_config",
        "config_path": str(
            config_path
        ),
        "region": config.aws.region,
        "collection_endpoint": (
            config.opensearch
            .collection_endpoint
        ),
        "index_name": (
            config.opensearch.index_name
        ),
        "book_id": config.book.book_id,
        "book_version": (
            config.book.version
        ),
        "vector_dimensions": (
            config.models.embedding
            .dimensions
        ),
    }


def configure_runtime(
    config_path: Path | None,
) -> dict[str, Any]:
    global REGION
    global COLLECTION_ENDPOINT
    global INDEX_NAME
    global BOOK_ID
    global BOOK_VERSION

    runtime = resolve_runtime_identity(
        config_path
    )

    REGION = str(runtime["region"])

    COLLECTION_ENDPOINT = str(
        runtime["collection_endpoint"]
    )

    INDEX_NAME = str(
        runtime["index_name"]
    )

    BOOK_ID = str(
        runtime["book_id"]
    )

    BOOK_VERSION = str(
        runtime["book_version"]
    )

    return runtime


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(
        value
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(
            1024 * 1024
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return value


def atomic_write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        path,
    )


def get_frozen_credentials() -> Any:
    credentials = (
        boto3.Session()
        .get_credentials()
    )

    if credentials is None:
        raise RuntimeError(
            "No AWS credentials were resolved."
        )

    return credentials.get_frozen_credentials()


def signed_request(
    http: urllib3.PoolManager,
    method: str,
    path: str,
    body: bytes = b"",
    content_type: str = "application/json",
    maximum_attempts: int = 12,
) -> tuple[int, dict[str, Any]]:
    url = (
        COLLECTION_ENDPOINT.rstrip("/")
        + "/"
        + path.lstrip("/")
    )

    payload_hash = sha256_bytes(body)

    for attempt in range(
        1,
        maximum_attempts + 1,
    ):
        credentials = get_frozen_credentials()

        headers = {
            "Content-Type": content_type,
            "Accept": "application/json",
            "x-amz-content-sha256": (
                payload_hash
            ),
        }

        aws_request = AWSRequest(
            method=method,
            url=url,
            data=body,
            headers=headers,
        )

        SigV4Auth(
            credentials,
            SERVICE,
            REGION,
        ).add_auth(aws_request)

        prepared = aws_request.prepare()

        try:
            response = http.request(
                method=method,
                url=url,
                body=body if body else None,
                headers=dict(
                    prepared.headers.items()
                ),
                preload_content=True,
            )

        except Exception as exc:
            if attempt == maximum_attempts:
                raise RuntimeError(
                    "HTTP request failed after "
                    f"{maximum_attempts} attempts: "
                    f"{exc}"
                ) from exc

            delay = min(
                2 * attempt,
                20,
            )

            print(
                f"Request attempt {attempt} failed: "
                f"{type(exc).__name__}. "
                f"Retrying in {delay}s."
            )

            time.sleep(delay)
            continue

        response_text = response.data.decode(
            "utf-8",
            errors="replace",
        )

        try:
            response_json = (
                json.loads(response_text)
                if response_text
                else {}
            )

        except json.JSONDecodeError:
            response_json = {
                "raw_response": response_text,
            }

        if (
            response.status
            in RETRYABLE_HTTP_STATUS
            and attempt < maximum_attempts
        ):
            delay = min(
                2 * attempt,
                20,
            )

            print(
                f"HTTP {response.status} on "
                f"attempt {attempt}. "
                f"Retrying in {delay}s."
            )

            time.sleep(delay)
            continue

        return (
            int(response.status),
            response_json,
        )

    raise RuntimeError(
        "Signed request retry loop ended "
        "unexpectedly."
    )


def validate_bulk_identity(
    bulk_path: Path,
    runtime: dict[str, Any],
) -> int:
    lines = bulk_path.read_text(
        encoding="utf-8"
    ).splitlines()

    if not lines or len(lines) % 2 != 0:
        raise RuntimeError(
            "Bulk payload must contain "
            "action/document line pairs."
        )

    expected_index = str(
        runtime["index_name"]
    )

    expected_book_id = str(
        runtime["book_id"]
    )

    expected_book_version = str(
        runtime["book_version"]
    )

    document_count = 0

    for position in range(
        0,
        len(lines),
        2,
    ):
        pair_number = (
            position // 2 + 1
        )

        try:
            action = json.loads(
                lines[position]
            )

            document = json.loads(
                lines[position + 1]
            )

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Invalid JSON in bulk payload "
                f"pair {pair_number}: {exc}"
            ) from exc

        if not isinstance(action, dict):
            raise RuntimeError(
                f"Bulk action {pair_number} "
                "is not an object."
            )

        operation = action.get("index")

        if not isinstance(operation, dict):
            raise RuntimeError(
                f"Bulk action {pair_number} "
                "does not contain an index "
                "operation."
            )

        actual_index = operation.get(
            "_index"
        )

        if actual_index != expected_index:
            raise RuntimeError(
                "Bulk target index mismatch: "
                f"expected={expected_index!r}, "
                f"actual={actual_index!r}"
            )

        if not isinstance(document, dict):
            raise RuntimeError(
                f"Bulk document {pair_number} "
                "is not an object."
            )

        if (
            document.get("book_id")
            != expected_book_id
        ):
            raise RuntimeError(
                "Bulk document book_id "
                "mismatch: "
                f"expected={expected_book_id!r}, "
                f"actual="
                f"{document.get('book_id')!r}"
            )

        if (
            document.get("book_version")
            != expected_book_version
        ):
            raise RuntimeError(
                "Bulk document book_version "
                "mismatch: "
                f"expected="
                f"{expected_book_version!r}, "
                f"actual="
                f"{document.get('book_version')!r}"
            )

        action_id = operation.get("_id")

        record_id = document.get(
            "record_id"
        )

        if action_id != record_id:
            raise RuntimeError(
                "Bulk action/document ID "
                "mismatch: "
                f"action={action_id!r}, "
                f"document={record_id!r}"
            )

        document_count += 1

    return document_count


def validate_preparation(
    bulk_path: Path,
    preparation_report: dict[str, Any],
    runtime: dict[str, Any],
) -> int:
    if preparation_report.get(
        "status"
    ) != "PREPARED":
        raise RuntimeError(
            "Bulk preparation report "
            "is not PREPARED."
        )

    expected_index = str(
        runtime["index_name"]
    )

    report_index = (
        preparation_report.get(
            "index_name"
        )
    )

    if report_index != expected_index:
        raise RuntimeError(
            "Bulk preparation index "
            "mismatch: "
            f"expected={expected_index!r}, "
            f"actual={report_index!r}"
        )

    report_configuration = (
        preparation_report.get(
            "configuration"
        )
    )

    if runtime["mode"] == "book_config":
        if not isinstance(
            report_configuration,
            dict,
        ):
            raise RuntimeError(
                "Config-driven bulk report "
                "contains no configuration "
                "identity."
            )

        identity_checks = {
            "book_id": runtime["book_id"],
            "book_version": (
                runtime["book_version"]
            ),
            "index_name": (
                runtime["index_name"]
            ),
            "vector_dimensions": (
                runtime[
                    "vector_dimensions"
                ]
            ),
        }

        for field, expected in (
            identity_checks.items()
        ):
            actual = (
                report_configuration.get(
                    field
                )
            )

            if actual != expected:
                raise RuntimeError(
                    "Bulk preparation "
                    "configuration mismatch "
                    f"for {field}: "
                    f"expected={expected!r}, "
                    f"actual={actual!r}"
                )

    validation = preparation_report.get(
        "validation",
        {},
    )

    expected_document_count = (
        validation.get(
            "document_count"
        )
    )

    if (
        not isinstance(
            expected_document_count,
            int,
        )
        or expected_document_count < 1
    ):
        raise RuntimeError(
            "Prepared document count "
            "is invalid: "
            f"{expected_document_count}"
        )

    unique_document_ids = (
        validation.get(
            "unique_document_ids"
        )
    )

    if (
        unique_document_ids
        != expected_document_count
    ):
        raise RuntimeError(
            "Prepared document IDs "
            "are not unique: "
            f"documents="
            f"{expected_document_count}, "
            f"unique_ids="
            f"{unique_document_ids}"
        )

    vector_dimensions = int(
        runtime["vector_dimensions"]
    )

    actual_dimensions = validation.get(
        "vector_dimensions"
    )

    if (
        actual_dimensions
        != vector_dimensions
    ):
        raise RuntimeError(
            "Prepared vector dimensions "
            "do not match configuration: "
            f"expected={vector_dimensions}, "
            f"actual={actual_dimensions}"
        )

    output = preparation_report.get(
        "output",
        {},
    )

    expected_sha256 = output.get(
        "sha256"
    )

    actual_sha256 = sha256_file(
        bulk_path
    )

    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Bulk payload checksum "
            "mismatch.\n"
            f"Expected: {expected_sha256}\n"
            f"Actual:   {actual_sha256}"
        )

    if not bulk_path.read_bytes().endswith(
        b"\n"
    ):
        raise RuntimeError(
            "Bulk payload does not "
            "end with newline."
        )

    actual_document_count = (
        validate_bulk_identity(
            bulk_path=bulk_path,
            runtime=runtime,
        )
    )

    if (
        actual_document_count
        != expected_document_count
    ):
        raise RuntimeError(
            "Bulk payload document count "
            "does not match report: "
            f"payload="
            f"{actual_document_count}, "
            f"report="
            f"{expected_document_count}"
        )

    return expected_document_count


def parse_bulk_result(
    response: dict[str, Any],
) -> dict[str, Any]:
    items = response.get(
        "items"
    )

    if not isinstance(items, list):
        raise RuntimeError(
            "Bulk response contains no items list."
        )

    failures: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    result_counts: dict[str, int] = {}

    returned_ids: set[str] = set()

    for position, item in enumerate(
        items,
        start=1,
    ):
        if not isinstance(item, dict):
            failures.append(
                {
                    "position": position,
                    "reason": (
                        "Bulk item is not an object."
                    ),
                    "item": item,
                }
            )
            continue

        operation = (
            item.get("index")
            or item.get("create")
            or item.get("update")
        )

        if not isinstance(
            operation,
            dict,
        ):
            failures.append(
                {
                    "position": position,
                    "reason": (
                        "Bulk item has no supported "
                        "operation result."
                    ),
                    "item": item,
                }
            )
            continue

        document_id = operation.get(
            "_id"
        )

        if isinstance(
            document_id,
            str,
        ):
            returned_ids.add(document_id)

        status = operation.get(
            "status"
        )

        status_key = str(status)

        status_counts[status_key] = (
            status_counts.get(
                status_key,
                0,
            )
            + 1
        )

        result = operation.get(
            "result",
            "unknown",
        )

        result_key = str(result)

        result_counts[result_key] = (
            result_counts.get(
                result_key,
                0,
            )
            + 1
        )

        if (
            not isinstance(status, int)
            or status < 200
            or status >= 300
            or operation.get("error")
        ):
            failures.append(
                {
                    "position": position,
                    "document_id": document_id,
                    "status": status,
                    "error": operation.get(
                        "error"
                    ),
                    "item": item,
                }
            )

    return {
        "item_count": len(items),
        "unique_returned_ids": len(
            returned_ids
        ),
        "status_counts": status_counts,
        "result_counts": result_counts,
        "failure_count": len(failures),
        "failures": failures,
    }



def build_bulk_batches(
    bulk_path: Path,
    max_batch_bytes: int,
) -> list[dict[str, Any]]:
    if max_batch_bytes < 1:
        raise ValueError(
            "max_batch_bytes must be positive."
        )

    payload = bulk_path.read_bytes()

    if not payload:
        raise RuntimeError(
            "Bulk payload is empty."
        )

    if not payload.endswith(b"\n"):
        raise RuntimeError(
            "Bulk payload does not end "
            "with newline."
        )

    lines = payload.splitlines(
        keepends=True
    )

    if len(lines) % 2 != 0:
        raise RuntimeError(
            "Bulk payload must contain "
            "action/document line pairs."
        )

    batches: list[
        dict[str, Any]
    ] = []

    current_parts: list[bytes] = []
    current_size = 0
    current_documents = 0

    def flush_batch() -> None:
        nonlocal current_parts
        nonlocal current_size
        nonlocal current_documents

        if not current_parts:
            return

        body = b"".join(
            current_parts
        )

        batches.append(
            {
                "batch_number": (
                    len(batches) + 1
                ),
                "document_count": (
                    current_documents
                ),
                "size_bytes": len(body),
                "sha256": sha256_bytes(
                    body
                ),
                "body": body,
            }
        )

        current_parts = []
        current_size = 0
        current_documents = 0

    for position in range(
        0,
        len(lines),
        2,
    ):
        pair = (
            lines[position]
            + lines[position + 1]
        )

        pair_number = (
            position // 2 + 1
        )

        if len(pair) > max_batch_bytes:
            raise RuntimeError(
                "A single bulk action/document "
                "pair exceeds the configured "
                "batch limit: "
                f"pair={pair_number}, "
                f"size={len(pair)}, "
                f"limit={max_batch_bytes}"
            )

        if (
            current_parts
            and current_size + len(pair)
            > max_batch_bytes
        ):
            flush_batch()

        current_parts.append(pair)
        current_size += len(pair)
        current_documents += 1

    flush_batch()

    if not batches:
        raise RuntimeError(
            "No bulk batches were produced."
        )

    return batches


def build_batch_plan(
    batches: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    return [
        {
            "batch_number": int(
                batch["batch_number"]
            ),
            "document_count": int(
                batch["document_count"]
            ),
            "size_bytes": int(
                batch["size_bytes"]
            ),
            "sha256": str(
                batch["sha256"]
            ),
        }
        for batch in batches
    ]


def checkpoint_runtime_identity(
    runtime: dict[str, Any],
) -> dict[str, Any]:
    fields = (
        "region",
        "collection_endpoint",
        "index_name",
        "book_id",
        "book_version",
        "vector_dimensions",
    )

    return {
        field: runtime[field]
        for field in fields
    }


def build_upload_checkpoint(
    *,
    runtime: dict[str, Any],
    bulk_sha256: str,
    expected_document_count: int,
    max_batch_bytes: int,
    batches: list[
        dict[str, Any]
    ],
    initial_count: int,
) -> dict[str, Any]:
    now = utc_now()

    return {
        "schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "status": "IN_PROGRESS",
        "started_at": now,
        "updated_at": now,
        "runtime_identity": (
            checkpoint_runtime_identity(
                runtime
            )
        ),
        "bulk_payload": {
            "sha256": bulk_sha256,
            "expected_document_count": (
                expected_document_count
            ),
        },
        "batching": {
            "max_batch_bytes": (
                max_batch_bytes
            ),
            "batch_count": len(
                batches
            ),
            "plan": build_batch_plan(
                batches
            ),
        },
        "initial_count": initial_count,
        "completed_batches": {},
    }


def validate_upload_checkpoint(
    checkpoint: dict[str, Any],
    *,
    runtime: dict[str, Any],
    bulk_sha256: str,
    expected_document_count: int,
    max_batch_bytes: int,
    batches: list[
        dict[str, Any]
    ],
) -> None:
    if checkpoint.get(
        "schema_version"
    ) != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            "Upload checkpoint schema "
            "version mismatch."
        )

    if checkpoint.get(
        "status"
    ) not in {
        "IN_PROGRESS",
        "COMPLETED",
    }:
        raise RuntimeError(
            "Upload checkpoint status "
            "is invalid."
        )

    expected_runtime = (
        checkpoint_runtime_identity(
            runtime
        )
    )

    if checkpoint.get(
        "runtime_identity"
    ) != expected_runtime:
        raise RuntimeError(
            "Upload checkpoint runtime "
            "identity does not match."
        )

    payload = checkpoint.get(
        "bulk_payload"
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Upload checkpoint contains "
            "no bulk payload identity."
        )

    if payload.get(
        "sha256"
    ) != bulk_sha256:
        raise RuntimeError(
            "Upload checkpoint bulk "
            "checksum does not match."
        )

    if payload.get(
        "expected_document_count"
    ) != expected_document_count:
        raise RuntimeError(
            "Upload checkpoint document "
            "count does not match."
        )

    batching = checkpoint.get(
        "batching"
    )

    if not isinstance(
        batching,
        dict,
    ):
        raise RuntimeError(
            "Upload checkpoint contains "
            "no batching plan."
        )

    expected_plan = build_batch_plan(
        batches
    )

    if batching.get(
        "max_batch_bytes"
    ) != max_batch_bytes:
        raise RuntimeError(
            "Upload checkpoint batch size "
            "does not match."
        )

    if batching.get(
        "batch_count"
    ) != len(batches):
        raise RuntimeError(
            "Upload checkpoint batch count "
            "does not match."
        )

    if batching.get(
        "plan"
    ) != expected_plan:
        raise RuntimeError(
            "Upload checkpoint batching "
            "plan does not match."
        )

    initial_count = checkpoint.get(
        "initial_count"
    )

    if (
        not isinstance(
            initial_count,
            int,
        )
        or initial_count < 0
        or initial_count
        > expected_document_count
    ):
        raise RuntimeError(
            "Upload checkpoint initial "
            "document count is invalid."
        )

    completed = checkpoint.get(
        "completed_batches"
    )

    if not isinstance(
        completed,
        dict,
    ):
        raise RuntimeError(
            "Upload checkpoint completed "
            "batch map is invalid."
        )

    planned = {
        f"{item['batch_number']:04d}": item
        for item in expected_plan
    }

    for key, entry in (
        completed.items()
    ):
        if key not in planned:
            raise RuntimeError(
                "Upload checkpoint contains "
                f"unknown batch {key}."
            )

        if not isinstance(
            entry,
            dict,
        ):
            raise RuntimeError(
                "Upload checkpoint batch "
                f"{key} is invalid."
            )

        if entry.get(
            "status"
        ) != "COMPLETED":
            raise RuntimeError(
                "Upload checkpoint batch "
                f"{key} is not COMPLETED."
            )

        expected = planned[key]

        for field in (
            "document_count",
            "size_bytes",
            "sha256",
        ):
            if entry.get(
                field
            ) != expected[field]:
                raise RuntimeError(
                    "Upload checkpoint batch "
                    f"{key} {field} mismatch."
                )

        result = entry.get(
            "bulk_result"
        )

        if not isinstance(
            result,
            dict,
        ):
            raise RuntimeError(
                "Upload checkpoint batch "
                f"{key} has no bulk result."
            )

        if result.get(
            "item_count"
        ) != expected[
            "document_count"
        ]:
            raise RuntimeError(
                "Upload checkpoint batch "
                f"{key} item count mismatch."
            )

        if result.get(
            "unique_returned_ids"
        ) != expected[
            "document_count"
        ]:
            raise RuntimeError(
                "Upload checkpoint batch "
                f"{key} unique ID count "
                "mismatch."
            )

        if (
            result.get(
                "failure_count"
            )
            != 0
            or result.get(
                "bulk_errors_flag"
            )
            is True
        ):
            raise RuntimeError(
                "Upload checkpoint batch "
                f"{key} contains failures."
            )


def aggregate_completed_batches(
    checkpoint: dict[str, Any],
    batches: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    completed = checkpoint[
        "completed_batches"
    ]

    status_counts: dict[
        str,
        int,
    ] = {}

    result_counts: dict[
        str,
        int,
    ] = {}

    item_count = 0
    unique_returned_ids = 0
    failure_count = 0
    failures: list[
        dict[str, Any]
    ] = []

    for batch in batches:
        key = (
            f"{batch['batch_number']:04d}"
        )

        entry = completed.get(key)

        if not isinstance(
            entry,
            dict,
        ):
            raise RuntimeError(
                "Completed upload is missing "
                f"batch {key}."
            )

        result = entry[
            "bulk_result"
        ]

        item_count += int(
            result["item_count"]
        )

        unique_returned_ids += int(
            result[
                "unique_returned_ids"
            ]
        )

        failure_count += int(
            result["failure_count"]
        )

        failures.extend(
            result.get(
                "failures",
                [],
            )
        )

        for status, count in (
            result.get(
                "status_counts",
                {},
            ).items()
        ):
            status_counts[
                str(status)
            ] = (
                status_counts.get(
                    str(status),
                    0,
                )
                + int(count)
            )

        for result_name, count in (
            result.get(
                "result_counts",
                {},
            ).items()
        ):
            result_counts[
                str(result_name)
            ] = (
                result_counts.get(
                    str(result_name),
                    0,
                )
                + int(count)
            )

    return {
        "item_count": item_count,
        "unique_returned_ids": (
            unique_returned_ids
        ),
        "status_counts": (
            status_counts
        ),
        "result_counts": (
            result_counts
        ),
        "failure_count": (
            failure_count
        ),
        "failures": failures,
        "bulk_errors_flag": (
            failure_count > 0
        ),
    }


def get_book_count(
    http: urllib3.PoolManager,
) -> tuple[int, dict[str, Any]]:
    query = {
        "query": {
            "bool": {
                "filter": [
                    {
                        "term": {
                            "book_id": BOOK_ID
                        }
                    },
                    {
                        "term": {
                            "book_version": (
                                BOOK_VERSION
                            )
                        }
                    },
                ]
            }
        }
    }

    body = json.dumps(
        query,
        separators=(",", ":"),
    ).encode("utf-8")

    return signed_request(
        http=http,
        method="POST",
        path=f"{INDEX_NAME}/_count",
        body=body,
        content_type="application/json",
    )


def wait_for_document_count(
    http: urllib3.PoolManager,
    expected_count: int,
    maximum_checks: int = 36,
    delay_seconds: int = 5,
) -> tuple[int, dict[str, Any]]:
    last_response: dict[str, Any] = {}

    for check in range(
        1,
        maximum_checks + 1,
    ):
        status, response = get_book_count(
            http
        )

        last_response = response

        if status < 200 or status >= 300:
            print(
                f"Count check {check}: "
                f"HTTP {status}"
            )

        else:
            count = response.get(
                "count"
            )

            print(
                f"Count check {check}: "
                f"{count}/{expected_count}"
            )

            if count == expected_count:
                return status, response

            if (
                isinstance(count, int)
                and count > expected_count
            ):
                raise RuntimeError(
                    "Indexed document count exceeds "
                    f"expected count: {count}"
                )

        if check < maximum_checks:
            time.sleep(delay_seconds)

    raise TimeoutError(
        "Documents did not become searchable at "
        f"the expected count. Last response: "
        f"{json.dumps(last_response)}"
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a validated NDJSON bulk "
            "payload to OpenSearch Serverless "
            "using resumable batches."
        )
    )

    parser.add_argument(
        "bulk_path",
        type=Path,
    )

    parser.add_argument(
        "--preparation-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional book configuration JSON. "
            "Endpoint, index, book identity "
            "and vector dimensions are derived "
            "from BookConfig."
        ),
    )

    parser.add_argument(
        "--max-batch-bytes",
        type=int,
        default=DEFAULT_MAX_BATCH_BYTES,
        help=(
            "Maximum NDJSON bytes per request. "
            "Action/document pairs are never "
            "split. Default: 5 MiB."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    runtime = configure_runtime(
        args.config
    )

    if not args.bulk_path.exists():
        raise FileNotFoundError(
            f"Bulk payload not found: "
            f"{args.bulk_path}"
        )

    if args.max_batch_bytes < 1:
        raise ValueError(
            "--max-batch-bytes must be "
            "positive."
        )

    preparation_report = load_json(
        args.preparation_report
    )

    expected_document_count = (
        validate_preparation(
            bulk_path=args.bulk_path,
            preparation_report=(
                preparation_report
            ),
            runtime=runtime,
        )
    )

    bulk_sha256 = sha256_file(
        args.bulk_path
    )

    payload_size = (
        args.bulk_path.stat().st_size
    )

    batches = build_bulk_batches(
        bulk_path=args.bulk_path,
        max_batch_bytes=(
            args.max_batch_bytes
        ),
    )

    planned_document_count = sum(
        int(batch["document_count"])
        for batch in batches
    )

    if (
        planned_document_count
        != expected_document_count
    ):
        raise RuntimeError(
            "Batch plan document count "
            "does not match prepared count: "
            f"planned={planned_document_count}, "
            f"expected="
            f"{expected_document_count}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    response_path = (
        args.output_dir
        / "bulk-response.json"
    )

    response_directory = (
        args.output_dir
        / "bulk-responses"
    )

    response_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        args.output_dir
        / "bulk-upload-report.json"
    )

    checkpoint_path = (
        args.output_dir
        / "bulk-upload-checkpoint.json"
    )

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(
            connect=15.0,
            read=180.0,
        ),
        retries=False,
    )

    print(
        "============================================"
    )
    print(
        "OPENSEARCH RESUMABLE BULK UPLOAD"
    )
    print(
        "============================================"
    )
    print(
        f"Mode:           {runtime['mode']}"
    )
    print(
        f"Endpoint:       {COLLECTION_ENDPOINT}"
    )
    print(
        f"Index:          {INDEX_NAME}"
    )
    print(
        f"Documents:      "
        f"{expected_document_count}"
    )
    print(
        f"Payload size:   "
        f"{payload_size:,} bytes"
    )
    print(
        f"Payload SHA256: {bulk_sha256}"
    )
    print(
        f"Batch limit:    "
        f"{args.max_batch_bytes:,} bytes"
    )
    print(
        f"Batch count:    {len(batches)}"
    )
    print("Signing service:aoss")
    print()

    if checkpoint_path.exists():
        checkpoint = load_json(
            checkpoint_path
        )

        validate_upload_checkpoint(
            checkpoint,
            runtime=runtime,
            bulk_sha256=bulk_sha256,
            expected_document_count=(
                expected_document_count
            ),
            max_batch_bytes=(
                args.max_batch_bytes
            ),
            batches=batches,
        )

        print(
            "Resume checkpoint: found"
        )

    else:
        initial_status, response = (
            get_book_count(http)
        )

        if (
            initial_status < 200
            or initial_status >= 300
        ):
            raise RuntimeError(
                "Initial OpenSearch count "
                f"failed with HTTP "
                f"{initial_status}: "
                f"{json.dumps(response)}"
            )

        initial_count = response.get(
            "count"
        )

        if (
            not isinstance(
                initial_count,
                int,
            )
            or initial_count < 0
            or initial_count
            > expected_document_count
        ):
            raise RuntimeError(
                "Initial OpenSearch count "
                "is invalid: "
                f"{initial_count}"
            )

        checkpoint = (
            build_upload_checkpoint(
                runtime=runtime,
                bulk_sha256=bulk_sha256,
                expected_document_count=(
                    expected_document_count
                ),
                max_batch_bytes=(
                    args.max_batch_bytes
                ),
                batches=batches,
                initial_count=(
                    initial_count
                ),
            )
        )

        atomic_write_json(
            checkpoint_path,
            checkpoint,
        )

        print(
            "Resume checkpoint: created"
        )

    initial_count = int(
        checkpoint["initial_count"]
    )

    print(
        f"Initial count:  {initial_count}"
    )
    print()

    skipped_batches = 0
    sent_batches = 0

    for batch in batches:
        number = int(
            batch["batch_number"]
        )

        key = f"{number:04d}"

        existing = checkpoint[
            "completed_batches"
        ].get(key)

        if isinstance(
            existing,
            dict,
        ):
            skipped_batches += 1

            print(
                f"Batch {number}/"
                f"{len(batches)}: "
                "already completed; skipped"
            )

            continue

        body = batch["body"]

        print(
            f"Batch {number}/"
            f"{len(batches)}: "
            f"{batch['document_count']} docs, "
            f"{batch['size_bytes']:,} bytes"
        )

        started_at = time.monotonic()

        bulk_status, bulk_response = (
            signed_request(
                http=http,
                method="POST",
                path="_bulk",
                body=body,
                content_type=(
                    "application/x-ndjson"
                ),
            )
        )

        duration_seconds = (
            time.monotonic()
            - started_at
        )

        batch_response_path = (
            response_directory
            / f"batch-{number:04d}.json"
        )

        atomic_write_json(
            batch_response_path,
            bulk_response,
        )

        if (
            bulk_status < 200
            or bulk_status >= 300
        ):
            raise RuntimeError(
                "Bulk batch failed with HTTP "
                f"{bulk_status}: "
                f"batch={number}"
            )

        parsed_bulk = parse_bulk_result(
            bulk_response
        )

        parsed_bulk[
            "bulk_errors_flag"
        ] = (
            bulk_response.get(
                "errors"
            )
            is True
        )

        if (
            parsed_bulk["item_count"]
            != batch["document_count"]
        ):
            raise RuntimeError(
                "Bulk batch response item "
                "count differs from expected: "
                f"batch={number}, "
                f"actual="
                f"{parsed_bulk['item_count']}, "
                f"expected="
                f"{batch['document_count']}"
            )

        if (
            parsed_bulk[
                "unique_returned_ids"
            ]
            != batch["document_count"]
        ):
            raise RuntimeError(
                "Bulk batch did not return "
                "the expected unique IDs: "
                f"batch={number}"
            )

        if (
            parsed_bulk[
                "failure_count"
            ]
            > 0
            or parsed_bulk[
                "bulk_errors_flag"
            ]
        ):
            raise RuntimeError(
                "One or more items failed "
                f"in batch {number}:\n"
                + json.dumps(
                    parsed_bulk,
                    indent=2,
                    default=str,
                )
            )

        checkpoint[
            "completed_batches"
        ][key] = {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "batch_number": number,
            "document_count": int(
                batch["document_count"]
            ),
            "size_bytes": int(
                batch["size_bytes"]
            ),
            "sha256": str(
                batch["sha256"]
            ),
            "http_status": (
                bulk_status
            ),
            "duration_seconds": (
                duration_seconds
            ),
            "response_path": str(
                batch_response_path
            ),
            "bulk_result": parsed_bulk,
        }

        checkpoint[
            "updated_at"
        ] = utc_now()

        atomic_write_json(
            checkpoint_path,
            checkpoint,
        )

        sent_batches += 1

        print(
            f"  HTTP status: {bulk_status}"
        )
        print(
            "  Result counts: "
            f"{parsed_bulk['result_counts']}"
        )
        print(
            "  Item failures: "
            f"{parsed_bulk['failure_count']}"
        )
        print(
            f"  Checkpoint: saved"
        )

    validate_upload_checkpoint(
        checkpoint,
        runtime=runtime,
        bulk_sha256=bulk_sha256,
        expected_document_count=(
            expected_document_count
        ),
        max_batch_bytes=(
            args.max_batch_bytes
        ),
        batches=batches,
    )

    aggregate = (
        aggregate_completed_batches(
            checkpoint,
            batches,
        )
    )

    if (
        aggregate["item_count"]
        != expected_document_count
    ):
        raise RuntimeError(
            "Aggregated bulk item count "
            "does not match expected count."
        )

    if (
        aggregate[
            "unique_returned_ids"
        ]
        != expected_document_count
    ):
        raise RuntimeError(
            "Aggregated unique ID count "
            "does not match expected count."
        )

    if (
        aggregate["failure_count"] > 0
        or aggregate[
            "bulk_errors_flag"
        ]
    ):
        raise RuntimeError(
            "Aggregated bulk result "
            "contains failures."
        )

    counted_results = sum(
        int(value)
        for value in aggregate[
            "result_counts"
        ].values()
    )

    if (
        counted_results
        != expected_document_count
    ):
        raise RuntimeError(
            "Aggregated result counts "
            "do not match expected count."
        )

    created_document_count = int(
        aggregate[
            "result_counts"
        ].get(
            "created",
            0,
        )
    )

    existing_document_count = (
        expected_document_count
        - created_document_count
    )

    expected_final_count = (
        expected_document_count
    )

    print()
    print(
        f"Created documents: "
        f"{created_document_count}"
    )
    print(
        f"Updated/existing:  "
        f"{existing_document_count}"
    )
    print(
        f"Expected final:    "
        f"{expected_final_count}"
    )
    print()
    print(
        "Waiting for searchable documents..."
    )

    count_status, count_response = (
        wait_for_document_count(
            http=http,
            expected_count=(
                expected_final_count
            ),
        )
    )

    final_count = count_response.get(
        "count"
    )

    completed_entries = [
        checkpoint[
            "completed_batches"
        ][
            f"{batch['batch_number']:04d}"
        ]
        for batch in batches
    ]

    total_duration = sum(
        float(
            entry[
                "duration_seconds"
            ]
        )
        for entry in completed_entries
    )

    response_summary = {
        "schema_version": "1.0",
        "errors": False,
        "batch_count": len(
            batches
        ),
        "aggregate": aggregate,
        "batches": [
            {
                "batch_number": (
                    entry[
                        "batch_number"
                    ]
                ),
                "document_count": (
                    entry[
                        "document_count"
                    ]
                ),
                "size_bytes": (
                    entry["size_bytes"]
                ),
                "sha256": (
                    entry["sha256"]
                ),
                "http_status": (
                    entry[
                        "http_status"
                    ]
                ),
                "duration_seconds": (
                    entry[
                        "duration_seconds"
                    ]
                ),
                "response_path": (
                    entry[
                        "response_path"
                    ]
                ),
            }
            for entry in (
                completed_entries
            )
        ],
    }

    atomic_write_json(
        response_path,
        response_summary,
    )

    report = {
        "schema_version": "1.1",
        "generated_at": utc_now(),
        "status": "COMPLETED",
        "configuration": runtime,
        "region": REGION,
        "service": SERVICE,
        "collection_endpoint": (
            COLLECTION_ENDPOINT
        ),
        "index_name": INDEX_NAME,
        "book_id": BOOK_ID,
        "book_version": BOOK_VERSION,
        "bulk_payload": {
            "path": str(
                args.bulk_path
            ),
            "sha256": bulk_sha256,
            "size_bytes": (
                payload_size
            ),
            "expected_document_count": (
                expected_document_count
            ),
        },
        "batching": {
            "max_batch_bytes": (
                args.max_batch_bytes
            ),
            "batch_count": len(
                batches
            ),
            "sent_batches_this_run": (
                sent_batches
            ),
            "resumed_batches_this_run": (
                skipped_batches
            ),
            "plan": build_batch_plan(
                batches
            ),
        },
        "initial_count": initial_count,
        "prepared_document_count": (
            expected_document_count
        ),
        "created_document_count": (
            created_document_count
        ),
        "existing_document_count": (
            existing_document_count
        ),
        "expected_final_count": (
            expected_final_count
        ),
        "bulk_http_status": 200,
        "bulk_duration_seconds": (
            total_duration
        ),
        "bulk_result": aggregate,
        "final_count_http_status": (
            count_status
        ),
        "final_count": final_count,
        "bulk_response_path": str(
            response_path
        ),
        "batch_response_directory": (
            str(response_directory)
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
        "uploaded": True,
    }

    atomic_write_json(
        report_path,
        report,
    )

    checkpoint[
        "status"
    ] = "COMPLETED"

    checkpoint[
        "updated_at"
    ] = utc_now()

    checkpoint[
        "completed_at"
    ] = utc_now()

    checkpoint[
        "final_count"
    ] = final_count

    checkpoint[
        "report_path"
    ] = str(report_path)

    atomic_write_json(
        checkpoint_path,
        checkpoint,
    )

    print()
    print(
        "============================================"
    )
    print(
        "RESUMABLE BULK UPLOAD COMPLETED"
    )
    print(
        "============================================"
    )
    print(
        f"Documents accepted: "
        f"{aggregate['item_count']}"
    )
    print(
        f"Unique IDs:         "
        f"{aggregate['unique_returned_ids']}"
    )
    print(
        f"Item failures:      "
        f"{aggregate['failure_count']}"
    )
    print(
        f"Initial count:      "
        f"{initial_count}"
    )
    print(
        f"Final count:        "
        f"{final_count}"
    )
    print(
        f"Batches sent:       "
        f"{sent_batches}"
    )
    print(
        f"Batches resumed:    "
        f"{skipped_batches}"
    )
    print(
        f"Bulk duration:      "
        f"{total_duration:.2f}s"
    )
    print(
        f"Response summary:   "
        f"{response_path}"
    )
    print(
        f"Checkpoint:         "
        f"{checkpoint_path}"
    )
    print(
        f"Report:             "
        f"{report_path}"
    )
    print(
        "Uploaded:           True"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (
        ClientError,
        BotoCoreError,
    ) as exc:
        print(
            f"AWS upload error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"Bulk upload failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
