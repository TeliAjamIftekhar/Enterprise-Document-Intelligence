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
            "Upload a validated NDJSON bulk payload "
            "to OpenSearch Serverless."
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

    bulk_body = args.bulk_path.read_bytes()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    response_path = (
        args.output_dir
        / "bulk-response.json"
    )

    report_path = (
        args.output_dir
        / "bulk-upload-report.json"
    )

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(
            connect=15.0,
            read=180.0,
        ),
        retries=False,
    )

    print("============================================")
    print("OPENSEARCH BULK UPLOAD")
    print("============================================")
    print(
        f"Mode:           {runtime['mode']}"
    )
    print(
        f"Endpoint:       {COLLECTION_ENDPOINT}"
    )
    print(f"Index:          {INDEX_NAME}")
    print(
        f"Documents:      {expected_document_count}"
    )
    print(
        f"Payload size:   {len(bulk_body):,} bytes"
    )
    print(
        f"Payload SHA256: {sha256_bytes(bulk_body)}"
    )
    print("Signing service:aoss")
    print()

    initial_status, initial_count_response = (
        get_book_count(http)
    )

    initial_count = (
        initial_count_response.get("count")
        if initial_status >= 200
        and initial_status < 300
        else None
    )

    print(
        f"Initial count:  {initial_count}"
    )
    print("Sending bulk request...")

    started_at = time.monotonic()

    bulk_status, bulk_response = signed_request(
        http=http,
        method="POST",
        path="_bulk",
        body=bulk_body,
        content_type=(
            "application/x-ndjson"
        ),
    )

    duration_seconds = (
        time.monotonic() - started_at
    )

    atomic_write_json(
        response_path,
        bulk_response,
    )

    if (
        bulk_status < 200
        or bulk_status >= 300
    ):
        raise RuntimeError(
            "Bulk request failed with HTTP "
            f"{bulk_status}:\n"
            + json.dumps(
                bulk_response,
                indent=2,
                default=str,
            )
        )

    parsed_bulk = parse_bulk_result(
        bulk_response
    )

    if bulk_response.get(
        "errors"
    ) is True:
        parsed_bulk["bulk_errors_flag"] = True

    else:
        parsed_bulk["bulk_errors_flag"] = False

    if (
        parsed_bulk["item_count"]
        != expected_document_count
    ):
        raise RuntimeError(
            "Bulk response item count differs "
            f"from expected: "
            f"{parsed_bulk['item_count']}"
        )

    if (
        parsed_bulk["unique_returned_ids"]
        != expected_document_count
    ):
        raise RuntimeError(
            "Bulk response did not return "
            f"{expected_document_count} unique "
            "document IDs."
        )

    if (
        parsed_bulk["failure_count"] > 0
        or parsed_bulk[
            "bulk_errors_flag"
        ]
    ):
        raise RuntimeError(
            "One or more bulk items failed:\n"
            + json.dumps(
                parsed_bulk,
                indent=2,
                default=str,
            )
        )

    print(
        "Bulk HTTP status:",
        bulk_status,
    )
    print(
        "Bulk items:      ",
        parsed_bulk["item_count"],
    )
    print(
        "Status counts:   ",
        parsed_bulk["status_counts"],
    )
    print(
        "Result counts:   ",
        parsed_bulk["result_counts"],
    )
    print(
        "Item failures:   ",
        parsed_bulk["failure_count"],
    )
    print()

    result_counts = parsed_bulk[
        "result_counts"
    ]

    counted_results = sum(
        int(value)
        for value in result_counts.values()
    )

    if (
        counted_results
        != expected_document_count
    ):
        raise RuntimeError(
            "Bulk result counts do not match "
            "the expected document count: "
            f"results={counted_results}, "
            f"expected={expected_document_count}"
        )

    created_document_count = int(
        result_counts.get(
            "created",
            0,
        )
    )

    existing_document_count = (
        expected_document_count
        - created_document_count
    )

    if not isinstance(
        initial_count,
        int,
    ):
        raise RuntimeError(
            "Initial OpenSearch count is not "
            f"an integer: {initial_count}"
        )

    expected_final_count = (
        initial_count
        + created_document_count
    )

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

    print("Waiting for searchable documents...")

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

    report = {
        "schema_version": "1.0",
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
            "path": str(args.bulk_path),
            "sha256": sha256_bytes(
                bulk_body
            ),
            "size_bytes": len(
                bulk_body
            ),
            "expected_document_count": (
                expected_document_count
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
        "bulk_http_status": bulk_status,
        "bulk_duration_seconds": (
            duration_seconds
        ),
        "bulk_result": parsed_bulk,
        "final_count_http_status": (
            count_status
        ),
        "final_count": final_count,
        "bulk_response_path": str(
            response_path
        ),
        "uploaded": True,
    }

    atomic_write_json(
        report_path,
        report,
    )

    print()
    print("============================================")
    print("BULK UPLOAD COMPLETED")
    print("============================================")
    print(
        f"Documents accepted: "
        f"{parsed_bulk['item_count']}"
    )
    print(
        f"Unique IDs:         "
        f"{parsed_bulk['unique_returned_ids']}"
    )
    print(
        f"Item failures:      "
        f"{parsed_bulk['failure_count']}"
    )
    print(
        f"Initial count:      {initial_count}"
    )
    print(
        f"Final count:        {final_count}"
    )
    print(
        f"Bulk duration:      "
        f"{duration_seconds:.2f}s"
    )
    print(
        f"Response:           {response_path}"
    )
    print(
        f"Report:             {report_path}"
    )
    print("Uploaded:           True")

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
