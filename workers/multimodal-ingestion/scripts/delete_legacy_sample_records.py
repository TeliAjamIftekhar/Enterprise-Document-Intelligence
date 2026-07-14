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

import urllib3

import upload_opensearch_bulk as opensearch


EXPECTED_BEFORE_COUNT = 2202
EXPECTED_AFTER_COUNT = 2162
EXPECTED_DELETE_COUNT = 40


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file missing: {path}"
        )

    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise RuntimeError(
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
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        path,
    )


def parse_delete_payload(
    payload: bytes,
) -> list[str]:
    if not payload.endswith(b"\n"):
        raise RuntimeError(
            "Delete payload does not end with newline."
        )

    document_ids: list[str] = []

    for line_number, line in enumerate(
        payload.decode("utf-8").splitlines(),
        start=1,
    ):
        item = json.loads(line)

        if set(item) != {"delete"}:
            raise RuntimeError(
                f"Payload line {line_number} is not "
                "a delete operation."
            )

        operation = item["delete"]

        if (
            operation.get("_index")
            != opensearch.INDEX_NAME
        ):
            raise RuntimeError(
                f"Payload line {line_number} targets "
                "the wrong index."
            )

        document_id = str(
            operation.get("_id", "")
        )

        if not document_id:
            raise RuntimeError(
                f"Payload line {line_number} "
                "contains no document ID."
            )

        document_ids.append(document_id)

    if len(document_ids) != EXPECTED_DELETE_COUNT:
        raise RuntimeError(
            "Unexpected delete payload count: "
            f"{len(document_ids)}"
        )

    if (
        len(set(document_ids))
        != EXPECTED_DELETE_COUNT
    ):
        raise RuntimeError(
            "Delete payload IDs are not unique."
        )

    return document_ids


def parse_delete_response(
    response: dict[str, Any],
    expected_ids: set[str],
) -> dict[str, Any]:
    items = response.get("items")

    if not isinstance(items, list):
        raise RuntimeError(
            "Bulk response contains no items list."
        )

    returned_ids: set[str] = set()
    failures: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    result_counts: dict[str, int] = {}

    for position, item in enumerate(
        items,
        start=1,
    ):
        if not isinstance(item, dict):
            failures.append(
                {
                    "position": position,
                    "reason": "Item is not an object.",
                }
            )
            continue

        operation = item.get("delete")

        if not isinstance(operation, dict):
            failures.append(
                {
                    "position": position,
                    "reason": (
                        "Item has no delete result."
                    ),
                    "item": item,
                }
            )
            continue

        document_id = str(
            operation.get("_id", "")
        )

        if document_id:
            returned_ids.add(document_id)

        status = operation.get("status")
        result = str(
            operation.get(
                "result",
                "unknown",
            )
        )

        status_counts[str(status)] = (
            status_counts.get(
                str(status),
                0,
            )
            + 1
        )

        result_counts[result] = (
            result_counts.get(
                result,
                0,
            )
            + 1
        )

        if (
            not isinstance(status, int)
            or status < 200
            or status >= 300
            or operation.get("error")
            or result != "deleted"
        ):
            failures.append(
                {
                    "position": position,
                    "document_id": document_id,
                    "status": status,
                    "result": result,
                    "error": operation.get("error"),
                }
            )

    if len(items) != EXPECTED_DELETE_COUNT:
        raise RuntimeError(
            "Bulk response item count mismatch: "
            f"{len(items)}"
        )

    if returned_ids != expected_ids:
        raise RuntimeError(
            "Bulk response IDs do not exactly match "
            "the requested legacy IDs."
        )

    if failures:
        raise RuntimeError(
            "One or more legacy deletions failed:\n"
            + json.dumps(
                failures,
                indent=2,
            )
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


def verify_ids_absent(
    http: urllib3.PoolManager,
    document_ids: list[str],
) -> dict[str, Any]:
    body = json.dumps(
        {
            "size": EXPECTED_DELETE_COUNT,
            "track_total_hits": True,
            "_source": False,
            "query": {
                "ids": {
                    "values": sorted(
                        document_ids
                    )
                }
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")

    status, response = opensearch.signed_request(
        http=http,
        method="POST",
        path=(
            f"{opensearch.INDEX_NAME}/_search"
        ),
        body=body,
        content_type="application/json",
    )

    if not 200 <= status < 300:
        raise RuntimeError(
            "Post-delete ID verification failed: "
            f"HTTP {status}"
        )

    hits = (
        response.get("hits", {})
        .get("hits", [])
    )

    if not isinstance(hits, list):
        raise RuntimeError(
            "Post-delete search has no hits list."
        )

    remaining_ids = [
        str(hit.get("_id", ""))
        for hit in hits
    ]

    if remaining_ids:
        raise RuntimeError(
            "Some legacy IDs remain after deletion:\n"
            + json.dumps(
                remaining_ids,
                indent=2,
            )
        )

    return {
        "http_status": status,
        "remaining_legacy_id_count": 0,
        "remaining_legacy_ids": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete exactly 40 audited legacy sample "
            "records from OpenSearch."
        )
    )

    parser.add_argument(
        "--preflight-report",
        type=Path,
        required=True,
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

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    preflight = load_json_object(
        args.preflight_report
    )

    preparation = load_json_object(
        args.preparation_report
    )

    if (
        preflight.get("status")
        != "PREFLIGHT_PASSED"
    ):
        raise RuntimeError(
            "Delete preflight is not passed."
        )

    if (
        int(
            preflight.get(
                "current_index_count",
                0,
            )
        )
        != EXPECTED_BEFORE_COUNT
    ):
        raise RuntimeError(
            "Preflight index count is not 2202."
        )

    if (
        int(
            preflight.get(
                "existing_legacy_id_count",
                0,
            )
        )
        != EXPECTED_DELETE_COUNT
    ):
        raise RuntimeError(
            "Preflight did not verify 40 legacy IDs."
        )

    if preparation.get("status") != "PREPARED":
        raise RuntimeError(
            "Delete payload is not PREPARED."
        )

    payload_path = Path(
        str(
            preparation["payload_path"]
        )
    )

    payload = payload_path.read_bytes()

    payload_sha256 = hashlib.sha256(
        payload
    ).hexdigest()

    if (
        payload_sha256
        != preparation.get("payload_sha256")
        or payload_sha256
        != preflight.get("payload_sha256")
    ):
        raise RuntimeError(
            "Delete payload checksum mismatch."
        )

    document_ids = parse_delete_payload(
        payload
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    response_path = (
        args.output_dir
        / "legacy-sample-delete-response.json"
    )

    report_path = (
        args.output_dir
        / "legacy-sample-delete-execution-report.json"
    )

    if (
        response_path.exists()
        or report_path.exists()
    ):
        raise RuntimeError(
            "Delete execution output already exists. "
            "Do not repeat the deletion."
        )

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(
            connect=10.0,
            read=120.0,
        )
    )

    count_status, count_response = (
        opensearch.get_book_count(http)
    )

    current_count = count_response.get(
        "count"
    )

    if (
        not 200 <= count_status < 300
        or current_count
        != EXPECTED_BEFORE_COUNT
    ):
        raise RuntimeError(
            "Current OpenSearch count changed before "
            "deletion: "
            f"status={count_status}, "
            f"count={current_count}"
        )

    print("=" * 64)
    print("DELETE LEGACY SAMPLE RECORDS")
    print("=" * 64)
    print(
        f"Index:                 "
        f"{opensearch.INDEX_NAME}"
    )
    print(
        f"Current count:         "
        f"{current_count:,}"
    )
    print(
        f"Legacy records:        "
        f"{len(document_ids)}"
    )
    print(
        f"Payload SHA256:        "
        f"{payload_sha256}"
    )
    print(
        f"Expected final count:  "
        f"{EXPECTED_AFTER_COUNT:,}"
    )
    print("Sending bulk delete request...")
    print()

    started_at = utc_now()
    started = time.monotonic()

    bulk_status, bulk_response = (
        opensearch.signed_request(
            http=http,
            method="POST",
            path="_bulk",
            body=payload,
            content_type=(
                "application/x-ndjson"
            ),
        )
    )

    duration_seconds = round(
        time.monotonic() - started,
        2,
    )

    atomic_write_json(
        response_path,
        bulk_response,
    )

    if not 200 <= bulk_status < 300:
        raise RuntimeError(
            "Bulk deletion request failed: "
            f"HTTP {bulk_status}\n"
            + json.dumps(
                bulk_response,
                indent=2,
            )
        )

    if bulk_response.get("errors") is True:
        raise RuntimeError(
            "OpenSearch bulk response reports errors."
        )

    parsed_result = parse_delete_response(
        response=bulk_response,
        expected_ids=set(document_ids),
    )

    print(
        f"Bulk HTTP status:      "
        f"{bulk_status}"
    )
    print(
        f"Delete items:          "
        f"{parsed_result['item_count']}"
    )
    print(
        f"Status counts:         "
        f"{parsed_result['status_counts']}"
    )
    print(
        f"Result counts:         "
        f"{parsed_result['result_counts']}"
    )
    print(
        f"Delete failures:       "
        f"{parsed_result['failure_count']}"
    )
    print()
    print("Waiting for searchable count...")

    final_status, final_response = (
        opensearch.wait_for_document_count(
            http=http,
            expected_count=(
                EXPECTED_AFTER_COUNT
            ),
        )
    )

    final_count = final_response.get(
        "count"
    )

    absence_validation = verify_ids_absent(
        http=http,
        document_ids=document_ids,
    )

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "started_at": started_at,
        "status": "COMPLETED",
        "index_name": opensearch.INDEX_NAME,
        "initial_count": current_count,
        "requested_delete_count": len(
            document_ids
        ),
        "payload_path": str(
            payload_path
        ),
        "payload_sha256": payload_sha256,
        "payload_size_bytes": len(
            payload
        ),
        "bulk_http_status": bulk_status,
        "bulk_duration_seconds": (
            duration_seconds
        ),
        "bulk_result": parsed_result,
        "expected_final_count": (
            EXPECTED_AFTER_COUNT
        ),
        "final_count_http_status": (
            final_status
        ),
        "final_count": final_count,
        "legacy_id_absence_validation": (
            absence_validation
        ),
        "response_path": str(
            response_path
        ),
        "records_deleted": (
            EXPECTED_DELETE_COUNT
        ),
        "opensearch_write": True,
    }

    atomic_write_json(
        report_path,
        report,
    )

    print()
    print("=" * 64)
    print("LEGACY SAMPLE CLEANUP COMPLETED")
    print("=" * 64)
    print(
        f"Records deleted:       "
        f"{EXPECTED_DELETE_COUNT}"
    )
    print(
        f"Delete failures:       "
        f"{parsed_result['failure_count']}"
    )
    print(
        f"Initial index count:   "
        f"{current_count:,}"
    )
    print(
        f"Final index count:     "
        f"{final_count:,}"
    )
    print(
        "Legacy IDs remaining: 0"
    )
    print(
        f"Report:                "
        f"{report_path}"
    )
    print("OpenSearch write:      True")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print(
            "Legacy cleanup interrupted.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except Exception as error:
        print(
            f"Legacy cleanup failed: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
