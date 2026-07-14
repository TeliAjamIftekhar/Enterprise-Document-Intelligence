from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError


REGION = "us-east-1"
BUCKET = "edi-documents-ajam-2026"

RUNNING_STATUSES = {
    "Created",
    "InProgress",
}

FAILURE_STATUSES = {
    "ServiceError",
    "ClientError",
}

SUCCESS_STATUS = "Success"

POLL_INTERVAL_SECONDS = 5
MAXIMUM_STATUS_CHECKS = 360
OUTPUT_LIST_RETRIES = 12
OUTPUT_LIST_DELAY_SECONDS = 5


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
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
            default=json_default,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        path,
    )


def parse_s3_uri(
    s3_uri: str,
) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(
            f"Invalid S3 URI: {s3_uri}"
        )

    without_scheme = s3_uri[5:]

    bucket, separator, key = (
        without_scheme.partition("/")
    )

    if not bucket:
        raise ValueError(
            f"S3 URI has no bucket: {s3_uri}"
        )

    return bucket, key if separator else ""


def verify_preflight(
    report: dict[str, Any],
) -> dict[str, Any]:
    if report.get("status") != "PREFLIGHT_PASSED":
        raise RuntimeError(
            "Pilot preflight status is not "
            "PREFLIGHT_PASSED."
        )

    if report.get("bda_invoked") is not False:
        raise RuntimeError(
            "Preflight report is already marked "
            "as BDA invoked."
        )

    checks = report.get(
        "checks",
        {},
    )

    required_checks = {
        "local_batch_verified",
        "remote_batch_verified",
        "remote_manifest_verified",
        "project_ready",
        "sdk_operations_ready",
        "output_prefix_empty",
        "existing_job_absent",
    }

    failed_checks = [
        check
        for check in sorted(
            required_checks
        )
        if checks.get(check) is not True
    ]

    if failed_checks:
        raise RuntimeError(
            "Preflight checks are not all valid:\n- "
            + "\n- ".join(failed_checks)
        )

    preview = report.get(
        "invocation_preview"
    )

    if not isinstance(preview, dict):
        raise RuntimeError(
            "Preflight contains no invocation preview."
        )

    required_preview_fields = {
        "client_token",
        "input_configuration",
        "output_configuration",
        "data_automation_configuration",
        "data_automation_profile_arn",
    }

    missing_fields = [
        field
        for field in sorted(
            required_preview_fields
        )
        if field not in preview
    ]

    if missing_fields:
        raise RuntimeError(
            "Invocation preview fields missing:\n- "
            + "\n- ".join(missing_fields)
        )

    return preview


def verify_input_object(
    s3_client: Any,
    report: dict[str, Any],
    input_s3_uri: str,
) -> dict[str, Any]:
    bucket, key = parse_s3_uri(
        input_s3_uri
    )

    if bucket != BUCKET:
        raise RuntimeError(
            f"Unexpected input bucket: {bucket}"
        )

    remote_expected = (
        report.get("batch", {})
        .get("remote", {})
    )

    expected_sha256 = (
        remote_expected.get("sha256")
    )

    expected_size = (
        remote_expected.get("size_bytes")
    )

    head = s3_client.head_object(
        Bucket=bucket,
        Key=key,
    )

    actual_size = int(
        head["ContentLength"]
    )

    actual_sha256 = (
        head.get(
            "Metadata",
            {},
        ).get("sha256")
    )

    if actual_size != expected_size:
        raise RuntimeError(
            "Pilot input size changed after "
            "preflight."
        )

    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Pilot input SHA256 changed after "
            "preflight."
        )

    if head.get("ContentType") != "application/pdf":
        raise RuntimeError(
            "Pilot input content type is not PDF."
        )

    return {
        "bucket": bucket,
        "key": key,
        "s3_uri": input_s3_uri,
        "size_bytes": actual_size,
        "sha256": actual_sha256,
        "etag": str(
            head.get("ETag", "")
        ).strip('"'),
        "version_id": head.get(
            "VersionId"
        ),
        "verified_at": utc_now(),
        "verified": True,
    }


def verify_output_prefix_empty(
    s3_client: Any,
    output_s3_uri: str,
) -> dict[str, Any]:
    bucket, prefix = parse_s3_uri(
        output_s3_uri
    )

    normalized_prefix = (
        prefix.rstrip("/")
        + "/"
    )

    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=normalized_prefix,
        MaxKeys=20,
    )

    objects = response.get(
        "Contents",
        [],
    )

    if objects:
        raise RuntimeError(
            "Pilot output prefix is no longer "
            "empty. Existing objects:\n- "
            + "\n- ".join(
                str(item.get("Key"))
                for item in objects
            )
        )

    return {
        "bucket": bucket,
        "prefix": prefix,
        "s3_uri": output_s3_uri,
        "existing_object_count": 0,
        "empty": True,
        "verified_at": utc_now(),
    }


def list_output_objects(
    s3_client: Any,
    output_s3_uri: str,
) -> list[dict[str, Any]]:
    bucket, prefix = parse_s3_uri(
        output_s3_uri
    )

    normalized_prefix = (
        prefix.rstrip("/")
        + "/"
    )

    objects: list[
        dict[str, Any]
    ] = []

    continuation_token: str | None = None

    while True:
        request: dict[str, Any] = {
            "Bucket": bucket,
            "Prefix": normalized_prefix,
            "MaxKeys": 1000,
        }

        if continuation_token:
            request[
                "ContinuationToken"
            ] = continuation_token

        response = s3_client.list_objects_v2(
            **request
        )

        for item in response.get(
            "Contents",
            [],
        ):
            objects.append(
                {
                    "key": item.get("Key"),
                    "size_bytes": int(
                        item.get("Size", 0)
                    ),
                    "etag": str(
                        item.get("ETag", "")
                    ).strip('"'),
                    "last_modified": (
                        item.get("LastModified")
                    ),
                }
            )

        if not response.get(
            "IsTruncated"
        ):
            break

        continuation_token = (
            response.get(
                "NextContinuationToken"
            )
        )

        if not continuation_token:
            raise RuntimeError(
                "S3 listing was truncated but "
                "returned no continuation token."
            )

    return sorted(
        objects,
        key=lambda item: str(
            item["key"]
        ),
    )


def wait_for_output_objects(
    s3_client: Any,
    output_s3_uri: str,
) -> list[dict[str, Any]]:
    for attempt in range(
        1,
        OUTPUT_LIST_RETRIES + 1,
    ):
        objects = list_output_objects(
            s3_client=s3_client,
            output_s3_uri=output_s3_uri,
        )

        print(
            f"Output check {attempt}: "
            f"{len(objects)} object(s)"
        )

        if objects:
            return objects

        if attempt < OUTPUT_LIST_RETRIES:
            time.sleep(
                OUTPUT_LIST_DELAY_SECONDS
            )

    raise RuntimeError(
        "BDA reported Success, but no S3 "
        "output objects were found."
    )


def poll_invocation(
    runtime_client: Any,
    job_record: dict[str, Any],
    job_record_path: Path,
) -> dict[str, Any]:
    invocation_arn = str(
        job_record["invocation_arn"]
    )

    for attempt in range(
        1,
        MAXIMUM_STATUS_CHECKS + 1,
    ):
        response = (
            runtime_client
            .get_data_automation_status(
                invocationArn=(
                    invocation_arn
                )
            )
        )

        status = response.get(
            "status"
        )

        print(
            f"Status check {attempt}: {status}"
        )

        history = job_record.setdefault(
            "status_history",
            [],
        )

        if (
            not history
            or history[-1].get("status")
            != status
        ):
            history.append(
                {
                    "checked_at": utc_now(),
                    "status": status,
                }
            )

        job_record["latest_status"] = (
            status
        )

        job_record[
            "latest_status_checked_at"
        ] = utc_now()

        job_record["status_response"] = (
            response
        )

        atomic_write_json(
            job_record_path,
            job_record,
        )

        if status == SUCCESS_STATUS:
            job_record[
                "completed_at"
            ] = utc_now()

            atomic_write_json(
                job_record_path,
                job_record,
            )

            return response

        if status in FAILURE_STATUSES:
            error_type = response.get(
                "errorType"
            )

            error_message = response.get(
                "errorMessage"
            )

            job_record["failed_at"] = (
                utc_now()
            )

            atomic_write_json(
                job_record_path,
                job_record,
            )

            raise RuntimeError(
                "BDA invocation failed. "
                f"Status={status}; "
                f"errorType={error_type}; "
                f"errorMessage={error_message}"
            )

        if status not in RUNNING_STATUSES:
            raise RuntimeError(
                "Unexpected BDA status: "
                f"{status}"
            )

        if attempt < MAXIMUM_STATUS_CHECKS:
            time.sleep(
                POLL_INTERVAL_SECONDS
            )

    raise TimeoutError(
        "BDA pilot did not reach a terminal "
        "status within the polling limit."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Invoke and monitor exactly one "
            "preflighted full-book BDA pilot."
        )
    )

    parser.add_argument(
        "--preflight-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--job-record",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    report = load_json_object(
        args.preflight_report
    )

    preview = verify_preflight(
        report
    )

    if args.job_record.exists():
        existing = load_json_object(
            args.job_record
        )

        raise RuntimeError(
            "Pilot job record already exists. "
            "Do not submit another invocation.\n"
            f"Path: {args.job_record}\n"
            f"Status: "
            f"{existing.get('latest_status')}\n"
            f"Invocation: "
            f"{existing.get('invocation_arn')}"
        )

    input_configuration = preview[
        "input_configuration"
    ]

    output_configuration = preview[
        "output_configuration"
    ]

    automation_configuration = preview[
        "data_automation_configuration"
    ]

    input_s3_uri = str(
        input_configuration["s3Uri"]
    )

    output_s3_uri = str(
        output_configuration["s3Uri"]
    )

    client_token = str(
        preview["client_token"]
    )

    profile_arn = str(
        preview[
            "data_automation_profile_arn"
        ]
    )

    s3_client = boto3.client(
        "s3",
        region_name=REGION,
    )

    runtime_client = boto3.client(
        "bedrock-data-automation-runtime",
        region_name=REGION,
    )

    input_validation = verify_input_object(
        s3_client=s3_client,
        report=report,
        input_s3_uri=input_s3_uri,
    )

    output_validation = (
        verify_output_prefix_empty(
            s3_client=s3_client,
            output_s3_uri=output_s3_uri,
        )
    )

    batch = report.get(
        "batch",
        {},
    )

    batch_id = str(
        batch.get("batch_id")
    )

    print("=" * 56)
    print("FULL BOOK BDA PILOT INVOCATION")
    print("=" * 56)
    print(f"Batch:        {batch_id}")
    print(
        "Source pages: "
        f"{batch.get('source_page_start')}-"
        f"{batch.get('source_page_end')}"
    )
    print(f"Input:        {input_s3_uri}")
    print(f"Output:       {output_s3_uri}")
    print(
        "Project:      "
        f"{automation_configuration.get('dataAutomationProjectArn')}"
    )
    print(
        "Stage:        "
        f"{automation_configuration.get('stage')}"
    )
    print(f"Client token: {client_token}")
    print("Submitting exactly one invocation...")
    print()

    submitted_at = utc_now()

    response = (
        runtime_client
        .invoke_data_automation_async(
            clientToken=client_token,
            inputConfiguration={
                "s3Uri": input_s3_uri,
            },
            outputConfiguration={
                "s3Uri": output_s3_uri,
            },
            dataAutomationConfiguration={
                "dataAutomationProjectArn": (
                    automation_configuration[
                        "dataAutomationProjectArn"
                    ]
                ),
                "stage": (
                    automation_configuration[
                        "stage"
                    ]
                ),
            },
            dataAutomationProfileArn=(
                profile_arn
            ),
        )
    )

    invocation_arn = response.get(
        "invocationArn"
    )

    if not isinstance(
        invocation_arn,
        str,
    ) or not invocation_arn:
        raise RuntimeError(
            "InvokeDataAutomationAsync returned "
            "no invocation ARN."
        )

    job_record: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": submitted_at,
        "updated_at": submitted_at,
        "status": "SUBMITTED",
        "region": REGION,
        "bucket": BUCKET,
        "book_id": report.get(
            "book_id"
        ),
        "book_version": report.get(
            "book_version"
        ),
        "batch_id": batch_id,
        "source_page_start": batch.get(
            "source_page_start"
        ),
        "source_page_end": batch.get(
            "source_page_end"
        ),
        "page_count": batch.get(
            "page_count"
        ),
        "preflight_report_path": str(
            args.preflight_report
        ),
        "invocation_arn": (
            invocation_arn
        ),
        "client_token": client_token,
        "input_configuration": {
            "s3Uri": input_s3_uri,
        },
        "output_configuration": {
            "s3Uri": output_s3_uri,
        },
        "data_automation_configuration": {
            "dataAutomationProjectArn": (
                automation_configuration[
                    "dataAutomationProjectArn"
                ]
            ),
            "stage": (
                automation_configuration[
                    "stage"
                ]
            ),
        },
        "data_automation_profile_arn": (
            profile_arn
        ),
        "input_validation": (
            input_validation
        ),
        "output_preflight": (
            output_validation
        ),
        "invoke_response": response,
        "submitted_at": submitted_at,
        "latest_status": "Submitted",
        "status_history": [
            {
                "checked_at": (
                    submitted_at
                ),
                "status": "Submitted",
            }
        ],
        "invocation_submitted": True,
        "bda_invoked": True,
    }

    atomic_write_json(
        args.job_record,
        job_record,
    )

    print(
        f"Invocation ARN: {invocation_arn}"
    )
    print(
        f"Job record:     {args.job_record}"
    )
    print()
    print("Monitoring invocation...")

    final_status = poll_invocation(
        runtime_client=runtime_client,
        job_record=job_record,
        job_record_path=args.job_record,
    )

    output_objects = (
        wait_for_output_objects(
            s3_client=s3_client,
            output_s3_uri=output_s3_uri,
        )
    )

    total_output_bytes = sum(
        int(item["size_bytes"])
        for item in output_objects
    )

    job_record["updated_at"] = (
        utc_now()
    )

    job_record["status"] = (
        "COMPLETED"
    )

    job_record["latest_status"] = (
        final_status.get("status")
    )

    job_record["final_status_response"] = (
        final_status
    )

    job_record["output_inventory"] = {
        "object_count": len(
            output_objects
        ),
        "total_size_bytes": (
            total_output_bytes
        ),
        "objects": output_objects,
    }

    job_record["completed"] = True

    atomic_write_json(
        args.job_record,
        job_record,
    )

    print()
    print("=" * 56)
    print("BDA PILOT RESULT")
    print("=" * 56)
    print(
        f"Status:          "
        f"{final_status.get('status')}"
    )
    print(
        f"Invocation ARN:  {invocation_arn}"
    )
    print(
        "Duration:        "
        f"{final_status.get('jobDurationInSeconds', 'Not returned')}"
    )
    print(
        f"Output objects:  "
        f"{len(output_objects)}"
    )
    print(
        f"Output bytes:    "
        f"{total_output_bytes:,}"
    )
    print(
        f"Requested output:{output_s3_uri}"
    )
    print(
        "Returned output: "
        f"{final_status.get('outputConfiguration', {}).get('s3Uri')}"
    )
    print(
        f"Job record:      {args.job_record}"
    )
    print("Pilot only:      True")
    print("Other batches:   Not invoked")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (
        ClientError,
        BotoCoreError,
    ) as exc:
        print(
            f"AWS BDA pilot error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"BDA pilot invocation failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
