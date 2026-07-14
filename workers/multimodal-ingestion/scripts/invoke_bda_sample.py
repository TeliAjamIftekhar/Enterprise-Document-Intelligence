from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError


REGION = "us-east-1"
ACCOUNT_ID = "334590195171"
BUCKET = "edi-documents-ajam-2026"

BOOK_ID = "grade-9-english-kaveri"
BOOK_VERSION = "v1"

PROJECT_STAGE = "DEVELOPMENT"

DATA_AUTOMATION_PROFILE_ARN = (
    f"arn:aws:bedrock:{REGION}:{ACCOUNT_ID}:"
    "data-automation-profile/us.data-automation-v1"
)

LOCAL_ROOT = (
    Path("data")
    / "multimodal-output"
    / BOOK_ID
    / BOOK_VERSION
)

PROJECT_METADATA_PATH = LOCAL_ROOT / "bda-project.json"

SAMPLE_NAME = "kaveri-pages-0001-0005.pdf"

LOCAL_SAMPLE_PATH = (
    LOCAL_ROOT
    / "bda-samples"
    / SAMPLE_NAME
)

LOCAL_SAMPLE_METADATA_PATH = (
    LOCAL_ROOT
    / "bda-samples"
    / f"{SAMPLE_NAME}.json"
)

INPUT_KEY = (
    f"bda-input/grade-9/{BOOK_ID}/{BOOK_VERSION}/"
    f"samples/{SAMPLE_NAME}"
)

INPUT_S3_URI = f"s3://{BUCKET}/{INPUT_KEY}"

OUTPUT_PREFIX = (
    f"derived-artifacts/grade-9/{BOOK_ID}/{BOOK_VERSION}/"
    "bda-output/samples/pages-0001-0005"
)

OUTPUT_S3_URI = f"s3://{BUCKET}/{OUTPUT_PREFIX}"

JOB_DIRECTORY = LOCAL_ROOT / "bda-jobs"
JOB_METADATA_PATH = (
    JOB_DIRECTORY
    / "kaveri-pages-0001-0005.json"
)


def json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )


def calculate_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return data


def save_job_record(
    record: dict[str, Any],
) -> None:
    JOB_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    JOB_METADATA_PATH.write_text(
        json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )


def validate_sample(
    s3_client: Any,
) -> str:
    """
    Confirm that the local sample, local metadata, and S3 object
    refer to the same immutable PDF bytes.
    """
    sample_metadata = load_json(
        LOCAL_SAMPLE_METADATA_PATH
    )

    if not LOCAL_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Local sample does not exist: {LOCAL_SAMPLE_PATH}"
        )

    local_size = LOCAL_SAMPLE_PATH.stat().st_size
    local_sha256 = calculate_sha256(LOCAL_SAMPLE_PATH)

    metadata_size = int(
        sample_metadata["sample_size_bytes"]
    )
    metadata_sha256 = str(
        sample_metadata["sample_sha256"]
    )

    if local_size != metadata_size:
        raise RuntimeError(
            "Local sample size does not match its metadata. "
            f"Local={local_size}, metadata={metadata_size}"
        )

    if local_sha256 != metadata_sha256:
        raise RuntimeError(
            "Local sample hash does not match its metadata. "
            f"Local={local_sha256}, "
            f"metadata={metadata_sha256}"
        )

    s3_head = s3_client.head_object(
        Bucket=BUCKET,
        Key=INPUT_KEY,
    )

    s3_size = int(s3_head["ContentLength"])
    s3_sha256 = str(
        s3_head.get("Metadata", {}).get("sha256", "")
    )

    if local_size != s3_size:
        raise RuntimeError(
            "Local and S3 sample sizes differ. "
            f"Local={local_size}, S3={s3_size}"
        )

    if local_sha256 != s3_sha256:
        raise RuntimeError(
            "Local and S3 sample hashes differ. "
            f"Local={local_sha256}, S3={s3_sha256}"
        )

    print("Sample integrity validation: PASSED")
    print(f"Sample size: {local_size:,} bytes")
    print(f"Sample SHA:  {local_sha256}")

    return local_sha256


def load_project_arn() -> str:
    project_metadata = load_json(
        PROJECT_METADATA_PATH
    )

    project = project_metadata.get("project")

    if not isinstance(project, dict):
        raise RuntimeError(
            "BDA project metadata contains no project object."
        )

    project_arn = project.get("projectArn")
    project_status = project.get("status")
    project_stage = project.get("projectStage")
    project_type = project.get("projectType")

    if not isinstance(project_arn, str) or not project_arn:
        raise RuntimeError(
            "BDA project ARN is missing."
        )

    if project_status != "COMPLETED":
        raise RuntimeError(
            f"BDA project is not ready: {project_status}"
        )

    if project_stage != PROJECT_STAGE:
        raise RuntimeError(
            "BDA project stage does not match. "
            f"Expected={PROJECT_STAGE}, "
            f"actual={project_stage}"
        )

    if project_type != "ASYNC":
        raise RuntimeError(
            f"Expected ASYNC project, received {project_type}"
        )

    return project_arn


def build_client_token(
    sample_sha256: str,
    project_arn: str,
) -> str:
    """
    Build a deterministic idempotency token.

    Re-running the same immutable sample with the same project
    and output location should not create a duplicate invocation.
    """
    canonical_request = json.dumps(
        {
            "input": INPUT_S3_URI,
            "output": OUTPUT_S3_URI,
            "project_arn": project_arn,
            "project_stage": PROJECT_STAGE,
            "profile_arn": DATA_AUTOMATION_PROFILE_ARN,
            "sample_sha256": sample_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    request_hash = hashlib.sha256(
        canonical_request.encode("utf-8")
    ).hexdigest()

    return f"bda-{request_hash}"


def invoke_job(
    runtime_client: Any,
    project_arn: str,
    client_token: str,
) -> str:
    response = runtime_client.invoke_data_automation_async(
        clientToken=client_token,
        inputConfiguration={
            "s3Uri": INPUT_S3_URI,
        },
        outputConfiguration={
            "s3Uri": OUTPUT_S3_URI,
        },
        dataAutomationConfiguration={
            "dataAutomationProjectArn": project_arn,
            "stage": PROJECT_STAGE,
        },
        dataAutomationProfileArn=(
            DATA_AUTOMATION_PROFILE_ARN
        ),
    )

    invocation_arn = response.get("invocationArn")

    if not isinstance(invocation_arn, str):
        raise RuntimeError(
            "InvokeDataAutomationAsync returned no invocation ARN."
        )

    return invocation_arn


def wait_for_job(
    runtime_client: Any,
    job_record: dict[str, Any],
) -> dict[str, Any]:
    maximum_attempts = 360
    polling_interval_seconds = 5

    valid_running_statuses = {
        "Created",
        "InProgress",
    }

    failure_statuses = {
        "ServiceError",
        "ClientError",
    }

    for attempt in range(1, maximum_attempts + 1):
        status_response = (
            runtime_client.get_data_automation_status(
                invocationArn=job_record["invocation_arn"]
            )
        )

        status_name = status_response.get("status")

        print(
            f"Status check {attempt}: {status_name}"
        )

        job_record["latest_status"] = status_name
        job_record["status_response"] = {
            key: value
            for key, value in status_response.items()
            if key != "ResponseMetadata"
        }
        job_record["last_checked_at"] = (
            datetime.utcnow().isoformat() + "Z"
        )

        save_job_record(job_record)

        if status_name == "Success":
            return status_response

        if status_name in failure_statuses:
            error_type = status_response.get(
                "errorType",
                "Unknown",
            )
            error_message = status_response.get(
                "errorMessage",
                "No error message returned.",
            )

            raise RuntimeError(
                "BDA processing failed. "
                f"Status={status_name}; "
                f"errorType={error_type}; "
                f"errorMessage={error_message}"
            )

        if status_name not in valid_running_statuses:
            raise RuntimeError(
                f"Unexpected BDA job status: {status_name}"
            )

        time.sleep(polling_interval_seconds)

    raise TimeoutError(
        "BDA invocation did not reach a terminal state "
        "within the configured polling attempts."
    )


def list_output_objects(
    s3_client: Any,
) -> list[dict[str, Any]]:
    paginator = s3_client.get_paginator(
        "list_objects_v2"
    )

    objects: list[dict[str, Any]] = []

    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=OUTPUT_PREFIX,
    ):
        objects.extend(page.get("Contents", []))

    return objects


def main() -> int:
    print("============================================")
    print("INVOKING BDA SAMPLE EXTRACTION")
    print("============================================")
    print(f"Region:       {REGION}")
    print(f"Input:        {INPUT_S3_URI}")
    print(f"Output:       {OUTPUT_S3_URI}")
    print(f"Stage:        {PROJECT_STAGE}")
    print()

    s3_client = boto3.client(
        "s3",
        region_name=REGION,
    )

    runtime_client = boto3.client(
        "bedrock-data-automation-runtime",
        region_name=REGION,
    )

    sample_sha256 = validate_sample(s3_client)
    project_arn = load_project_arn()

    print(f"Project ARN:  {project_arn}")
    print(f"Profile ARN:  {DATA_AUTOMATION_PROFILE_ARN}")

    client_token = build_client_token(
        sample_sha256=sample_sha256,
        project_arn=project_arn,
    )

    print(f"Client token: {client_token}")
    print()
    print("Submitting asynchronous BDA invocation...")

    invocation_arn = invoke_job(
        runtime_client=runtime_client,
        project_arn=project_arn,
        client_token=client_token,
    )

    job_record: dict[str, Any] = {
        "book_id": BOOK_ID,
        "book_version": BOOK_VERSION,
        "sample_name": SAMPLE_NAME,
        "sample_sha256": sample_sha256,
        "input_s3_uri": INPUT_S3_URI,
        "output_s3_uri": OUTPUT_S3_URI,
        "project_arn": project_arn,
        "project_stage": PROJECT_STAGE,
        "data_automation_profile_arn": (
            DATA_AUTOMATION_PROFILE_ARN
        ),
        "client_token": client_token,
        "invocation_arn": invocation_arn,
        "latest_status": "Submitted",
        "submitted_at": (
            datetime.utcnow().isoformat() + "Z"
        ),
    }

    save_job_record(job_record)

    print(f"Invocation ARN: {invocation_arn}")
    print(f"Job metadata:   {JOB_METADATA_PATH}")
    print()

    final_status = wait_for_job(
        runtime_client=runtime_client,
        job_record=job_record,
    )

    returned_output_uri = (
        final_status
        .get("outputConfiguration", {})
        .get("s3Uri")
    )

    objects = list_output_objects(s3_client)

    print()
    print("============================================")
    print("BDA SAMPLE EXTRACTION COMPLETED")
    print("============================================")
    print(f"Status:          {final_status['status']}")
    print(f"Invocation ARN:  {invocation_arn}")
    print(f"Requested output:{OUTPUT_S3_URI}")
    print(f"Returned output: {returned_output_uri}")
    print(
        "Duration:        "
        f"{final_status.get('jobDurationInSeconds', 'Not returned')}"
    )
    print(f"Output objects:  {len(objects)}")
    print()

    if objects:
        print("Generated S3 objects:")

        for item in sorted(
            objects,
            key=lambda value: value["Key"],
        ):
            print(
                f"- s3://{BUCKET}/{item['Key']} "
                f"({item['Size']:,} bytes)"
            )
    else:
        print(
            "No objects were found under the requested "
            "output prefix."
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (ClientError, BotoCoreError) as exc:
        print(
            f"AWS error during BDA invocation: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"BDA invocation failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
