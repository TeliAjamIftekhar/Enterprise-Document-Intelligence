from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

PROJECT_METADATA_PATH = (
    LOCAL_ROOT / "bda-project.json"
)


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected a JSON object in {path}"
        )

    return value


def calculate_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def build_paths(
    start_page: int,
    end_page: int,
) -> dict[str, Any]:
    range_name = (
        f"pages-{start_page:04d}-{end_page:04d}"
    )

    sample_name = (
        f"kaveri-{range_name}.pdf"
    )

    local_sample_path = (
        LOCAL_ROOT
        / "bda-samples"
        / sample_name
    )

    sample_metadata_path = (
        LOCAL_ROOT
        / "bda-samples"
        / f"{sample_name}.json"
    )

    input_key = (
        f"bda-input/grade-9/{BOOK_ID}/"
        f"{BOOK_VERSION}/samples/{sample_name}"
    )

    output_prefix = (
        f"derived-artifacts/grade-9/{BOOK_ID}/"
        f"{BOOK_VERSION}/bda-output/samples/"
        f"{range_name}"
    )

    job_metadata_path = (
        LOCAL_ROOT
        / "bda-jobs"
        / f"{sample_name}.json"
    )

    return {
        "range_name": range_name,
        "sample_name": sample_name,
        "local_sample_path": local_sample_path,
        "sample_metadata_path": (
            sample_metadata_path
        ),
        "input_key": input_key,
        "input_s3_uri": (
            f"s3://{BUCKET}/{input_key}"
        ),
        "output_prefix": output_prefix,
        "output_s3_uri": (
            f"s3://{BUCKET}/{output_prefix}"
        ),
        "job_metadata_path": (
            job_metadata_path
        ),
    }


def save_job_record(
    path: Path,
    record: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
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
    paths: dict[str, Any],
    start_page: int,
    end_page: int,
) -> str:
    local_sample_path: Path = paths[
        "local_sample_path"
    ]

    metadata_path: Path = paths[
        "sample_metadata_path"
    ]

    sample_metadata = load_json(
        metadata_path
    )

    if not local_sample_path.exists():
        raise FileNotFoundError(
            f"Local sample not found: "
            f"{local_sample_path}"
        )

    metadata_start_page = int(
        sample_metadata.get(
            "source_start_page",
            -1,
        )
    )

    metadata_end_page = int(
        sample_metadata.get(
            "source_end_page",
            -1,
        )
    )

    if metadata_start_page != start_page:
        raise RuntimeError(
            "Sample start page does not match. "
            f"Expected={start_page}, "
            f"metadata={metadata_start_page}"
        )

    if metadata_end_page != end_page:
        raise RuntimeError(
            "Sample end page does not match. "
            f"Expected={end_page}, "
            f"metadata={metadata_end_page}"
        )

    expected_page_count = (
        end_page - start_page + 1
    )

    metadata_page_count = int(
        sample_metadata.get(
            "sample_page_count",
            -1,
        )
    )

    if metadata_page_count != expected_page_count:
        raise RuntimeError(
            "Sample page count does not match. "
            f"Expected={expected_page_count}, "
            f"metadata={metadata_page_count}"
        )

    local_size = (
        local_sample_path.stat().st_size
    )

    local_sha256 = calculate_sha256(
        local_sample_path
    )

    metadata_size = int(
        sample_metadata.get(
            "sample_size_bytes",
            -1,
        )
    )

    metadata_sha256 = str(
        sample_metadata.get(
            "sample_sha256",
            "",
        )
    )

    if local_size != metadata_size:
        raise RuntimeError(
            "Local sample size differs from metadata. "
            f"Local={local_size}, "
            f"metadata={metadata_size}"
        )

    if local_sha256 != metadata_sha256:
        raise RuntimeError(
            "Local sample hash differs from metadata. "
            f"Local={local_sha256}, "
            f"metadata={metadata_sha256}"
        )

    head = s3_client.head_object(
        Bucket=BUCKET,
        Key=paths["input_key"],
    )

    s3_size = int(head["ContentLength"])

    s3_sha256 = str(
        head.get(
            "Metadata",
            {},
        ).get(
            "sha256",
            "",
        )
    )

    if local_size != s3_size:
        raise RuntimeError(
            "Local and S3 sample sizes differ. "
            f"Local={local_size}, S3={s3_size}"
        )

    if local_sha256 != s3_sha256:
        raise RuntimeError(
            "Local and S3 sample hashes differ. "
            f"Local={local_sha256}, "
            f"S3={s3_sha256}"
        )

    print("Sample integrity validation: PASSED")
    print(f"Sample pages: {start_page}-{end_page}")
    print(f"Sample size:  {local_size:,} bytes")
    print(f"Sample SHA:   {local_sha256}")

    return local_sha256


def load_project_arn() -> str:
    metadata = load_json(
        PROJECT_METADATA_PATH
    )

    project = metadata.get("project")

    if not isinstance(project, dict):
        raise RuntimeError(
            "Project metadata contains no project object."
        )

    project_arn = project.get(
        "projectArn"
    )

    if not isinstance(
        project_arn,
        str,
    ) or not project_arn:
        raise RuntimeError(
            "Project ARN is missing."
        )

    if project.get("status") != "COMPLETED":
        raise RuntimeError(
            "BDA project is not ready. "
            f"Status={project.get('status')}"
        )

    if (
        project.get("projectStage")
        != PROJECT_STAGE
    ):
        raise RuntimeError(
            "BDA project stage does not match."
        )

    if project.get("projectType") != "ASYNC":
        raise RuntimeError(
            "BDA project is not ASYNC."
        )

    return project_arn


def build_client_token(
    paths: dict[str, Any],
    project_arn: str,
    sample_sha256: str,
) -> str:
    canonical = json.dumps(
        {
            "input": paths["input_s3_uri"],
            "output": paths["output_s3_uri"],
            "project_arn": project_arn,
            "project_stage": PROJECT_STAGE,
            "profile_arn": (
                DATA_AUTOMATION_PROFILE_ARN
            ),
            "sample_sha256": sample_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    return f"bda-{digest}"


def remove_response_metadata(
    response: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in response.items()
        if key != "ResponseMetadata"
    }


def list_output_objects(
    s3_client: Any,
    output_prefix: str,
) -> list[dict[str, Any]]:
    paginator = s3_client.get_paginator(
        "list_objects_v2"
    )

    objects: list[dict[str, Any]] = []

    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=output_prefix,
    ):
        objects.extend(
            page.get("Contents", [])
        )

    return objects


def print_output_objects(
    objects: list[dict[str, Any]],
) -> None:
    print(f"Output objects:  {len(objects)}")

    for item in sorted(
        objects,
        key=lambda value: value["Key"],
    ):
        print(
            f"- s3://{BUCKET}/{item['Key']} "
            f"({item['Size']:,} bytes)"
        )


def poll_job(
    runtime_client: Any,
    s3_client: Any,
    paths: dict[str, Any],
    job_record: dict[str, Any],
    poll_interval: int,
    max_attempts: int,
) -> int:
    running_statuses = {
        "Created",
        "InProgress",
    }

    failure_statuses = {
        "ClientError",
        "ServiceError",
    }

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        response = (
            runtime_client
            .get_data_automation_status(
                invocationArn=(
                    job_record["invocation_arn"]
                )
            )
        )

        status = response.get("status")

        print(
            f"Status check {attempt}: {status}"
        )

        job_record["latest_status"] = status
        job_record["status_response"] = (
            remove_response_metadata(response)
        )
        job_record["last_checked_at"] = (
            utc_now()
        )

        save_job_record(
            paths["job_metadata_path"],
            job_record,
        )

        if status == "Success":
            objects = list_output_objects(
                s3_client=s3_client,
                output_prefix=(
                    paths["output_prefix"]
                ),
            )

            print()
            print("============================================")
            print("BDA RANGE EXTRACTION COMPLETED")
            print("============================================")
            print(f"Status:          {status}")
            print(
                "Invocation ARN:  "
                f"{job_record['invocation_arn']}"
            )
            print(
                "Requested output:"
                f"{paths['output_s3_uri']}"
            )
            print(
                "Returned output: "
                f"{response.get('outputConfiguration', {}).get('s3Uri')}"
            )
            print(
                "Duration:        "
                f"{response.get('jobDurationInSeconds')}"
            )

            print_output_objects(objects)
            return 0

        if status in failure_statuses:
            raise RuntimeError(
                "BDA processing failed. "
                f"Status={status}; "
                f"errorType={response.get('errorType')}; "
                f"errorMessage="
                f"{response.get('errorMessage')}"
            )

        if status not in running_statuses:
            raise RuntimeError(
                f"Unexpected BDA status: {status}"
            )

        time.sleep(poll_interval)

    raise TimeoutError(
        "BDA job did not reach a terminal status."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Invoke BDA asynchronously for a prepared "
            "textbook page range."
        )
    )

    parser.add_argument(
        "--start-page",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--end-page",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--invoke",
        action="store_true",
        help=(
            "Submit the paid BDA asynchronous request. "
            "Without this option, only preflight checks run."
        ),
    )

    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=360,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.start_page < 1:
        raise ValueError(
            "start-page must be at least 1."
        )

    if args.end_page < args.start_page:
        raise ValueError(
            "end-page must be greater than or equal "
            "to start-page."
        )

    paths = build_paths(
        start_page=args.start_page,
        end_page=args.end_page,
    )

    print("============================================")
    print("BDA RANGE INVOCATION")
    print("============================================")
    print(f"Region:       {REGION}")
    print(
        f"Pages:        "
        f"{args.start_page}-{args.end_page}"
    )
    print(
        f"Input:        {paths['input_s3_uri']}"
    )
    print(
        f"Output:       {paths['output_s3_uri']}"
    )
    print(f"Invoke:       {args.invoke}")
    print()

    s3_client = boto3.client(
        "s3",
        region_name=REGION,
    )

    runtime_client = boto3.client(
        "bedrock-data-automation-runtime",
        region_name=REGION,
    )

    sample_sha256 = validate_sample(
        s3_client=s3_client,
        paths=paths,
        start_page=args.start_page,
        end_page=args.end_page,
    )

    project_arn = load_project_arn()

    client_token = build_client_token(
        paths=paths,
        project_arn=project_arn,
        sample_sha256=sample_sha256,
    )

    print(f"Project ARN:  {project_arn}")
    print(
        f"Profile ARN:  "
        f"{DATA_AUTOMATION_PROFILE_ARN}"
    )
    print(f"Client token: {client_token}")
    print(
        f"Job metadata: "
        f"{paths['job_metadata_path']}"
    )

    job_metadata_path: Path = paths[
        "job_metadata_path"
    ]

    if job_metadata_path.exists():
        job_record = load_json(
            job_metadata_path
        )

        if (
            job_record.get("client_token")
            != client_token
        ):
            raise RuntimeError(
                "Existing job metadata has a different "
                "client token."
            )

        invocation_arn = job_record.get(
            "invocation_arn"
        )

        if not isinstance(
            invocation_arn,
            str,
        ) or not invocation_arn:
            raise RuntimeError(
                "Existing job metadata contains no "
                "invocation ARN."
            )

        print()
        print("Existing invocation found.")
        print(
            f"Invocation ARN: {invocation_arn}"
        )
        print("Resuming status checks.")

        return poll_job(
            runtime_client=runtime_client,
            s3_client=s3_client,
            paths=paths,
            job_record=job_record,
            poll_interval=args.poll_interval,
            max_attempts=args.max_attempts,
        )

    existing_objects = list_output_objects(
        s3_client=s3_client,
        output_prefix=paths["output_prefix"],
    )

    if existing_objects:
        raise RuntimeError(
            "The output prefix already contains objects, "
            "but no local job metadata was found."
        )

    if not args.invoke:
        print()
        print("============================================")
        print("BDA PREFLIGHT PASSED")
        print("============================================")
        print("No BDA request was submitted.")
        print()
        print("Submit with:")
        print(
            "PYTHONPATH=workers/multimodal-ingestion "
            "python "
            "workers/multimodal-ingestion/scripts/"
            "invoke_bda_range.py "
            f"--start-page {args.start_page} "
            f"--end-page {args.end_page} "
            "--invoke"
        )

        return 0

    print()
    print("Submitting asynchronous BDA invocation...")

    response = (
        runtime_client
        .invoke_data_automation_async(
            clientToken=client_token,
            inputConfiguration={
                "s3Uri": paths[
                    "input_s3_uri"
                ],
            },
            outputConfiguration={
                "s3Uri": paths[
                    "output_s3_uri"
                ],
            },
            dataAutomationConfiguration={
                "dataAutomationProjectArn": (
                    project_arn
                ),
                "stage": PROJECT_STAGE,
            },
            dataAutomationProfileArn=(
                DATA_AUTOMATION_PROFILE_ARN
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
            "BDA returned no invocation ARN."
        )

    job_record = {
        "book_id": BOOK_ID,
        "book_version": BOOK_VERSION,
        "source_start_page": args.start_page,
        "source_end_page": args.end_page,
        "sample_name": paths["sample_name"],
        "sample_sha256": sample_sha256,
        "input_s3_uri": paths["input_s3_uri"],
        "output_s3_uri": paths["output_s3_uri"],
        "project_arn": project_arn,
        "project_stage": PROJECT_STAGE,
        "data_automation_profile_arn": (
            DATA_AUTOMATION_PROFILE_ARN
        ),
        "client_token": client_token,
        "invocation_arn": invocation_arn,
        "latest_status": "Submitted",
        "submitted_at": utc_now(),
    }

    save_job_record(
        job_metadata_path,
        job_record,
    )

    print(
        f"Invocation ARN: {invocation_arn}"
    )
    print(
        f"Job metadata:   {job_metadata_path}"
    )
    print()

    return poll_job(
        runtime_client=runtime_client,
        s3_client=s3_client,
        paths=paths,
        job_record=job_record,
        poll_interval=args.poll_interval,
        max_attempts=args.max_attempts,
    )


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
            f"BDA range invocation failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
