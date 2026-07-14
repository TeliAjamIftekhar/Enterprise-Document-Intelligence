from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import fitz
from botocore.exceptions import BotoCoreError, ClientError


REGION = "us-east-1"
ACCOUNT_ID = "334590195171"
BUCKET = "edi-documents-ajam-2026"

BOOK_ID = "grade-9-english-kaveri"
BOOK_VERSION = "v1"
GRADE = "grade-9"

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
    LOCAL_ROOT
    / "bda-project.json"
)

DEFAULT_MANIFEST_PATH = (
    LOCAL_ROOT
    / "full-book"
    / "full-book-batch-manifest.json"
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


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
            allow_nan=False,
            default=str,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        path,
    )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(
            chunk_size
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_project_arn() -> tuple[
    str,
    dict[str, Any],
]:
    metadata = load_json_object(
        PROJECT_METADATA_PATH
    )

    project = metadata.get(
        "project"
    )

    if not isinstance(project, dict):
        raise RuntimeError(
            "BDA project metadata contains "
            "no project object."
        )

    project_arn = project.get(
        "projectArn"
    )

    if not isinstance(
        project_arn,
        str,
    ) or not project_arn:
        raise RuntimeError(
            "BDA project ARN is missing."
        )

    if project.get("status") != "COMPLETED":
        raise RuntimeError(
            "BDA project is not ready: "
            f"{project.get('status')}"
        )

    if (
        project.get("projectStage")
        != PROJECT_STAGE
    ):
        raise RuntimeError(
            "BDA project stage mismatch: "
            f"{project.get('projectStage')}"
        )

    if project.get("projectType") != "ASYNC":
        raise RuntimeError(
            "BDA project is not ASYNC: "
            f"{project.get('projectType')}"
        )

    return project_arn, project


def select_batch(
    manifest: dict[str, Any],
    batch_id: str,
) -> dict[str, Any]:
    if manifest.get("status") != "UPLOADED":
        raise RuntimeError(
            "Full-book manifest status is not "
            f"UPLOADED: {manifest.get('status')}"
        )

    if manifest.get("uploaded") is not True:
        raise RuntimeError(
            "Full-book manifest is not marked "
            "as uploaded."
        )

    batches = manifest.get(
        "batches"
    )

    if not isinstance(batches, list):
        raise RuntimeError(
            "Manifest contains no batches list."
        )

    matches = [
        batch
        for batch in batches
        if batch.get("batch_id")
        == batch_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {batch_id} entry, "
            f"found {len(matches)}."
        )

    batch = matches[0]

    if batch.get("uploaded") is not True:
        raise RuntimeError(
            f"{batch_id} is not marked uploaded."
        )

    if batch.get("s3_verified") is not True:
        raise RuntimeError(
            f"{batch_id} is not S3 verified."
        )

    if batch.get("bda_invoked") is True:
        raise RuntimeError(
            f"{batch_id} is already marked "
            "as BDA invoked."
        )

    return batch


def validate_local_batch(
    batch: dict[str, Any],
) -> dict[str, Any]:
    local_path = Path(
        str(batch["local_path"])
    )

    if not local_path.is_file():
        raise FileNotFoundError(
            f"Pilot PDF missing: {local_path}"
        )

    actual_size = (
        local_path.stat().st_size
    )

    expected_size = int(
        batch["size_bytes"]
    )

    if actual_size != expected_size:
        raise RuntimeError(
            "Pilot local size mismatch: "
            f"expected={expected_size}, "
            f"actual={actual_size}"
        )

    actual_sha256 = sha256_file(
        local_path
    )

    expected_sha256 = str(
        batch["sha256"]
    )

    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Pilot local SHA256 mismatch."
        )

    with fitz.open(
        str(local_path)
    ) as document:
        page_count = len(document)

    expected_page_count = int(
        batch["page_count"]
    )

    if page_count != expected_page_count:
        raise RuntimeError(
            "Pilot local page count mismatch: "
            f"expected={expected_page_count}, "
            f"actual={page_count}"
        )

    return {
        "local_path": str(local_path),
        "size_bytes": actual_size,
        "sha256": actual_sha256,
        "page_count": page_count,
        "verified": True,
    }


def validate_remote_manifest(
    s3_client: Any,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    upload_data = manifest.get(
        "s3_upload",
        {},
    )

    manifest_s3_key = upload_data.get(
        "manifest_s3_key"
    )

    if not isinstance(
        manifest_s3_key,
        str,
    ) or not manifest_s3_key:
        raise RuntimeError(
            "Manifest S3 key is missing."
        )

    local_sha256 = sha256_file(
        manifest_path
    )

    head = s3_client.head_object(
        Bucket=BUCKET,
        Key=manifest_s3_key,
    )

    remote_sha256 = (
        head.get(
            "Metadata",
            {},
        ).get("sha256")
    )

    if remote_sha256 != local_sha256:
        raise RuntimeError(
            "Local and S3 manifest SHA256 "
            "metadata do not match."
        )

    if (
        head.get("ContentType")
        != "application/json"
    ):
        raise RuntimeError(
            "S3 manifest content type mismatch."
        )

    if (
        head.get(
            "ServerSideEncryption"
        )
        != "AES256"
    ):
        raise RuntimeError(
            "S3 manifest encryption mismatch."
        )

    return {
        "s3_key": manifest_s3_key,
        "s3_uri": (
            f"s3://{BUCKET}/"
            f"{manifest_s3_key}"
        ),
        "sha256": local_sha256,
        "size_bytes": int(
            head["ContentLength"]
        ),
        "version_id": head.get(
            "VersionId"
        ),
        "verified": True,
    }


def validate_remote_batch(
    s3_client: Any,
    batch: dict[str, Any],
    local_validation: dict[str, Any],
) -> dict[str, Any]:
    s3_key = str(
        batch["s3_key"]
    )

    expected_s3_uri = (
        f"s3://{BUCKET}/{s3_key}"
    )

    if batch.get("s3_uri") != expected_s3_uri:
        raise RuntimeError(
            "Pilot S3 URI does not match "
            "bucket and key."
        )

    head = s3_client.head_object(
        Bucket=BUCKET,
        Key=s3_key,
    )

    remote_size = int(
        head["ContentLength"]
    )

    remote_sha256 = (
        head.get(
            "Metadata",
            {},
        ).get("sha256")
    )

    if (
        remote_size
        != local_validation["size_bytes"]
    ):
        raise RuntimeError(
            "Pilot local/S3 size mismatch."
        )

    if (
        remote_sha256
        != local_validation["sha256"]
    ):
        raise RuntimeError(
            "Pilot local/S3 SHA256 mismatch."
        )

    if (
        head.get("ContentType")
        != "application/pdf"
    ):
        raise RuntimeError(
            "Pilot S3 content type mismatch."
        )

    if (
        head.get(
            "ServerSideEncryption"
        )
        != "AES256"
    ):
        raise RuntimeError(
            "Pilot S3 encryption mismatch."
        )

    metadata = head.get(
        "Metadata",
        {},
    )

    if (
        metadata.get("batch-id")
        != batch["batch_id"]
    ):
        raise RuntimeError(
            "Pilot S3 batch ID metadata "
            "mismatch."
        )

    return {
        "s3_key": s3_key,
        "s3_uri": expected_s3_uri,
        "size_bytes": remote_size,
        "sha256": remote_sha256,
        "etag": str(
            head.get("ETag", "")
        ).strip('"'),
        "version_id": head.get(
            "VersionId"
        ),
        "verified": True,
    }


def build_output_location(
    batch_id: str,
) -> tuple[str, str]:
    output_prefix = (
        f"derived-artifacts/{GRADE}/"
        f"{BOOK_ID}/{BOOK_VERSION}/"
        "bda-output/full-book/batches/"
        f"{batch_id}"
    )

    output_s3_uri = (
        f"s3://{BUCKET}/{output_prefix}"
    )

    return output_prefix, output_s3_uri


def verify_output_is_empty(
    s3_client: Any,
    output_prefix: str,
) -> dict[str, Any]:
    normalized_prefix = (
        output_prefix.rstrip("/")
        + "/"
    )

    response = s3_client.list_objects_v2(
        Bucket=BUCKET,
        Prefix=normalized_prefix,
        MaxKeys=20,
    )

    objects = response.get(
        "Contents",
        [],
    )

    if objects:
        object_keys = [
            item.get("Key")
            for item in objects
        ]

        raise RuntimeError(
            "Pilot output prefix is not empty. "
            "No invocation is safe until these "
            "objects are reviewed: "
            + json.dumps(object_keys)
        )

    return {
        "output_prefix": output_prefix,
        "output_s3_uri": (
            f"s3://{BUCKET}/"
            f"{output_prefix}"
        ),
        "existing_object_count": 0,
        "empty": True,
    }


def verify_no_existing_job(
    batch_id: str,
) -> dict[str, Any]:
    job_path = (
        LOCAL_ROOT
        / "full-book"
        / "bda-jobs"
        / f"{batch_id}.json"
    )

    if job_path.exists():
        existing = load_json_object(
            job_path
        )

        raise RuntimeError(
            "Existing pilot job record found. "
            "Review before creating another job: "
            f"{job_path}; "
            f"status={existing.get('latest_status')}; "
            f"invocation={existing.get('invocation_arn')}"
        )

    return {
        "job_record_path": str(
            job_path
        ),
        "exists": False,
    }


def build_client_token(
    input_s3_uri: str,
    output_s3_uri: str,
    project_arn: str,
    batch_sha256: str,
) -> tuple[str, str]:
    canonical_request = json.dumps(
        {
            "input_s3_uri": input_s3_uri,
            "output_s3_uri": output_s3_uri,
            "project_arn": project_arn,
            "project_stage": PROJECT_STAGE,
            "profile_arn": (
                DATA_AUTOMATION_PROFILE_ARN
            ),
            "batch_sha256": batch_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    request_sha256 = hashlib.sha256(
        canonical_request.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        f"bda-{request_sha256}",
        request_sha256,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-invocation preflight for "
            "one full-book BDA pilot batch."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )

    parser.add_argument(
        "--batch-id",
        default="batch-0001",
    )

    parser.add_argument(
        "--report",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest = load_json_object(
        args.manifest
    )

    batch = select_batch(
        manifest=manifest,
        batch_id=args.batch_id,
    )

    project_arn, project = (
        load_project_arn()
    )

    s3_client = boto3.client(
        "s3",
        region_name=REGION,
    )

    runtime_client = boto3.client(
        "bedrock-data-automation-runtime",
        region_name=REGION,
    )

    operations = set(
        runtime_client.meta.service_model
        .operation_names
    )

    required_operations = {
        "InvokeDataAutomationAsync",
        "GetDataAutomationStatus",
    }

    missing_operations = (
        required_operations
        - operations
    )

    if missing_operations:
        raise RuntimeError(
            "Required BDA SDK operations missing: "
            + json.dumps(
                sorted(missing_operations)
            )
        )

    local_validation = (
        validate_local_batch(batch)
    )

    remote_manifest = (
        validate_remote_manifest(
            s3_client=s3_client,
            manifest_path=args.manifest,
            manifest=manifest,
        )
    )

    remote_batch = (
        validate_remote_batch(
            s3_client=s3_client,
            batch=batch,
            local_validation=(
                local_validation
            ),
        )
    )

    output_prefix, output_s3_uri = (
        build_output_location(
            args.batch_id
        )
    )

    output_validation = (
        verify_output_is_empty(
            s3_client=s3_client,
            output_prefix=output_prefix,
        )
    )

    job_validation = (
        verify_no_existing_job(
            args.batch_id
        )
    )

    client_token, request_sha256 = (
        build_client_token(
            input_s3_uri=(
                remote_batch["s3_uri"]
            ),
            output_s3_uri=(
                output_s3_uri
            ),
            project_arn=project_arn,
            batch_sha256=(
                local_validation["sha256"]
            ),
        )
    )

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": "PREFLIGHT_PASSED",
        "region": REGION,
        "bucket": BUCKET,
        "book_id": BOOK_ID,
        "book_version": BOOK_VERSION,
        "batch": {
            "batch_id": batch[
                "batch_id"
            ],
            "source_page_start": batch[
                "source_page_start"
            ],
            "source_page_end": batch[
                "source_page_end"
            ],
            "page_count": batch[
                "page_count"
            ],
            "local": local_validation,
            "remote": remote_batch,
        },
        "manifest": remote_manifest,
        "project": {
            "project_arn": project_arn,
            "project_status": project.get(
                "status"
            ),
            "project_stage": project.get(
                "projectStage"
            ),
            "project_type": project.get(
                "projectType"
            ),
            "profile_arn": (
                DATA_AUTOMATION_PROFILE_ARN
            ),
        },
        "output": output_validation,
        "job_record": job_validation,
        "invocation_preview": {
            "client_token": client_token,
            "request_sha256": (
                request_sha256
            ),
            "input_configuration": {
                "s3Uri": remote_batch[
                    "s3_uri"
                ],
            },
            "output_configuration": {
                "s3Uri": output_s3_uri,
            },
            "data_automation_configuration": {
                "dataAutomationProjectArn": (
                    project_arn
                ),
                "stage": PROJECT_STAGE,
            },
            "data_automation_profile_arn": (
                DATA_AUTOMATION_PROFILE_ARN
            ),
        },
        "checks": {
            "local_batch_verified": True,
            "remote_batch_verified": True,
            "remote_manifest_verified": True,
            "project_ready": True,
            "sdk_operations_ready": True,
            "output_prefix_empty": True,
            "existing_job_absent": True,
        },
        "invocation_submitted": False,
        "bda_invoked": False,
    }

    atomic_write_json(
        args.report,
        report,
    )

    print("=" * 52)
    print("FULL BOOK BDA PILOT PREFLIGHT")
    print("=" * 52)
    print(f"Batch:        {args.batch_id}")
    print(
        "Source pages: "
        f"{batch['source_page_start']}-"
        f"{batch['source_page_end']}"
    )
    print(
        f"Input:        "
        f"{remote_batch['s3_uri']}"
    )
    print(
        f"Output:       {output_s3_uri}"
    )
    print(f"Project:      {project_arn}")
    print(
        f"Project stage:{PROJECT_STAGE}"
    )
    print(
        f"Client token: {client_token}"
    )
    print()

    print("=" * 52)
    print("BDA PILOT PREFLIGHT RESULT")
    print("=" * 52)
    print("Status:              PREFLIGHT_PASSED")
    print("Local batch:         VERIFIED")
    print("S3 batch:            VERIFIED")
    print("S3 manifest:         VERIFIED")
    print("BDA project:         READY")
    print("SDK operations:      READY")
    print("Output prefix:       EMPTY")
    print("Existing job record: NONE")
    print("Invocation submitted:False")
    print("BDA invoked:         False")
    print(f"Report:              {args.report}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (
        ClientError,
        BotoCoreError,
    ) as exc:
        print(
            f"AWS pilot preflight error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"BDA pilot preflight failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
