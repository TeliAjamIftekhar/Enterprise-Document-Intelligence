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
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
)


REGION = "us-east-1"
BUCKET = "edi-documents-ajam-2026"

BOOK_ID = "grade-9-english-kaveri"
BOOK_VERSION = "v1"
GRADE = "grade-9"


ARTIFACT_SPECS = (
    (
        "content-units.jsonl",
        "normalized/content-units.jsonl",
        "application/x-ndjson",
        "normalized_content_units",
    ),
    (
        "figures.jsonl",
        "normalized/figures.jsonl",
        "application/x-ndjson",
        "normalized_figures",
    ),
    (
        "tables.jsonl",
        "normalized/tables.jsonl",
        "application/x-ndjson",
        "normalized_tables",
    ),
    (
        "normalization-report.json",
        "normalized/normalization-report.json",
        "application/json",
        "normalization_report",
    ),
    (
        "embedding-ready/embedding-records.jsonl",
        "embedding-ready/embedding-records.jsonl",
        "application/x-ndjson",
        "embedding_records",
    ),
    (
        "embedding-ready/skipped-records.jsonl",
        "embedding-ready/skipped-records.jsonl",
        "application/x-ndjson",
        "skipped_records",
    ),
    (
        "embedding-ready/embedding-preparation-report.json",
        "embedding-ready/embedding-preparation-report.json",
        "application/json",
        "embedding_preparation_report",
    ),
    (
        "embedding-ready/titan-text-v2/embeddings.jsonl",
        "titan-text-v2/embeddings.jsonl",
        "application/x-ndjson",
        "titan_embeddings",
    ),
    (
        "embedding-ready/titan-text-v2/embedding-manifest.json",
        "titan-text-v2/embedding-manifest.json",
        "application/json",
        "embedding_manifest",
    ),
    (
        "embedding-ready/titan-text-v2/retrieval-evaluation-report.json",
        "titan-text-v2/retrieval-evaluation-report.json",
        "application/json",
        "retrieval_evaluation_report",
    ),
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(
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


def count_jsonl(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
        )

    count = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object in {path} "
                    f"at line {line_number}."
                )

            count += 1

    return count


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
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        path,
    )


def validate_bundle(
    normalized_dir: Path,
) -> dict[str, Any]:
    normalization_report_path = (
        normalized_dir
        / "normalization-report.json"
    )

    embedding_ready_dir = (
        normalized_dir
        / "embedding-ready"
    )

    embedding_preparation_report_path = (
        embedding_ready_dir
        / "embedding-preparation-report.json"
    )

    titan_dir = (
        embedding_ready_dir
        / "titan-text-v2"
    )

    embedding_manifest_path = (
        titan_dir
        / "embedding-manifest.json"
    )

    evaluation_report_path = (
        titan_dir
        / "retrieval-evaluation-report.json"
    )

    normalization_report = load_json(
        normalization_report_path
    )

    embedding_preparation_report = load_json(
        embedding_preparation_report_path
    )

    embedding_manifest = load_json(
        embedding_manifest_path
    )

    evaluation_report = load_json(
        evaluation_report_path
    )

    content_unit_count = count_jsonl(
        normalized_dir
        / "content-units.jsonl"
    )

    figure_count = count_jsonl(
        normalized_dir
        / "figures.jsonl"
    )

    table_count = count_jsonl(
        normalized_dir
        / "tables.jsonl"
    )

    embedding_record_count = count_jsonl(
        embedding_ready_dir
        / "embedding-records.jsonl"
    )

    skipped_record_count = count_jsonl(
        embedding_ready_dir
        / "skipped-records.jsonl"
    )

    titan_embedding_count = count_jsonl(
        titan_dir
        / "embeddings.jsonl"
    )

    errors: list[str] = []

    expected_normalized_count = (
        normalization_report.get(
            "normalized_content_unit_count"
        )
    )

    if content_unit_count != expected_normalized_count:
        errors.append(
            "content-units.jsonl count differs from "
            "normalization report."
        )

    if figure_count != normalization_report.get(
        "normalized_figure_count"
    ):
        errors.append(
            "figures.jsonl count differs from "
            "normalization report."
        )

    if table_count != normalization_report.get(
        "normalized_table_count"
    ):
        errors.append(
            "tables.jsonl count differs from "
            "normalization report."
        )

    if normalization_report.get(
        "missing_page_reference_count"
    ) != 0:
        errors.append(
            "Normalization report contains missing "
            "page references."
        )

    if normalization_report.get(
        "missing_asset_count"
    ) != 0:
        errors.append(
            "Normalization report contains missing assets."
        )

    if embedding_record_count != (
        embedding_preparation_report.get(
            "embedding_record_count"
        )
    ):
        errors.append(
            "Embedding-record count differs from "
            "preparation report."
        )

    if skipped_record_count != (
        embedding_preparation_report.get(
            "skipped_unit_count"
        )
    ):
        errors.append(
            "Skipped-record count differs from "
            "preparation report."
        )

    if embedding_manifest.get(
        "status"
    ) != "COMPLETED":
        errors.append(
            "Embedding manifest is not COMPLETED."
        )

    if titan_embedding_count != (
        embedding_manifest.get(
            "completed_record_count"
        )
    ):
        errors.append(
            "Titan embedding count differs from "
            "embedding manifest."
        )

    if titan_embedding_count != embedding_record_count:
        errors.append(
            "Not every embedding-ready record has a "
            "Titan embedding."
        )

    configuration = embedding_manifest.get(
        "configuration",
        {},
    )

    if configuration.get("dimensions") != 1024:
        errors.append(
            "Embedding dimension is not 1024."
        )

    if configuration.get("normalize") is not True:
        errors.append(
            "Embeddings are not marked normalized."
        )

    if evaluation_report.get(
        "all_tests_passed"
    ) is not True:
        errors.append(
            "Retrieval evaluation did not pass."
        )

    if evaluation_report.get(
        "embedding_record_count"
    ) != titan_embedding_count:
        errors.append(
            "Evaluation record count differs from "
            "Titan embedding count."
        )

    if errors:
        raise RuntimeError(
            "Retrieval bundle validation failed:\n- "
            + "\n- ".join(errors)
        )

    return {
        "content_unit_count": content_unit_count,
        "figure_count": figure_count,
        "table_count": table_count,
        "embedding_record_count": (
            embedding_record_count
        ),
        "skipped_record_count": (
            skipped_record_count
        ),
        "titan_embedding_count": (
            titan_embedding_count
        ),
        "embedding_dimensions": (
            configuration["dimensions"]
        ),
        "embedding_model_id": (
            configuration["model_id"]
        ),
        "retrieval_test_count": (
            evaluation_report["test_count"]
        ),
        "retrieval_passed_test_count": (
            evaluation_report[
                "passed_test_count"
            ]
        ),
        "top_1_accuracy": (
            evaluation_report[
                "top_1_accuracy"
            ]
        ),
        "mean_reciprocal_rank": (
            evaluation_report[
                "mean_reciprocal_rank"
            ]
        ),
    }


def object_status(
    s3_client: Any,
    key: str,
    expected_sha256: str,
    expected_size: int,
) -> str:
    try:
        response = s3_client.head_object(
            Bucket=BUCKET,
            Key=key,
        )

    except ClientError as exc:
        error_code = str(
            exc.response.get(
                "Error",
                {},
            ).get(
                "Code",
                "",
            )
        )

        if error_code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return "missing"

        raise

    actual_sha256 = str(
        response.get(
            "Metadata",
            {},
        ).get(
            "sha256",
            "",
        )
    )

    actual_size = int(
        response.get(
            "ContentLength",
            -1,
        )
    )

    if (
        actual_sha256 == expected_sha256
        and actual_size == expected_size
    ):
        return "matching"

    return "different"


def upload_artifact(
    s3_client: Any,
    local_path: Path,
    key: str,
    content_type: str,
    artifact_kind: str,
    sha256: str,
    range_name: str,
    invocation_id: str,
) -> None:
    s3_client.upload_file(
        Filename=str(local_path),
        Bucket=BUCKET,
        Key=key,
        ExtraArgs={
            "ContentType": content_type,
            "ServerSideEncryption": "AES256",
            "Metadata": {
                "sha256": sha256,
                "book-id": BOOK_ID,
                "book-version": BOOK_VERSION,
                "range-name": range_name,
                "invocation-id": invocation_id,
                "artifact-kind": artifact_kind,
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and publish retrieval artifacts "
            "to a versioned S3 prefix."
        )
    )

    parser.add_argument(
        "normalized_dir",
        type=Path,
    )

    parser.add_argument(
        "--range-name",
        required=True,
    )

    parser.add_argument(
        "--invocation-id",
        required=True,
    )

    parser.add_argument(
        "--upload",
        action="store_true",
        help=(
            "Upload artifacts. Without this option, "
            "only validation and planning are performed."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacement when an existing object "
            "has a different checksum."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.normalized_dir.exists():
        raise FileNotFoundError(
            f"Normalized directory not found: "
            f"{args.normalized_dir}"
        )

    validation = validate_bundle(
        args.normalized_dir
    )

    s3_prefix = (
        f"derived-artifacts/{GRADE}/{BOOK_ID}/"
        f"{BOOK_VERSION}/retrieval/samples/"
        f"{args.range_name}/{args.invocation_id}"
    )

    print("============================================")
    print("RETRIEVAL ARTIFACT PUBLICATION")
    print("============================================")
    print(f"Book:          {BOOK_ID}")
    print(f"Version:       {BOOK_VERSION}")
    print(f"Range:         {args.range_name}")
    print(
        f"Invocation:    {args.invocation_id}"
    )
    print(
        f"Destination:   s3://{BUCKET}/{s3_prefix}"
    )
    print(f"Upload:        {args.upload}")
    print()

    print("Validated bundle:")
    print(
        "Content units: "
        f"{validation['content_unit_count']}"
    )
    print(
        "Figures:       "
        f"{validation['figure_count']}"
    )
    print(
        "Tables:        "
        f"{validation['table_count']}"
    )
    print(
        "Embeddings:    "
        f"{validation['titan_embedding_count']}"
    )
    print(
        "Dimensions:    "
        f"{validation['embedding_dimensions']}"
    )
    print(
        "Retrieval:     "
        f"{validation['retrieval_passed_test_count']}/"
        f"{validation['retrieval_test_count']} passed"
    )
    print()

    s3_client = boto3.client(
        "s3",
        region_name=REGION,
    )

    artifacts: list[dict[str, Any]] = []

    uploaded_count = 0
    matching_count = 0
    planned_count = 0

    for (
        local_relative_path,
        s3_relative_path,
        content_type,
        artifact_kind,
    ) in ARTIFACT_SPECS:
        local_path = (
            args.normalized_dir
            / local_relative_path
        )

        if not local_path.exists():
            raise FileNotFoundError(
                f"Required artifact missing: {local_path}"
            )

        size_bytes = local_path.stat().st_size
        file_sha256 = sha256_file(local_path)

        key = (
            f"{s3_prefix}/{s3_relative_path}"
        )

        status = object_status(
            s3_client=s3_client,
            key=key,
            expected_sha256=file_sha256,
            expected_size=size_bytes,
        )

        if status == "different" and not args.overwrite:
            raise RuntimeError(
                "An existing S3 object has a different "
                "checksum. Refusing to overwrite:\n"
                f"s3://{BUCKET}/{key}"
            )

        action = "skip"

        if status == "matching":
            matching_count += 1
            action = "matching"

        elif args.upload:
            upload_artifact(
                s3_client=s3_client,
                local_path=local_path,
                key=key,
                content_type=content_type,
                artifact_kind=artifact_kind,
                sha256=file_sha256,
                range_name=args.range_name,
                invocation_id=args.invocation_id,
            )

            verification_status = object_status(
                s3_client=s3_client,
                key=key,
                expected_sha256=file_sha256,
                expected_size=size_bytes,
            )

            if verification_status != "matching":
                raise RuntimeError(
                    "Uploaded object verification failed: "
                    f"s3://{BUCKET}/{key}"
                )

            uploaded_count += 1
            action = "uploaded"

        else:
            planned_count += 1
            action = "planned"

        print(
            f"{action.upper():8} "
            f"{s3_relative_path} "
            f"({size_bytes:,} bytes)"
        )

        artifacts.append(
            {
                "artifact_kind": artifact_kind,
                "local_path": str(local_path),
                "relative_path": (
                    s3_relative_path
                ),
                "s3_uri": (
                    f"s3://{BUCKET}/{key}"
                ),
                "content_type": content_type,
                "size_bytes": size_bytes,
                "sha256": file_sha256,
                "publication_action": action,
            }
        )

    publication_status = (
        "PUBLISHED"
        if args.upload
        else "PLANNED"
    )

    manifest = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "publication_status": (
            publication_status
        ),
        "region": REGION,
        "bucket": BUCKET,
        "prefix": s3_prefix,
        "book_id": BOOK_ID,
        "book_version": BOOK_VERSION,
        "grade": GRADE,
        "range_name": args.range_name,
        "bda_invocation_id": (
            args.invocation_id
        ),
        "validation": validation,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }

    local_manifest_path = (
        args.normalized_dir
        / "retrieval-publication-manifest.json"
    )

    atomic_write_json(
        local_manifest_path,
        manifest,
    )

    manifest_sha256 = sha256_file(
        local_manifest_path
    )

    manifest_key = (
        f"{s3_prefix}/"
        "retrieval-publication-manifest.json"
    )

    if args.upload:
        upload_artifact(
            s3_client=s3_client,
            local_path=local_manifest_path,
            key=manifest_key,
            content_type="application/json",
            artifact_kind=(
                "retrieval_publication_manifest"
            ),
            sha256=manifest_sha256,
            range_name=args.range_name,
            invocation_id=args.invocation_id,
        )

        manifest_status = object_status(
            s3_client=s3_client,
            key=manifest_key,
            expected_sha256=manifest_sha256,
            expected_size=(
                local_manifest_path.stat().st_size
            ),
        )

        if manifest_status != "matching":
            raise RuntimeError(
                "Publication manifest verification failed."
            )

    print()
    print("============================================")
    print("PUBLICATION COMPLETED")
    print("============================================")
    print(f"Status:          {publication_status}")
    print(f"Uploaded:        {uploaded_count}")
    print(f"Already matching:{matching_count}")
    print(f"Planned:         {planned_count}")
    print(f"Artifacts:       {len(artifacts)}")
    print(
        f"Local manifest:  {local_manifest_path}"
    )
    print(
        "S3 manifest:     "
        f"s3://{BUCKET}/{manifest_key}"
    )
    print(
        f"Manifest SHA:    {manifest_sha256}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (ClientError, BotoCoreError) as exc:
        print(
            f"AWS publication error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"Retrieval publication failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
