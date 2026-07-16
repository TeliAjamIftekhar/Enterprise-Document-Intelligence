from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError


REGION = "us-east-1"
BUCKET = "edi-documents-ajam-2026"


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()

    raise TypeError(
        f"Unsupported JSON value: "
        f"{type(value).__name__}"
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


def parse_s3_uri(
    s3_uri: str,
) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(
            f"Invalid S3 URI: {s3_uri}"
        )

    value = s3_uri[5:]

    bucket, separator, key = (
        value.partition("/")
    )

    if not bucket:
        raise ValueError(
            f"S3 URI has no bucket: {s3_uri}"
        )

    return bucket, key if separator else ""


def get_invocation_id(
    invocation_arn: str,
) -> str:
    invocation_id = (
        invocation_arn.rstrip("/")
        .rsplit("/", 1)[-1]
    )

    if not invocation_id:
        raise RuntimeError(
            "Could not extract invocation ID."
        )

    return invocation_id


def select_batch(
    manifest: dict[str, Any],
    batch_id: str,
) -> dict[str, Any]:
    batches = manifest.get(
        "batches"
    )

    if not isinstance(batches, list):
        raise RuntimeError(
            "Manifest contains no batches list."
        )

    matches = [
        item
        for item in batches
        if item.get("batch_id")
        == batch_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {batch_id}, "
            f"found {len(matches)}."
        )

    batch = matches[0]

    if batch.get("uploaded") is not True:
        raise RuntimeError(
            f"{batch_id} is not uploaded."
        )

    if batch.get("s3_verified") is not True:
        raise RuntimeError(
            f"{batch_id} is not S3 verified."
        )

    return batch


def validate_job(
    job: dict[str, Any],
    batch_id: str,
) -> tuple[str, str, str]:
    if job.get("batch_id") != batch_id:
        raise RuntimeError(
            "Job batch ID mismatch."
        )

    if job.get("latest_status") != "Success":
        raise RuntimeError(
            "BDA job is not successful: "
            f"{job.get('latest_status')}"
        )

    if job.get("completed") is not True:
        raise RuntimeError(
            "BDA job is not marked completed."
        )

    invocation_arn = str(
        job.get("invocation_arn", "")
    )

    if not invocation_arn:
        raise RuntimeError(
            "Job invocation ARN is missing."
        )

    requested_output_uri = str(
        job.get(
            "output_configuration",
            {},
        ).get(
            "s3Uri",
            "",
        )
    )

    if not requested_output_uri:
        raise RuntimeError(
            "Requested output S3 URI missing."
        )

    returned_output_uri = str(
        job.get(
            "final_status_response",
            {},
        ).get(
            "outputConfiguration",
            {},
        ).get(
            "s3Uri",
            "",
        )
    )

    if not returned_output_uri:
        raise RuntimeError(
            "Returned output S3 URI missing."
        )

    return (
        invocation_arn,
        requested_output_uri,
        returned_output_uri,
    )


def list_output_objects(
    s3_client: Any,
    bucket: str,
    prefix: str,
) -> list[dict[str, Any]]:
    normalized_prefix = (
        prefix.rstrip("/")
        + "/"
    )

    paginator = s3_client.get_paginator(
        "list_objects_v2"
    )

    objects: list[
        dict[str, Any]
    ] = []

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=normalized_prefix,
    ):
        for item in page.get(
            "Contents",
            [],
        ):
            key = str(item["Key"])

            objects.append(
                {
                    "key": key,
                    "relative_key": key[
                        len(normalized_prefix):
                    ],
                    "size_bytes": int(
                        item.get("Size", 0)
                    ),
                    "etag": str(
                        item.get("ETag", "")
                    ).strip('"'),
                    "last_modified": (
                        item.get(
                            "LastModified"
                        )
                    ),
                }
            )

    return sorted(
        objects,
        key=lambda item: str(
            item["key"]
        ),
    )


def download_objects(
    s3_client: Any,
    bucket: str,
    objects: list[dict[str, Any]],
    local_root: Path,
) -> tuple[
    list[dict[str, Any]],
    int,
]:
    downloaded: list[
        dict[str, Any]
    ] = []

    directory_marker_count = 0

    for index, item in enumerate(
        objects,
        start=1,
    ):
        relative_key = str(
            item["relative_key"]
        )

        if not relative_key:
            raise RuntimeError(
                "Output object has empty "
                "relative key."
            )

        local_path = (
            local_root / relative_key
        )

        if (
            item["key"].endswith("/")
            or (
                item["size_bytes"] == 0
                and not Path(
                    relative_key
                ).suffix
            )
        ):
            local_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            directory_marker_count += 1

            downloaded.append(
                {
                    **item,
                    "local_path": str(
                        local_path
                    ),
                    "directory_marker": True,
                    "downloaded": False,
                }
            )

            print(
                f"[{index:02d}/"
                f"{len(objects):02d}] "
                f"DIR  {relative_key}"
            )

            continue

        local_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        should_download = True

        if local_path.is_file():
            if (
                local_path.stat().st_size
                == item["size_bytes"]
            ):
                should_download = False

        if should_download:
            temporary_path = (
                local_path.with_suffix(
                    local_path.suffix
                    + ".download"
                )
            )

            if temporary_path.exists():
                temporary_path.unlink()

            s3_client.download_file(
                Bucket=bucket,
                Key=item["key"],
                Filename=str(
                    temporary_path
                ),
            )

            if (
                temporary_path.stat().st_size
                != item["size_bytes"]
            ):
                raise RuntimeError(
                    "Downloaded size mismatch: "
                    f"{relative_key}"
                )

            os.replace(
                temporary_path,
                local_path,
            )

            action = "GET "

        else:
            action = "SKIP"

        local_size = (
            local_path.stat().st_size
        )

        if local_size != item["size_bytes"]:
            raise RuntimeError(
                "Local output size mismatch: "
                f"{relative_key}"
            )

        local_sha256 = sha256_file(
            local_path
        )

        downloaded.append(
            {
                **item,
                "local_path": str(
                    local_path
                ),
                "local_size_bytes": (
                    local_size
                ),
                "local_sha256": (
                    local_sha256
                ),
                "directory_marker": False,
                "downloaded": (
                    action == "GET "
                ),
                "verified": True,
            }
        )

        print(
            f"[{index:02d}/"
            f"{len(objects):02d}] "
            f"{action} {relative_key} | "
            f"{local_size:,} bytes"
        )

    return (
        downloaded,
        directory_marker_count,
    )



def load_batch_context_metadata(
    manifest: dict[str, Any],
    batch: dict[str, Any],
) -> dict[str, Any]:
    metadata_path_value = batch.get(
        "normalizer_metadata_path"
    )

    # Preserve legacy manifests that do not have
    # chapter/page context sidecars.
    if not metadata_path_value:
        return {}

    metadata_path = Path(
        str(metadata_path_value)
    )

    sidecar = load_json_object(
        metadata_path
    )

    actual_sha256 = sha256_file(
        metadata_path
    )

    expected_sha256 = batch.get(
        "normalizer_metadata_sha256"
    )

    if (
        expected_sha256
        and actual_sha256
        != str(expected_sha256)
    ):
        raise RuntimeError(
            "Batch context sidecar SHA256 "
            "does not match the manifest."
        )

    expected_identity = {
        "book_id": manifest["book_id"],
        "book_version": (
            manifest["book_version"]
        ),
        "batch_id": batch["batch_id"],
        "batch_number": (
            batch["batch_number"]
        ),
        "source_start_page": (
            batch["source_page_start"]
        ),
        "source_end_page": (
            batch["source_page_end"]
        ),
        "source_page_offset": (
            batch["source_page_offset"]
        ),
        "batch_page_count": (
            batch["page_count"]
        ),
    }

    for field, expected in (
        expected_identity.items()
    ):
        actual = sidecar.get(field)

        if actual != expected:
            raise RuntimeError(
                "Batch context sidecar identity "
                f"mismatch for {field}: "
                f"expected={expected!r}, "
                f"actual={actual!r}"
            )

    page_contexts = sidecar.get(
        "page_contexts"
    )

    if not isinstance(
        page_contexts,
        list,
    ):
        raise RuntimeError(
            "Batch context sidecar contains "
            "no page_contexts list."
        )

    expected_count = int(
        batch["page_count"]
    )

    if len(page_contexts) != expected_count:
        raise RuntimeError(
            "Batch page-context count mismatch: "
            f"expected={expected_count}, "
            f"actual={len(page_contexts)}"
        )

    batch_pages: list[int] = []

    for context in page_contexts:
        if not isinstance(context, dict):
            raise RuntimeError(
                "Invalid page-context record."
            )

        batch_page = context.get(
            "batch_page"
        )

        if not isinstance(batch_page, int):
            raise RuntimeError(
                "Page-context record has no "
                "integer batch_page."
            )

        batch_pages.append(batch_page)

    expected_batch_pages = list(
        range(1, expected_count + 1)
    )

    if sorted(batch_pages) != (
        expected_batch_pages
    ):
        raise RuntimeError(
            "Page-context batch-page coverage "
            "is not complete and contiguous."
        )

    context_fields = (
        "metadata_type",
        "source_pdf_uri",
        "chapter_page_map_path",
        "document_ids",
        "chapter_ids",
        "page_contexts",
        "chapter_spans",
    )

    result = {
        field: sidecar[field]
        for field in context_fields
        if field in sidecar
    }

    result[
        "page_context_metadata_path"
    ] = str(metadata_path)

    result[
        "page_context_metadata_sha256"
    ] = actual_sha256

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download one successful full-book "
            "BDA batch and create normalizer "
            "metadata."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--job-record",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--batch-id",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest = load_json_object(
        args.manifest
    )

    job = load_json_object(
        args.job_record
    )

    batch = select_batch(
        manifest=manifest,
        batch_id=args.batch_id,
    )

    (
        invocation_arn,
        requested_output_uri,
        returned_output_uri,
    ) = validate_job(
        job=job,
        batch_id=args.batch_id,
    )

    invocation_id = get_invocation_id(
        invocation_arn
    )

    local_invocation_root = (
        args.output_root
        / args.batch_id
        / invocation_id
    )

    local_invocation_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    bucket, prefix = parse_s3_uri(
        requested_output_uri
    )

    if bucket != BUCKET:
        raise RuntimeError(
            f"Unexpected output bucket: {bucket}"
        )

    s3_client = boto3.client(
        "s3",
        region_name=REGION,
    )

    objects = list_output_objects(
        s3_client=s3_client,
        bucket=bucket,
        prefix=prefix,
    )

    if not objects:
        raise RuntimeError(
            "No BDA output objects found."
        )

    expected_inventory = job.get(
        "output_inventory",
        {},
    )

    expected_object_count = int(
        expected_inventory.get(
            "object_count",
            -1,
        )
    )

    expected_total_bytes = int(
        expected_inventory.get(
            "total_size_bytes",
            -1,
        )
    )

    actual_total_bytes = sum(
        int(item["size_bytes"])
        for item in objects
    )

    if (
        expected_object_count != -1
        and len(objects)
        != expected_object_count
    ):
        raise RuntimeError(
            "S3 output object count changed: "
            f"expected={expected_object_count}, "
            f"actual={len(objects)}"
        )

    if (
        expected_total_bytes != -1
        and actual_total_bytes
        != expected_total_bytes
    ):
        raise RuntimeError(
            "S3 output byte count changed: "
            f"expected={expected_total_bytes}, "
            f"actual={actual_total_bytes}"
        )

    returned_bucket, returned_key = (
        parse_s3_uri(
            returned_output_uri
        )
    )

    if returned_bucket != bucket:
        raise RuntimeError(
            "Returned output bucket mismatch."
        )

    if not any(
        item["key"] == returned_key
        for item in objects
    ):
        raise RuntimeError(
            "Returned job metadata object "
            "not found in output inventory."
        )

    print("=" * 60)
    print("DOWNLOAD FULL BOOK BDA BATCH")
    print("=" * 60)
    print(f"Batch:       {args.batch_id}")
    print(
        f"Pages:       "
        f"{batch['source_page_start']}-"
        f"{batch['source_page_end']}"
    )
    print(
        f"Invocation:  {invocation_id}"
    )
    print(
        f"S3 objects:  {len(objects)}"
    )
    print(
        f"S3 bytes:    "
        f"{actual_total_bytes:,}"
    )
    print(
        f"Local root:  "
        f"{local_invocation_root}"
    )
    print()

    (
        downloaded_objects,
        directory_marker_count,
    ) = download_objects(
        s3_client=s3_client,
        bucket=bucket,
        objects=objects,
        local_root=local_invocation_root,
    )

    downloaded_files = [
        item
        for item in downloaded_objects
        if not item[
            "directory_marker"
        ]
    ]

    result_json_files = [
        Path(item["local_path"])
        for item in downloaded_files
        if Path(
            item["local_path"]
        ).name == "result.json"
    ]

    job_metadata_files = [
        Path(item["local_path"])
        for item in downloaded_files
        if Path(
            item["local_path"]
        ).name
        == "job_metadata.json"
    ]

    if len(result_json_files) != 1:
        raise RuntimeError(
            "Expected exactly one result.json, "
            f"found {len(result_json_files)}."
        )

    if len(job_metadata_files) != 1:
        raise RuntimeError(
            "Expected exactly one "
            "job_metadata.json, found "
            f"{len(job_metadata_files)}."
        )

    result_json_path = (
        result_json_files[0]
    )

    result = load_json_object(
        result_json_path
    )

    elements = result.get(
        "elements"
    )

    if not isinstance(elements, list):
        raise RuntimeError(
            "BDA result contains no "
            "elements list."
        )

    result_metadata = result.get(
        "metadata",
        {},
    )

    if not isinstance(
        result_metadata,
        dict,
    ):
        result_metadata = {}

    result_page_count = (
        result_metadata.get(
            "number_of_pages"
        )
    )

    expected_page_count = int(
        batch["page_count"]
    )

    if (
        result_page_count is not None
        and int(result_page_count)
        != expected_page_count
    ):
        raise RuntimeError(
            "BDA result page count mismatch: "
            f"expected={expected_page_count}, "
            f"actual={result_page_count}"
        )

    asset_directory = (
        result_json_path.parent
        / "assets"
    )

    if not asset_directory.is_dir():
        raise RuntimeError(
            "Expected result assets directory "
            f"not found: {asset_directory}"
        )

    extension_counts: Counter[str] = (
        Counter()
    )

    for item in downloaded_files:
        suffix = Path(
            item["local_path"]
        ).suffix.lower()

        extension_counts[
            suffix or "<none>"
        ] += 1

    chapter_context_metadata = (
        load_batch_context_metadata(
            manifest=manifest,
            batch=batch,
        )
    )

    normalizer_metadata = {
        "schema_version": "1.0",
        "book_id": manifest[
            "book_id"
        ],
        "book_version": manifest[
            "book_version"
        ],
        "source_pdf": manifest[
            "source"
        ]["local_path"],
        "source_start_page": batch[
            "source_page_start"
        ],
        "source_end_page": batch[
            "source_page_end"
        ],
        "sample_page_count": batch[
            "page_count"
        ],
        "sample_size_bytes": batch[
            "size_bytes"
        ],
        "sample_sha256": batch[
            "sha256"
        ],
        "local_sample_path": batch[
            "local_path"
        ],
        "sample_s3_uri": batch[
            "s3_uri"
        ],
        "batch_id": batch[
            "batch_id"
        ],
        "batch_number": batch[
            "batch_number"
        ],
        "source_page_offset": batch[
            "source_page_offset"
        ],
        "bda_invocation_arn": (
            invocation_arn
        ),
        "bda_output_s3_uri": (
            requested_output_uri
        ),
        "local_result_json": str(
            result_json_path
        ),
        "created_at": utc_now(),
    }

    normalizer_metadata.update(
        chapter_context_metadata
    )

    metadata_adapter_path = (
        local_invocation_root
        / "normalizer-metadata.json"
    )

    atomic_write_json(
        metadata_adapter_path,
        normalizer_metadata,
    )

    report_path = (
        local_invocation_root
        / "download-report.json"
    )

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": "VALIDATED",
        "book_id": manifest[
            "book_id"
        ],
        "book_version": manifest[
            "book_version"
        ],
        "batch_id": args.batch_id,
        "source_page_start": batch[
            "source_page_start"
        ],
        "source_page_end": batch[
            "source_page_end"
        ],
        "page_count": (
            expected_page_count
        ),
        "invocation_arn": (
            invocation_arn
        ),
        "requested_output_s3_uri": (
            requested_output_uri
        ),
        "returned_output_s3_uri": (
            returned_output_uri
        ),
        "local_invocation_root": str(
            local_invocation_root
        ),
        "result_json_path": str(
            result_json_path
        ),
        "job_metadata_path": str(
            job_metadata_files[0]
        ),
        "normalizer_metadata_path": str(
            metadata_adapter_path
        ),
        "page_context_metadata_available": (
            bool(
                normalizer_metadata.get(
                    "page_contexts"
                )
            )
        ),
        "page_context_record_count": len(
            normalizer_metadata.get(
                "page_contexts",
                [],
            )
        ),
        "asset_directory": str(
            asset_directory
        ),
        "source_object_count": len(
            objects
        ),
        "downloaded_file_count": len(
            downloaded_files
        ),
        "directory_marker_count": (
            directory_marker_count
        ),
        "total_size_bytes": (
            actual_total_bytes
        ),
        "element_count": len(
            elements
        ),
        "result_page_count": (
            result_page_count
        ),
        "extension_counts": dict(
            sorted(
                extension_counts.items()
            )
        ),
        "objects": downloaded_objects,
        "validated": True,
        "new_bda_invocation": False,
    }

    atomic_write_json(
        report_path,
        report,
    )

    print()
    print("=" * 60)
    print("BDA BATCH DOWNLOAD RESULT")
    print("=" * 60)
    print("Status:              VALIDATED")
    print(
        f"Batch:               "
        f"{args.batch_id}"
    )
    print(
        f"Source objects:      "
        f"{len(objects)}"
    )
    print(
        f"Local files:         "
        f"{len(downloaded_files)}"
    )
    print(
        f"Directory markers:   "
        f"{directory_marker_count}"
    )
    print(
        f"Total bytes:         "
        f"{actual_total_bytes:,}"
    )
    print(
        f"Result pages:        "
        f"{result_page_count}"
    )
    print(
        f"Elements:            "
        f"{len(elements)}"
    )
    print(
        f"PNG assets:          "
        f"{extension_counts.get('.png', 0)}"
    )
    print(
        f"CSV assets:          "
        f"{extension_counts.get('.csv', 0)}"
    )
    print(
        f"Result JSON:         "
        f"{result_json_path}"
    )
    print(
        f"Normalizer metadata: "
        f"{metadata_adapter_path}"
    )
    print(
        f"Download report:     "
        f"{report_path}"
    )
    print(
        "New BDA invocation:  False"
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
            f"AWS download error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"BDA batch download failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
