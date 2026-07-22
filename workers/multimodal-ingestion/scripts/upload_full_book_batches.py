from __future__ import annotations

import argparse
import copy
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


DEFAULT_REGION = "us-east-1"
DEFAULT_BUCKET = "edi-documents-ajam-2026"

def derive_manifest_s3_key(
    manifest: dict[str, Any],
) -> str:
    """
    Derive the manifest key from this book's batch
    S3 prefix. Never use another book's default.
    """

    batching = manifest.get("batching")

    if isinstance(batching, dict):
        prefix = batching.get("s3_prefix")

        if isinstance(prefix, str) and prefix:
            normalized = prefix.strip()

            if normalized.startswith("s3://"):
                without_scheme = normalized[5:]

                if "/" not in without_scheme:
                    raise ValueError(
                        "Invalid batching S3 prefix: "
                        f"{prefix!r}"
                    )

                _, normalized = (
                    without_scheme.split("/", 1)
                )

            normalized = normalized.strip("/")

            if normalized.endswith("/batches"):
                root = normalized[:-len("/batches")]

                return (
                    f"{root}/"
                    "full-book-batch-manifest.json"
                )

    batches = manifest.get("batches")

    if isinstance(batches, list) and batches:
        first_batch = batches[0]

        if isinstance(first_batch, dict):
            batch_key = first_batch.get("s3_key")

            if (
                isinstance(batch_key, str)
                and "/batches/" in batch_key
            ):
                root = batch_key.split(
                    "/batches/",
                    1,
                )[0].strip("/")

                return (
                    f"{root}/"
                    "full-book-batch-manifest.json"
                )

    raise ValueError(
        "Unable to derive manifest S3 key from "
        "the batch manifest."
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
        while chunk := file.read(
            chunk_size
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
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


def validate_manifest(
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    status = manifest.get("status")

    if status not in {
        "PREPARED",
        "UPLOADED",
    }:
        raise RuntimeError(
            "Manifest status must be PREPARED "
            f"or UPLOADED, received: {status}"
        )

    validation = manifest.get(
        "validation",
        {},
    )

    expected_pages = validation.get(
        "expected_pages"
    )

    expected_batch_count = validation.get(
        "expected_batch_count"
    )

    actual_batch_count = validation.get(
        "actual_batch_count"
    )

    for field_name, value in (
        ("expected_pages", expected_pages),
        (
            "expected_batch_count",
            expected_batch_count,
        ),
        (
            "actual_batch_count",
            actual_batch_count,
        ),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise RuntimeError(
                "Manifest validation has invalid "
                f"{field_name}: {value}"
            )

    if expected_batch_count != actual_batch_count:
        raise RuntimeError(
            "Manifest expected and actual batch "
            "counts differ: "
            f"expected={expected_batch_count}, "
            f"actual={actual_batch_count}"
        )

    required_checks = {
        "contiguous": (
            validation.get("contiguous")
            is True
        ),
        "missing_pages_empty": (
            validation.get(
                "missing_pages"
            )
            == []
        ),
        "overlapping_pages_empty": (
            validation.get(
                "overlapping_pages"
            )
            == []
        ),
        "expected_pages_positive": (
            expected_pages > 0
        ),
        "expected_batch_count_positive": (
            expected_batch_count > 0
        ),
        "actual_batch_count_positive": (
            actual_batch_count > 0
        ),
        "all_text_verified": (
            validation.get(
                "all_batch_text_verified"
            )
            is True
        ),
        "all_geometry_verified": (
            validation.get(
                "all_geometry_verified"
            )
            is True
        ),
        "all_visual_verified": (
            validation.get(
                "all_visual_verified"
            )
            is True
        ),
        "all_fidelity_verified": (
            validation.get(
                "all_fidelity_verified"
            )
            is True
        ),
    }

    failed_checks = [
        name
        for name, passed
        in required_checks.items()
        if not passed
    ]

    if failed_checks:
        raise RuntimeError(
            "Manifest validation failed:\n- "
            + "\n- ".join(
                failed_checks
            )
        )

    batches = manifest.get("batches")

    if not isinstance(batches, list):
        raise RuntimeError(
            "Manifest has no batches list."
        )

    if len(batches) != actual_batch_count:
        raise RuntimeError(
            "Manifest actual batch count differs "
            "from the batches list: "
            f"validation={actual_batch_count}, "
            f"actual={len(batches)}"
        )

    if len(batches) != expected_batch_count:
        raise RuntimeError(
            "Manifest batch count differs from "
            "the expected batch count: "
            f"expected={expected_batch_count}, "
            f"actual={len(batches)}"
        )

    sorted_batches = sorted(
        batches,
        key=lambda item: int(
            item["batch_number"]
        ),
    )

    expected_start_page = 1
    local_results: list[
        dict[str, Any]
    ] = []

    batch_ids: set[str] = set()
    s3_keys: set[str] = set()

    for batch in sorted_batches:
        batch_id = str(
            batch.get("batch_id", "")
        )

        if not batch_id:
            raise RuntimeError(
                "Batch has no batch_id."
            )

        if batch_id in batch_ids:
            raise RuntimeError(
                f"Duplicate batch ID: {batch_id}"
            )

        batch_ids.add(batch_id)

        start_page = int(
            batch["source_page_start"]
        )

        end_page = int(
            batch["source_page_end"]
        )

        if start_page != expected_start_page:
            raise RuntimeError(
                f"{batch_id} starts at page "
                f"{start_page}; expected "
                f"{expected_start_page}."
            )

        if end_page < start_page:
            raise RuntimeError(
                f"{batch_id} has invalid range."
            )

        expected_page_count = (
            end_page - start_page + 1
        )

        if (
            int(batch["page_count"])
            != expected_page_count
        ):
            raise RuntimeError(
                f"{batch_id} page-count metadata "
                "does not match its range."
            )

        expected_start_page = (
            end_page + 1
        )

        local_path = Path(
            str(batch["local_path"])
        )

        if not local_path.is_file():
            raise FileNotFoundError(
                f"{batch_id} local file missing: "
                f"{local_path}"
            )

        actual_size = (
            local_path.stat().st_size
        )

        expected_size = int(
            batch["size_bytes"]
        )

        if actual_size != expected_size:
            raise RuntimeError(
                f"{batch_id} size mismatch: "
                f"manifest={expected_size}, "
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
                f"{batch_id} SHA256 mismatch."
            )

        with fitz.open(
            str(local_path)
        ) as document:
            actual_page_count = len(
                document
            )

        if (
            actual_page_count
            != expected_page_count
        ):
            raise RuntimeError(
                f"{batch_id} PDF page-count "
                f"mismatch: expected="
                f"{expected_page_count}, "
                f"actual={actual_page_count}"
            )

        s3_key = str(
            batch.get("s3_key", "")
        )

        if not s3_key:
            raise RuntimeError(
                f"{batch_id} has no S3 key."
            )

        if s3_key in s3_keys:
            raise RuntimeError(
                f"Duplicate S3 key: {s3_key}"
            )

        s3_keys.add(s3_key)

        local_results.append(
            {
                "batch_id": batch_id,
                "batch_number": int(
                    batch["batch_number"]
                ),
                "local_path": str(
                    local_path
                ),
                "s3_key": s3_key,
                "source_page_start": (
                    start_page
                ),
                "source_page_end": (
                    end_page
                ),
                "page_count": (
                    expected_page_count
                ),
                "size_bytes": actual_size,
                "sha256": actual_sha256,
                "local_verified": True,
            }
        )

    if expected_start_page != expected_pages + 1:
        raise RuntimeError(
            "Final source page is not "
            f"{expected_pages}."
        )

    return local_results


def inspect_remote_object(
    s3_client: Any,
    bucket: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    key = str(item["s3_key"])

    try:
        head = s3_client.head_object(
            Bucket=bucket,
            Key=key,
        )

    except ClientError as exc:
        error = exc.response.get(
            "Error",
            {},
        )

        code = str(
            error.get("Code", "")
        )

        http_status = (
            exc.response.get(
                "ResponseMetadata",
                {},
            ).get(
                "HTTPStatusCode"
            )
        )

        if (
            code in {
                "404",
                "NoSuchKey",
                "NotFound",
            }
            or http_status == 404
        ):
            return {
                **item,
                "remote_state": "missing",
                "remote_exists": False,
                "remote_verified": False,
                "conflict_reasons": [],
            }

        raise

    metadata = head.get(
        "Metadata",
        {},
    )

    remote_size = int(
        head.get("ContentLength", -1)
    )

    remote_sha256 = metadata.get(
        "sha256"
    )

    content_type = head.get(
        "ContentType"
    )

    encryption = head.get(
        "ServerSideEncryption"
    )

    conflict_reasons: list[str] = []

    if (
        remote_size
        != int(item["size_bytes"])
    ):
        conflict_reasons.append(
            "size_mismatch"
        )

    if (
        remote_sha256
        != item["sha256"]
    ):
        conflict_reasons.append(
            "sha256_metadata_mismatch"
        )

    if content_type != "application/pdf":
        conflict_reasons.append(
            "content_type_mismatch"
        )

    if encryption != "AES256":
        conflict_reasons.append(
            "encryption_mismatch"
        )

    state = (
        "matching"
        if not conflict_reasons
        else "conflict"
    )

    return {
        **item,
        "remote_state": state,
        "remote_exists": True,
        "remote_verified": (
            state == "matching"
        ),
        "remote_size_bytes": (
            remote_size
        ),
        "remote_sha256": (
            remote_sha256
        ),
        "content_type": content_type,
        "server_side_encryption": (
            encryption
        ),
        "etag": str(
            head.get("ETag", "")
        ).strip('"'),
        "version_id": head.get(
            "VersionId"
        ),
        "last_modified": head.get(
            "LastModified"
        ),
        "conflict_reasons": (
            conflict_reasons
        ),
    }


def upload_batch(
    s3_client: Any,
    bucket: str,
    item: dict[str, Any],
    book_id: str,
    book_version: str,
) -> dict[str, Any]:
    local_path = Path(
        item["local_path"]
    )

    s3_client.upload_file(
        Filename=str(local_path),
        Bucket=bucket,
        Key=item["s3_key"],
        ExtraArgs={
            "ContentType": "application/pdf",
            "ServerSideEncryption": "AES256",
            "Metadata": {
                "sha256": item["sha256"],
                "book-id": book_id,
                "book-version": (
                    book_version
                ),
                "batch-id": (
                    item["batch_id"]
                ),
                "source-page-start": str(
                    item[
                        "source_page_start"
                    ]
                ),
                "source-page-end": str(
                    item[
                        "source_page_end"
                    ]
                ),
                "page-count": str(
                    item["page_count"]
                ),
            },
        },
    )

    verified = inspect_remote_object(
        s3_client=s3_client,
        bucket=bucket,
        item=item,
    )

    if (
        verified["remote_state"]
        != "matching"
    ):
        raise RuntimeError(
            f"{item['batch_id']} upload "
            "verification failed: "
            + json.dumps(
                verified[
                    "conflict_reasons"
                ]
            )
        )

    return verified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or upload validated "
            "full-book BDA batches to S3."
        )
    )

    parser.add_argument(
        "manifest",
        type=Path,
    )

    parser.add_argument(
        "--report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
    )

    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
    )

    parser.add_argument(
        "--manifest-s3-key",
        default=None,
        help=(
            "Optional manifest S3 key override. "
            "When omitted, the key is derived "
            "from this book's batch S3 prefix."
        ),
    )

    parser.add_argument(
        "--upload",
        action="store_true",
        help=(
            "Upload missing batches. Without this "
            "flag only preflight checks are run."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest = load_json_object(
        args.manifest
    )


    manifest_s3_key = (
        args.manifest_s3_key
        or derive_manifest_s3_key(
            manifest
        )
    )
    local_items = validate_manifest(
        manifest
    )

    s3_client = boto3.client(
        "s3",
        region_name=args.region,
    )

    print("=" * 48)
    print("FULL BOOK S3 BATCH PREFLIGHT")
    print("=" * 48)
    print(f"Region:       {args.region}")
    print(f"Bucket:       {args.bucket}")
    print(
        f"Manifest:     {args.manifest}"
    )
    print(
        f"Local batches:{len(local_items)}"
    )
    print(
        f"Upload mode:  {args.upload}"
    )
    print(
        "BDA invoked:  False"
    )
    print()

    inspected_items: list[
        dict[str, Any]
    ] = []

    for index, item in enumerate(
        local_items,
        start=1,
    ):
        inspected = inspect_remote_object(
            s3_client=s3_client,
            bucket=args.bucket,
            item=item,
        )

        inspected_items.append(
            inspected
        )

        print(
            f"[{index:02d}/"
            f"{len(local_items):02d}] "
            f"{item['batch_id']} | "
            f"pages "
            f"{item['source_page_start']:04d}-"
            f"{item['source_page_end']:04d} | "
            f"{inspected['remote_state'].upper()}"
        )

    missing_count = sum(
        1
        for item in inspected_items
        if item["remote_state"]
        == "missing"
    )

    matching_count = sum(
        1
        for item in inspected_items
        if item["remote_state"]
        == "matching"
    )

    conflict_items = [
        item
        for item in inspected_items
        if item["remote_state"]
        == "conflict"
    ]

    if conflict_items:
        report = {
            "schema_version": "1.0",
            "generated_at": utc_now(),
            "status": (
                "CONFLICTS_DETECTED"
            ),
            "region": args.region,
            "bucket": args.bucket,
            "manifest_path": str(
                args.manifest
            ),
            "upload_requested": (
                args.upload
            ),
            "local_batch_count": len(
                local_items
            ),
            "missing_count": (
                missing_count
            ),
            "matching_count": (
                matching_count
            ),
            "conflict_count": len(
                conflict_items
            ),
            "items": inspected_items,
            "uploaded": False,
            "bda_invoked": False,
        }

        atomic_write_json(
            args.report,
            report,
        )

        raise RuntimeError(
            "S3 conflicts detected. No uploads "
            "were performed."
        )

    if not args.upload:
        report = {
            "schema_version": "1.0",
            "generated_at": utc_now(),
            "status": "PREFLIGHT_PASSED",
            "region": args.region,
            "bucket": args.bucket,
            "manifest_path": str(
                args.manifest
            ),
            "manifest_sha256": (
                sha256_file(
                    args.manifest
                )
            ),
            "upload_requested": False,
            "local_batch_count": len(
                local_items
            ),
            "local_verified_count": len(
                local_items
            ),
            "missing_count": missing_count,
            "matching_count": (
                matching_count
            ),
            "conflict_count": 0,
            "total_size_bytes": sum(
                int(item["size_bytes"])
                for item in local_items
            ),
            "items": inspected_items,
            "uploaded": False,
            "bda_invoked": False,
        }

        atomic_write_json(
            args.report,
            report,
        )

        print()
        print("=" * 48)
        print("S3 PREFLIGHT RESULT")
        print("=" * 48)
        print(
            "Status:          PREFLIGHT_PASSED"
        )
        print(
            f"Local verified:  "
            f"{len(local_items)}/"
            f"{len(local_items)}"
        )
        print(
            f"Missing in S3:   {missing_count}"
        )
        print(
            f"Already matching:{matching_count}"
        )
        print(
            "Conflicts:       0"
        )
        print(
            "Uploaded:        False"
        )
        print(
            "BDA invoked:     False"
        )
        print(
            f"Report:          {args.report}"
        )

        return 0

    uploaded_results: list[
        dict[str, Any]
    ] = []

    for index, item in enumerate(
        inspected_items,
        start=1,
    ):
        if (
            item["remote_state"]
            == "matching"
        ):
            uploaded_results.append(
                item
            )

            print(
                f"SKIP {item['batch_id']}: "
                "already matching"
            )

            continue

        print(
            f"UPLOAD [{index:02d}/"
            f"{len(inspected_items):02d}] "
            f"{item['batch_id']}"
        )

        uploaded = upload_batch(
            s3_client=s3_client,
            bucket=args.bucket,
            item=item,
            book_id=str(
                manifest["book_id"]
            ),
            book_version=str(
                manifest[
                    "book_version"
                ]
            ),
        )

        uploaded_results.append(
            uploaded
        )

    if not all(
        item["remote_state"]
        == "matching"
        for item in uploaded_results
    ):
        raise RuntimeError(
            "One or more uploaded objects "
            "failed final verification."
        )

    updated_manifest = copy.deepcopy(
        manifest
    )

    remote_by_batch_id = {
        item["batch_id"]: item
        for item in uploaded_results
    }

    for batch in updated_manifest[
        "batches"
    ]:
        remote = remote_by_batch_id[
            batch["batch_id"]
        ]

        batch["uploaded"] = True
        batch["s3_verified"] = True
        batch["s3_etag"] = remote.get(
            "etag"
        )
        batch["s3_version_id"] = (
            remote.get("version_id")
        )
        batch["uploaded_at"] = utc_now()

    updated_manifest["status"] = (
        "UPLOADED"
    )

    updated_manifest["uploaded"] = True
    updated_manifest["uploaded_at"] = (
        utc_now()
    )

    updated_manifest["bda_invoked"] = (
        False
    )

    updated_manifest[
        "s3_upload"
    ] = {
        "bucket": args.bucket,
        "batch_count": len(
            uploaded_results
        ),
        "verified_count": len(
            uploaded_results
        ),
        "manifest_s3_key": (
            manifest_s3_key
        ),
    }

    atomic_write_json(
        args.manifest,
        updated_manifest,
    )

    manifest_sha256 = sha256_file(
        args.manifest
    )

    s3_client.upload_file(
        Filename=str(args.manifest),
        Bucket=args.bucket,
        Key=manifest_s3_key,
        ExtraArgs={
            "ContentType": (
                "application/json"
            ),
            "ServerSideEncryption": (
                "AES256"
            ),
            "Metadata": {
                "sha256": (
                    manifest_sha256
                ),
                "book-id": str(
                    manifest["book_id"]
                ),
                "book-version": str(
                    manifest[
                        "book_version"
                    ]
                ),
                "status": "uploaded",
            },
        },
    )

    manifest_head = (
        s3_client.head_object(
            Bucket=args.bucket,
            Key=manifest_s3_key,
        )
    )

    remote_manifest_sha256 = (
        manifest_head.get(
            "Metadata",
            {},
        ).get("sha256")
    )

    if (
        remote_manifest_sha256
        != manifest_sha256
    ):
        raise RuntimeError(
            "Uploaded manifest checksum "
            "metadata mismatch."
        )

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": "UPLOADED",
        "region": args.region,
        "bucket": args.bucket,
        "manifest_path": str(
            args.manifest
        ),
        "manifest_sha256": (
            manifest_sha256
        ),
        "manifest_s3_uri": (
            f"s3://{args.bucket}/"
            f"{manifest_s3_key}"
        ),
        "upload_requested": True,
        "batch_count": len(
            uploaded_results
        ),
        "verified_count": len(
            uploaded_results
        ),
        "items": uploaded_results,
        "uploaded": True,
        "bda_invoked": False,
    }

    atomic_write_json(
        args.report,
        report,
    )

    print()
    print("=" * 48)
    print("S3 UPLOAD RESULT")
    print("=" * 48)
    print("Status:          UPLOADED")
    print(
        f"Batches:         "
        f"{len(uploaded_results)}"
    )
    print(
        f"Verified:        "
        f"{len(uploaded_results)}"
    )
    print(
        f"Manifest SHA256: "
        f"{manifest_sha256}"
    )
    print(
        f"Manifest S3:     "
        f"s3://{args.bucket}/"
        f"{manifest_s3_key}"
    )
    print("BDA invoked:     False")
    print(f"Report:          {args.report}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (
        ClientError,
        BotoCoreError,
    ) as exc:
        print(
            f"AWS S3 error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"Full-book S3 operation "
            f"failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
