from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
import boto3
import fitz
from botocore.exceptions import ClientError

from src.models import BookManifest


def calculate_sha256(
    file_path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """Calculate SHA-256 without loading the complete PDF into memory."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def get_object_head(
    s3_client: Any,
    bucket: str,
    key: str,
) -> dict[str, Any] | None:
    """Return S3 object metadata, or None when the object does not exist."""
    try:
        return s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")

        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return None

        raise


def copy_source_object(
    s3_client: Any,
    bucket: str,
    source_key: str,
    destination_key: str,
) -> dict[str, Any]:
    """
    Copy the source PDF to an immutable, versioned location.

    If the destination already exists with the same size, the copy is skipped.
    If it exists with a different size, registration fails rather than silently
    overwriting the versioned object.
    """
    source_head = get_object_head(s3_client, bucket, source_key)

    if source_head is None:
        raise FileNotFoundError(
            f"Source object does not exist: s3://{bucket}/{source_key}"
        )

    destination_head = get_object_head(
        s3_client,
        bucket,
        destination_key,
    )

    if destination_head is not None:
        source_size = source_head["ContentLength"]
        destination_size = destination_head["ContentLength"]

        if source_size != destination_size:
            raise RuntimeError(
                "The versioned destination already exists with a different "
                f"size. Source={source_size}, destination={destination_size}. "
                "Use a new version instead of overwriting this version."
            )

        print("Versioned source already exists; copy skipped.")
        return destination_head

    print("Copying PDF to versioned S3 path...")

    s3_client.copy_object(
        Bucket=bucket,
        Key=destination_key,
        CopySource={
            "Bucket": bucket,
            "Key": source_key,
        },
        ContentType="application/pdf",
        MetadataDirective="REPLACE",
        Metadata={
            "book-id": "grade-9-english-kaveri",
            "book-version": "v1",
            "document-type": "textbook",
        },
        ServerSideEncryption="AES256",
    )

    copied_head = get_object_head(
        s3_client,
        bucket,
        destination_key,
    )

    if copied_head is None:
        raise RuntimeError("S3 copy completed but destination was not found.")

    return copied_head


def download_validation_copy(
    s3_client: Any,
    bucket: str,
    key: str,
    local_path: Path,
    expected_size: int,
) -> None:
    """Download the versioned PDF when the local copy is missing or incomplete."""
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and local_path.stat().st_size == expected_size:
        print("Local validation copy already exists; download skipped.")
        return

    if local_path.exists():
        print("Removing incomplete local validation copy.")
        local_path.unlink()

    print("Downloading validation copy...")
    s3_client.download_file(bucket, key, str(local_path))

    actual_size = local_path.stat().st_size

    if actual_size != expected_size:
        raise RuntimeError(
            f"Downloaded PDF size mismatch. "
            f"Expected={expected_size}, actual={actual_size}"
        )


def inspect_pdf(local_path: Path) -> int:
    """Open the PDF with PyMuPDF and return its verified page count."""
    try:
        with fitz.open(local_path) as document:
            if document.needs_pass:
                raise RuntimeError("The source PDF is password protected.")

            page_count = document.page_count

            if page_count < 1:
                raise RuntimeError("The PDF contains no readable pages.")

            # Force PyMuPDF to access the first and last page.
            document.load_page(0)
            document.load_page(page_count - 1)

            return page_count

    except fitz.FileDataError as exc:
        raise RuntimeError(
            f"PyMuPDF could not read the PDF: {local_path}"
        ) from exc


def save_manifest(
    manifest: BookManifest,
    local_path: Path,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)

    local_path.write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register and version a textbook stored in Amazon S3."
    )

    parser.add_argument(
        "--bucket",
        default="edi-documents-ajam-2026",
    )
    parser.add_argument(
        "--source-key",
        default="Textbooks/Kaveri_English_Text_Book_Class_9.pdf",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
    )
    parser.add_argument(
        "--book-id",
        default="grade-9-english-kaveri",
    )
    parser.add_argument(
        "--version",
        default="v1",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    destination_key = (
        f"source-documents/grade-9/{args.book_id}/"
        f"versions/{args.version}/textbook.pdf"
    )

    derived_prefix = (
        f"derived-artifacts/grade-9/{args.book_id}/{args.version}"
    )

    manifest_key = f"{derived_prefix}/manifest.json"

    local_root = (
        Path("data")
        / "multimodal-output"
        / args.book_id
        / args.version
    )

    local_pdf = local_root / "source" / "textbook.pdf"
    local_manifest = local_root / "manifest.json"

    s3_client = boto3.client(
        "s3",
        region_name=args.region,
    )

    print("============================================")
    print("REGISTERING TEXTBOOK")
    print("============================================")
    print(f"Book ID:       {args.book_id}")
    print(f"Version:       {args.version}")
    print(f"Original:      s3://{args.bucket}/{args.source_key}")
    print(f"Versioned PDF: s3://{args.bucket}/{destination_key}")
    print()

    destination_head = copy_source_object(
        s3_client=s3_client,
        bucket=args.bucket,
        source_key=args.source_key,
        destination_key=destination_key,
    )

    source_size = int(destination_head["ContentLength"])

    download_validation_copy(
        s3_client=s3_client,
        bucket=args.bucket,
        key=destination_key,
        local_path=local_pdf,
        expected_size=source_size,
    )

    print("Validating PDF with PyMuPDF...")
    page_count = inspect_pdf(local_pdf)

    print("Calculating SHA-256...")
    sha256 = calculate_sha256(local_pdf)

    etag = str(destination_head.get("ETag", "")).strip('"') or None

    manifest = BookManifest(
        book_id=args.book_id,
        title="Kaveri English Textbook",
        grade=9,
        subject="english",
        language="english",
        board="maharashtra-state-board",
        book_version=args.version,
        source_pdf_uri=f"s3://{args.bucket}/{destination_key}",
        original_source_key=args.source_key,
        source_size_bytes=source_size,
        source_sha256=sha256,
        source_etag=etag,
        page_count=page_count,
        content_units_key=(
            f"{derived_prefix}/metadata/content-units.json"
        ),
        page_images_prefix=f"{derived_prefix}/pages/",
        figures_prefix=f"{derived_prefix}/figures/",
        tables_prefix=f"{derived_prefix}/tables/",
        ingestion_status="registered",
    )

    save_manifest(manifest, local_manifest)

    print("Uploading manifest...")

    s3_client.upload_file(
        str(local_manifest),
        args.bucket,
        manifest_key,
        ExtraArgs={
            "ContentType": "application/json",
            "ServerSideEncryption": "AES256",
            "Metadata": {
                "book-id": args.book_id,
                "book-version": args.version,
                "artifact-type": "manifest",
            },
        },
    )

    print()
    print("============================================")
    print("REGISTRATION COMPLETED")
    print("============================================")
    print(f"Page count:       {page_count}")
    print(f"Source size:      {source_size:,} bytes")
    print(f"SHA-256:          {sha256}")
    print(f"Local PDF:        {local_pdf}")
    print(f"Local manifest:   {local_manifest}")
    print(f"S3 manifest:      s3://{args.bucket}/{manifest_key}")
    print("Ingestion status: registered")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Registration failed: {exc}", file=sys.stderr)
        sys.exit(1)