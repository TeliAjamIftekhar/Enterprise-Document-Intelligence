from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import fitz


DEFAULT_BUCKET = "edi-documents-ajam-2026"
DEFAULT_REGION = "us-east-1"
DEFAULT_BOOK_ID = "grade-9-english-kaveri"
DEFAULT_VERSION = "v1"


def calculate_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def create_sample_pdf(
    source_pdf: Path,
    output_pdf: Path,
    start_page: int,
    end_page: int,
    overwrite: bool = False,
) -> int:
    """
    Create an inclusive, one-based PDF sample.

    Existing valid samples are reused unless overwrite=True.
    This keeps versioned sample artifacts byte-for-byte stable.
    """
    if not source_pdf.exists():
        raise FileNotFoundError(
            f"Source PDF not found: {source_pdf}"
        )

    if start_page < 1:
        raise ValueError("start_page must be at least 1.")

    if end_page < start_page:
        raise ValueError(
            "end_page must be greater than or equal to start_page."
        )

    expected_count = end_page - start_page + 1
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    if output_pdf.exists() and not overwrite:
        with fitz.open(output_pdf) as existing:
            if existing.needs_pass:
                raise RuntimeError(
                    "The existing sample PDF is password protected."
                )

            if existing.page_count != expected_count:
                raise RuntimeError(
                    "Existing sample has an unexpected page count. "
                    f"Expected={expected_count}, "
                    f"actual={existing.page_count}. "
                    "Use --overwrite to recreate it."
                )

            existing.load_page(0)
            existing.load_page(existing.page_count - 1)

        print(
            "Existing sample PDF is valid; "
            "sample creation skipped."
        )
        return expected_count

    if output_pdf.exists():
        print("Removing existing sample because --overwrite was used.")
        output_pdf.unlink()

    with fitz.open(source_pdf) as source:
        if source.needs_pass:
            raise RuntimeError(
                "The source PDF is password protected."
            )

        if end_page > source.page_count:
            raise ValueError(
                f"Requested page {end_page} exceeds "
                f"the PDF page count of {source.page_count}."
            )

        with fitz.open() as sample:
            sample.insert_pdf(
                source,
                from_page=start_page - 1,
                to_page=end_page - 1,
            )

            sample.set_metadata(
                {
                    "title": (
                        f"Kaveri Grade 9 English "
                        f"Pages {start_page}-{end_page}"
                    ),
                    "subject": (
                        "Bedrock Data Automation sample"
                    ),
                    "author": "Textbook AI Assistant",
                    "keywords": (
                        "grade-9,kaveri,"
                        "bedrock-data-automation,sample"
                    ),
                }
            )

            sample.save(
                output_pdf,
                garbage=4,
                deflate=True,
                clean=True,
            )

    with fitz.open(output_pdf) as validation:
        if validation.page_count != expected_count:
            raise RuntimeError(
                "Sample page-count mismatch. "
                f"Expected={expected_count}, "
                f"actual={validation.page_count}"
            )

        validation.load_page(0)
        validation.load_page(validation.page_count - 1)

    return expected_count


def upload_sample(
    s3_client: Any,
    local_pdf: Path,
    bucket: str,
    s3_key: str,
    book_id: str,
    book_version: str,
    start_page: int,
    end_page: int,
    sha256: str,
) -> None:
    s3_client.upload_file(
        str(local_pdf),
        bucket,
        s3_key,
        ExtraArgs={
            "ContentType": "application/pdf",
            "ServerSideEncryption": "AES256",
            "Metadata": {
                "book-id": book_id,
                "book-version": book_version,
                "artifact-type": "bda-sample-input",
                "source-start-page": str(start_page),
                "source-end-page": str(end_page),
                "sha256": sha256,
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a small PDF sample for Bedrock Data Automation."
        )
    )

    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--book-id", default=DEFAULT_BOOK_ID)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=5)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Recreate the sample even when a valid "
            "local sample already exists."
        ),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the sample PDF and metadata to Amazon S3.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    local_root = (
        Path("data")
        / "multimodal-output"
        / args.book_id
        / args.version
    )

    source_pdf = local_root / "source" / "textbook.pdf"

    sample_name = (
        f"kaveri-pages-"
        f"{args.start_page:04d}-{args.end_page:04d}.pdf"
    )

    sample_root = local_root / "bda-samples"
    output_pdf = sample_root / sample_name
    metadata_path = sample_root / f"{sample_name}.json"

    s3_key = (
        f"bda-input/grade-9/{args.book_id}/{args.version}/"
        f"samples/{sample_name}"
    )

    print("============================================")
    print("CREATING BDA SAMPLE")
    print("============================================")
    print(f"Source PDF:  {source_pdf}")
    print(f"Pages:       {args.start_page}-{args.end_page}")
    print(f"Output PDF:  {output_pdf}")
    print(f"Upload:      {args.upload}")
    print()

    page_count = create_sample_pdf(
        source_pdf=source_pdf,
        output_pdf=output_pdf,
        start_page=args.start_page,
        end_page=args.end_page,
        overwrite=args.overwrite,
    )

    file_size = output_pdf.stat().st_size
    sha256 = calculate_sha256(output_pdf)

    metadata = {
        "book_id": args.book_id,
        "book_version": args.version,
        "source_pdf": str(source_pdf),
        "source_start_page": args.start_page,
        "source_end_page": args.end_page,
        "sample_page_count": page_count,
        "sample_size_bytes": file_size,
        "sample_sha256": sha256,
        "local_sample_path": str(output_pdf),
        "sample_s3_uri": f"s3://{args.bucket}/{s3_key}",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if args.upload:
        s3_client = boto3.client(
            "s3",
            region_name=args.region,
        )

        print("Uploading sample PDF...")

        upload_sample(
            s3_client=s3_client,
            local_pdf=output_pdf,
            bucket=args.bucket,
            s3_key=s3_key,
            book_id=args.book_id,
            book_version=args.version,
            start_page=args.start_page,
            end_page=args.end_page,
            sha256=sha256,
        )

        metadata_s3_key = f"{s3_key}.json"

        s3_client.upload_file(
            str(metadata_path),
            args.bucket,
            metadata_s3_key,
            ExtraArgs={
                "ContentType": "application/json",
                "ServerSideEncryption": "AES256",
                "Metadata": {
                    "book-id": args.book_id,
                    "book-version": args.version,
                    "artifact-type": "bda-sample-metadata",
                },
            },
        )

    print()
    print("============================================")
    print("BDA SAMPLE COMPLETED")
    print("============================================")
    print(f"Sample pages:  {page_count}")
    print(f"Sample size:   {file_size:,} bytes")
    print(f"SHA-256:       {sha256}")
    print(f"Local PDF:     {output_pdf}")
    print(f"Metadata:      {metadata_path}")
    print(f"S3 URI:        s3://{args.bucket}/{s3_key}")

    if not args.upload:
        print("S3 status:     Not uploaded")
    else:
        print("S3 status:     Uploaded")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(
            f"BDA sample creation failed: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
