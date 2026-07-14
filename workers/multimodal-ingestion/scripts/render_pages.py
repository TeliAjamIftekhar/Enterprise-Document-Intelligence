from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3

from src.page_renderer import render_pages, upload_page_artifacts


DEFAULT_BOOK_ID = "grade-9-english-kaveri"
DEFAULT_VERSION = "v1"
DEFAULT_BUCKET = "edi-documents-ajam-2026"
DEFAULT_REGION = "us-east-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render textbook PDF pages as versioned image artifacts."
    )

    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--book-id", default=DEFAULT_BOOK_ID)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload rendered images and metadata to S3.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerender local images that already exist.",
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

    pdf_path = local_root / "source" / "textbook.pdf"
    manifest_path = local_root / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Run register_book.py first."
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest["book_id"] != args.book_id:
        raise RuntimeError("Manifest book_id does not match the request.")

    if manifest["book_version"] != args.version:
        raise RuntimeError("Manifest book_version does not match the request.")

    derived_prefix = (
        f"derived-artifacts/grade-9/"
        f"{args.book_id}/{args.version}"
    )

    print("============================================")
    print("RENDERING TEXTBOOK PAGES")
    print("============================================")
    print(f"Book:          {args.book_id}")
    print(f"Version:       {args.version}")
    print(f"Pages:         {args.start_page}-{args.end_page}")
    print(f"DPI:           {args.dpi}")
    print(f"JPEG quality:  {args.jpeg_quality}")
    print(f"Upload to S3:  {args.upload}")
    print()

    artifacts = render_pages(
        pdf_path=pdf_path,
        output_root=local_root,
        bucket=args.bucket,
        source_pdf_uri=manifest["source_pdf_uri"],
        book_id=args.book_id,
        book_version=args.version,
        derived_prefix=derived_prefix,
        start_page=args.start_page,
        end_page=args.end_page,
        dpi=args.dpi,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )

    if args.upload:
        s3_client = boto3.client(
            "s3",
            region_name=args.region,
        )

        upload_page_artifacts(
            s3_client=s3_client,
            bucket=args.bucket,
            output_root=local_root,
            artifacts=artifacts,
        )

    total_size = sum(
        artifact.file_size_bytes
        for artifact in artifacts
    )

    print()
    print("============================================")
    print("PAGE RENDERING COMPLETED")
    print("============================================")
    print(f"Pages rendered: {len(artifacts)}")
    print(f"Total size:     {total_size:,} bytes")

    for artifact in artifacts:
        print(
            f"Page {artifact.page_number:04d}: "
            f"{artifact.pixel_width}x{artifact.pixel_height}, "
            f"{artifact.file_size_bytes:,} bytes"
        )

    if not args.upload:
        print()
        print("Artifacts were created locally but not uploaded.")
        print("Rerun with --upload after reviewing the images.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Page rendering failed: {exc}", file=sys.stderr)
        sys.exit(1)
