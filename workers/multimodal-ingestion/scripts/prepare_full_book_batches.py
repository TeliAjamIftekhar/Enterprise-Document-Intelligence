from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.book_config import load_book_config

import fitz
import numpy as np


DEFAULT_BUCKET = "edi-documents-ajam-2026"

BOOK_ID = "grade-9-english-kaveri"
BOOK_VERSION = "v1"
GRADE = "grade-9"

DEFAULT_BATCH_SIZE = 20
DEFAULT_EXPECTED_PAGES = 300

MINIMUM_TEXT_SIMILARITY = 0.999
MAXIMUM_MEAN_PIXEL_DIFFERENCE = 3.0
MAXIMUM_CHANGED_PIXEL_PERCENT = 10.0
PIXEL_CHANGE_THRESHOLD = 5
RENDER_SCALE = 1.5

DEFAULT_S3_PREFIX = (
    f"bda-input/{GRADE}/{BOOK_ID}/"
    f"{BOOK_VERSION}/full-book/batches"
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


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


def normalize_extracted_text(
    value: str,
) -> str:
    value = value.replace(
        "\u00a0",
        " ",
    )

    value = value.replace(
        "\u200b",
        "",
    )

    value = value.replace(
        "\ufeff",
        "",
    )

    # Remove non-printing C0 control characters.
    # Newlines and tabs are handled by whitespace
    # normalization below.
    value = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        "",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def document_normalized_text(
    document: fitz.Document,
    from_page: int,
    to_page: int,
) -> str:
    page_text: list[str] = []

    for page_index in range(
        from_page,
        to_page + 1,
    ):
        page = document.load_page(
            page_index
        )

        page_text.append(
            normalize_extracted_text(
                page.get_text("text")
            )
        )

    return "\n\f\n".join(
        page_text
    )


def document_text_digest(
    document: fitz.Document,
    from_page: int,
    to_page: int,
) -> str:
    return sha256_text(
        document_normalized_text(
            document=document,
            from_page=from_page,
            to_page=to_page,
        )
    )


def document_page_geometry(
    document: fitz.Document,
    from_page: int,
    to_page: int,
) -> list[dict[str, float]]:
    geometry: list[dict[str, float]] = []

    for page_index in range(
        from_page,
        to_page + 1,
    ):
        page = document.load_page(
            page_index
        )

        geometry.append(
            {
                "width": round(
                    float(page.rect.width),
                    4,
                ),
                "height": round(
                    float(page.rect.height),
                    4,
                ),
            }
        )

    return geometry


def render_page_array(
    page: fitz.Page,
) -> np.ndarray:
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(
            RENDER_SCALE,
            RENDER_SCALE,
        ),
        alpha=False,
        colorspace=fitz.csRGB,
    )

    return np.frombuffer(
        pixmap.samples,
        dtype=np.uint8,
    ).reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )


def compare_batch_fidelity(
    source_document: fitz.Document,
    batch_document: fitz.Document,
    source_start_index: int,
    source_end_index: int,
) -> dict[str, Any]:
    page_metrics: list[
        dict[str, Any]
    ] = []

    for batch_page_index, source_page_index in enumerate(
        range(
            source_start_index,
            source_end_index + 1,
        )
    ):
        source_page = source_document.load_page(
            source_page_index
        )

        batch_page = batch_document.load_page(
            batch_page_index
        )

        source_text = normalize_extracted_text(
            source_page.get_text("text")
        )

        batch_text = normalize_extracted_text(
            batch_page.get_text("text")
        )

        text_similarity = (
            difflib.SequenceMatcher(
                None,
                source_text,
                batch_text,
                autojunk=False,
            ).ratio()
        )

        source_geometry = {
            "width": round(
                float(source_page.rect.width),
                4,
            ),
            "height": round(
                float(source_page.rect.height),
                4,
            ),
        }

        batch_geometry = {
            "width": round(
                float(batch_page.rect.width),
                4,
            ),
            "height": round(
                float(batch_page.rect.height),
                4,
            ),
        }

        geometry_verified = (
            source_geometry
            == batch_geometry
        )

        source_image = render_page_array(
            source_page
        )

        batch_image = render_page_array(
            batch_page
        )

        image_shape_verified = (
            source_image.shape
            == batch_image.shape
        )

        if image_shape_verified:
            difference = np.abs(
                source_image.astype(
                    np.int16
                )
                - batch_image.astype(
                    np.int16
                )
            )

            mean_pixel_difference = float(
                difference.mean()
            )

            changed_pixel_percent = float(
                np.any(
                    difference
                    > PIXEL_CHANGE_THRESHOLD,
                    axis=2,
                ).mean()
                * 100.0
            )

        else:
            mean_pixel_difference = float(
                "inf"
            )

            changed_pixel_percent = float(
                "inf"
            )

        text_verified = (
            text_similarity
            >= MINIMUM_TEXT_SIMILARITY
        )

        visual_verified = (
            image_shape_verified
            and mean_pixel_difference
            <= MAXIMUM_MEAN_PIXEL_DIFFERENCE
            and changed_pixel_percent
            <= MAXIMUM_CHANGED_PIXEL_PERCENT
        )

        page_verified = (
            text_verified
            and geometry_verified
            and visual_verified
        )

        page_metrics.append(
            {
                "batch_page_number": (
                    batch_page_index + 1
                ),
                "source_page_number": (
                    source_page_index + 1
                ),
                "text_similarity": (
                    text_similarity
                ),
                "normalized_text_exact": (
                    source_text
                    == batch_text
                ),
                "source_geometry": (
                    source_geometry
                ),
                "batch_geometry": (
                    batch_geometry
                ),
                "geometry_verified": (
                    geometry_verified
                ),
                "image_shape_verified": (
                    image_shape_verified
                ),
                "mean_pixel_difference": (
                    mean_pixel_difference
                ),
                "changed_pixel_percent": (
                    changed_pixel_percent
                ),
                "text_verified": (
                    text_verified
                ),
                "visual_verified": (
                    visual_verified
                ),
                "verified": page_verified,
            }
        )

    failed_pages = [
        metric
        for metric in page_metrics
        if not metric["verified"]
    ]

    minimum_text_similarity = min(
        float(metric["text_similarity"])
        for metric in page_metrics
    )

    mean_text_similarity = sum(
        float(metric["text_similarity"])
        for metric in page_metrics
    ) / len(page_metrics)

    maximum_mean_pixel_difference = max(
        float(
            metric[
                "mean_pixel_difference"
            ]
        )
        for metric in page_metrics
    )

    maximum_changed_pixel_percent = max(
        float(
            metric[
                "changed_pixel_percent"
            ]
        )
        for metric in page_metrics
    )

    return {
        "page_count": len(
            page_metrics
        ),
        "minimum_text_similarity": (
            minimum_text_similarity
        ),
        "mean_text_similarity": (
            mean_text_similarity
        ),
        "maximum_mean_pixel_difference": (
            maximum_mean_pixel_difference
        ),
        "maximum_changed_pixel_percent": (
            maximum_changed_pixel_percent
        ),
        "normalized_text_exact_pages": sum(
            1
            for metric in page_metrics
            if metric[
                "normalized_text_exact"
            ]
        ),
        "text_verified": all(
            metric["text_verified"]
            for metric in page_metrics
        ),
        "geometry_verified": all(
            metric["geometry_verified"]
            for metric in page_metrics
        ),
        "visual_verified": all(
            metric["visual_verified"]
            for metric in page_metrics
        ),
        "fidelity_verified": (
            not failed_pages
        ),
        "failed_page_count": len(
            failed_pages
        ),
        "failed_pages": failed_pages,
        "page_metrics": page_metrics,
        "thresholds": {
            "minimum_text_similarity": (
                MINIMUM_TEXT_SIMILARITY
            ),
            "maximum_mean_pixel_difference": (
                MAXIMUM_MEAN_PIXEL_DIFFERENCE
            ),
            "maximum_changed_pixel_percent": (
                MAXIMUM_CHANGED_PIXEL_PERCENT
            ),
            "pixel_change_threshold": (
                PIXEL_CHANGE_THRESHOLD
            ),
            "render_scale": RENDER_SCALE,
        },
    }


def create_batch_pdf(
    source_document: fitz.Document,
    output_path: Path,
    source_start_index: int,
    source_end_index: int,
    batch_id: str,
    book_id: str,
    book_version: str,
) -> dict[str, Any]:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        ".pdf.tmp"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    batch_document = fitz.open()

    try:
        batch_document.insert_pdf(
            source_document,
            from_page=source_start_index,
            to_page=source_end_index,
        )

        batch_document.set_metadata(
            {
                "title": (
                    f"{book_id} {book_version} "
                    f"{batch_id}"
                ),
                "author": "",
                "subject": (
                    "Deterministic BDA ingestion batch"
                ),
                "keywords": (
                    f"{book_id}, {book_version}, "
                    f"{batch_id}"
                ),
                "creator": (
                    "Enterprise Document Intelligence"
                ),
                "producer": "PyMuPDF",
                "creationDate": "",
                "modDate": "",
                "trapped": "",
            }
        )

        # Avoid aggressive PDF cleanup because it can
        # unnecessarily rewrite page resources.
        batch_document.save(
            str(temporary_path),
            garbage=0,
            deflate=False,
            clean=False,
        )

    finally:
        batch_document.close()

    expected_page_count = (
        source_end_index
        - source_start_index
        + 1
    )

    source_text_sha256 = (
        document_text_digest(
            source_document,
            source_start_index,
            source_end_index,
        )
    )

    with fitz.open(
        str(temporary_path)
    ) as validation_document:
        actual_page_count = len(
            validation_document
        )

        if (
            actual_page_count
            != expected_page_count
        ):
            raise RuntimeError(
                f"{batch_id} page count mismatch: "
                f"expected={expected_page_count}, "
                f"actual={actual_page_count}"
            )

        batch_text_sha256 = (
            document_text_digest(
                validation_document,
                0,
                actual_page_count - 1,
            )
        )

        fidelity = compare_batch_fidelity(
            source_document=(
                source_document
            ),
            batch_document=(
                validation_document
            ),
            source_start_index=(
                source_start_index
            ),
            source_end_index=(
                source_end_index
            ),
        )

    if not fidelity[
        "fidelity_verified"
    ]:
        failure_summary = [
            {
                "source_page_number": (
                    item[
                        "source_page_number"
                    ]
                ),
                "text_similarity": (
                    item[
                        "text_similarity"
                    ]
                ),
                "mean_pixel_difference": (
                    item[
                        "mean_pixel_difference"
                    ]
                ),
                "changed_pixel_percent": (
                    item[
                        "changed_pixel_percent"
                    ]
                ),
                "text_verified": (
                    item[
                        "text_verified"
                    ]
                ),
                "geometry_verified": (
                    item[
                        "geometry_verified"
                    ]
                ),
                "visual_verified": (
                    item[
                        "visual_verified"
                    ]
                ),
            }
            for item in fidelity[
                "failed_pages"
            ]
        ]

        raise RuntimeError(
            f"{batch_id} fidelity verification "
            "failed: "
            + json.dumps(
                failure_summary,
                ensure_ascii=False,
            )
        )

    os.replace(
        temporary_path,
        output_path,
    )

    return {
        "page_count": expected_page_count,
        "size_bytes": (
            output_path.stat().st_size
        ),
        "sha256": sha256_file(
            output_path
        ),
        "page_text_sha256": (
            batch_text_sha256
        ),
        "source_text_sha256": (
            source_text_sha256
        ),
        "normalized_text_hash_exact": (
            source_text_sha256
            == batch_text_sha256
        ),
        "minimum_text_similarity": (
            fidelity[
                "minimum_text_similarity"
            ]
        ),
        "mean_text_similarity": (
            fidelity[
                "mean_text_similarity"
            ]
        ),
        "maximum_mean_pixel_difference": (
            fidelity[
                "maximum_mean_pixel_difference"
            ]
        ),
        "maximum_changed_pixel_percent": (
            fidelity[
                "maximum_changed_pixel_percent"
            ]
        ),
        "normalized_text_exact_pages": (
            fidelity[
                "normalized_text_exact_pages"
            ]
        ),
        "text_verified": (
            fidelity["text_verified"]
        ),
        "geometry_verified": (
            fidelity[
                "geometry_verified"
            ]
        ),
        "visual_verified": (
            fidelity[
                "visual_verified"
            ]
        ),
        "fidelity_verified": (
            fidelity[
                "fidelity_verified"
            ]
        ),
        "fidelity_thresholds": (
            fidelity["thresholds"]
        ),
        "page_fidelity": (
            fidelity["page_metrics"]
        ),
    }


def validate_page_ranges(
    batches: list[dict[str, Any]],
    expected_pages: int,
) -> dict[str, Any]:
    page_occurrences: dict[int, int] = {}

    for batch in batches:
        start_page = int(
            batch["source_page_start"]
        )

        end_page = int(
            batch["source_page_end"]
        )

        for page_number in range(
            start_page,
            end_page + 1,
        ):
            page_occurrences[page_number] = (
                page_occurrences.get(
                    page_number,
                    0,
                )
                + 1
            )

    missing_pages = [
        page_number
        for page_number in range(
            1,
            expected_pages + 1,
        )
        if page_number not in page_occurrences
    ]

    overlapping_pages = [
        page_number
        for page_number, count
        in sorted(
            page_occurrences.items()
        )
        if count > 1
    ]

    unexpected_pages = [
        page_number
        for page_number
        in sorted(page_occurrences)
        if (
            page_number < 1
            or page_number > expected_pages
        )
    ]

    contiguous = (
        not missing_pages
        and not overlapping_pages
        and not unexpected_pages
        and len(page_occurrences)
        == expected_pages
    )

    return {
        "contiguous": contiguous,
        "missing_pages": missing_pages,
        "overlapping_pages": (
            overlapping_pages
        ),
        "unexpected_pages": (
            unexpected_pages
        ),
        "unique_source_pages": len(
            page_occurrences
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the complete textbook into "
            "deterministic BDA ingestion batches."
        )
    )

    parser.add_argument(
        "source_pdf",
        nargs="?",
        type=Path,
        help=(
            "Legacy mode source PDF. Omit when "
            "--config is provided."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Book configuration JSON. In this "
            "mode all paths and identity values "
            "are derived from BookConfig."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Legacy mode batch output directory."
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Legacy mode batch manifest path."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help=(
            "Legacy mode page count per batch."
        ),
    )

    parser.add_argument(
        "--expected-pages",
        type=int,
        help=(
            "Legacy mode expected source pages."
        ),
    )

    parser.add_argument(
        "--s3-prefix",
        help=(
            "Legacy mode S3 key prefix."
        ),
    )

    return parser.parse_args()


def resolve_runtime_args(
    args: argparse.Namespace,
) -> argparse.Namespace:
    if args.config is not None:
        conflicts = {
            "source_pdf": args.source_pdf,
            "output_dir": args.output_dir,
            "manifest": args.manifest,
            "batch_size": args.batch_size,
            "expected_pages": (
                args.expected_pages
            ),
            "s3_prefix": args.s3_prefix,
        }

        supplied_conflicts = [
            name
            for name, value in conflicts.items()
            if value is not None
        ]

        if supplied_conflicts:
            raise ValueError(
                "--config cannot be combined "
                "with legacy batch arguments: "
                + ", ".join(
                    supplied_conflicts
                )
            )

        config = load_book_config(
            args.config
        )

        args.source_pdf = (
            config.source_pdf_path
        )
        args.output_dir = (
            config.full_book_root
            / "batches"
        )
        args.manifest = (
            config.batch_manifest_path
        )
        args.batch_size = (
            config.processing
            .page_batch_size
        )
        args.expected_pages = (
            config.book.page_count
        )
        args.s3_prefix = (
            config.storage
            .bda_input_prefix
            .rstrip("/")
            + "/full-book/batches"
        )
        args.bucket = config.aws.bucket
        args.book_id = (
            config.book.book_id
        )
        args.book_version = (
            config.book.version
        )
        args.grade = config.book.grade
        args.configuration_mode = (
            "book_config"
        )
        args.config_path = str(
            args.config
        )
        args.source_s3_uri = (
            config.source_s3_uri
        )

        return args

    missing_arguments = []

    if args.source_pdf is None:
        missing_arguments.append(
            "source_pdf"
        )

    if args.output_dir is None:
        missing_arguments.append(
            "--output-dir"
        )

    if args.manifest is None:
        missing_arguments.append(
            "--manifest"
        )

    if missing_arguments:
        raise ValueError(
            "Legacy mode requires: "
            + ", ".join(
                missing_arguments
            )
            + ". Alternatively provide "
            "--config."
        )

    if args.batch_size is None:
        args.batch_size = (
            DEFAULT_BATCH_SIZE
        )

    if args.expected_pages is None:
        args.expected_pages = (
            DEFAULT_EXPECTED_PAGES
        )

    if args.s3_prefix is None:
        args.s3_prefix = (
            DEFAULT_S3_PREFIX
        )

    args.bucket = DEFAULT_BUCKET
    args.book_id = BOOK_ID
    args.book_version = BOOK_VERSION
    args.grade = GRADE
    args.configuration_mode = "legacy"
    args.config_path = None
    args.source_s3_uri = None

    return args


def main() -> int:
    args = resolve_runtime_args(
        parse_args()
    )

    if not args.source_pdf.is_file():
        raise FileNotFoundError(
            f"Source PDF not found: "
            f"{args.source_pdf}"
        )

    if args.batch_size < 1:
        raise ValueError(
            "Batch size must be at least 1."
        )

    if args.batch_size > 20:
        raise ValueError(
            "Batch size must not exceed 20 pages."
        )

    if args.expected_pages < 1:
        raise ValueError(
            "Expected pages must be positive."
        )

    source_size_bytes = (
        args.source_pdf.stat().st_size
    )

    source_sha256 = sha256_file(
        args.source_pdf
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 44)
    print("FULL BOOK BATCH PREPARATION")
    print("=" * 44)
    print(
        f"Book:           {args.book_id}"
    )
    print(
        f"Version:        "
        f"{args.book_version}"
    )
    print(f"Source:         {args.source_pdf}")
    print(
        f"Source size:    "
        f"{source_size_bytes:,} bytes"
    )
    print(f"Source SHA256:  {source_sha256}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Expected pages: {args.expected_pages}")
    print()

    batches: list[dict[str, Any]] = []

    with fitz.open(
        str(args.source_pdf)
    ) as source_document:
        actual_page_count = len(
            source_document
        )

        if (
            actual_page_count
            != args.expected_pages
        ):
            raise RuntimeError(
                "Source page count mismatch: "
                f"expected={args.expected_pages}, "
                f"actual={actual_page_count}"
            )

        batch_number = 0

        for source_start_index in range(
            0,
            actual_page_count,
            args.batch_size,
        ):
            batch_number += 1

            source_end_index = min(
                source_start_index
                + args.batch_size
                - 1,
                actual_page_count - 1,
            )

            source_page_start = (
                source_start_index + 1
            )

            source_page_end = (
                source_end_index + 1
            )

            batch_id = (
                f"batch-{batch_number:04d}"
            )

            filename = (
                f"{batch_id}-pages-"
                f"{source_page_start:04d}-"
                f"{source_page_end:04d}.pdf"
            )

            output_path = (
                args.output_dir
                / filename
            )

            print(
                f"[{batch_number:02d}] "
                f"Pages "
                f"{source_page_start:04d}-"
                f"{source_page_end:04d}"
            )

            result = create_batch_pdf(
                source_document=(
                    source_document
                ),
                output_path=output_path,
                source_start_index=(
                    source_start_index
                ),
                source_end_index=(
                    source_end_index
                ),
                batch_id=batch_id,
                book_id=args.book_id,
                book_version=(
                    args.book_version
                ),
            )

            s3_key = (
                args.s3_prefix.rstrip("/")
                + "/"
                + filename
            )

            batches.append(
                {
                    "batch_id": batch_id,
                    "batch_number": (
                        batch_number
                    ),
                    "filename": filename,
                    "local_path": str(
                        output_path
                    ),
                    "s3_key": s3_key,
                    "s3_uri": (
                        f"s3://{args.bucket}/"
                        f"{s3_key}"
                    ),
                    "source_page_start": (
                        source_page_start
                    ),
                    "source_page_end": (
                        source_page_end
                    ),
                    "source_page_offset": (
                        source_start_index
                    ),
                    "batch_page_start": 1,
                    "batch_page_end": (
                        result["page_count"]
                    ),
                    **result,
                    "uploaded": False,
                    "bda_invoked": False,
                }
            )

            print(
                f"     VERIFIED | "
                f"{result['page_count']} pages | "
                f"{result['size_bytes']:,} bytes | "
                f"text="
                f"{result['minimum_text_similarity']:.6f} | "
                f"pixel="
                f"{result['maximum_mean_pixel_difference']:.4f}"
            )

    range_validation = (
        validate_page_ranges(
            batches=batches,
            expected_pages=(
                args.expected_pages
            ),
        )
    )

    if not range_validation[
        "contiguous"
    ]:
        raise RuntimeError(
            "Full-book page-range validation "
            "failed: "
            + json.dumps(
                range_validation
            )
        )

    expected_batch_count = (
        args.expected_pages
        + args.batch_size
        - 1
    ) // args.batch_size

    if (
        len(batches)
        != expected_batch_count
    ):
        raise RuntimeError(
            "Unexpected batch count: "
            f"expected={expected_batch_count}, "
            f"actual={len(batches)}"
        )

    total_batch_bytes = sum(
        int(batch["size_bytes"])
        for batch in batches
    )

    manifest = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": "PREPARED",
        "book_id": args.book_id,
        "book_version": (
            args.book_version
        ),
        "grade": args.grade,
        "configuration": {
            "mode": (
                args.configuration_mode
            ),
            "config_path": (
                args.config_path
            ),
        },
        "source": {
            "local_path": str(
                args.source_pdf
            ),
            "configured_s3_uri": (
                args.source_s3_uri
            ),
            "page_count": (
                args.expected_pages
            ),
            "size_bytes": (
                source_size_bytes
            ),
            "sha256": source_sha256,
        },
        "batching": {
            "strategy": (
                "fixed_contiguous_page_ranges"
            ),
            "batch_size_pages": (
                args.batch_size
            ),
            "batch_count": len(
                batches
            ),
            "source_page_formula": (
                "source_page_number = "
                "source_page_offset + "
                "batch_page_number"
            ),
            "s3_prefix": (
                f"s3://{args.bucket}/"
                f"{args.s3_prefix.rstrip('/')}/"
            ),
        },
        "validation": {
            **range_validation,
            "expected_pages": (
                args.expected_pages
            ),
            "expected_batch_count": (
                expected_batch_count
            ),
            "actual_batch_count": (
                len(batches)
            ),
            "all_batch_text_verified": all(
                batch["text_verified"]
                for batch in batches
            ),
            "all_geometry_verified": all(
                batch[
                    "geometry_verified"
                ]
                for batch in batches
            ),
            "all_visual_verified": all(
                batch[
                    "visual_verified"
                ]
                for batch in batches
            ),
            "all_fidelity_verified": all(
                batch[
                    "fidelity_verified"
                ]
                for batch in batches
            ),
            "minimum_text_similarity": min(
                float(
                    batch[
                        "minimum_text_similarity"
                    ]
                )
                for batch in batches
            ),
            "maximum_mean_pixel_difference": max(
                float(
                    batch[
                        "maximum_mean_pixel_difference"
                    ]
                )
                for batch in batches
            ),
            "maximum_changed_pixel_percent": max(
                float(
                    batch[
                        "maximum_changed_pixel_percent"
                    ]
                )
                for batch in batches
            ),
            "maximum_batch_pages": max(
                int(batch["page_count"])
                for batch in batches
            ),
            "minimum_batch_pages": min(
                int(batch["page_count"])
                for batch in batches
            ),
            "total_batch_bytes": (
                total_batch_bytes
            ),
            "errors": [],
        },
        "batches": batches,
        "uploaded": False,
        "bda_invoked": False,
        "aws_calls": 0,
    }

    atomic_write_json(
        args.manifest,
        manifest,
    )

    print()
    print("=" * 44)
    print("FULL BOOK BATCH RESULT")
    print("=" * 44)
    print("Status:            PREPARED")
    print(
        f"Source pages:      "
        f"{args.expected_pages}"
    )
    print(
        f"Batches:           "
        f"{len(batches)}"
    )
    print(
        f"Pages per batch:   "
        f"{args.batch_size}"
    )
    print(
        f"Missing pages:     "
        f"{len(range_validation['missing_pages'])}"
    )
    print(
        f"Overlapping pages: "
        f"{len(range_validation['overlapping_pages'])}"
    )
    print(
        f"Text verified:     "
        f"{len(batches)}/{len(batches)}"
    )
    print(
        f"Geometry verified: "
        f"{len(batches)}/{len(batches)}"
    )
    print(
        f"Visual verified:   "
        f"{len(batches)}/{len(batches)}"
    )
    print(
        f"Fidelity verified: "
        f"{len(batches)}/{len(batches)}"
    )
    print(
        "Minimum text sim: "
        f"{manifest['validation']['minimum_text_similarity']:.9f}"
    )
    print(
        "Max mean pixel:   "
        f"{manifest['validation']['maximum_mean_pixel_difference']:.6f}"
    )
    print(
        "Max changed px:   "
        f"{manifest['validation']['maximum_changed_pixel_percent']:.4f}%"
    )
    print(
        f"Batch bytes:       "
        f"{total_batch_bytes:,}"
    )
    print(f"Output directory:  {args.output_dir}")
    print(f"Manifest:          {args.manifest}")
    print("Uploaded:          False")
    print("BDA invoked:       False")
    print("AWS calls:         0")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as exc:
        print(
            f"Full-book batch preparation "
            f"failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
