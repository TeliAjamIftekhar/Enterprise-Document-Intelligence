#!/usr/bin/env python3
"""Merge normalized BDA outputs with verified Surya OCR pages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.bda_surya_merge import (
    merge_bda_surya_outputs,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace fallback-page BDA text with "
            "verified Surya OCR while preserving "
            "BDA figures, tables and metadata."
        )
    )

    parser.add_argument(
        "--normalized-root",
        type=Path,
        action="append",
        required=True,
        help=(
            "Normalized BDA directory. Repeat for "
            "multiple batches."
        ),
    )

    parser.add_argument(
        "--ocr-plan",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--surya-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--page-map",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--source-pdf",
        default=None,
    )

    parser.add_argument(
        "--replace",
        action="store_true",
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    report = merge_bda_surya_outputs(
        normalized_roots=(
            args.normalized_root
        ),
        ocr_plan_path=args.ocr_plan,
        surya_report_path=(
            args.surya_report
        ),
        page_map_path=args.page_map,
        output_dir=args.output_dir,
        source_pdf=args.source_pdf,
        replace=args.replace,
    )

    print("=" * 80)
    print("BDA + SURYA UNIFIED MERGE")
    print("=" * 80)
    print("Status:               ", report.status)
    print("Book:                 ", report.book_id)
    print("Version:              ", report.book_version)
    print(
        "Fallback pages:       ",
        list(report.fallback_pages),
    )
    print(
        "Input content units:  ",
        report.input_content_units,
    )
    print(
        "Removed BDA text:     ",
        report.removed_bda_text_units,
    )
    print(
        "Created Surya units:  ",
        report.created_surya_text_units,
    )
    print(
        "Created native units: ",
        report.created_native_pdf_text_units,
    )
    print(
        "Preserved figures:    ",
        report.preserved_figures,
    )
    print(
        "Preserved tables:     ",
        report.preserved_tables,
    )
    print(
        "Output content units: ",
        report.output_content_units,
    )
    print(
        "Output:",
        args.output_dir,
    )
    print("AWS API calls: 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
