#!/usr/bin/env python3
"""Create page-level Surya fallback plan from normalized BDA output."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.ocr_fallback_planner import (
    load_normalized_records,
    plan_ocr_fallback,
    write_fallback_plan,
)


def parse_page_spec(value: str) -> tuple[int, ...]:
    pages: set[int] = set()

    for raw_token in value.split(","):
        token = raw_token.strip()

        if not token:
            continue

        if "-" in token:
            parts = token.split("-")

            if len(parts) != 2:
                raise argparse.ArgumentTypeError(
                    f"Invalid page range: {token}"
                )

            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"Invalid page range: {token}"
                ) from error

            if start <= 0 or end < start:
                raise argparse.ArgumentTypeError(
                    f"Invalid page range: {token}"
                )

            pages.update(
                range(start, end + 1)
            )

        else:
            try:
                page = int(token)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"Invalid page: {token}"
                ) from error

            if page <= 0:
                raise argparse.ArgumentTypeError(
                    "Pages must be positive"
                )

            pages.add(page)

    if not pages:
        raise argparse.ArgumentTypeError(
            "At least one page is required"
        )

    return tuple(sorted(pages))


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate normalized BDA page text and "
            "plan Surya OCR fallback pages."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Normalized JSON, JSONL or directory.",
    )

    parser.add_argument(
        "--expected-language",
        required=True,
    )

    parser.add_argument(
        "--expected-pages",
        type=parse_page_spec,
        default=None,
        help="Optional one-based page range, such as 1-120.",
    )

    parser.add_argument(
        "--canonical-pdf",
        type=Path,
        default=None,
        help=(
            "Canonical textbook PDF used only for "
            "approved text-layout recovery."
        ),
    )

    parser.add_argument(
        "--allow-native-text-recovery",
        action="store_true",
        help=(
            "Enable guarded native text recovery for "
            "a verified text-layout textbook."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    records = load_normalized_records(
        args.input
    )

    plan = plan_ocr_fallback(
        records,
        expected_language=args.expected_language,
        expected_pages=args.expected_pages,
        canonical_pdf_path=args.canonical_pdf,
        allow_native_text_recovery=(
            args.allow_native_text_recovery
        ),
    )

    write_fallback_plan(
        plan,
        args.output,
    )

    print("=" * 80)
    print("OCR FALLBACK PAGE PLAN")
    print("=" * 80)
    print("Input records: ", len(records))
    print("Language:      ", args.expected_language)
    print("Classification:", plan.classification)
    print("Expected:      ", len(plan.expected_pages))
    print("Discovered:    ", len(plan.discovered_pages))
    print("BDA accepted:  ", len(plan.accepted_bda_pages))
    print("Fallback:      ", len(plan.fallback_pages))
    print("Review:        ", len(plan.review_pages))
    print("Failed:        ", len(plan.failed_pages))
    print("Missing:       ", len(plan.missing_pages))
    print(
        "Native recovered:",
        len(plan.canonical_recovered_pages),
    )
    print("Fallback pages:", list(plan.fallback_pages))
    print("Plan:          ", args.output)
    print("AWS API calls: 0")

    if plan.classification == "INVALID_INPUT":
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
