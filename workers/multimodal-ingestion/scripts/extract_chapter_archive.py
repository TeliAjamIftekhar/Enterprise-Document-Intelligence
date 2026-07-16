from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.book_config import (
    load_book_config,
)
from src.chapter_manifest import (
    load_chapter_manifest,
    validate_manifest_for_book_config,
)
from src.chapter_source import (
    extract_chapter_archive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely extract and validate a "
            "chapter-wise textbook ZIP archive."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help=(
            "Path to the chapter-folder "
            "book configuration JSON."
        ),
    )

    parser.add_argument(
        "--archive",
        required=True,
        type=Path,
        help=(
            "Path to the downloaded source "
            "ZIP archive."
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "Optional extraction report path."
        ),
    )

    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Replace an existing extraction "
            "directory."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_book_config(
        args.config
    )

    if (
        config.source.mode
        != "chapter_folder"
    ):
        raise ValueError(
            "Book source mode must be "
            "chapter_folder."
        )

    if not config.source.chapter_manifest:
        raise ValueError(
            "Book configuration does not "
            "define a chapter manifest."
        )

    manifest_path = Path(
        config.source.chapter_manifest
    )

    manifest = load_chapter_manifest(
        manifest_path
    )

    validate_manifest_for_book_config(
        manifest,
        config,
    )

    target_directory = (
        config.chapter_directory_path
    )

    if target_directory is None:
        raise ValueError(
            "Chapter directory is missing."
        )

    report_path = (
        args.report
        if args.report
        else (
            config.local_root
            / "source"
            / "chapter-extraction-report.json"
        )
    )

    report = extract_chapter_archive(
        args.archive,
        target_directory,
        manifest,
        report_path=report_path,
        replace=args.replace,
    )

    print("=" * 72)
    print("CHAPTER ARCHIVE EXTRACTION")
    print("=" * 72)
    print("Status:             VALID")
    print(
        f"Book ID:            "
        f"{manifest.book_id}"
    )
    print(
        f"Version:            "
        f"{manifest.book_version}"
    )
    print(
        f"Documents:          "
        f"{report['document_count']}"
    )
    print(
        f"Source pages:       "
        f"{report['source_page_count']}"
    )
    print(
        f"Target directory:   "
        f"{target_directory}"
    )
    print(
        f"Report:             "
        f"{report_path}"
    )
    print("AWS calls:          0")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as error:
        print(
            "Chapter archive extraction "
            f"failed: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
