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
from src.chapter_merge import (
    build_chapter_textbook,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a chapter-aware canonical "
            "textbook PDF and page map."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help=(
            "Path to the chapter-folder "
            "book configuration."
        ),
    )

    parser.add_argument(
        "--reference-pdf",
        type=Path,
        help=(
            "Optional completed PDF used only "
            "for comparison."
        ),
    )

    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Replace existing generated files."
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
            "Chapter manifest is missing."
        )

    manifest = load_chapter_manifest(
        config.source.chapter_manifest
    )

    validate_manifest_for_book_config(
        manifest,
        config,
    )

    source_directory = (
        config.chapter_directory_path
    )

    if source_directory is None:
        raise ValueError(
            "Chapter directory is missing."
        )

    page_map_path = (
        config.local_root
        / "source"
        / "chapter-page-map.json"
    )

    report_path = (
        config.local_root
        / "source"
        / "chapter-merge-report.json"
    )

    report = build_chapter_textbook(
        source_directory,
        config.source_pdf_path,
        page_map_path,
        report_path,
        manifest,
        reference_pdf_path=(
            args.reference_pdf
        ),
        replace=args.replace,
    )

    comparison = report.get(
        "reference_comparison"
    )

    print("=" * 72)
    print("CHAPTER-AWARE TEXTBOOK BUILD")
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
        f"Named chapters:     "
        f"{report['chapter_count']}"
    )
    print(
        f"Canonical pages:    "
        f"{report['canonical_page_count']}"
    )
    print(
        f"Source pages valid: "
        f"{report['source_equivalence']['matching_source_pages']}"
    )
    print(
        f"Output PDF:         "
        f"{config.source_pdf_path}"
    )
    print(
        f"Page map:           "
        f"{page_map_path}"
    )
    print(
        f"Report:             "
        f"{report_path}"
    )

    if comparison:
        print(
            "Reference exact:    "
            f"{comparison['exact_text_pages']}/"
            f"{comparison['page_count']}"
        )
        print(
            "Reference mean:     "
            f"{comparison['mean_text_similarity']}"
        )
        print(
            "Reference minimum:  "
            f"{comparison['minimum_text_similarity']}"
        )

    print("AWS calls:          0")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as error:
        print(
            "Chapter-aware textbook build "
            f"failed: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
