from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.book_config import (
    load_book_config,
)
from src.chapter_batch_metadata import (
    enrich_batch_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach page-level document and "
            "chapter metadata to full-book "
            "BDA batches."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Book configuration JSON.",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional batch manifest override."
        ),
    )

    parser.add_argument(
        "--page-map",
        type=Path,
        help=(
            "Optional chapter page-map override."
        ),
    )

    parser.add_argument(
        "--metadata-dir",
        type=Path,
        help=(
            "Optional per-batch metadata "
            "directory override."
        ),
    )

    parser.add_argument(
        "--output-manifest",
        type=Path,
        help=(
            "Optional enriched manifest output. "
            "The source manifest is updated "
            "in-place when omitted."
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
            "Chapter metadata requires "
            "chapter_folder source mode."
        )

    batch_manifest_path = (
        args.manifest
        if args.manifest
        else config.batch_manifest_path
    )

    page_map_path = (
        args.page_map
        if args.page_map
        else (
            config.local_root
            / "source"
            / "chapter-page-map.json"
        )
    )

    metadata_directory = (
        args.metadata_dir
        if args.metadata_dir
        else (
            config.full_book_root
            / "batch-metadata"
        )
    )

    manifest = enrich_batch_manifest(
        batch_manifest_path,
        page_map_path,
        metadata_directory,
        output_manifest_path=(
            args.output_manifest
        ),
    )

    chapter_metadata = manifest[
        "chapter_metadata"
    ]

    output_path = (
        args.output_manifest
        if args.output_manifest
        else batch_manifest_path
    )

    print("=" * 76)
    print("BATCH CHAPTER METADATA")
    print("=" * 76)
    print("Status:               ATTACHED")
    print(
        f"Book ID:              "
        f"{manifest['book_id']}"
    )
    print(
        f"Version:              "
        f"{manifest['book_version']}"
    )
    print(
        f"Batches:              "
        f"{len(manifest['batches'])}"
    )
    print(
        f"Page contexts:        "
        f"{chapter_metadata['page_context_count']}"
    )
    print(
        f"Chapter spans:        "
        f"{chapter_metadata['span_count']}"
    )
    print(
        f"Document pages:       "
        f"{chapter_metadata['document_page_count']}"
    )
    print(
        f"Named chapter pages:  "
        f"{chapter_metadata['chapter_page_count']}"
    )
    print(
        f"Blank pages:          "
        f"{chapter_metadata['blank_page_count']}"
    )
    print(
        f"Unique chapters:      "
        f"{chapter_metadata['unique_chapter_count']}"
    )
    print(
        f"Batch metadata files: "
        f"{chapter_metadata['batch_metadata_count']}"
    )
    print(
        f"Metadata directory:   "
        f"{metadata_directory}"
    )
    print(
        f"Manifest:             "
        f"{output_path}"
    )
    print("AWS calls:            0")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as error:
        print(
            "Batch chapter metadata failed: "
            f"{error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
