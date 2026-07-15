from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.book_config import load_book_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a textbook configuration "
            "and display its derived paths."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to the book configuration JSON.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional path for the validation "
            "report JSON."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_book_config(
        args.config
    )

    report = {
        "schema_version": "1.0",
        "status": "VALID",
        "config_path": str(args.config),
        "book": {
            "book_id": config.book.book_id,
            "title": config.book.title,
            "grade": config.book.grade,
            "subject": config.book.subject,
            "language": config.book.language,
            "board": config.book.board,
            "version": config.book.version,
            "page_count": (
                config.book.page_count
            ),
            "status": config.book.status,
        },
        "aws": {
            "region": config.aws.region,
            "bucket": config.aws.bucket,
        },
        "bda": {
            "project_arn": (
                config.bda.project_arn
            ),
            "profile_arn": (
                config.bda.profile_arn
            ),
            "stage": config.bda.stage,
        },
        "opensearch": {
            "collection_endpoint": (
                config.opensearch
                .collection_endpoint
            ),
            "index_name": (
                config.opensearch.index_name
            ),
            "vector_field": (
                config.opensearch.vector_field
            ),
        },
        "storage": {
            "source_s3_uri": (
                config.source_s3_uri
            ),
            "local_root": str(
                config.local_root
            ),
            "source_pdf_path": str(
                config.source_pdf_path
            ),
            "full_book_root": str(
                config.full_book_root
            ),
            "batch_manifest_path": str(
                config.batch_manifest_path
            ),
            "jobs_dir": str(
                config.jobs_dir
            ),
            "results_root": str(
                config.results_root
            ),
        },
        "models": {
            "embedding_model_id": (
                config.models.embedding.model_id
            ),
            "embedding_dimensions": (
                config.models.embedding.dimensions
            ),
            "generation_model_id": (
                config.models.generation.model_id
            ),
        },
        "processing": {
            "page_batch_size": (
                config.processing.page_batch_size
            ),
            "expected_batch_count": (
                (
                    config.book.page_count
                    + config.processing.page_batch_size
                    - 1
                )
                // config.processing.page_batch_size
            ),
        },
        "aws_calls": 0,
        "file_changes": (
            1 if args.output else 0
        ),
    }

    if args.output:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    print("=" * 68)
    print("BOOK CONFIGURATION VALIDATION")
    print("=" * 68)
    print("Status:            VALID")
    print(
        f"Book ID:           "
        f"{config.book.book_id}"
    )
    print(
        f"Title:             "
        f"{config.book.title}"
    )
    print(
        f"Grade/subject:     "
        f"{config.book.grade} / "
        f"{config.book.subject}"
    )
    print(
        f"Language:          "
        f"{config.book.language}"
    )
    print(
        f"Version:           "
        f"{config.book.version}"
    )
    print(
        f"Pages:             "
        f"{config.book.page_count}"
    )
    print(
        f"Batch size:        "
        f"{config.processing.page_batch_size}"
    )
    print(
        f"Expected batches:  "
        f"{report['processing']['expected_batch_count']}"
    )
    print(
        f"OpenSearch index:  "
        f"{config.opensearch.index_name}"
    )
    print(
        f"Source PDF:        "
        f"{config.source_s3_uri}"
    )
    print(
        f"Local root:        "
        f"{config.local_root}"
    )

    if args.output:
        print(
            f"Report:            "
            f"{args.output}"
        )

    print("AWS calls:         0")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as error:
        print(
            f"Book configuration is invalid: "
            f"{error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
