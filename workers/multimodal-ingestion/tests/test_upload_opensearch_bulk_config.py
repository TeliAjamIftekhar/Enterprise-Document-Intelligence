import json
from pathlib import Path

import pytest

from scripts.upload_opensearch_bulk import (
    resolve_runtime_identity,
    sha256_file,
    validate_preparation,
)


CONFIG_PATH = Path(
    "workers/multimodal-ingestion/"
    "config/books/"
    "grade-9-english-kaveri-"
    "v1-chapter-test.json"
)


def write_bulk(
    tmp_path: Path,
    *,
    index_name: str,
    book_version: str,
) -> Path:
    record_id = "record-1"

    action = {
        "index": {
            "_index": index_name,
            "_id": record_id,
        }
    }

    document = {
        "record_id": record_id,
        "book_id": (
            "grade-9-english-kaveri"
        ),
        "book_version": book_version,
    }

    path = tmp_path / "bulk.ndjson"

    path.write_text(
        json.dumps(action)
        + "\n"
        + json.dumps(document)
        + "\n",
        encoding="utf-8",
    )

    return path


def build_report(
    bulk_path: Path,
    runtime: dict,
) -> dict:
    return {
        "status": "PREPARED",
        "configuration": runtime,
        "index_name": runtime[
            "index_name"
        ],
        "validation": {
            "document_count": 1,
            "unique_document_ids": 1,
            "vector_dimensions": runtime[
                "vector_dimensions"
            ],
        },
        "output": {
            "sha256": sha256_file(
                bulk_path
            ),
        },
    }


def test_resolves_chapter_runtime():
    runtime = resolve_runtime_identity(
        CONFIG_PATH
    )

    assert runtime["index_name"] == (
        "grade-9-english-kaveri-"
        "v1-chapter-test"
    )

    assert runtime["book_version"] == (
        "v1-chapter-test"
    )

    assert runtime[
        "vector_dimensions"
    ] == 1024


def test_preserves_legacy_runtime():
    runtime = resolve_runtime_identity(
        None
    )

    assert runtime["index_name"] == (
        "grade-9-english-kaveri-v1"
    )

    assert runtime["book_version"] == "v1"


def test_accepts_isolated_chapter_payload(
    tmp_path,
):
    runtime = resolve_runtime_identity(
        CONFIG_PATH
    )

    bulk_path = write_bulk(
        tmp_path,
        index_name=runtime[
            "index_name"
        ],
        book_version=runtime[
            "book_version"
        ],
    )

    report = build_report(
        bulk_path,
        runtime,
    )

    result = validate_preparation(
        bulk_path=bulk_path,
        preparation_report=report,
        runtime=runtime,
    )

    assert result == 1


def test_rejects_old_index_target(
    tmp_path,
):
    runtime = resolve_runtime_identity(
        CONFIG_PATH
    )

    bulk_path = write_bulk(
        tmp_path,
        index_name=(
            "grade-9-english-kaveri-v1"
        ),
        book_version=runtime[
            "book_version"
        ],
    )

    report = build_report(
        bulk_path,
        runtime,
    )

    with pytest.raises(
        RuntimeError,
        match="target index mismatch",
    ):
        validate_preparation(
            bulk_path=bulk_path,
            preparation_report=report,
            runtime=runtime,
        )


def test_rejects_old_book_version(
    tmp_path,
):
    runtime = resolve_runtime_identity(
        CONFIG_PATH
    )

    bulk_path = write_bulk(
        tmp_path,
        index_name=runtime[
            "index_name"
        ],
        book_version="v1",
    )

    report = build_report(
        bulk_path,
        runtime,
    )

    with pytest.raises(
        RuntimeError,
        match="book_version mismatch",
    ):
        validate_preparation(
            bulk_path=bulk_path,
            preparation_report=report,
            runtime=runtime,
        )
