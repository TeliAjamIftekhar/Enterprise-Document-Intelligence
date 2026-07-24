import json
from pathlib import Path

import pytest

from scripts.upload_opensearch_bulk import (
    build_bulk_batches,
    build_upload_checkpoint,
    resolve_runtime_identity,
    sha256_file,
    validate_preparation,
    validate_upload_checkpoint,
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



def build_test_payload(
    tmp_path: Path,
    document_count: int = 4,
) -> tuple[Path, list[bytes]]:
    pairs: list[bytes] = []

    for number in range(
        1,
        document_count + 1,
    ):
        record_id = (
            f"record-{number}"
        )

        action = {
            "index": {
                "_index": (
                    "test-index"
                ),
                "_id": record_id,
            }
        }

        document = {
            "record_id": record_id,
            "book_id": "test-book",
            "book_version": "v1",
            "text": "x" * (
                40 + number
            ),
        }

        pair = (
            json.dumps(action)
            + "\n"
            + json.dumps(document)
            + "\n"
        ).encode("utf-8")

        pairs.append(pair)

    path = tmp_path / "many.ndjson"

    path.write_bytes(
        b"".join(pairs)
    )

    return path, pairs


def test_build_bulk_batches_preserves_pairs(
    tmp_path,
):
    bulk_path, pairs = (
        build_test_payload(
            tmp_path
        )
    )

    maximum_pair_size = max(
        len(pair)
        for pair in pairs
    )

    batches = build_bulk_batches(
        bulk_path=bulk_path,
        max_batch_bytes=(
            maximum_pair_size + 1
        ),
    )

    assert len(batches) == len(
        pairs
    )

    assert sum(
        batch["document_count"]
        for batch in batches
    ) == len(pairs)

    assert b"".join(
        batch["body"]
        for batch in batches
    ) == bulk_path.read_bytes()

    assert all(
        batch["size_bytes"]
        <= maximum_pair_size + 1
        for batch in batches
    )


def test_rejects_oversized_bulk_pair(
    tmp_path,
):
    bulk_path, pairs = (
        build_test_payload(
            tmp_path,
            document_count=1,
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "single bulk action/document "
            "pair exceeds"
        ),
    ):
        build_bulk_batches(
            bulk_path=bulk_path,
            max_batch_bytes=(
                len(pairs[0]) - 1
            ),
        )


def test_validates_matching_upload_checkpoint(
    tmp_path,
):
    bulk_path, _ = (
        build_test_payload(
            tmp_path
        )
    )

    batches = build_bulk_batches(
        bulk_path=bulk_path,
        max_batch_bytes=300,
    )

    runtime = {
        "region": "us-east-1",
        "collection_endpoint": (
            "https://example.invalid"
        ),
        "index_name": "test-index",
        "book_id": "test-book",
        "book_version": "v1",
        "vector_dimensions": 1024,
    }

    bulk_sha256 = sha256_file(
        bulk_path
    )

    checkpoint = (
        build_upload_checkpoint(
            runtime=runtime,
            bulk_sha256=bulk_sha256,
            expected_document_count=4,
            max_batch_bytes=300,
            batches=batches,
            initial_count=0,
        )
    )

    validate_upload_checkpoint(
        checkpoint,
        runtime=runtime,
        bulk_sha256=bulk_sha256,
        expected_document_count=4,
        max_batch_bytes=300,
        batches=batches,
    )


def test_rejects_changed_checkpoint_plan(
    tmp_path,
):
    bulk_path, _ = (
        build_test_payload(
            tmp_path
        )
    )

    batches = build_bulk_batches(
        bulk_path=bulk_path,
        max_batch_bytes=300,
    )

    runtime = {
        "region": "us-east-1",
        "collection_endpoint": (
            "https://example.invalid"
        ),
        "index_name": "test-index",
        "book_id": "test-book",
        "book_version": "v1",
        "vector_dimensions": 1024,
    }

    bulk_sha256 = sha256_file(
        bulk_path
    )

    checkpoint = (
        build_upload_checkpoint(
            runtime=runtime,
            bulk_sha256=bulk_sha256,
            expected_document_count=4,
            max_batch_bytes=300,
            batches=batches,
            initial_count=0,
        )
    )

    checkpoint[
        "batching"
    ]["plan"][0]["sha256"] = "tampered"

    with pytest.raises(
        RuntimeError,
        match=(
            "batching plan does not match"
        ),
    ):
        validate_upload_checkpoint(
            checkpoint,
            runtime=runtime,
            bulk_sha256=bulk_sha256,
            expected_document_count=4,
            max_batch_bytes=300,
            batches=batches,
        )
