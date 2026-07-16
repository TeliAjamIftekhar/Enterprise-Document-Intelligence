from pathlib import Path

import pytest

from scripts.run_full_book_batches_sequentially import (
    build_bulk_command,
    build_titan_command,
    build_upload_command,
    manifest_batch_bounds,
    resolve_runner_runtime,
    validate_manifest_identity,
)


CONFIG_PATH = Path(
    "workers/multimodal-ingestion/"
    "config/books/"
    "grade-9-english-kaveri-"
    "v1-chapter-test.json"
)


def test_resolves_chapter_test_runtime():
    runtime = resolve_runner_runtime(
        CONFIG_PATH
    )

    assert runtime["mode"] == (
        "book_config"
    )

    assert runtime["book_version"] == (
        "v1-chapter-test"
    )

    assert runtime[
        "vector_dimensions"
    ] == 1024

    assert runtime[
        "manifest_path"
    ].endswith(
        "v1-chapter-test/full-book/"
        "full-book-batch-manifest.json"
    )

    assert runtime["start_batch"] is None
    assert runtime["end_batch"] is None


def test_preserves_legacy_runtime():
    runtime = resolve_runner_runtime(
        None
    )

    assert runtime["mode"] == "legacy"
    assert runtime["book_version"] == "v1"

    assert runtime["start_batch"] == (
        "batch-0003"
    )

    assert runtime["end_batch"] == (
        "batch-0015"
    )


def test_rejects_manifest_version_mismatch():
    runtime = resolve_runner_runtime(
        CONFIG_PATH
    )

    manifest = {
        "book_id": (
            "grade-9-english-kaveri"
        ),
        "book_version": "v1",
    }

    with pytest.raises(
        RuntimeError,
        match="book_version",
    ):
        validate_manifest_identity(
            manifest,
            runtime,
        )


def test_resolves_manifest_batch_bounds():
    manifest = {
        "batches": [
            {"batch_id": "batch-0015"},
            {"batch_id": "batch-0001"},
            {"batch_id": "batch-0008"},
        ]
    }

    assert manifest_batch_bounds(
        manifest
    ) == (
        "batch-0001",
        "batch-0015",
    )


def test_commands_propagate_config():
    bulk_command = build_bulk_command(
        embeddings_path=Path(
            "/tmp/embeddings.jsonl"
        ),
        bulk_dir=Path("/tmp/bulk"),
        config_path=CONFIG_PATH,
    )

    upload_command = build_upload_command(
        bulk_dir=Path("/tmp/bulk"),
        upload_dir=Path("/tmp/upload"),
        config_path=CONFIG_PATH,
    )

    assert "--config" in bulk_command
    assert str(CONFIG_PATH) in bulk_command

    assert "--config" in upload_command
    assert str(CONFIG_PATH) in upload_command


def test_titan_command_uses_runtime_dimensions():
    command = build_titan_command(
        records_path=Path(
            "/tmp/records.jsonl"
        ),
        titan_dir=Path("/tmp/titan"),
        vector_dimensions=768,
    )

    dimension_position = (
        command.index("--dimensions") + 1
    )

    assert command[
        dimension_position
    ] == "768"
