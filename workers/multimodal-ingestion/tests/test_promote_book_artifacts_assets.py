from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(
    "workers/multimodal-ingestion/scripts/"
    "promote_book_artifacts.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "promote_book_artifacts_assets",
        SCRIPT_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def write_jsonl(
    path: Path,
    values: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "".join(
            json.dumps(value) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def test_preserves_asset_references(
    tmp_path: Path,
) -> None:
    module = load_module()

    source_root = tmp_path / "source-version"

    ready_root = (
        source_root
        / "full-book/bda-results/"
        "batch-0001/job/normalized/"
        "embedding-ready"
    )

    source_id = (
        "test-book:v1-source:bda:unit-1:"
        "chunk-0001"
    )

    local_asset = (
        "data/test-book/v1-source/"
        "assets/figure.png"
    )

    s3_asset = (
        "s3://test-bucket/test-book/"
        "v1-source/assets/figure.png"
    )

    record = {
        "record_id": source_id,
        "source_unit_id": (
            "test-book:v1-source:bda:unit-1"
        ),
        "book_id": "test-book",
        "book_version": "v1-source",
        "text": "Test record",
        "citation_label": "test-book, page 1",
        "context_citation_label": (
            "Chapter One (test-book, page 1)"
        ),
        "source_page_numbers": [1],
        "locations": [
            {
                "source_page_number": 1,
            }
        ],
        "chapter_id": "chapter-one",
        "chapter_ids": ["chapter-one"],
        "chapter_title": "Chapter One",
        "chapter_titles": ["Chapter One"],
        "asset_local_paths": [local_asset],
        "asset_s3_uris": [s3_asset],
    }

    embedding = {
        "record_id": source_id,
        "book_id": "test-book",
        "book_version": "v1-source",
        "embedding": [0.25] * 1024,
    }

    write_jsonl(
        ready_root / "embedding-records.jsonl",
        [record],
    )

    write_jsonl(
        (
            ready_root
            / "titan-text-v2/embeddings.jsonl"
        ),
        [embedding],
    )

    source_dir = source_root / "source"
    source_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        source_dir / "chapter-page-map.json"
    ).write_text(
        json.dumps(
            {
                "book_id": "test-book",
                "book_version": "v1-source",
                "canonical_page_count": 1,
                "pages": [
                    {
                        "canonical_page": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    (
        source_dir / "chapter-merge-report.json"
    ).write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "book_version": "v1-source",
                "page_count": 1,
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "promoted"

    report = module.promote(
        source_root=source_root,
        output_root=output_root,
        book_id="test-book",
        source_version="v1-source",
        target_version="v1",
        expected_count=1,
        expected_dimension=1024,
    )

    promoted_record = json.loads(
        (
            output_root
            / "embedding-records.jsonl"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        promoted_record["record_id"]
        == "test-book:v1:bda:unit-1:chunk-0001"
    )

    assert promoted_record[
        "source_unit_id"
    ] == "test-book:v1:bda:unit-1"

    assert promoted_record[
        "book_version"
    ] == "v1"

    assert promoted_record[
        "asset_local_paths"
    ] == [local_asset]

    assert promoted_record[
        "asset_s3_uris"
    ] == [s3_asset]

    assert report[
        "preserved_reference_fields"
    ] == [
        "asset_local_paths",
        "asset_s3_uris",
    ]

    assert report[
        "preserved_source_reference_count"
    ] == 2

    assert report[
        "source_version_strings_remaining"
    ] == 0

    second_report = module.promote(
        source_root=source_root,
        output_root=output_root,
        book_id="test-book",
        source_version="v1-source",
        target_version="v1",
        expected_count=1,
        expected_dimension=1024,
    )

    assert (
        second_report["record_sha256"]
        == report["record_sha256"]
    )
