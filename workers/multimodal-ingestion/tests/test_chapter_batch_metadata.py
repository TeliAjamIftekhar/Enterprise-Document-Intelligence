import json
from pathlib import Path

import pytest

from src.chapter_batch_metadata import (
    enrich_batch_manifest,
)


def write_json(
    path: Path,
    value: dict,
) -> None:
    path.write_text(
        json.dumps(value),
        encoding="utf-8",
    )


def build_batch_manifest() -> dict:
    return {
        "schema_version": "1.0",
        "book_id": "test-book",
        "book_version": "v1-test",
        "source": {
            "local_path": "source.pdf",
            "configured_s3_uri": (
                "s3://example/source.pdf"
            ),
            "page_count": 6
        },
        "batches": [
            {
                "batch_id": "batch-0001",
                "batch_number": 1,
                "local_path": "batch-1.pdf",
                "s3_uri": (
                    "s3://example/batch-1.pdf"
                ),
                "source_page_start": 1,
                "source_page_end": 3,
                "source_page_offset": 0,
                "page_count": 3
            },
            {
                "batch_id": "batch-0002",
                "batch_number": 2,
                "local_path": "batch-2.pdf",
                "s3_uri": (
                    "s3://example/batch-2.pdf"
                ),
                "source_page_start": 4,
                "source_page_end": 6,
                "source_page_offset": 3,
                "page_count": 3
            }
        ],
        "aws_calls": 0
    }


def page(
    number: int,
    *,
    page_type: str,
    document_id: str | None,
    chapter_id: str | None,
) -> dict:
    return {
        "canonical_page": number,
        "page_type": page_type,
        "document_order": (
            1 if document_id else None
        ),
        "document_id": document_id,
        "document_type": (
            "unit"
            if document_id
            else None
        ),
        "document_title": (
            "Unit 1"
            if document_id
            else None
        ),
        "source_filename": (
            "unit.pdf"
            if document_id
            else None
        ),
        "source_page": (
            number - 1
            if document_id
            else None
        ),
        "unit_number": (
            1 if document_id else None
        ),
        "chapter_id": chapter_id,
        "chapter_title": (
            chapter_id
            if chapter_id
            else None
        ),
        "chapter_page": (
            number - 1
            if chapter_id
            else None
        )
    }


def build_page_map() -> dict:
    return {
        "schema_version": "1.0",
        "book_id": "test-book",
        "book_version": "v1-test",
        "canonical_page_count": 6,
        "pages": [
            page(
                1,
                page_type="leading_blank",
                document_id=None,
                chapter_id=None,
            ),
            page(
                2,
                page_type="unit",
                document_id="unit-1",
                chapter_id="chapter-a",
            ),
            page(
                3,
                page_type="unit",
                document_id="unit-1",
                chapter_id="chapter-a",
            ),
            page(
                4,
                page_type="unit",
                document_id="unit-1",
                chapter_id="chapter-b",
            ),
            page(
                5,
                page_type="appendix",
                document_id="appendix",
                chapter_id=None,
            ),
            page(
                6,
                page_type="trailing_blank",
                document_id=None,
                chapter_id=None,
            )
        ]
    }


def test_enriches_batches_and_writes_sidecars(
    tmp_path: Path,
):
    manifest_path = (
        tmp_path / "manifest.json"
    )
    page_map_path = (
        tmp_path / "page-map.json"
    )
    metadata_directory = (
        tmp_path / "metadata"
    )

    write_json(
        manifest_path,
        build_batch_manifest(),
    )
    write_json(
        page_map_path,
        build_page_map(),
    )

    manifest = enrich_batch_manifest(
        manifest_path,
        page_map_path,
        metadata_directory,
    )

    summary = manifest[
        "chapter_metadata"
    ]

    assert summary[
        "page_context_count"
    ] == 6
    assert summary["span_count"] == 5
    assert summary[
        "document_page_count"
    ] == 4
    assert summary[
        "chapter_page_count"
    ] == 3
    assert summary[
        "blank_page_count"
    ] == 2
    assert summary[
        "unique_chapter_count"
    ] == 2

    first_batch = manifest[
        "batches"
    ][0]

    assert first_batch[
        "chapter_metadata"
    ]["span_count"] == 2

    sidecar_path = (
        metadata_directory
        / "batch-0001.json"
    )

    assert sidecar_path.is_file()

    sidecar = json.loads(
        sidecar_path.read_text(
            encoding="utf-8"
        )
    )

    assert sidecar[
        "source_start_page"
    ] == 1
    assert len(
        sidecar["page_contexts"]
    ) == 3
    assert sidecar[
        "page_contexts"
    ][1]["chapter_id"] == (
        "chapter-a"
    )
    assert sidecar["aws_calls"] == 0


def test_rejects_page_map_identity_mismatch(
    tmp_path: Path,
):
    manifest_path = (
        tmp_path / "manifest.json"
    )
    page_map_path = (
        tmp_path / "page-map.json"
    )

    page_map = build_page_map()
    page_map["book_version"] = (
        "wrong-version"
    )

    write_json(
        manifest_path,
        build_batch_manifest(),
    )
    write_json(
        page_map_path,
        page_map,
    )

    with pytest.raises(
        ValueError,
        match="book versions",
    ):
        enrich_batch_manifest(
            manifest_path,
            page_map_path,
            tmp_path / "metadata",
        )


def test_rejects_missing_page_context(
    tmp_path: Path,
):
    manifest_path = (
        tmp_path / "manifest.json"
    )
    page_map_path = (
        tmp_path / "page-map.json"
    )

    page_map = build_page_map()
    page_map["pages"] = (
        page_map["pages"][:-1]
    )

    write_json(
        manifest_path,
        build_batch_manifest(),
    )
    write_json(
        page_map_path,
        page_map,
    )

    with pytest.raises(
        ValueError,
        match="page count",
    ):
        enrich_batch_manifest(
            manifest_path,
            page_map_path,
            tmp_path / "metadata",
        )


def test_rejects_batch_coverage_gap(
    tmp_path: Path,
):
    manifest_path = (
        tmp_path / "manifest.json"
    )
    page_map_path = (
        tmp_path / "page-map.json"
    )

    manifest = build_batch_manifest()

    manifest["batches"][1][
        "source_page_start"
    ] = 5

    manifest["batches"][1][
        "source_page_offset"
    ] = 4

    manifest["batches"][1][
        "page_count"
    ] = 2

    write_json(
        manifest_path,
        manifest,
    )
    write_json(
        page_map_path,
        build_page_map(),
    )

    with pytest.raises(
        ValueError,
        match="contiguous",
    ):
        enrich_batch_manifest(
            manifest_path,
            page_map_path,
            tmp_path / "metadata",
        )
