from pathlib import Path

import fitz
import pytest

import src.chapter_merge as chapter_merge
from src.chapter_manifest import ChapterManifest
from src.chapter_merge import (
    build_chapter_textbook,
    render_page_difference_metrics,
)


def create_pdf(
    path: Path,
    texts: list[str],
) -> None:
    document = fitz.open()

    for text in texts:
        page = document.new_page()
        page.insert_text((72, 72), text)

    document.save(path)
    document.close()


def manifest() -> ChapterManifest:
    return ChapterManifest.model_validate({
        "schema_version": "1.0",
        "book_id": "preserving-test",
        "book_version": "v1",
        "title": "Preserving Test",
        "ordering_strategy": "manifest",
        "source_archive": {
            "bucket": "example",
            "key": "example.zip",
            "archive_root": "example",
            "supplementary_assets": [],
        },
        "canonical_layout": {
            "leading_blank_pages": 1,
            "source_document_pages": 3,
            "trailing_blank_pages": 1,
            "canonical_page_count": 5,
            "source_to_canonical_page_offset": 1,
        },
        "documents": [
            {
                "order": 1,
                "document_id": "unit-1",
                "document_type": "unit",
                "unit_number": 1,
                "source_filename": "unit-1.pdf",
                "source_page_count": 2,
                "canonical_start_page": 2,
                "canonical_end_page": 3,
                "title": "Unit 1",
                "chapters": [],
            },
            {
                "order": 2,
                "document_id": "appendix",
                "document_type": "appendix",
                "unit_number": None,
                "source_filename": "appendix.pdf",
                "source_page_count": 1,
                "canonical_start_page": 4,
                "canonical_end_page": 4,
                "title": "Appendix",
                "chapters": [],
            },
        ],
    })


def prepare_sources(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()

    create_pdf(
        source / "unit-1.pdf",
        ["First page", "Second page"],
    )
    create_pdf(
        source / "appendix.pdf",
        ["Appendix"],
    )

    return source


def test_first_source_remains_pixel_identical(
    tmp_path: Path,
):
    source = prepare_sources(tmp_path)
    output = tmp_path / "textbook.pdf"

    result = build_chapter_textbook(
        source,
        output,
        tmp_path / "page-map.json",
        tmp_path / "report.json",
        manifest(),
    )

    assert result["status"] == "VALID"

    with fitz.open(
        source / "unit-1.pdf"
    ) as original:
        with fitz.open(output) as merged:
            metrics = (
                render_page_difference_metrics(
                    original[0],
                    merged[1],
                )
            )

    assert metrics[
        "maximum_channel_difference"
    ] == 0


def test_failed_validation_removes_temporary_pdf(
    tmp_path: Path,
    monkeypatch,
):
    source = prepare_sources(tmp_path)

    def fail_validation(*args, **kwargs):
        raise ValueError("forced validation failure")

    monkeypatch.setattr(
        chapter_merge,
        "validate_source_equivalence",
        fail_validation,
    )

    with pytest.raises(
        ValueError,
        match="forced validation failure",
    ):
        build_chapter_textbook(
            source,
            tmp_path / "textbook.pdf",
            tmp_path / "page-map.json",
            tmp_path / "report.json",
            manifest(),
        )

    assert list(
        tmp_path.glob(".textbook.*.tmp.pdf")
    ) == []
