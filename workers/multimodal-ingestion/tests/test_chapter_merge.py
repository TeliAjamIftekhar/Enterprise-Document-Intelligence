from pathlib import Path

import fitz
import pytest

from src.chapter_manifest import (
    ChapterManifest,
)
from src.chapter_merge import (
    build_chapter_textbook,
)


def create_pdf(
    path: Path,
    texts: list[str],
) -> None:
    document = fitz.open()

    for text in texts:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            text,
        )

    document.save(path)
    document.close()


def build_manifest() -> ChapterManifest:
    return ChapterManifest.model_validate({
        "schema_version": "1.0",
        "book_id": "grade-9-test-book",
        "book_version": "v1-test",
        "title": "Test Book",
        "ordering_strategy": "manifest",
        "source_archive": {
            "bucket": "example-book-bucket",
            "key": "books/test.zip",
            "archive_root": "test",
            "supplementary_assets": []
        },
        "canonical_layout": {
            "leading_blank_pages": 1,
            "source_document_pages": 3,
            "trailing_blank_pages": 1,
            "canonical_page_count": 5,
            "source_to_canonical_page_offset": 1
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
                "chapters": [
                    {
                        "chapter_id": "first-reading",
                        "chapter_title": "First Reading",
                        "source_start_page": 1,
                        "source_end_page": 1,
                        "canonical_start_page": 2,
                        "canonical_end_page": 2
                    },
                    {
                        "chapter_id": "second-reading",
                        "chapter_title": "Second Reading",
                        "source_start_page": 2,
                        "source_end_page": 2,
                        "canonical_start_page": 3,
                        "canonical_end_page": 3
                    }
                ]
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
                "chapters": []
            }
        ]
    })


def test_builds_canonical_pdf_and_page_map(
    tmp_path: Path,
):
    source_directory = (
        tmp_path / "source"
    )
    source_directory.mkdir()

    create_pdf(
        source_directory / "unit-1.pdf",
        ["First reading", "Second reading"],
    )
    create_pdf(
        source_directory / "appendix.pdf",
        ["Appendix text"],
    )

    output_pdf = tmp_path / "textbook.pdf"
    page_map = tmp_path / "page-map.json"
    report = tmp_path / "report.json"

    result = build_chapter_textbook(
        source_directory,
        output_pdf,
        page_map,
        report,
        build_manifest(),
    )

    assert result["status"] == "VALID"
    assert result["canonical_page_count"] == 5
    assert (
        result["source_equivalence"]
        ["matching_source_pages"]
        == 3
    )
    assert (
        result["source_equivalence"]
        ["matching_source_text_pages"]
        == 3
    )
    assert (
        result["source_equivalence"]
        ["matching_source_geometry_pages"]
        == 3
    )
    assert (
        result["source_equivalence"]
        ["matching_source_render_pages"]
        == 3
    )
    assert (
        result["source_equivalence"]
        ["mismatching_source_render_pages"]
        == 0
    )
    assert (
        result["source_equivalence"]
        ["render_validation_mode"]
        == "tolerance"
    )

    with fitz.open(output_pdf) as document:
        assert document.page_count == 5
        assert (
            document[0]
            .get_text("text")
            .strip()
            == ""
        )
        assert "First reading" in (
            document[1].get_text("text")
        )
        assert "Second reading" in (
            document[2].get_text("text")
        )
        assert "Appendix text" in (
            document[3].get_text("text")
        )
        assert (
            document[4]
            .get_text("text")
            .strip()
            == ""
        )

    assert page_map.is_file()
    assert report.is_file()


def test_rejects_existing_outputs(
    tmp_path: Path,
):
    source_directory = (
        tmp_path / "source"
    )
    source_directory.mkdir()

    create_pdf(
        source_directory / "unit-1.pdf",
        ["One", "Two"],
    )
    create_pdf(
        source_directory / "appendix.pdf",
        ["Appendix"],
    )

    output_pdf = tmp_path / "textbook.pdf"
    output_pdf.write_bytes(b"existing")

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        build_chapter_textbook(
            source_directory,
            output_pdf,
            tmp_path / "page-map.json",
            tmp_path / "report.json",
            build_manifest(),
        )


def test_rejects_missing_source_pdf(
    tmp_path: Path,
):
    source_directory = (
        tmp_path / "source"
    )
    source_directory.mkdir()

    create_pdf(
        source_directory / "unit-1.pdf",
        ["One", "Two"],
    )

    with pytest.raises(
        FileNotFoundError,
        match="appendix.pdf",
    ):
        build_chapter_textbook(
            source_directory,
            tmp_path / "textbook.pdf",
            tmp_path / "page-map.json",
            tmp_path / "report.json",
            build_manifest(),
        )
