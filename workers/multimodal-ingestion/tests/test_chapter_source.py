from pathlib import Path
from zipfile import ZipFile

import fitz
import pytest

from src.chapter_manifest import (
    ChapterManifest,
)
from src.chapter_source import (
    extract_chapter_archive,
)


def build_pdf_bytes(
    page_count: int,
) -> bytes:
    document = fitz.open()

    for page_number in range(
        1,
        page_count + 1,
    ):
        page = document.new_page()
        page.insert_text(
            (72, 72),
            f"Test page {page_number}",
        )

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def build_manifest() -> ChapterManifest:
    return ChapterManifest.model_validate({
        "schema_version": "1.0",
        "book_id": "grade-9-test-book",
        "book_version": "v1-test",
        "title": "Test Book",
        "ordering_strategy": "manifest",
        "source_archive": {
            "bucket": "example-book-bucket",
            "key": "books/test-book.zip",
            "archive_root": "test-book",
            "supplementary_assets": [
                "cover.png"
            ]
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
                        "chapter_id": "chapter-one",
                        "chapter_title": "Chapter One",
                        "source_start_page": 1,
                        "source_end_page": 2,
                        "canonical_start_page": 2,
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


def create_test_archive(
    path: Path,
    *,
    unit_pages: int = 2,
    include_unexpected: bool = False,
) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "test-book/unit-1.pdf",
            build_pdf_bytes(unit_pages),
        )
        archive.writestr(
            "test-book/appendix.pdf",
            build_pdf_bytes(1),
        )
        archive.writestr(
            "test-book/cover.png",
            b"test-image",
        )

        if include_unexpected:
            archive.writestr(
                "test-book/unexpected.txt",
                b"unexpected",
            )


def test_extracts_and_validates_archive(
    tmp_path: Path,
):
    archive_path = (
        tmp_path / "book.zip"
    )
    target_directory = (
        tmp_path / "extracted"
    )
    report_path = (
        tmp_path / "report.json"
    )

    create_test_archive(
        archive_path
    )

    report = extract_chapter_archive(
        archive_path,
        target_directory,
        build_manifest(),
        report_path=report_path,
    )

    assert report["status"] == "VALID"
    assert report["document_count"] == 2
    assert report["source_page_count"] == 3

    assert (
        target_directory
        / "unit-1.pdf"
    ).is_file()
    assert (
        target_directory
        / "appendix.pdf"
    ).is_file()
    assert (
        target_directory
        / "cover.png"
    ).is_file()
    assert report_path.is_file()


def test_rejects_unexpected_archive_file(
    tmp_path: Path,
):
    archive_path = (
        tmp_path / "book.zip"
    )

    create_test_archive(
        archive_path,
        include_unexpected=True,
    )

    with pytest.raises(
        ValueError,
        match="unexpected files",
    ):
        extract_chapter_archive(
            archive_path,
            tmp_path / "extracted",
            build_manifest(),
            report_path=(
                tmp_path / "report.json"
            ),
        )


def test_rejects_pdf_page_count_mismatch(
    tmp_path: Path,
):
    archive_path = (
        tmp_path / "book.zip"
    )

    create_test_archive(
        archive_path,
        unit_pages=1,
    )

    with pytest.raises(
        ValueError,
        match="page count mismatch",
    ):
        extract_chapter_archive(
            archive_path,
            tmp_path / "extracted",
            build_manifest(),
            report_path=(
                tmp_path / "report.json"
            ),
        )


def test_rejects_existing_target_directory(
    tmp_path: Path,
):
    archive_path = (
        tmp_path / "book.zip"
    )
    target_directory = (
        tmp_path / "extracted"
    )

    create_test_archive(
        archive_path
    )
    target_directory.mkdir()

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        extract_chapter_archive(
            archive_path,
            target_directory,
            build_manifest(),
            report_path=(
                tmp_path / "report.json"
            ),
        )


def test_rejects_unsafe_zip_path(
    tmp_path: Path,
):
    archive_path = (
        tmp_path / "unsafe.zip"
    )

    with ZipFile(
        archive_path,
        "w",
    ) as archive:
        archive.writestr(
            "../unsafe.txt",
            b"unsafe",
        )

    with pytest.raises(
        ValueError,
        match="unsafe parent path",
    ):
        extract_chapter_archive(
            archive_path,
            tmp_path / "extracted",
            build_manifest(),
            report_path=(
                tmp_path / "report.json"
            ),
        )



def build_root_level_manifest(
) -> ChapterManifest:
    return ChapterManifest.model_validate({
        "schema_version": "1.0",
        "book_id": "grade-1-root-book",
        "book_version": "v1-test",
        "title": "Root-Level Test Book",
        "ordering_strategy": "manifest",
        "source_archive": {
            "bucket": "example-book-bucket",
            "key": "books/root-book.zip",
            "archive_root": None,
            "supplementary_assets": []
        },
        "canonical_layout": {
            "leading_blank_pages": 0,
            "source_document_pages": 3,
            "trailing_blank_pages": 0,
            "canonical_page_count": 3,
            "source_to_canonical_page_offset": 0
        },
        "documents": [
            {
                "order": 1,
                "document_id": "front-matter",
                "document_type": "front_matter",
                "unit_number": None,
                "source_filename": "root-front.pdf",
                "source_page_count": 1,
                "canonical_start_page": 1,
                "canonical_end_page": 1,
                "title": "Front Matter",
                "chapters": []
            },
            {
                "order": 2,
                "document_id": "unit-1",
                "document_type": "unit",
                "unit_number": 1,
                "source_filename": "root-unit-1.pdf",
                "source_page_count": 2,
                "canonical_start_page": 2,
                "canonical_end_page": 3,
                "title": "Unit 1",
                "chapters": [
                    {
                        "chapter_id": "unit-one",
                        "chapter_title": "Unit One",
                        "source_start_page": 1,
                        "source_end_page": 2,
                        "canonical_start_page": 2,
                        "canonical_end_page": 3
                    }
                ]
            }
        ]
    })


def test_extracts_root_level_archive(
    tmp_path: Path,
):
    archive_path = (
        tmp_path / "root-book.zip"
    )

    with ZipFile(
        archive_path,
        "w",
    ) as archive:
        archive.writestr(
            "root-front.pdf",
            build_pdf_bytes(1),
        )
        archive.writestr(
            "root-unit-1.pdf",
            build_pdf_bytes(2),
        )

    target_directory = (
        tmp_path / "root-extracted"
    )

    report_path = (
        tmp_path / "root-report.json"
    )

    report = extract_chapter_archive(
        archive_path,
        target_directory,
        build_root_level_manifest(),
        report_path=report_path,
    )

    assert report["status"] == "VALID"
    assert report["document_count"] == 2
    assert report["source_page_count"] == 3

    assert (
        target_directory
        / "root-front.pdf"
    ).is_file()

    assert (
        target_directory
        / "root-unit-1.pdf"
    ).is_file()

    assert report_path.is_file()
