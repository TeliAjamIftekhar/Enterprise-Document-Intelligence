from pathlib import Path
from zipfile import ZipFile

import fitz

from generate_book_config_from_inspection import (
    extract_numbered_toc_entries,
)


def test_extract_numbered_toc_entries(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "front.pdf"
    zip_path = tmp_path / "source.zip"

    document = fitz.open()
    page = document.new_page(
        width=600,
        height=800,
    )

    page.insert_text(
        (55, 120),
        "1.",
        fontsize=22,
        fontname="hebo",
    )
    page.insert_text(
        (95, 120),
        "First Lesson",
        fontsize=22,
        fontname="hebo",
    )
    page.insert_text(
        (230, 120),
        "10",
        fontsize=22,
        fontname="hebo",
    )

    page.insert_text(
        (55, 200),
        "2.",
        fontsize=22,
        fontname="hebo",
    )
    page.insert_text(
        (95, 200),
        "Second Lesson",
        fontsize=22,
        fontname="hebo",
    )
    page.insert_text(
        (95, 226),
        "Continued",
        fontsize=22,
        fontname="hebo",
    )
    page.insert_text(
        (230, 226),
        "20",
        fontsize=22,
        fontname="hebo",
    )

    page.insert_text(
        (55, 300),
        "3. Third Lesson",
        fontsize=22,
        fontname="hebo",
    )
    page.insert_text(
        (230, 300),
        "30",
        fontsize=22,
        fontname="hebo",
    )

    document.save(pdf_path)
    document.close()

    with ZipFile(
        zip_path,
        "w",
    ) as archive:
        archive.write(
            pdf_path,
            arcname="front.pdf",
        )

    with ZipFile(
        zip_path,
        "r",
    ) as archive:
        entries = (
            extract_numbered_toc_entries(
                archive,
                "front.pdf",
            )
        )

    assert entries == {
        1: "First Lesson",
        2: "Second Lesson Continued",
        3: "Third Lesson",
    }
