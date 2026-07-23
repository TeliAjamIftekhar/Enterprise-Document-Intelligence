from pathlib import Path

import fitz
import pytest

from scripts import prepare_full_book_batches as batches


def create_source_pdf(
    path: Path,
) -> None:
    document = fitz.open()

    for page_number in range(1, 4):
        page = document.new_page(
            width=595,
            height=842,
        )

        page.insert_text(
            (72, 90),
            f"Preserving batch test page {page_number}",
            fontsize=18,
        )

        page.draw_rect(
            fitz.Rect(
                70,
                120,
                300 + page_number,
                220 + page_number,
            ),
            width=2,
        )

    document.save(
        str(path),
        garbage=0,
        deflate=False,
        clean=False,
    )

    document.close()


def test_batch_pages_are_render_exact(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.pdf"
    output_path = tmp_path / "batch.pdf"

    create_source_pdf(
        source_path
    )

    with fitz.open(
        str(source_path)
    ) as source_document:
        result = batches.create_batch_pdf(
            source_document=source_document,
            output_path=output_path,
            source_start_index=1,
            source_end_index=2,
            batch_id="batch-0001",
            book_id="grade-6-test-book",
            book_version="v1",
        )

    assert output_path.is_file()
    assert not output_path.with_suffix(
        ".pdf.tmp"
    ).exists()

    assert result["page_count"] == 2
    assert result["text_verified"] is True
    assert result["geometry_verified"] is True
    assert result["visual_verified"] is True
    assert result["fidelity_verified"] is True

    assert (
        result[
            "maximum_mean_pixel_difference"
        ]
        == 0.0
    )

    assert (
        result[
            "maximum_changed_pixel_percent"
        ]
        == 0.0
    )


def test_failed_validation_removes_temporary_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.pdf"
    output_path = tmp_path / "batch.pdf"

    create_source_pdf(
        source_path
    )

    monkeypatch.setattr(
        batches,
        "compare_batch_fidelity",
        lambda **_: {
            "fidelity_verified": False,
            "failed_pages": [
                {
                    "source_page_number": 1,
                    "text_similarity": 1.0,
                    "mean_pixel_difference": 4.0,
                    "changed_pixel_percent": 11.0,
                    "text_verified": True,
                    "geometry_verified": True,
                    "visual_verified": False,
                }
            ],
        },
    )

    with fitz.open(
        str(source_path)
    ) as source_document:
        with pytest.raises(
            RuntimeError,
            match="fidelity verification failed",
        ):
            batches.create_batch_pdf(
                source_document=source_document,
                output_path=output_path,
                source_start_index=0,
                source_end_index=0,
                batch_id="batch-0001",
                book_id="grade-6-test-book",
                book_version="v1",
            )

    assert not output_path.exists()
    assert not output_path.with_suffix(
        ".pdf.tmp"
    ).exists()
