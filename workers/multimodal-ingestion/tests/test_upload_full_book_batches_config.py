from __future__ import annotations

import hashlib
from pathlib import Path

import fitz
import pytest

from scripts.upload_full_book_batches import (
    validate_manifest,
)


def create_pdf(
    path: Path,
    page_count: int,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = fitz.open()

    try:
        for _ in range(page_count):
            document.new_page()

        document.save(str(path))

    finally:
        document.close()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def build_manifest(
    tmp_path: Path,
    expected_pages: int,
    batch_count: int,
) -> dict:
    assert expected_pages % batch_count == 0

    pages_per_batch = (
        expected_pages // batch_count
    )

    batches = []
    next_page = 1

    for batch_number in range(
        1,
        batch_count + 1,
    ):
        batch_id = (
            f"batch-{batch_number:04d}"
        )

        page_start = next_page
        page_end = (
            page_start
            + pages_per_batch
            - 1
        )

        pdf_path = (
            tmp_path
            / "batches"
            / f"{batch_id}.pdf"
        )

        create_pdf(
            pdf_path,
            pages_per_batch,
        )

        batches.append({
            "batch_id": batch_id,
            "batch_number": batch_number,
            "source_page_start": page_start,
            "source_page_end": page_end,
            "page_count": pages_per_batch,
            "local_path": str(pdf_path),
            "size_bytes": (
                pdf_path.stat().st_size
            ),
            "sha256": sha256_file(
                pdf_path
            ),
            "s3_key": (
                "bda-input/test-book/"
                f"{batch_id}.pdf"
            ),
        })

        next_page = page_end + 1

    return {
        "status": "PREPARED",
        "validation": {
            "expected_pages": expected_pages,
            "expected_batch_count": (
                batch_count
            ),
            "actual_batch_count": (
                batch_count
            ),
            "contiguous": True,
            "missing_pages": [],
            "overlapping_pages": [],
            "all_batch_text_verified": True,
            "all_geometry_verified": True,
            "all_visual_verified": True,
            "all_fidelity_verified": True,
        },
        "batches": batches,
    }


@pytest.mark.parametrize(
    (
        "expected_pages",
        "batch_count",
    ),
    [
        (220, 11),
        (300, 15),
    ],
)
def test_validate_manifest_supports_dynamic_books(
    tmp_path: Path,
    expected_pages: int,
    batch_count: int,
) -> None:
    manifest = build_manifest(
        tmp_path,
        expected_pages,
        batch_count,
    )

    results = validate_manifest(
        manifest
    )

    assert len(results) == batch_count

    assert results[0][
        "source_page_start"
    ] == 1

    assert results[-1][
        "source_page_end"
    ] == expected_pages

    assert all(
        result["local_verified"] is True
        for result in results
    )


def test_rejects_expected_actual_count_mismatch(
    tmp_path: Path,
) -> None:
    manifest = build_manifest(
        tmp_path,
        expected_pages=20,
        batch_count=1,
    )

    manifest["validation"][
        "expected_batch_count"
    ] = 2

    with pytest.raises(
        RuntimeError,
        match=(
            "expected and actual batch "
            "counts differ"
        ),
    ):
        validate_manifest(manifest)


def test_rejects_actual_count_list_mismatch(
    tmp_path: Path,
) -> None:
    manifest = build_manifest(
        tmp_path,
        expected_pages=20,
        batch_count=1,
    )

    manifest["validation"][
        "expected_batch_count"
    ] = 2

    manifest["validation"][
        "actual_batch_count"
    ] = 2

    with pytest.raises(
        RuntimeError,
        match=(
            "actual batch count differs "
            "from the batches list"
        ),
    ):
        validate_manifest(manifest)


def test_rejects_final_page_mismatch(
    tmp_path: Path,
) -> None:
    manifest = build_manifest(
        tmp_path,
        expected_pages=20,
        batch_count=1,
    )

    manifest["validation"][
        "expected_pages"
    ] = 21

    with pytest.raises(
        RuntimeError,
        match="Final source page is not 21",
    ):
        validate_manifest(manifest)
