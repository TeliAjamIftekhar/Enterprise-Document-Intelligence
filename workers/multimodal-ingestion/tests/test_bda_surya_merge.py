import copy
import json
from pathlib import Path

import pytest

from src.bda_surya_merge import (
    build_surya_content_unit,
    deterministic_surya_unit_id,
    load_verified_surya_pages,
    merge_records,
    record_page_numbers,
)


def page_context(
    page: int,
    *,
    chapter_id: str | None = None,
    chapter_title: str | None = None,
) -> dict:
    return {
        "canonical_page": page,
        "page_type": (
            "chapter"
            if chapter_id
            else "front_matter"
        ),
        "document_order": 1,
        "document_id": (
            chapter_id or "front-matter"
        ),
        "document_type": (
            "chapter"
            if chapter_id
            else "front_matter"
        ),
        "document_title": (
            chapter_title
            or "Shahnai Front Matter"
        ),
        "source_filename": "aush1ps.pdf",
        "source_page": page,
        "unit_number": None,
        "chapter_id": chapter_id,
        "chapter_title": chapter_title,
        "chapter_page": (
            1 if chapter_id else None
        ),
    }


def bda_text(
    unit_id: str,
    page: int,
    text: str,
) -> dict:
    return {
        "schema_version": "1.1",
        "unit_id": unit_id,
        "book_id": "grade-1-urdu-test",
        "book_version": "v1",
        "source_kind": "bda_standard_output",
        "source_pdf": "/tmp/textbook.pdf",
        "bda_element_id": unit_id,
        "element_index": page,
        "element_type": "TEXT",
        "element_sub_type": "PARAGRAPH",
        "modality": "paragraph",
        "reading_order": 1,
        "source_page_numbers": [page],
        "locations": [],
        "raw_text": text,
        "markdown": text,
        "search_text": text,
        "asset_s3_uris": [],
        "asset_local_paths": [],
        "quality_flags": [],
    }


def surya_page(
    page: int,
    text: str,
) -> dict:
    return {
        "page_key": f"page-{page:04d}",
        "canonical_page": page,
        "clean_text": text,
        "confidence": 0.98,
        "decision": {
            "classification": "PASS",
            "accepted": True,
            "fallback_recommended": False,
            "sparse_page": False,
            "expected_language": "urdu",
            "expected_script": "arabic",
            "source": "surya",
            "reasons": [
                "expected_script_quality_gate_passed"
            ],
            "metrics": {
                "expected_script": "arabic",
                "expected_script_ratio": 0.95,
            },
            "clean_text": text,
        },
    }


def test_record_page_numbers_uses_resolved_context() -> None:
    record = {
        "resolved_page_contexts": [
            {
                "canonical_page": 17,
            }
        ]
    }

    assert record_page_numbers(record) == (17,)


def test_deterministic_surya_id() -> None:
    first = deterministic_surya_unit_id(
        book_id="book",
        book_version="v1",
        canonical_page=5,
        clean_text="اردو عبارت",
    )

    second = deterministic_surya_unit_id(
        book_id="book",
        book_version="v1",
        canonical_page=5,
        clean_text="اردو عبارت",
    )

    assert first == second
    assert ":surya:page-0005:" in first


def test_surya_unit_contains_chapter_metadata() -> None:
    unit = build_surya_content_unit(
        book_id="grade-1-urdu-test",
        book_version="v1",
        source_pdf="/tmp/textbook.pdf",
        canonical_page=17,
        page_context=page_context(
            17,
            chapter_id="chapter-01",
            chapter_title="پہلا سبق",
        ),
        surya_page=surya_page(
            17,
            "یہ ایک مکمل اردو عبارت ہے۔",
        ),
    )

    assert unit["text_source"] == "surya"
    assert unit["source_page_numbers"] == [17]
    assert unit["chapter_id"] == "chapter-01"
    assert unit["chapter_title"] == "پہلا سبق"
    assert unit["document_id"] == "chapter-01"
    assert (
        "surya_ocr_fallback_verified"
        in unit["quality_flags"]
    )


def test_merge_replaces_only_fallback_text() -> None:
    page_1 = bda_text(
        "unit-1",
        1,
        "Valid BDA text",
    )

    page_2 = bda_text(
        "unit-2",
        2,
        "Corrupted Latin output",
    )

    plan = {
        "fallback_pages": (2,),
        "accepted_bda_pages": (1,),
        "assessments": [],
    }

    (
        output,
        figures,
        tables,
        stats,
    ) = merge_records(
        content_units=[
            page_1,
            page_2,
        ],
        figures=[],
        tables=[],
        plan=plan,
        surya_pages={
            2: surya_page(
                2,
                "یہ درست اردو عبارت ہے۔",
            )
        },
        page_lookup={
            1: page_context(1),
            2: page_context(
                2,
                chapter_id="chapter-01",
                chapter_title="سبق",
            ),
        },
        book_id="grade-1-urdu-test",
        book_version="v1",
        source_pdf="/tmp/textbook.pdf",
    )

    assert figures == []
    assert tables == []
    assert stats[
        "removed_bda_text_units"
    ] == 1
    assert stats[
        "created_surya_text_units"
    ] == 1

    assert len(output) == 2

    bda_units = [
        record
        for record in output
        if record["text_source"] == "bda"
    ]

    surya_units = [
        record
        for record in output
        if record["text_source"] == "surya"
    ]

    assert len(bda_units) == 1
    assert bda_units[0]["unit_id"] == "unit-1"
    assert len(surya_units) == 1
    assert (
        surya_units[0]["search_text"]
        == "یہ درست اردو عبارت ہے۔"
    )


def test_figures_and_tables_are_preserved_exactly() -> None:
    figure = {
        "figure_id": "figure-1",
        "source_page_numbers": [2],
        "crop_s3_uris": [
            "s3://bucket/figure.png"
        ],
        "crop_local_paths": [
            "/tmp/figure.png"
        ],
    }

    table = {
        "table_id": "table-1",
        "source_page_numbers": [2],
        "markdown": "| A | B |",
    }

    original_figure = copy.deepcopy(
        figure
    )

    original_table = copy.deepcopy(
        table
    )

    (
        _,
        figures,
        tables,
        stats,
    ) = merge_records(
        content_units=[
            bda_text(
                "unit-2",
                2,
                "Corrupted output",
            )
        ],
        figures=[figure],
        tables=[table],
        plan={
            "fallback_pages": (2,),
            "accepted_bda_pages": (),
            "assessments": [],
        },
        surya_pages={
            2: surya_page(
                2,
                "درست اردو عبارت",
            )
        },
        page_lookup={
            2: page_context(2),
        },
        book_id="grade-1-urdu-test",
        book_version="v1",
        source_pdf="/tmp/textbook.pdf",
    )

    assert figures == [original_figure]
    assert tables == [original_table]
    assert stats["preserved_figures"] == 1
    assert stats["preserved_tables"] == 1


def test_cross_page_text_unit_is_rejected() -> None:
    unit = bda_text(
        "multi-page",
        1,
        "Text spanning pages",
    )

    unit["source_page_numbers"] = [
        1,
        2,
    ]

    with pytest.raises(
        ValueError,
        match="spans fallback and non-fallback",
    ):
        merge_records(
            content_units=[unit],
            figures=[],
            tables=[],
            plan={
                "fallback_pages": (2,),
                "accepted_bda_pages": (1,),
                "assessments": [],
            },
            surya_pages={
                2: surya_page(
                    2,
                    "اردو",
                )
            },
            page_lookup={
                1: page_context(1),
                2: page_context(2),
            },
            book_id="grade-1-urdu-test",
            book_version="v1",
            source_pdf="/tmp/textbook.pdf",
        )


def test_unverified_surya_report_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"

    path.write_text(
        json.dumps(
            {
                "classification": "FAIL",
                "accepted_for_pipeline": False,
                "pages": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="not PASS",
    ):
        load_verified_surya_pages(
            path,
            fallback_pages=(1,),
        )



def test_native_pdf_recovery_injects_only_missing_bda_text() -> None:
    existing_bda = bda_text(
        "unit-page-1",
        1,
        "Existing valid BDA mathematics text.",
    )

    plan = {
        "fallback_pages": (),
        "accepted_bda_pages": (1, 2),
        "canonical_recovered_pages": (1, 2),
        "assessments": [],
    }

    (
        output,
        figures,
        tables,
        stats,
    ) = merge_records(
        content_units=[existing_bda],
        figures=[],
        tables=[],
        plan=plan,
        surya_pages={},
        page_lookup={
            1: page_context(1),
            2: page_context(
                2,
                chapter_id="chapter-01",
                chapter_title="Patterns",
            ),
        },
        book_id="grade-6-math-test",
        book_version="v1",
        source_pdf="/tmp/textbook.pdf",
        native_pdf_pages={
            1: "This must not create a duplicate.",
            2: (
                "Learning Material Sheets "
                "Reprint 2026-27"
            ),
        },
    )

    assert figures == []
    assert tables == []
    assert len(output) == 2

    bda_units = [
        record
        for record in output
        if record["text_source"] == "bda"
    ]

    native_units = [
        record
        for record in output
        if (
            record["text_source"]
            == "canonical_pdf"
        )
    ]

    assert len(bda_units) == 1
    assert bda_units[0]["unit_id"] == "unit-page-1"

    assert len(native_units) == 1
    assert native_units[0][
        "source_page_numbers"
    ] == [2]
    assert native_units[0][
        "search_text"
    ] == (
        "Learning Material Sheets "
        "Reprint 2026-27"
    )
    assert native_units[0][
        "chapter_id"
    ] == "chapter-01"

    assert stats[
        "created_native_pdf_text_units"
    ] == 1
    assert stats[
        "created_surya_text_units"
    ] == 0


def test_native_pdf_recovery_requires_text_for_missing_page() -> None:
    with pytest.raises(
        ValueError,
        match="has no native PDF text",
    ):
        merge_records(
            content_units=[],
            figures=[],
            tables=[],
            plan={
                "fallback_pages": (),
                "accepted_bda_pages": (1,),
                "canonical_recovered_pages": (1,),
                "assessments": [],
            },
            surya_pages={},
            page_lookup={
                1: page_context(1),
            },
            book_id="grade-6-math-test",
            book_version="v1",
            source_pdf="/tmp/textbook.pdf",
            native_pdf_pages={},
        )


def test_load_native_pdf_page_texts(
    tmp_path: Path,
) -> None:
    import fitz

    from src.bda_surya_merge import (
        load_native_pdf_page_texts,
    )

    pdf_path = tmp_path / "textbook.pdf"

    document = fitz.open()

    first = document.new_page()
    first.insert_text(
        (72, 72),
        "First textbook page",
    )

    second = document.new_page()
    second.insert_text(
        (72, 72),
        "Learning Material Sheets Reprint 2026-27",
    )

    document.save(str(pdf_path))
    document.close()

    recovered = load_native_pdf_page_texts(
        str(pdf_path),
        [2],
    )

    assert tuple(recovered) == (2,)
    assert (
        "Learning Material Sheets"
        in recovered[2]
    )
