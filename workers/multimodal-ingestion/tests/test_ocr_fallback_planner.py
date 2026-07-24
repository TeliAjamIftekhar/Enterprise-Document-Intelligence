import json
from pathlib import Path

from src.ocr_fallback_planner import (
    extract_page_number,
    load_normalized_records,
    plan_ocr_fallback,
    records_from_payload,
    select_page_candidates,
)


URDU_PASS_TEXT = """
اساتذہ کے لیے ہدایات۔
اس سبق کی مدد سے بچوں کو اردو حروف اور آسان الفاظ سکھائے جائیں۔
بچے سوالوں کے جواب دیں اور اپنی کتاب میں مکمل الفاظ لکھیں۔
"""


def test_extract_nested_page_number() -> None:
    assert extract_page_number(
        {
            "metadata": {
                "canonical_page": "17",
            }
        }
    ) == 17


def test_records_from_page_key_payload() -> None:
    records = records_from_payload(
        {
            "page-0005": {
                "text": URDU_PASS_TEXT,
            },
            "page-0017": {
                "text": URDU_PASS_TEXT,
            },
        }
    )

    assert len(records) == 2
    assert records[0]["canonical_page"] == 5
    assert records[1]["canonical_page"] == 17


def test_corrupted_bda_urdu_requires_fallback() -> None:
    records = [
        {
            "canonical_page": 1,
            "text": (
                "This is corrupted English output "
                "from an Urdu scanned textbook."
            ),
        },
        {
            "canonical_page": 2,
            "text": URDU_PASS_TEXT,
        },
    ]

    plan = plan_ocr_fallback(
        records,
        expected_language="Urdu",
        expected_pages=[1, 2],
    )

    assert plan.classification == (
        "OCR_FALLBACK_REQUIRED"
    )
    assert plan.fallback_pages == (1,)
    assert plan.failed_pages == (1,)
    assert plan.accepted_bda_pages == (2,)


def test_missing_page_requires_fallback() -> None:
    records = [
        {
            "canonical_page": 1,
            "text": URDU_PASS_TEXT,
        }
    ]

    plan = plan_ocr_fallback(
        records,
        expected_language="Urdu",
        expected_pages=[1, 2],
    )

    assert plan.missing_pages == (2,)
    assert plan.fallback_pages == (2,)


def test_sparse_urdu_page_is_accepted() -> None:
    records = [
        {
            "canonical_page": 60,
            "html": """
            <p>
            تصویروں کی مدد سے خالی خانوں میں
            حروف لکھوا کر پورا لفظ لکھوائیں
            </p>
            <p>
            = ..... ..... ..... ..... .....
            ج ز ..... ..... ..... ..... .....
            ل ک ..... ..... ..... ..... .....
            ر ی ..... ..... ..... ..... .....
            شہنائی 46
            </p>
            """,
        }
    ]

    plan = plan_ocr_fallback(
        records,
        expected_language="Urdu",
        expected_pages=[60],
    )

    assert plan.classification == "BDA_ACCEPTED"
    assert plan.accepted_bda_pages == (60,)
    assert plan.fallback_pages == ()


def test_duplicate_page_records_are_combined() -> None:
    records = [
        {
            "canonical_page": 5,
            "text": "اساتذہ کے لیے ہدایات۔",
        },
        {
            "canonical_page": 5,
            "text": URDU_PASS_TEXT,
            "confidence": 0.98,
        },
    ]

    candidates = select_page_candidates(records)

    assert len(candidates) == 1
    assert candidates[0].canonical_page == 5
    assert candidates[0].source_record_count == 2
    assert "بچوں کو اردو حروف" in candidates[0].clean_text


def test_mathematics_mixed_page_is_accepted() -> None:
    records = [
        {
            "canonical_page": 10,
            "text": (
                "Area = length × breadth\n"
                "8 × 7 = 56\n"
                "Solve the equations."
            ),
        }
    ]

    plan = plan_ocr_fallback(
        records,
        expected_language="Mathematics",
        expected_pages=[10],
    )

    assert plan.classification == "BDA_ACCEPTED"
    assert plan.accepted_bda_pages == (10,)


def test_load_jsonl_records(tmp_path: Path) -> None:
    source = tmp_path / "normalized.jsonl"

    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "canonical_page": 1,
                        "text": URDU_PASS_TEXT,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "canonical_page": 2,
                        "text": URDU_PASS_TEXT,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_normalized_records(source)

    assert len(records) == 2



def test_figure_descriptions_are_excluded_from_quality_text() -> None:
    valid_text = (
        "This mathematics lesson explains number "
        "patterns and asks students to solve the "
        "given classroom exercises."
    )

    records = [
        {
            "canonical_page": 1,
            "element_type": "TEXT",
            "raw_text": valid_text,
        },
        {
            "canonical_page": 1,
            "element_type": "FIGURE",
            "raw_text": (
                "hallucinated repeated phrase " * 20
            ),
        },
    ]

    plan = plan_ocr_fallback(
        records,
        expected_language="Mathematics",
        expected_pages=[1],
    )

    assert plan.classification == "BDA_ACCEPTED"
    assert plan.fallback_pages == ()


def test_quality_text_uses_one_primary_field() -> None:
    records = [
        {
            "canonical_page": 1,
            "element_type": "TEXT",
            "raw_text": (
                "This complete mathematics paragraph "
                "contains enough meaningful content "
                "for a textbook quality check."
            ),
            "markdown": (
                "repeated duplicate phrase " * 20
            ),
            "search_text": (
                "repeated duplicate phrase " * 20
            ),
        }
    ]

    plan = plan_ocr_fallback(
        records,
        expected_language="Mathematics",
        expected_pages=[1],
    )

    assert plan.classification == "BDA_ACCEPTED"
    assert plan.fallback_pages == ()


def test_native_pdf_recovers_figure_only_text_layout_page(
    tmp_path: Path,
) -> None:
    import fitz

    pdf_path = tmp_path / "textbook.pdf"

    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        (
            "Learning Material Sheets "
            "Reprint 2026-27"
        ),
    )
    document.save(str(pdf_path))
    document.close()

    records = [
        {
            "canonical_page": 1,
            "element_type": "FIGURE",
            "raw_text": "Decorative page image",
        }
    ]

    plan = plan_ocr_fallback(
        records,
        expected_language="Mathematics",
        expected_pages=[1],
        canonical_pdf_path=pdf_path,
        allow_native_text_recovery=True,
    )

    assert plan.classification == "BDA_ACCEPTED"
    assert plan.fallback_pages == ()
    assert plan.missing_pages == ()
    assert plan.canonical_recovered_pages == (1,)


def test_native_pdf_verifies_legitimate_math_repetition(
    tmp_path: Path,
) -> None:
    import fitz

    repeated = (
        "The total number of wickets "
        "The total number of wickets "
        "The total number of wickets "
        "The total number of wickets "
        "The total number of wickets "
        "The total number of wickets "
        "Students compare the table and explain "
        "their mathematical reasoning."
    )

    pdf_path = tmp_path / "textbook.pdf"

    document = fitz.open()
    page = document.new_page(
        width=800,
        height=1000,
    )
    page.insert_textbox(
        fitz.Rect(50, 50, 750, 950),
        repeated,
        fontsize=11,
    )
    document.save(str(pdf_path))
    document.close()

    records = [
        {
            "canonical_page": 1,
            "element_type": "TEXT",
            "raw_text": repeated,
        }
    ]

    plan = plan_ocr_fallback(
        records,
        expected_language="Mathematics",
        expected_pages=[1],
        canonical_pdf_path=pdf_path,
        allow_native_text_recovery=True,
    )

    assert plan.classification == "BDA_ACCEPTED"
    assert plan.fallback_pages == ()
    assert plan.canonical_recovered_pages == (1,)



def test_native_recovery_does_not_replace_passing_bda_text(
    tmp_path: Path,
) -> None:
    import fitz

    pdf_path = tmp_path / "textbook.pdf"

    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        (
            "Canonical mathematics textbook text "
            "with sufficient meaningful content."
        ),
    )
    document.save(str(pdf_path))
    document.close()

    records = [
        {
            "canonical_page": 1,
            "element_type": "TEXT",
            "raw_text": (
                "This mathematics lesson explains "
                "number patterns and gives students "
                "several meaningful exercises."
            ),
        }
    ]

    plan = plan_ocr_fallback(
        records,
        expected_language="Mathematics",
        expected_pages=[1],
        canonical_pdf_path=pdf_path,
        allow_native_text_recovery=True,
    )

    assert plan.classification == "BDA_ACCEPTED"
    assert plan.accepted_bda_pages == (1,)
    assert plan.canonical_recovered_pages == ()
    assert plan.fallback_pages == ()
