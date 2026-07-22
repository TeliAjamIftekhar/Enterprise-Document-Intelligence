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
