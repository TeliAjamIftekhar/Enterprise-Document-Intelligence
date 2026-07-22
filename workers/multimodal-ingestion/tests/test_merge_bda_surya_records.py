import json
from pathlib import Path

from scripts.merge_bda_surya_records import (
    main,
)


def write_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def test_cli_creates_embedding_compatible_output(
    tmp_path: Path,
) -> None:
    normalized = tmp_path / "normalized"
    output = tmp_path / "unified"

    write_jsonl(
        normalized / "content-units.jsonl",
        [
            {
                "schema_version": "1.1",
                "unit_id": "unit-page-1",
                "book_id": "grade-1-urdu-test",
                "book_version": "v1",
                "source_kind": "bda_standard_output",
                "source_pdf": "/tmp/textbook.pdf",
                "bda_element_id": "element-1",
                "element_index": 1,
                "element_type": "TEXT",
                "element_sub_type": "PARAGRAPH",
                "modality": "paragraph",
                "reading_order": 1,
                "source_page_numbers": [1],
                "locations": [],
                "raw_text": "Corrupted English",
                "markdown": "Corrupted English",
                "search_text": "Corrupted English",
                "asset_s3_uris": [],
                "asset_local_paths": [],
                "quality_flags": [],
            }
        ],
    )

    write_jsonl(
        normalized / "figures.jsonl",
        [
            {
                "figure_id": "figure-1",
                "book_id": "grade-1-urdu-test",
                "book_version": "v1",
                "source_page_numbers": [1],
                "crop_s3_uris": [
                    "s3://bucket/figure.png"
                ],
                "crop_local_paths": [
                    "/tmp/figure.png"
                ],
            }
        ],
    )

    write_jsonl(
        normalized / "tables.jsonl",
        [],
    )

    plan = tmp_path / "plan.json"

    write_json(
        plan,
        {
            "classification": (
                "OCR_FALLBACK_REQUIRED"
            ),
            "fallback_pages": [1],
            "accepted_bda_pages": [],
            "assessments": [
                {
                    "canonical_page": 1,
                    "decision": {
                        "classification": "FAIL",
                    },
                }
            ],
        },
    )

    surya_report = (
        tmp_path / "surya-report.json"
    )

    urdu_text = (
        "یہ مکمل اور درست اردو عبارت ہے۔"
    )

    write_json(
        surya_report,
        {
            "classification": "PASS",
            "accepted_for_pipeline": True,
            "pages": [
                {
                    "page_key": "page-0001",
                    "canonical_page": 1,
                    "clean_text": urdu_text,
                    "decision": {
                        "classification": "PASS",
                        "accepted": True,
                        "expected_language": "urdu",
                        "expected_script": "arabic",
                    },
                }
            ],
        },
    )

    page_map = tmp_path / "page-map.json"

    write_json(
        page_map,
        {
            "schema_version": "1.0",
            "book_id": "grade-1-urdu-test",
            "book_version": "v1",
            "pages": [
                {
                    "canonical_page": 1,
                    "page_type": "chapter",
                    "document_order": 1,
                    "document_id": "chapter-01",
                    "document_type": "chapter",
                    "document_title": "سبق",
                    "source_filename": "chapter.pdf",
                    "source_page": 1,
                    "unit_number": None,
                    "chapter_id": "chapter-01",
                    "chapter_title": "سبق",
                    "chapter_page": 1,
                }
            ],
        },
    )

    exit_code = main(
        [
            "--normalized-root",
            str(normalized),
            "--ocr-plan",
            str(plan),
            "--surya-report",
            str(surya_report),
            "--page-map",
            str(page_map),
            "--output-dir",
            str(output),
        ]
    )

    assert exit_code == 0

    content_lines = (
        output / "content-units.jsonl"
    ).read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(content_lines) == 1

    unit = json.loads(
        content_lines[0]
    )

    assert unit["text_source"] == "surya"
    assert unit["search_text"] == urdu_text
    assert unit["chapter_id"] == "chapter-01"
    assert unit["source_page_numbers"] == [1]

    figures = (
        output / "figures.jsonl"
    ).read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(figures) == 1

    assert (
        output
        / "tables.jsonl"
    ).is_file()

    report = json.loads(
        (
            output
            / "bda-surya-merge-report.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "VALID"
    assert report[
        "removed_bda_text_units"
    ] == 1
    assert report[
        "created_surya_text_units"
    ] == 1
    assert report["preserved_figures"] == 1

    assert (
        output
        / "BDA_SURYA_MERGE_VALID"
    ).is_file()
