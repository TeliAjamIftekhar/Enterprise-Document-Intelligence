import json
from pathlib import Path

from scripts.plan_ocr_fallback_pages import (
    main,
    parse_page_spec,
)


def test_parse_page_spec() -> None:
    assert parse_page_spec(
        "1,3-5"
    ) == (1, 3, 4, 5)


def test_cli_writes_fallback_plan(
    tmp_path: Path,
) -> None:
    source = tmp_path / "normalized.json"
    output = tmp_path / "plan.json"

    source.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "canonical_page": 1,
                        "text": (
                            "Corrupted English output "
                            "instead of expected Urdu."
                        ),
                    },
                    {
                        "canonical_page": 2,
                        "text": (
                            "اساتذہ کے لیے ہدایات۔ "
                            "بچوں کو اردو حروف اور آسان "
                            "الفاظ پڑھنے اور لکھنے کی "
                            "مشق کرائی جائے۔"
                        ),
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(source),
            "--expected-language",
            "Urdu",
            "--expected-pages",
            "1-3",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.is_file()

    plan = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert plan["classification"] == (
        "OCR_FALLBACK_REQUIRED"
    )
    assert plan["fallback_pages"] == [1, 3]
    assert plan["failed_pages"] == [1]
    assert plan["missing_pages"] == [3]
