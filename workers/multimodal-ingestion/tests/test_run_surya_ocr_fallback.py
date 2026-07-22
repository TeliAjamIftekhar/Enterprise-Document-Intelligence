import json
import os
import sys
from pathlib import Path

import fitz
import pytest

from scripts.run_surya_ocr_fallback import (
    main,
    parse_page_spec,
)


def create_pdf(
    path: Path,
    *,
    pages: int = 2,
) -> None:
    document = fitz.open()

    for number in range(1, pages + 1):
        page = document.new_page()

        page.insert_text(
            (72, 72),
            f"Page {number}",
        )

    document.save(path)
    document.close()


def create_approval_record(
    path: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "ocr_engine": "surya-ocr",
                "book_id": "grade-1-urdu-test",
                "version": "v1",
                "approved_for_pipeline_integration": True,
                "full_book_run_authorized": False,
                "representative_pages": [1, 2],
            }
        ),
        encoding="utf-8",
    )


def create_fake_surya(
    path: Path,
) -> None:
    urdu_text = (
        "اساتذہ کے لیے ہدایات۔ "
        "اس سبق کی مدد سے بچوں کو اردو "
        "حروف اور آسان الفاظ سکھائے جائیں۔ "
        "بچے سوالوں کے جواب دیں اور اپنی "
        "کتاب میں مکمل الفاظ لکھیں۔"
    )

    script = f"""#!{sys.executable}
import json
import sys
from pathlib import Path

input_dir = Path(sys.argv[1])
output_dir = Path(
    sys.argv[
        sys.argv.index("--output_dir") + 1
    ]
)

payload = {{}}

for image in sorted(
    input_dir.glob("page-*.png")
):
    payload[image.stem] = [
        {{
            "blocks": [
                {{
                    "html": "<p>{urdu_text}</p>",
                    "confidence": 0.98
                }}
            ]
        }}
    ]

target = output_dir / input_dir.name / "results.json"
target.parent.mkdir(
    parents=True,
    exist_ok=True
)

target.write_text(
    json.dumps(
        payload,
        ensure_ascii=False
    ),
    encoding="utf-8"
)
"""

    path.write_text(
        script,
        encoding="utf-8",
    )

    path.chmod(0o755)


def test_parse_page_spec_with_ranges() -> None:
    assert parse_page_spec(
        "1,3-5,3"
    ) == (1, 3, 4, 5)


def test_parse_page_spec_rejects_invalid_values() -> None:
    with pytest.raises(
        Exception,
        match="positive",
    ):
        parse_page_spec("0")

    with pytest.raises(
        Exception,
        match="Descending",
    ):
        parse_page_spec("5-3")


def test_dry_run_renders_pages_without_surya(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "textbook.pdf"
    approval_path = tmp_path / "approval.json"
    output_root = tmp_path / "output"

    create_pdf(
        pdf_path,
        pages=2,
    )

    create_approval_record(
        approval_path
    )

    exit_code = main(
        [
            "--book-id",
            "grade-1-urdu-test",
            "--version",
            "v1",
            "--pdf",
            str(pdf_path),
            "--output-root",
            str(output_root),
            "--expected-language",
            "Urdu",
            "--pages",
            "1-2",
            "--approval-record",
            str(approval_path),
            "--project-root",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert exit_code == 0

    state = json.loads(
        (
            output_root
            / "run-state.json"
        ).read_text(encoding="utf-8")
    )

    assert state["status"] == "DRY_RUN_READY"

    assert (
        output_root
        / "input/page-0001.png"
    ).is_file()

    assert (
        output_root
        / "input/page-0002.png"
    ).is_file()


def test_fake_surya_execution_passes(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "textbook.pdf"
    approval_path = tmp_path / "approval.json"
    executable = tmp_path / "surya_ocr"
    output_root = tmp_path / "output"

    create_pdf(
        pdf_path,
        pages=2,
    )

    create_approval_record(
        approval_path
    )

    create_fake_surya(
        executable
    )

    exit_code = main(
        [
            "--book-id",
            "grade-1-urdu-test",
            "--version",
            "v1",
            "--pdf",
            str(pdf_path),
            "--output-root",
            str(output_root),
            "--expected-language",
            "Urdu",
            "--pages",
            "1-2",
            "--approval-record",
            str(approval_path),
            "--project-root",
            str(tmp_path),
            "--surya-executable",
            str(executable),
        ]
    )

    assert exit_code == 0

    state = json.loads(
        (
            output_root
            / "run-state.json"
        ).read_text(encoding="utf-8")
    )

    assert (
        state["status"]
        == "OCR_FALLBACK_VERIFIED"
    )

    assert state["classification"] == "PASS"
    assert state["passed"] == 2
    assert state["failed"] == 0

    verified_dir = (
        output_root
        / "verified"
    )

    assert (
        verified_dir
        / "SURYA_OCR_FALLBACK_VERIFIED"
    ).is_file()

    assert (
        verified_dir
        / "pages/page-0001.txt"
    ).is_file()

    assert (
        verified_dir
        / "pages/page-0002.txt"
    ).is_file()


def test_resume_skips_completed_execution(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "textbook.pdf"
    approval_path = tmp_path / "approval.json"
    executable = tmp_path / "surya_ocr"
    output_root = tmp_path / "output"

    create_pdf(
        pdf_path,
        pages=1,
    )

    create_approval_record(
        approval_path
    )

    create_fake_surya(
        executable
    )

    arguments = [
        "--book-id",
        "grade-1-urdu-test",
        "--version",
        "v1",
        "--pdf",
        str(pdf_path),
        "--output-root",
        str(output_root),
        "--expected-language",
        "Urdu",
        "--pages",
        "1",
        "--approval-record",
        str(approval_path),
        "--project-root",
        str(tmp_path),
        "--surya-executable",
        str(executable),
    ]

    assert main(arguments) == 0

    executable.write_text(
        "#!/bin/sh\nexit 99\n",
        encoding="utf-8",
    )

    executable.chmod(0o755)

    assert main(
        arguments + ["--resume"]
    ) == 0

    state = json.loads(
        (
            output_root
            / "run-state.json"
        ).read_text(encoding="utf-8")
    )

    assert (
        state["status"]
        == "OCR_FALLBACK_VERIFIED"
    )
