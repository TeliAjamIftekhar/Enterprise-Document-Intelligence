import json
from pathlib import Path

import fitz
import pytest
from PIL import Image

from src.surya_ocr_fallback import (
    SuryaRuntimeConfig,
    build_surya_command,
    build_surya_environment,
    load_approval_record,
    locate_results_json,
    normalize_page_numbers,
    page_number_from_key,
    parse_surya_results,
    render_pdf_pages,
    validate_runtime_config,
    write_fallback_artifacts,
)


def runtime_config(
    tmp_path: Path,
) -> SuryaRuntimeConfig:
    executable = tmp_path / "surya_ocr"
    executable.write_text(
        "#!/bin/sh\n",
        encoding="utf-8",
    )

    return SuryaRuntimeConfig(
        executable=executable,
        project_root=tmp_path,
    )


def test_build_environment_uses_validated_t4_settings(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path)

    environment = build_surya_environment(
        config,
        base_environment={"PATH": "/usr/bin"},
    )

    assert environment["VLLM_GPU_TYPE"] == "t4"
    assert environment["VLLM_DTYPE"] == "float16"
    assert (
        environment["SURYA_INFERENCE_BACKEND"]
        == "vllm"
    )
    assert (
        environment[
            "SURYA_INFERENCE_KEEP_ALIVE"
        ]
        == "false"
    )
    assert environment["PATH"] == "/usr/bin"


def test_uppercase_gpu_type_is_rejected(
    tmp_path: Path,
) -> None:
    config = SuryaRuntimeConfig(
        executable=tmp_path / "surya_ocr",
        project_root=tmp_path,
        gpu_type="T4",
    )

    with pytest.raises(
        ValueError,
        match="lowercase",
    ):
        validate_runtime_config(
            config,
            require_executable=False,
        )


def test_build_surya_command(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path)

    command = build_surya_command(
        config,
        input_path=tmp_path / "input",
        output_dir=tmp_path / "output",
    )

    assert command == [
        str(config.executable),
        str(tmp_path / "input"),
        "--output_dir",
        str(tmp_path / "output"),
        "--images",
    ]


def test_page_number_from_key() -> None:
    assert page_number_from_key(
        "page-0017"
    ) == 17

    assert page_number_from_key(
        "book-page-0110"
    ) == 110


def test_normalize_page_numbers() -> None:
    assert normalize_page_numbers(
        [5, 1, 5, 3],
        page_count=10,
    ) == (1, 3, 5)

    with pytest.raises(ValueError):
        normalize_page_numbers(
            [0],
            page_count=10,
        )


def test_locate_nested_results_json(
    tmp_path: Path,
) -> None:
    results_path = (
        tmp_path
        / "output/input/results.json"
    )

    results_path.parent.mkdir(
        parents=True
    )

    results_path.write_text(
        "{}",
        encoding="utf-8",
    )

    assert (
        locate_results_json(tmp_path / "output")
        == results_path
    )


def test_render_selected_pdf_pages(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "textbook.pdf"

    document = fitz.open()

    for index in range(3):
        page = document.new_page()
        page.insert_text(
            (72, 72),
            f"Page {index + 1}",
        )

    document.save(pdf_path)
    document.close()

    rendered = render_pdf_pages(
        pdf_path,
        tmp_path / "images",
        page_numbers=[3, 1],
        dpi=144,
    )

    assert [
        page.canonical_page
        for page in rendered
    ] == [1, 3]

    assert all(
        page.image_path.is_file()
        for page in rendered
    )

    assert all(
        page.width > 0
        and page.height > 0
        and page.byte_size > 0
        for page in rendered
    )


def test_parse_valid_urdu_surya_results(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.json"

    urdu_text = (
        "اساتذہ کے لیے ہدایات۔ "
        "اس کہانی کی مدد سے بچوں کو اردو حروف "
        "اور آسان الفاظ سکھائے جائیں۔ "
        "بچے سوالوں کے جواب دیں اور الفاظ لکھیں۔"
    )

    payload = {
        "page-0005": [
            {
                "blocks": [
                    {
                        "html": (
                            f"<h2>{urdu_text}</h2>"
                        ),
                        "confidence": 0.98,
                    }
                ]
            }
        ]
    }

    results_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = parse_surya_results(
        results_path,
        expected_language="Urdu",
        expected_pages=[5],
    )

    assert report.classification == "PASS"
    assert report.accepted_for_pipeline is True
    assert report.passed == 1
    assert report.review == 0
    assert report.failed == 0
    assert report.missing_pages == ()
    assert report.pages[0].confidence == 0.98
    assert (
        report.pages[0].decision.classification
        == "PASS"
    )


def test_sparse_urdu_worksheet_is_accepted(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.json"

    payload = {
        "page-0060": [
            {
                "blocks": [
                    {
                        "html": """
                        <p>
                        8. تصویروں کی مدد سے خالی خانوں میں
                        حروف لکھوا کر پورا لفظ لکھوائیں
                        </p>
                        <p>
                        = ..... ..... ..... ..... .....
                        </p>
                        <p>
                        ج ز ..... ..... ..... ..... .....
                        </p>
                        <p>
                        ج ز ..... ..... ..... ..... .....
                        </p>
                        <p>
                        ج ش ..... ..... ..... ..... .....
                        </p>
                        <p>
                        ل ..... ..... ..... ..... .....
                        </p>
                        <p>
                        ل ک ..... ..... ..... ..... .....
                        </p>
                        <p>
                        ر ی ..... ..... ..... ..... .....
                        </p>
                        <p>
                        ر ر ..... ..... ..... ..... .....
                        </p>
                        <p>
                        ل س ..... ..... ..... ..... .....
                        </p>
                        <p>
                        ل ا ..... ..... ..... ..... .....
                        </p>
                        <p>شہنائی 46</p>
                        """,
                        "confidence": 0.92,
                    }
                ]
            }
        ]
    }

    results_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = parse_surya_results(
        results_path,
        expected_language="Urdu",
        expected_pages=[60],
    )

    assert report.classification == "PASS"
    assert (
        report.pages[0].decision.sparse_page
        is True
    )


def test_missing_expected_page_fails_report(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.json"

    payload = {
        "page-0005": [
            {
                "blocks": [
                    {
                        "html": (
                            "<p>"
                            "اساتذہ بچوں کو اردو الفاظ "
                            "پڑھنے اور لکھنے کی مشق کرائیں۔ "
                            "بچے سبق کے سوالات کے جواب دیں۔"
                            "</p>"
                        )
                    }
                ]
            }
        ]
    }

    results_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = parse_surya_results(
        results_path,
        expected_language="Urdu",
        expected_pages=[5, 17],
    )

    assert report.classification == "FAIL"
    assert report.accepted_for_pipeline is False
    assert report.missing_pages == (17,)


def test_latin_output_for_urdu_fails(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.json"

    payload = {
        "page-0017": [
            {
                "blocks": [
                    {
                        "html": (
                            "<p>This is corrupted English "
                            "instead of Urdu OCR output.</p>"
                        ),
                        "confidence": 0.99,
                    }
                ]
            }
        ]
    }

    results_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    report = parse_surya_results(
        results_path,
        expected_language="Urdu",
        expected_pages=[17],
    )

    assert report.classification == "FAIL"
    assert report.failed == 1
    assert report.accepted_for_pipeline is False


def test_write_fallback_artifacts(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.json"

    payload = {
        "page-0005": [
            {
                "blocks": [
                    {
                        "html": (
                            "<p>"
                            "یہ ایک مکمل اردو عبارت ہے جسے "
                            "طلبہ پڑھیں گے اور سوالات کے "
                            "جواب اپنی کتاب میں لکھیں گے۔"
                            "</p>"
                        )
                    }
                ]
            }
        ]
    }

    results_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = parse_surya_results(
        results_path,
        expected_language="Urdu",
        expected_pages=[5],
    )

    artifacts = write_fallback_artifacts(
        report,
        tmp_path / "artifacts",
    )

    assert artifacts["report"].is_file()
    assert artifacts["marker"].name == (
        "SURYA_OCR_FALLBACK_VERIFIED"
    )
    assert (
        artifacts["pages_dir"]
        / "page-0005.txt"
    ).is_file()


def test_load_approved_pilot_record(
    tmp_path: Path,
) -> None:
    approval_path = tmp_path / "approval.json"

    approval_path.write_text(
        json.dumps(
            {
                "approved_for_pipeline_integration": True,
                "ocr_engine": "surya-ocr",
            }
        ),
        encoding="utf-8",
    )

    record = load_approval_record(
        approval_path
    )

    assert record["ocr_engine"] == "surya-ocr"


def test_unapproved_pilot_record_is_rejected(
    tmp_path: Path,
) -> None:
    approval_path = tmp_path / "approval.json"

    approval_path.write_text(
        json.dumps(
            {
                "approved_for_pipeline_integration": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="not been approved",
    ):
        load_approval_record(
            approval_path
        )



def test_empty_surya_result_recovers_canonical_pdf_text(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "textbook.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        (
            "This canonical textbook page contains "
            "valid lesson content for students."
        ),
    )
    document.save(pdf_path)
    document.close()

    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "page-0001": [
                    {
                        "html": "",
                        "confidence": 0.98,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = parse_surya_results(
        results_path,
        expected_language="English",
        expected_pages=[1],
        canonical_pdf_path=pdf_path,
    )

    page_result = report.pages[0]

    assert report.classification == "PASS"
    assert (
        "canonical textbook page"
        in page_result.clean_text
    )
    assert (
        "canonical_pdf_text_recovered"
        in page_result.decision.reasons
    )


def test_extremely_blank_decorative_page_is_accepted(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    Image.new(
        "RGB",
        (300, 400),
        "white",
    ).save(
        input_dir / "page-0001.png"
    )

    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "page-0001": [
                    {
                        "html": (
                            "<p>Reprint 2026-27</p>"
                        ),
                        "confidence": 0.99,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = parse_surya_results(
        results_path,
        expected_language="Sanskrit",
        expected_pages=[1],
        input_dir=input_dir,
    )

    assert report.classification == "PASS"
    assert (
        report.pages[0].decision.reasons
        == (
            "visually_blank_or_decorative_page_accepted",
        )
    )


def test_latex_control_words_do_not_fail_sanskrit(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.json"

    formulas = (
        "\\text{क्ष} = \\boxed{\\text{क्}} + "
        "\\boxed{\\text{ष्}} + \\boxed{\\text{अ}}",
        "\\text{कक्षा} = \\boxed{\\text{क्}} + "
        "\\boxed{\\text{अ}} + \\boxed{\\text{क्ष}}",
        "\\text{त्र} = \\boxed{\\text{त्}} + "
        "\\boxed{\\text{र्}} + \\boxed{\\text{अ}}",
        "\\text{पत्रम्} = \\boxed{\\text{प्}} + "
        "\\boxed{\\text{त्र}} + \\boxed{\\text{म्}}",
        "\\text{ज्ञ} = \\boxed{\\text{ज्}} + "
        "\\boxed{\\text{ञ्}} + \\boxed{\\text{अ}}",
        "\\text{ज्ञानम्} = \\boxed{\\text{ज्ञ}} + "
        "\\boxed{\\text{आ}} + \\boxed{\\text{नम्}}",
        "\\text{द्य} = \\boxed{\\text{द्}} + "
        "\\boxed{\\text{य्}} + \\boxed{\\text{अ}}",
        "\\text{विद्या} = \\boxed{\\text{वि}} + "
        "\\boxed{\\text{द्या}}",
        "\\text{श्र} = \\boxed{\\text{श्}} + "
        "\\boxed{\\text{र्}} + \\boxed{\\text{अ}}",
        "\\text{श्रमः} = \\boxed{\\text{श्र}} + "
        "\\boxed{\\text{मः}}",
    )

    results_path.write_text(
        json.dumps(
            {
                "page-0001": [
                    {
                        "html": (
                            "<p>"
                            "संयुक्त व्यञ्जनानि विद्यार्थिनः "
                            "ध्यानपूर्वक पठन्तु।"
                            "</p><p>"
                            + " ".join(formulas)
                            + "</p>"
                        ),
                        "confidence": 0.97,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = parse_surya_results(
        results_path,
        expected_language="Sanskrit",
        expected_pages=[1],
    )

    assert report.classification == "PASS"
    assert (
        "latex_control_words_ignored_for_validation"
        in report.pages[0].decision.reasons
    )


def test_structured_table_duplicate_label_is_accepted(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.json"

    repeated_rows = "".join(
        (
            "<tr><td>"
            "द्वितीया विभक्तिः"
            "</td></tr>"
        )
        for _ in range(5)
    )

    results_path.write_text(
        json.dumps(
            {
                "page-0001": [
                    {
                        "html": (
                            "<p>"
                            "एतानि शब्दरूपाणि पठन्तु "
                            "अवगच्छन्तु स्मरन्तु च।"
                            "</p><table>"
                            + repeated_rows
                            + "".join(
                                (
                                    "<tr><td>"
                                    f"शब्दरूपम् {index}"
                                    "</td></tr>"
                                )
                                for index in range(20)
                            )
                            + "</table>"
                        ),
                        "confidence": 0.98,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = parse_surya_results(
        results_path,
        expected_language="Sanskrit",
        expected_pages=[1],
    )

    assert report.classification == "PASS"
    assert (
        report.pages[0].decision.reasons
        == (
            "structured_table_duplicate_label_accepted",
        )
    )


def test_structured_workbook_repetition_is_accepted(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.json"

    repeated_instruction = (
        "अभ्यास प्रश्न उत्तर लिखें ..... "
    )

    results_path.write_text(
        json.dumps(
            {
                "page-0001": [
                    {
                        "html": (
                            "<p>"
                            "वर्ण वियोग अभ्यासः"
                            "</p><table><tr><td>"
                            + repeated_instruction * 10
                            + "</td></tr>"
                            + "".join(
                                (
                                    "<tr><td>"
                                    f"रामः = ..... + ..... {index}"
                                    "</td></tr>"
                                )
                                for index in range(20)
                            )
                            + "</table>"
                        ),
                        "confidence": 0.97,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = parse_surya_results(
        results_path,
        expected_language="Sanskrit",
        expected_pages=[1],
    )

    assert report.classification == "PASS"
    assert (
        report.pages[0].decision.reasons
        == (
            "structured_workbook_repetition_accepted",
        )
    )



def test_blank_image_does_not_accept_hallucinated_repetition(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    Image.new(
        "RGB",
        (300, 400),
        "white",
    ).save(
        input_dir / "page-0001.png"
    )

    repeated = (
        "विद्यार्थी दिए गए अभ्यास को ध्यान से "
        "पढ़कर उत्तर लिखेंगे। "
    ) * 10

    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "page-0001": [
                    {
                        "html": f"<p>{repeated}</p>",
                        "confidence": 0.99,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = parse_surya_results(
        results_path,
        expected_language="Sanskrit",
        expected_pages=[1],
        input_dir=input_dir,
    )

    assert report.classification == "FAIL"
    assert report.accepted_for_pipeline is False
    assert (
        "runaway_phrase_repetition"
        in report.pages[0].decision.reasons
    )
