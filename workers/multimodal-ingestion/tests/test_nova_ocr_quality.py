from invoke_nova_ocr_page import (
    assess_ocr_quality,
)


def test_accepts_completed_urdu_ocr():
    text = "\n".join([
        "کتاب ہماری بہترین دوست ہے۔",
        "ہم کتاب سے نئی باتیں سیکھتے ہیں۔",
        "بچے روزانہ کتاب پڑھتے ہیں۔",
        "علم انسان کو کامیاب بناتا ہے۔",
    ])

    result = assess_ocr_quality(
        text,
        stop_reason="end_turn",
    )

    assert result["status"] == "OCR_VALID"
    assert result["max_line_repeat"] == 1


def test_rejects_repetitive_output():
    text = "\n".join(
        ["ہم نے کیا سیکھا؟"] * 50
    )

    result = assess_ocr_quality(
        text,
        stop_reason="end_turn",
    )

    assert result["status"] == "NEEDS_REVIEW"
    assert result["max_line_repeat"] == 50


def test_rejects_truncated_output():
    result = assess_ocr_quality(
        (
            "کتاب ہماری بہترین دوست ہے۔ "
            "ہم اس سے نئی باتیں سیکھتے ہیں۔"
        ),
        stop_reason="max_tokens",
    )

    assert result["status"] == "NEEDS_REVIEW"
    assert (
        result["checks"][
            "response_completed"
        ]
        is False
    )


def test_rejects_latin_description():
    result = assess_ocr_quality(
        (
            "This image shows children "
            "reading a school textbook."
        ),
        stop_reason="end_turn",
    )

    assert result["status"] == "NEEDS_REVIEW"
    assert (
        result["checks"][
            "arabic_dominant"
        ]
        is False
    )
