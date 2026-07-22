from src.ocr_quality import (
    clean_ocr_text,
    evaluate_ocr_text,
    expected_script_for_language,
    requires_ocr_fallback,
)


def test_expected_script_language_mapping() -> None:
    assert expected_script_for_language("Urdu") == "arabic"
    assert expected_script_for_language("Hindi") == "devanagari"
    assert expected_script_for_language("Marathi") == "devanagari"
    assert expected_script_for_language("Sanskrit") == "devanagari"
    assert expected_script_for_language("English") == "latin"
    assert expected_script_for_language("Mathematics") == "mixed"


def test_clean_ocr_html_removes_markup() -> None:
    value = clean_ocr_text(
        "<h2>اساتذہ کے لیے ہدایات</h2>"
        "<ul><li>پہلا سبق</li><li>دوسرا سبق</li></ul>"
    )

    assert "<h2>" not in value
    assert "<li>" not in value
    assert "اساتذہ کے لیے ہدایات" in value
    assert "پہلا سبق" in value
    assert "دوسرا سبق" in value


def test_coherent_urdu_page_passes() -> None:
    text = """
    اساتذہ کے لیے ہدایات

    اس کہانی کی مدد سے ہمیں بچوں کو اردو حروف سکھانے ہیں۔
    بچوں سے سوال کریں اور تختہ سیاہ پر آسان الفاظ لکھیں۔
    اردو میں نقطوں کی بڑی اہمیت ہے اور حروف کی شکلیں مختلف ہیں۔
    بچے بارش، بوتل، بادل اور دوسرے الفاظ کے نام بتائیں گے۔
    """

    decision = evaluate_ocr_text(
        text,
        expected_language="Urdu",
        source="surya",
        confidence=0.97,
    )

    assert decision.classification == "PASS"
    assert decision.accepted is True
    assert decision.fallback_recommended is False
    assert decision.metrics.arabic_characters > 40
    assert decision.metrics.script_only_expected_ratio > 0.95


def test_sparse_urdu_worksheet_passes() -> None:
    text = """
    8. تصویروں کی مدد سے خالی خانوں میں حروف لکھوا کر
    پورا لفظ لکھوائیں

    = .....
    ج ز .....
    ج ش .....
    ل ک .....
    ر ی .....
    شہنائی 46
    """

    decision = evaluate_ocr_text(
        text,
        expected_language="Urdu",
        source="surya",
        confidence=0.92,
    )

    assert decision.classification == "PASS"
    assert decision.accepted is True
    assert decision.sparse_page is True
    assert decision.metrics.expected_script_ratio < 0.70
    assert decision.metrics.script_only_expected_ratio == 1.0


def test_latin_garbage_for_urdu_fails() -> None:
    text = """
    This page contains only corrupted English OCR output.
    The expected Urdu textbook script was not extracted.
    """

    decision = evaluate_ocr_text(
        text,
        expected_language="Urdu",
        source="bda",
    )

    assert decision.classification == "FAIL"
    assert decision.accepted is False
    assert decision.fallback_recommended is True
    assert "expected_script_not_detected" in decision.reasons
    assert requires_ocr_fallback(
        text,
        expected_language="Urdu",
    )


def test_repeated_hallucination_fails() -> None:
    repeated = "ہم نے کیا سیکھا آج کے سبق میں " * 12

    decision = evaluate_ocr_text(
        repeated,
        expected_language="Urdu",
        source="nova",
    )

    assert decision.classification == "FAIL"
    assert decision.accepted is False
    assert decision.fallback_recommended is True
    assert (
        "runaway_phrase_repetition"
        in decision.reasons
    )


def test_empty_text_fails() -> None:
    decision = evaluate_ocr_text(
        "   \n\t",
        expected_language="English",
        source="bda",
    )

    assert decision.classification == "FAIL"
    assert decision.fallback_recommended is True


def test_hindi_devanagari_page_passes() -> None:
    text = """
    यह पाठ बच्चों को भाषा सीखने और सरल शब्द पढ़ने में सहायता करता है।
    शिक्षक विद्यार्थियों से कहानी के बारे में प्रश्न पूछ सकते हैं।
    विद्यार्थी दिए गए अभ्यास को ध्यान से पढ़कर उत्तर लिखेंगे।
    """

    decision = evaluate_ocr_text(
        text,
        expected_language="Hindi",
        source="surya",
    )

    assert decision.classification == "PASS"
    assert decision.metrics.devanagari_characters > 40
    assert decision.metrics.script_only_expected_ratio > 0.95


def test_english_page_passes() -> None:
    text = """
    This chapter introduces learners to a simple story about friendship.
    Students should read the passage carefully and answer the questions.
    Teachers may use the activity to discuss vocabulary and comprehension.
    """

    decision = evaluate_ocr_text(
        text,
        expected_language="English",
        source="bda",
    )

    assert decision.classification == "PASS"
    assert decision.metrics.latin_characters > 40


def test_mathematics_mixed_content_passes() -> None:
    text = """
    8 × 7 = 56
    Area = length × breadth
    4 + 6 + 8 = 18
    Solve the equations and write the answers.
    """

    decision = evaluate_ocr_text(
        text,
        expected_language="Mathematics",
        source="bda",
    )

    assert decision.classification == "PASS"
    assert decision.expected_script == "mixed"


def test_decision_serializes_to_dictionary() -> None:
    decision = evaluate_ocr_text(
        "This is a complete English textbook paragraph for testing.",
        expected_language="English",
        source="test",
    )

    payload = decision.to_dict()

    assert payload["classification"] in {
        "PASS",
        "REVIEW",
        "FAIL",
    }
    assert isinstance(payload["reasons"], list)
    assert payload["metrics"]["expected_script"] == "latin"


def test_sanskrit_bilingual_cover_passes() -> None:
    text = """
    दीपकम्
    षष्ठकक्षायाः संस्कृत-पाठ्यपुस्तकम्
    0672
    राष्ट्रीय शैक्षिक अनुसंधान और प्रशिक्षण परिषद्
    NATIONAL COUNCIL OF EDUCATIONAL RESEARCH AND TRAINING
    Reprint 2026-27
    """

    decision = evaluate_ocr_text(
        text,
        expected_language="Sanskrit",
        source="surya",
        confidence=0.91,
    )

    assert decision.classification == "PASS"
    assert decision.accepted is True
    assert (
        "coherent_bilingual_page_accepted"
        in decision.reasons
    )


def test_sanskrit_workbook_placeholders_are_not_duplicates() -> None:
    placeholder = (
        "= ..... + ..... + ..... + ..... + ....."
    )

    text = """
    संयुक्त-व्यञ्जनानि
    वर्णः कर्म सप्तर्षिः सिद्धार्थः पद्यम् शब्दः
    व्यञ्जनम् कण्ठः सन्धिः प्रारम्भः संवादः
    """ + "\n".join([placeholder] * 7)

    decision = evaluate_ocr_text(
        text,
        expected_language="Sanskrit",
        source="surya",
        confidence=0.98,
    )

    assert decision.classification == "PASS"
    assert decision.accepted is True
    assert (
        decision.metrics.max_duplicate_line_count
        < 5
    )
    assert (
        "runaway_duplicate_lines"
        not in decision.reasons
    )


def test_genuine_duplicate_text_still_fails() -> None:
    repeated_line = (
        "विद्यार्थी दिए गए अभ्यास को ध्यान से पढ़कर उत्तर लिखेंगे।"
    )

    decision = evaluate_ocr_text(
        "\n".join([repeated_line] * 5),
        expected_language="Hindi",
        source="surya",
        confidence=0.95,
    )

    assert decision.classification == "FAIL"
    assert decision.accepted is False
    assert (
        "runaway_duplicate_lines"
        in decision.reasons
    )

