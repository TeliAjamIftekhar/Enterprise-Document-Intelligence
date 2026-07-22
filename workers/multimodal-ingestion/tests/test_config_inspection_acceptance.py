from generate_book_config_from_inspection import (
    inspection_allows_config_generation,
)


def scanned_inspection():
    return {
        "inspection_status": (
            "PASSED_WITH_WARNING"
        ),
        "book": {
            "detected_script": "unknown",
            "script_verification": (
                "unverified"
            ),
            "extraction_route": "visual-bda",
        },
        "script_evidence": {
            "text_evidence_status": (
                "insufficient"
            ),
            "extraction_route": "visual-bda",
        },
        "documents": [
            {
                "source_filename": "chapter.pdf",
                "pdf_status": "valid",
            }
        ],
        "warnings": [
            (
                "PDF text layer does not provide "
                "enough meaningful script evidence; "
                "visual/BDA extraction required."
            )
        ],
    }


def test_accepts_normal_passed_inspection():
    accepted, mode = (
        inspection_allows_config_generation({
            "inspection_status": "PASSED"
        })
    )

    assert accepted is True
    assert mode == "passed"


def test_accepts_scanned_visual_bda_book():
    accepted, mode = (
        inspection_allows_config_generation(
            scanned_inspection()
        )
    )

    assert accepted is True
    assert mode == (
        "visual-bda-scanned-textbook"
    )


def test_rejects_invalid_pdf():
    inspection = scanned_inspection()

    inspection["documents"][0][
        "pdf_status"
    ] = "invalid"

    accepted, _ = (
        inspection_allows_config_generation(
            inspection
        )
    )

    assert accepted is False


def test_rejects_manual_review_route():
    inspection = scanned_inspection()

    inspection["book"][
        "extraction_route"
    ] = "manual-review"

    accepted, _ = (
        inspection_allows_config_generation(
            inspection
        )
    )

    assert accepted is False


def test_rejects_additional_warning():
    inspection = scanned_inspection()

    inspection["warnings"].append(
        "Unsupported supplementary file detected"
    )

    accepted, _ = (
        inspection_allows_config_generation(
            inspection
        )
    )

    assert accepted is False
