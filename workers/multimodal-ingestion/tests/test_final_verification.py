from src.final_verification import (
    evaluate_final_verification,
)


def passing_evaluation() -> dict:
    return {
        "status": "PASSED",
        "all_tests_passed": True,
        "test_count": 3,
        "passed_test_count": 3,
        "failed_test_count": 0,
    }


def passing_bulk_report() -> dict:
    return {
        "status": "COMPLETED",
        "uploaded": True,
        "prepared_document_count": 17,
        "expected_final_count": 17,
        "final_count": 17,
        "bulk_result": {
            "failure_count": 0,
        },
    }


def test_generic_final_verification_passes() -> None:
    result = evaluate_final_verification(
        book_id="grade-1-english-test",
        book_version="v1",
        bulk_upload_report=(
            passing_bulk_report()
        ),
        vector_report=passing_evaluation(),
        hybrid_report=passing_evaluation(),
        rag_report=passing_evaluation(),
    )

    assert result["status"] == "VERIFIED"
    assert (
        result["all_checks_passed"]
        is True
    )
    assert result["failed_checks"] == []
    assert (
        result["document_counts"]["final"]
        == 17
    )


def test_dynamic_count_mismatch_fails() -> None:
    bulk = passing_bulk_report()
    bulk["final_count"] = 16

    result = evaluate_final_verification(
        book_id="grade-1-english-test",
        book_version="v1",
        bulk_upload_report=bulk,
        vector_report=passing_evaluation(),
        hybrid_report=passing_evaluation(),
        rag_report=passing_evaluation(),
    )

    assert result["status"] == "FAILED"
    assert (
        "final_count_matches_expected"
        in result["failed_checks"]
    )


def test_failed_rag_report_blocks_verified() -> None:
    rag = passing_evaluation()
    rag["status"] = "FAILED"
    rag["all_tests_passed"] = False
    rag["failed_test_count"] = 1

    result = evaluate_final_verification(
        book_id="grade-1-english-test",
        book_version="v1",
        bulk_upload_report=(
            passing_bulk_report()
        ),
        vector_report=passing_evaluation(),
        hybrid_report=passing_evaluation(),
        rag_report=rag,
    )

    assert result["status"] == "FAILED"
    assert (
        "rag_status_passed"
        in result["failed_checks"]
    )
    assert (
        "rag_all_tests_passed"
        in result["failed_checks"]
    )
