from __future__ import annotations

from typing import Any


def _positive_integer(
    value: Any,
) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def _evaluation_checks(
    report: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, bool]:
    return {
        f"{prefix}_status_passed": (
            report.get("status") == "PASSED"
        ),
        f"{prefix}_all_tests_passed": (
            report.get("all_tests_passed")
            is True
        ),
        f"{prefix}_test_count_positive": (
            _positive_integer(
                report.get("test_count")
            )
        ),
        f"{prefix}_failed_tests_zero": (
            report.get("failed_test_count")
            == 0
        ),
    }


def evaluate_final_verification(
    *,
    book_id: str,
    book_version: str,
    bulk_upload_report: dict[str, Any],
    vector_report: dict[str, Any],
    hybrid_report: dict[str, Any],
    rag_report: dict[str, Any],
) -> dict[str, Any]:
    prepared_count = bulk_upload_report.get(
        "prepared_document_count"
    )
    expected_final_count = (
        bulk_upload_report.get(
            "expected_final_count"
        )
    )
    final_count = bulk_upload_report.get(
        "final_count"
    )

    bulk_result = bulk_upload_report.get(
        "bulk_result"
    )

    if not isinstance(bulk_result, dict):
        bulk_result = {}

    checks: dict[str, bool] = {
        "bulk_status_completed": (
            bulk_upload_report.get("status")
            == "COMPLETED"
        ),
        "bulk_uploaded": (
            bulk_upload_report.get("uploaded")
            is True
        ),
        "prepared_document_count_positive": (
            _positive_integer(
                prepared_count
            )
        ),
        "expected_final_count_positive": (
            _positive_integer(
                expected_final_count
            )
        ),
        "final_count_matches_expected": (
            _positive_integer(final_count)
            and final_count
            == expected_final_count
        ),
        "bulk_failure_count_zero": (
            bulk_result.get(
                "failure_count",
                0,
            )
            == 0
        ),
    }

    checks.update(
        _evaluation_checks(
            vector_report,
            prefix="vector",
        )
    )
    checks.update(
        _evaluation_checks(
            hybrid_report,
            prefix="hybrid",
        )
    )
    checks.update(
        _evaluation_checks(
            rag_report,
            prefix="rag",
        )
    )

    all_checks_passed = all(
        checks.values()
    )

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    return {
        "schema_version": "1.0",
        "status": (
            "VERIFIED"
            if all_checks_passed
            else "FAILED"
        ),
        "book_id": book_id,
        "book_version": book_version,
        "all_checks_passed": (
            all_checks_passed
        ),
        "checks": checks,
        "failed_checks": failed_checks,
        "document_counts": {
            "prepared": prepared_count,
            "expected_final": (
                expected_final_count
            ),
            "final": final_count,
        },
        "evaluation_summary": {
            "vector_test_count": (
                vector_report.get(
                    "test_count"
                )
            ),
            "hybrid_test_count": (
                hybrid_report.get(
                    "test_count"
                )
            ),
            "rag_test_count": (
                rag_report.get(
                    "test_count"
                )
            ),
        },
    }
