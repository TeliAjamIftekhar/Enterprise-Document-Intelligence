from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import evaluate_opensearch_rag as rag_eval


TEST_CASES: list[dict[str, Any]] = [
    {
        "test_id": "kaveri-policy-alignment",
        "question": (
            "Which education policy and curriculum "
            "framework is the Kaveri Grade 9 "
            "textbook aligned with?"
        ),
        "top_k": 5,
        "modality": "paragraph",
        "expected_citation_pages": [7],
        "required_term_groups": [
            ["NEP 2020", "National Education Policy"],
            ["NCF-SE 2023", "National Curriculum Framework"],
        ],
        "expect_insufficient": False,
    },
    {
        "test_id": "reading-for-meaning-purpose",
        "question": (
            "What does the Reading for Meaning "
            "section encourage students to do?"
        ),
        "top_k": 5,
        "modality": "paragraph",
        "expected_citation_pages": [8],
        "required_term_groups": [
            ["critically", "critical thinking"],
            ["connections"],
            ["inferences"],
        ],
        "expect_insufficient": False,
    },
    {
        "test_id": "winds-of-change-start-page",
        "question": (
            "According to the table of contents, "
            "on which page does Winds of Change begin?"
        ),
        "top_k": 5,
        "modality": "table",
        "expected_citation_pages": [19],
        "required_term_groups": [
            ["69"],
            ["Winds of Change"],
        ],
        "expect_insufficient": False,
    },
    {
        "test_id": "fundamental-duty-national-symbols",
        "question": (
            "What is the fundamental duty regarding "
            "the Constitution, National Flag and "
            "National Anthem?"
        ),
        "top_k": 5,
        "modality": "list",
        "expected_citation_pages": [20],
        "required_term_groups": [
            ["Constitution"],
            ["National Flag"],
            ["National Anthem"],
            ["respect", "abide"],
        ],
        "expect_insufficient": False,
    },
    {
        "test_id": "unsupported-capital-france",
        "question": (
            "What is the capital of France?"
        ),
        "top_k": 5,
        "modality": None,
        "expected_citation_pages": [],
        "required_term_groups": [],
        "expect_insufficient": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate grounded RAG for the "
            "full-book pilot pages 1-20."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 52)
    print("FULL-BOOK PILOT GROUNDED RAG REGRESSION")
    print("=" * 52)
    print(
        f"Generation model: "
        f"{rag_eval.rag.GENERATION_MODEL_ID}"
    )
    print(
        f"Tests:            {len(TEST_CASES)}"
    )
    print("Supported pages:  7, 8, 19, 20")
    print("Unsupported test: capital of France")
    print()

    results: list[
        dict[str, Any]
    ] = []

    for number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        print(
            f"[{number}/{len(TEST_CASES)}] "
            f"{test_case['test_id']}"
        )

        result = rag_eval.run_test(
            test_case
        )

        results.append(result)

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        cited_pages = result[
            "citation_validation"
        ][
            "unique_cited_pages"
        ]

        print(
            f"    {status} | "
            f"cited_pages={cited_pages} | "
            f"insufficient="
            f"{result['insufficient_returned']}"
        )

        answer_preview = " ".join(
            result["answer"].split()
        )[:500]

        print(
            f"    Answer: {answer_preview}"
        )

        for error in result["errors"]:
            print(
                f"    Error: {error}"
            )

        print()

    passed_count = sum(
        1
        for result in results
        if result["passed"]
    )

    test_count = len(results)

    failed_count = (
        test_count - passed_count
    )

    pass_rate = (
        passed_count / test_count
        if test_count
        else 0.0
    )

    all_tests_passed = (
        passed_count == test_count
    )

    report = {
        "schema_version": "1.0",
        "generated_at": (
            rag_eval.utc_now()
        ),
        "status": (
            "PASSED"
            if all_tests_passed
            else "FAILED"
        ),
        "evaluation": (
            "full-book-pilot-grounded-rag"
        ),
        "generation_model_id": (
            rag_eval.rag.GENERATION_MODEL_ID
        ),
        "pilot_page_range": {
            "start": 1,
            "end": 20,
        },
        "test_count": test_count,
        "passed_test_count": passed_count,
        "failed_test_count": failed_count,
        "pass_rate": pass_rate,
        "all_tests_passed": (
            all_tests_passed
        ),
        "tests": results,
        "opensearch_write": False,
    }

    report_path = (
        args.output_dir
        / "pilot-rag-evaluation-report.json"
    )

    write_json(
        report_path,
        report,
    )

    print("=" * 52)
    print("PILOT RAG REGRESSION RESULT")
    print("=" * 52)
    print(
        f"Passed:           "
        f"{passed_count}/{test_count}"
    )
    print(
        f"Failed:           {failed_count}"
    )
    print(
        f"Pass rate:        {pass_rate:.3f}"
    )
    print(
        f"Result:           "
        f"{report['status']}"
    )
    print(
        f"Report:           {report_path}"
    )
    print("OpenSearch write: False")

    return (
        0
        if all_tests_passed
        else 1
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print(
            "Pilot RAG evaluation interrupted.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except Exception as error:
        print(
            "Pilot RAG evaluation failed: "
            f"{error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
