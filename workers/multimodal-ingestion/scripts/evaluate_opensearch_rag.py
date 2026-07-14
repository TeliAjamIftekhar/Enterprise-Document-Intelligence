from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import answer_opensearch_rag as rag


OUTPUT_PATH = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/"
    "opensearch-serverless/"
    "rag-evaluation-report.json"
)


TEST_CASES: list[dict[str, Any]] = [
    {
        "test_id": "indigenous-meaning",
        "question": (
            "What does indigenous mean and where "
            "is this meaning given?"
        ),
        "top_k": 5,
        "modality": None,
        "expected_citation_pages": [90],
        "required_term_groups": [
            ["local"],
            ["origin"],
        ],
        "expect_insufficient": False,
    },
    {
        "test_id": "ancient-pankhi-evidence",
        "question": (
            "Where can evidence of ancient pankhi "
            "fans be found?"
        ),
        "top_k": 5,
        "modality": None,
        "expected_citation_pages": [90],
        "required_term_groups": [
            ["ajanta"],
            ["buddhist", "wall painting"],
        ],
        "expect_insufficient": False,
    },
    {
        "test_id": "zardozi-postage-stamp",
        "question": (
            "What is shown in the Zardozi hand fan "
            "postage stamp?"
        ),
        "top_k": 5,
        "modality": "figure",
        "expected_citation_pages": [91],
        "required_term_groups": [
            ["zardozi"],
            ["rajasthan"],
            ["stamp", "postage"],
        ],
        "expect_insufficient": False,
    },
    {
        "test_id": "mirror-work-fans",
        "question": (
            "Which state makes mirror-work hand fans "
            "and what material is used?"
        ),
        "top_k": 10,
        "modality": None,
        "expected_citation_pages": [91],
        "required_term_groups": [
            ["gujarat"],
            ["mirror"],
            ["cotton", "cloth"],
        ],
        "expect_insufficient": False,
    },
    {
        "test_id": "unsupported-outside-textbook",
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


INSUFFICIENT_MESSAGE = (
    "The provided textbook sources do not "
    "contain enough information to answer this."
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def normalize_text(value: str) -> str:
    return " ".join(
        value.lower().split()
    )


def validate_required_terms(
    answer: str,
    required_term_groups: list[list[str]],
) -> list[dict[str, Any]]:
    normalized_answer = normalize_text(
        answer
    )

    group_results: list[
        dict[str, Any]
    ] = []

    for alternatives in required_term_groups:
        matched_terms = [
            term
            for term in alternatives
            if normalize_text(term)
            in normalized_answer
        ]

        group_results.append(
            {
                "alternatives": alternatives,
                "matched_terms": matched_terms,
                "passed": bool(
                    matched_terms
                ),
            }
        )

    return group_results


def run_test(
    test_case: dict[str, Any],
) -> dict[str, Any]:
    question = str(
        test_case["question"]
    )

    results = rag.retrieve_context(
        query=question,
        top_k=int(
            test_case["top_k"]
        ),
        modality=test_case[
            "modality"
        ],
    )

    context, allowed_pages = (
        rag.build_context(results)
    )

    raw_answer, generation_metadata = (
        rag.generate_answer(
            question=question,
            context=context,
        )
    )

    normalized_answer, normalization = (
        rag.normalize_generated_answer(
            raw_answer
        )
    )

    answer, citation_alignment = (
        rag.realign_answer_citations(
            answer=normalized_answer,
            sources=results,
        )
    )

    answer_postprocessing = {
        "normalization": normalization,
        "citation_alignment": (
            citation_alignment
        ),
    }

    citation_validation = (
        rag.validate_citations(
            answer=answer,
            allowed_pages=allowed_pages,
        )
    )

    normalized_answer = (
        normalize_text(answer)
    )

    expect_insufficient = bool(
        test_case[
            "expect_insufficient"
        ]
    )

    insufficient_returned = (
        normalize_text(
            INSUFFICIENT_MESSAGE
        )
        in normalized_answer
    )

    term_results = (
        validate_required_terms(
            answer=answer,
            required_term_groups=(
                test_case[
                    "required_term_groups"
                ]
            ),
        )
    )

    expected_pages = set(
        int(page)
        for page in test_case[
            "expected_citation_pages"
        ]
    )

    cited_pages = set(
        citation_validation[
            "unique_cited_pages"
        ]
    )

    errors: list[str] = []

    if not citation_validation[
        "passed"
    ]:
        errors.extend(
            citation_validation[
                "errors"
            ]
        )

    if expect_insufficient:
        if not insufficient_returned:
            errors.append(
                "Expected insufficient-information "
                "response was not returned."
            )

        if cited_pages:
            errors.append(
                "Unsupported answer should not "
                "contain page citations."
            )

    else:
        if insufficient_returned:
            errors.append(
                "Model returned insufficient "
                "information for an answerable "
                "question."
            )

        if (
            expected_pages
            and not cited_pages
        ):
            errors.append(
                "Expected citation page is missing."
            )

        elif (
            expected_pages
            and not cited_pages.issubset(
                expected_pages
            )
        ):
            errors.append(
                "Answer contains an unsupported "
                "citation page. "
                f"Allowed "
                f"{sorted(expected_pages)}, "
                f"received "
                f"{sorted(cited_pages)}."
            )

        failed_groups = [
            result
            for result in term_results
            if not result["passed"]
        ]

        if failed_groups:
            errors.append(
                "Required answer terms are missing: "
                + json.dumps(
                    failed_groups,
                    ensure_ascii=False,
                )
            )

    return {
        "test_id": test_case[
            "test_id"
        ],
        "question": question,
        "passed": not errors,
        "errors": errors,
        "answer": answer,
        "expect_insufficient": (
            expect_insufficient
        ),
        "insufficient_returned": (
            insufficient_returned
        ),
        "expected_citation_pages": sorted(
            expected_pages
        ),
        "citation_validation": (
            citation_validation
        ),
        "required_term_results": (
            term_results
        ),
        "retrieved_sources": [
            {
                "rank": result["rank"],
                "record_id": result[
                    "record_id"
                ],
                "modality": result[
                    "modality"
                ],
                "source_page_numbers": (
                    result[
                        "source_page_numbers"
                    ]
                ),
                "citation_label": result[
                    "citation_label"
                ],
            }
            for result in results
        ],
        "generation": {
            **generation_metadata,
            "raw_answer": raw_answer,
            "postprocessing": (
                answer_postprocessing
            ),
        },
    }


def main() -> int:
    print(
        "============================================"
    )
    print(
        "GROUNDED RAG REGRESSION"
    )
    print(
        "============================================"
    )
    print(
        f"Tests: {len(TEST_CASES)}"
    )
    print()

    results: list[
        dict[str, Any]
    ] = []

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        print(
            f"[{test_number}/"
            f"{len(TEST_CASES)}] "
            f"{test_case['test_id']}"
        )

        result = run_test(
            test_case
        )

        results.append(result)

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        cited_pages = (
            result[
                "citation_validation"
            ][
                "unique_cited_pages"
            ]
        )

        print(
            f"    {status} | "
            f"cited_pages={cited_pages} | "
            f"insufficient="
            f"{result['insufficient_returned']}"
        )

        print(
            "    Answer: "
            + " ".join(
                result["answer"].split()
            )[:350]
        )

        if result["errors"]:
            for error in result[
                "errors"
            ]:
                print(
                    f"    Error: {error}"
                )

        print()

    passed_count = sum(
        1
        for result in results
        if result["passed"]
    )

    failed_count = (
        len(results)
        - passed_count
    )

    pass_rate = (
        passed_count
        / len(results)
    )

    all_tests_passed = (
        passed_count
        == len(results)
    )

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": (
            "PASSED"
            if all_tests_passed
            else "FAILED"
        ),
        "generation_model_id": (
            rag.GENERATION_MODEL_ID
        ),
        "test_count": len(
            results
        ),
        "passed_test_count": (
            passed_count
        ),
        "failed_test_count": (
            failed_count
        ),
        "pass_rate": pass_rate,
        "all_tests_passed": (
            all_tests_passed
        ),
        "tests": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        "============================================"
    )
    print(
        "RAG REGRESSION RESULT"
    )
    print(
        "============================================"
    )
    print(
        f"Passed:     "
        f"{passed_count}/{len(results)}"
    )
    print(
        f"Failed:     {failed_count}"
    )
    print(
        f"Pass rate:  {pass_rate:.3f}"
    )
    print(
        "Result:     "
        + (
            "PASSED"
            if all_tests_passed
            else "FAILED"
        )
    )
    print(
        f"Report:     {OUTPUT_PATH}"
    )

    return (
        0
        if all_tests_passed
        else 1
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as exc:
        print(
            f"RAG evaluation failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
