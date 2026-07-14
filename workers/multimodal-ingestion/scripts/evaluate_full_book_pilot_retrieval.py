from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import boto3
import urllib3

import evaluate_opensearch_hybrid_retrieval as hybrid_eval
import evaluate_opensearch_vector_retrieval as vector_eval


TEST_CASES: list[dict[str, Any]] = [
    {
        "test_id": "kaveri-policy-alignment",
        "query": (
            "Which national education policy and "
            "curriculum framework is the Kaveri "
            "Grade 9 textbook aligned with?"
        ),
        "expected_record_id": (
            "grade-9-english-kaveri:v1:bda:"
            "5b97eb23-fc84-42bd-92ec-76b001c7029f:"
            "chunk-0001"
        ),
        "expected_page": 7,
        "expected_modality": "paragraph",
        "candidate_modality": "paragraph",
    },
    {
        "test_id": "reading-for-meaning-purpose",
        "query": (
            "What does Reading for Meaning encourage "
            "students to do?"
        ),
        "expected_record_id": (
            "grade-9-english-kaveri:v1:bda:"
            "e0cfe640-1fc2-40c4-9af6-f2bf3f775d8a:"
            "chunk-0001"
        ),
        "expected_page": 8,
        "expected_modality": "paragraph",
        "candidate_modality": "paragraph",
    },
    {
        "test_id": "winds-of-change-contents-page",
        "query": (
            "According to the table of contents, "
            "on which page does Winds of Change begin?"
        ),
        "expected_record_id": (
            "grade-9-english-kaveri:v1:bda:"
            "f9846ff6-8891-439a-b69d-8d206159e514:"
            "chunk-0001"
        ),
        "expected_page": 19,
        "expected_modality": "table",
        "candidate_modality": "table",
    },
    {
        "test_id": "fundamental-duties-list",
        "query": (
            "What duties must every citizen of India "
            "follow regarding the Constitution, "
            "National Flag and National Anthem?"
        ),
        "expected_record_id": (
            "grade-9-english-kaveri:v1:bda:"
            "e988a832-23ef-4148-83aa-7e77beef4c72:"
            "chunk-0001"
        ),
        "expected_page": 20,
        "expected_modality": "list",
        "candidate_modality": "list",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate vector and hybrid retrieval "
            "for full-book pilot pages 1-20."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def reciprocal_rank(
    rank: int | None,
) -> float:
    if (
        not isinstance(rank, int)
        or rank < 1
    ):
        return 0.0

    return 1.0 / rank


def build_summary(
    results: list[dict[str, Any]],
    rank_field: str,
) -> dict[str, Any]:
    passed_count = sum(
        1
        for result in results
        if result.get("passed") is True
    )

    reciprocal_ranks = [
        reciprocal_rank(
            result.get(rank_field)
        )
        for result in results
    ]

    test_count = len(results)

    return {
        "test_count": test_count,
        "passed_count": passed_count,
        "failed_count": (
            test_count - passed_count
        ),
        "top_1_accuracy": (
            passed_count / test_count
            if test_count
            else 0.0
        ),
        "mean_reciprocal_rank": (
            sum(reciprocal_ranks) / test_count
            if test_count
            else 0.0
        ),
        "all_tests_passed": (
            passed_count == test_count
        ),
    }


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

    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=vector_eval.REGION,
    )

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(
            connect=10.0,
            read=60.0,
        ),
    )

    print("=" * 52)
    print("FULL-BOOK PILOT RETRIEVAL REGRESSION")
    print("=" * 52)
    print(
        f"Endpoint: "
        f"{vector_eval.COLLECTION_ENDPOINT}"
    )
    print(
        f"Index:    {vector_eval.INDEX_NAME}"
    )
    print(
        f"Tests:    {len(TEST_CASES)}"
    )
    print("Pages:    7, 8, 19, 20")
    print()

    vector_results: list[
        dict[str, Any]
    ] = []

    print("VECTOR RETRIEVAL")
    print("-" * 52)

    for number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        result = vector_eval.run_test(
            bedrock_client=bedrock_client,
            http=http,
            test_case=test_case,
        )

        vector_results.append(result)

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"[{number}/{len(TEST_CASES)}] "
            f"{test_case['test_id']}"
        )
        print(
            f"    {status} | "
            f"rank={result.get('actual_rank')} | "
            f"score={result.get('actual_score')}"
        )

        if result.get("errors"):
            print(
                f"    Errors: "
                f"{result['errors']}"
            )

    vector_summary = build_summary(
        vector_results,
        rank_field="actual_rank",
    )

    print()
    print("HYBRID RETRIEVAL")
    print("-" * 52)

    hybrid_results: list[
        dict[str, Any]
    ] = []

    for number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        result = hybrid_eval.run_test(
            bedrock_client=bedrock_client,
            http=http,
            test_case=test_case,
        )

        hybrid_results.append(result)

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"[{number}/{len(TEST_CASES)}] "
            f"{test_case['test_id']}"
        )
        print(
            f"    {status} | "
            f"vector={result.get('vector_rank')} | "
            f"bm25={result.get('bm25_rank')} | "
            f"hybrid={result.get('hybrid_rank')}"
        )

        if result.get("errors"):
            print(
                f"    Errors: "
                f"{result['errors']}"
            )

    hybrid_summary = build_summary(
        hybrid_results,
        rank_field="hybrid_rank",
    )

    generated_at = vector_eval.utc_now()

    vector_report = {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "evaluation": (
            "full-book-pilot-vector-retrieval"
        ),
        "index_name": vector_eval.INDEX_NAME,
        "page_range": {
            "start": 1,
            "end": 20,
        },
        "summary": vector_summary,
        "results": vector_results,
        "opensearch_write": False,
    }

    hybrid_report = {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "evaluation": (
            "full-book-pilot-hybrid-retrieval"
        ),
        "index_name": vector_eval.INDEX_NAME,
        "page_range": {
            "start": 1,
            "end": 20,
        },
        "summary": hybrid_summary,
        "results": hybrid_results,
        "opensearch_write": False,
    }

    combined_report = {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "status": (
            "PASSED"
            if (
                vector_summary[
                    "all_tests_passed"
                ]
                and hybrid_summary[
                    "all_tests_passed"
                ]
            )
            else "FAILED"
        ),
        "index_name": vector_eval.INDEX_NAME,
        "pilot_page_range": {
            "start": 1,
            "end": 20,
        },
        "vector": vector_summary,
        "hybrid": hybrid_summary,
        "opensearch_write": False,
    }

    vector_path = (
        args.output_dir
        / "pilot-vector-retrieval-report.json"
    )

    hybrid_path = (
        args.output_dir
        / "pilot-hybrid-retrieval-report.json"
    )

    combined_path = (
        args.output_dir
        / "pilot-retrieval-summary.json"
    )

    write_json(
        vector_path,
        vector_report,
    )

    write_json(
        hybrid_path,
        hybrid_report,
    )

    write_json(
        combined_path,
        combined_report,
    )

    print()
    print("=" * 52)
    print("PILOT RETRIEVAL RESULT")
    print("=" * 52)
    print(
        "Vector passed: "
        f"{vector_summary['passed_count']}/"
        f"{vector_summary['test_count']}"
    )
    print(
        "Vector top-1:  "
        f"{vector_summary['top_1_accuracy']:.3f}"
    )
    print(
        "Vector MRR:    "
        f"{vector_summary['mean_reciprocal_rank']:.3f}"
    )
    print(
        "Hybrid passed: "
        f"{hybrid_summary['passed_count']}/"
        f"{hybrid_summary['test_count']}"
    )
    print(
        "Hybrid top-1:  "
        f"{hybrid_summary['top_1_accuracy']:.3f}"
    )
    print(
        "Hybrid MRR:    "
        f"{hybrid_summary['mean_reciprocal_rank']:.3f}"
    )
    print(
        f"Result:         "
        f"{combined_report['status']}"
    )
    print(
        f"Summary:        {combined_path}"
    )
    print("OpenSearch write: False")

    return (
        0
        if combined_report["status"]
        == "PASSED"
        else 1
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print(
            "Evaluation interrupted.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except Exception as error:
        print(
            f"Pilot retrieval evaluation failed: "
            f"{error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
