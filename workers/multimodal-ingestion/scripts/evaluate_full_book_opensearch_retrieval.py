from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import urllib3

import evaluate_opensearch_hybrid_retrieval as hybrid_eval
import evaluate_opensearch_vector_retrieval as vector_eval


DEFAULT_OUTPUT = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/"
    "opensearch-serverless/"
    "full-book-retrieval-regression-report.json"
)


TEST_CASES: list[dict[str, Any]] = [
    {
        "test_id": "mirror-work-table",
        "query": (
            "Which state makes mirror work hand fans "
            "and what materials are used?"
        ),
        "expected_page": 93,
        "expected_modality": "table",
        "required_terms": [
            "gujarat",
            "mirror work hand fans",
            "beads",
            "leather",
        ],
        "vector_max_rank": 1,
        "hybrid_max_rank": 1,
    },
    {
        "test_id": "indigenous-vocabulary",
        "query": (
            "What does the word indigenous mean?"
        ),
        "expected_page": 90,
        "expected_modality": "table",
        "required_terms": [
            "indigenous",
            "local",
            "originated",
        ],
        "vector_max_rank": 1,
        # The query is ambiguous across the full book:
        # page 213 contains indigenous instruments.
        # The correct vocabulary table must remain in top 2.
        "hybrid_max_rank": 2,
    },
    {
        "test_id": "ancient-pankhi-evidence",
        "query": (
            "Where can evidence of ancient pankhi "
            "fans be found?"
        ),
        "expected_page": 90,
        "expected_modality": "paragraph",
        "required_terms": [
            "pankhi",
            "buddhist",
            "ajanta",
        ],
        "vector_max_rank": 1,
        "hybrid_max_rank": 1,
    },
    {
        "test_id": "zardozi-visual",
        "query": (
            "Show the postage stamp depicting a "
            "Zardozi hand fan from Rajasthan."
        ),
        "expected_page": 91,
        "expected_modality": "figure",
        "required_terms": [
            "zardozi",
            "rajasthan",
            "hand fan",
        ],
        "candidate_modality": "figure",
        "vector_max_rank": 1,
        "hybrid_max_rank": 1,
    },
]


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def normalize_text(value: Any) -> str:
    return " ".join(
        str(value).casefold().split()
    )


def source_matches(
    source: dict[str, Any],
    test_case: dict[str, Any],
) -> bool:
    pages = source.get(
        "source_page_numbers",
        [],
    )

    if not isinstance(pages, list):
        return False

    if test_case["expected_page"] not in pages:
        return False

    if (
        source.get("modality")
        != test_case["expected_modality"]
    ):
        return False

    embedding_text = normalize_text(
        source.get(
            "embedding_text",
            "",
        )
    )

    required_terms = [
        normalize_text(term)
        for term in test_case.get(
            "required_terms",
            [],
        )
    ]

    return all(
        term in embedding_text
        for term in required_terms
    )


def find_raw_hit_rank(
    hits: list[dict[str, Any]],
    test_case: dict[str, Any],
) -> tuple[
    int | None,
    str | None,
    dict[str, Any] | None,
    float | None,
]:
    for rank, hit in enumerate(
        hits,
        start=1,
    ):
        source = hit.get(
            "_source",
            {},
        )

        if not isinstance(source, dict):
            continue

        if not source_matches(
            source,
            test_case,
        ):
            continue

        score_value = hit.get("_score")

        score = (
            float(score_value)
            if isinstance(
                score_value,
                (int, float),
            )
            else None
        )

        return (
            rank,
            str(hit.get("_id", "")),
            source,
            score,
        )

    return None, None, None, None


def find_fused_rank(
    results: list[dict[str, Any]],
    test_case: dict[str, Any],
) -> tuple[
    int | None,
    str | None,
    dict[str, Any] | None,
    float | None,
]:
    for rank, result in enumerate(
        results,
        start=1,
    ):
        source = result.get(
            "source",
            {},
        )

        if not isinstance(source, dict):
            continue

        if not source_matches(
            source,
            test_case,
        ):
            continue

        score_value = result.get(
            "rrf_score"
        )

        score = (
            float(score_value)
            if isinstance(
                score_value,
                (int, float),
            )
            else None
        )

        return (
            rank,
            str(
                result.get(
                    "record_id",
                    "",
                )
            ),
            source,
            score,
        )

    return None, None, None, None


def raw_top_results(
    hits: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for rank, hit in enumerate(
        hits[:limit],
        start=1,
    ):
        source = hit.get(
            "_source",
            {},
        )

        if not isinstance(source, dict):
            source = {}

        results.append(
            {
                "rank": rank,
                "record_id": str(
                    hit.get("_id", "")
                ),
                "score": hit.get("_score"),
                "modality": source.get(
                    "modality"
                ),
                "source_page_numbers": (
                    source.get(
                        "source_page_numbers",
                        [],
                    )
                ),
                "citation_label": source.get(
                    "citation_label"
                ),
                "text_preview": " ".join(
                    str(
                        source.get(
                            "embedding_text",
                            "",
                        )
                    ).split()
                )[:300],
            }
        )

    return results


def fused_top_results(
    results: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    top_results: list[dict[str, Any]] = []

    for rank, result in enumerate(
        results[:limit],
        start=1,
    ):
        source = result.get(
            "source",
            {},
        )

        if not isinstance(source, dict):
            source = {}

        top_results.append(
            {
                "rank": rank,
                "record_id": str(
                    result.get(
                        "record_id",
                        "",
                    )
                ),
                "rrf_score": result.get(
                    "rrf_score"
                ),
                "vector_rank": result.get(
                    "vector_rank"
                ),
                "vector_score": result.get(
                    "vector_score"
                ),
                "bm25_rank": result.get(
                    "bm25_rank"
                ),
                "bm25_score": result.get(
                    "bm25_score"
                ),
                "modality": source.get(
                    "modality"
                ),
                "source_page_numbers": (
                    source.get(
                        "source_page_numbers",
                        [],
                    )
                ),
                "citation_label": source.get(
                    "citation_label"
                ),
                "text_preview": " ".join(
                    str(
                        source.get(
                            "embedding_text",
                            "",
                        )
                    ).split()
                )[:300],
            }
        )

    return top_results


def reciprocal_rank(
    rank: int | None,
) -> float:
    if not isinstance(rank, int):
        return 0.0

    if rank < 1:
        return 0.0

    return 1.0 / rank


def run_test(
    bedrock_client: Any,
    http: urllib3.PoolManager,
    test_case: dict[str, Any],
) -> dict[str, Any]:
    query = str(
        test_case["query"]
    )

    query_vector, token_count = (
        vector_eval.create_query_embedding(
            client=bedrock_client,
            query=query,
        )
    )

    candidate_modality = test_case.get(
        "candidate_modality"
    )

    if not isinstance(
        candidate_modality,
        str,
    ):
        candidate_modality = None

    vector_hits = hybrid_eval.vector_search(
        http=http,
        query_vector=query_vector,
        candidate_modality=(
            candidate_modality
        ),
    )

    bm25_hits = hybrid_eval.bm25_search(
        http=http,
        query=query,
        candidate_modality=(
            candidate_modality
        ),
    )

    fused_results = hybrid_eval.fuse_results(
        vector_hits=vector_hits,
        bm25_hits=bm25_hits,
    )

    (
        vector_rank,
        vector_record_id,
        vector_source,
        vector_score,
    ) = find_raw_hit_rank(
        vector_hits,
        test_case,
    )

    (
        bm25_rank,
        bm25_record_id,
        bm25_source,
        bm25_score,
    ) = find_raw_hit_rank(
        bm25_hits,
        test_case,
    )

    (
        hybrid_rank,
        hybrid_record_id,
        hybrid_source,
        hybrid_score,
    ) = find_fused_rank(
        fused_results,
        test_case,
    )

    vector_max_rank = int(
        test_case["vector_max_rank"]
    )

    hybrid_max_rank = int(
        test_case["hybrid_max_rank"]
    )

    vector_errors: list[str] = []
    hybrid_errors: list[str] = []

    if vector_rank is None:
        vector_errors.append(
            "Expected semantic target was absent "
            "from vector candidates."
        )

    elif vector_rank > vector_max_rank:
        vector_errors.append(
            "Expected semantic target ranked below "
            f"the vector threshold: rank={vector_rank}, "
            f"maximum={vector_max_rank}."
        )

    if hybrid_rank is None:
        hybrid_errors.append(
            "Expected semantic target was absent "
            "from hybrid candidates."
        )

    elif hybrid_rank > hybrid_max_rank:
        hybrid_errors.append(
            "Expected semantic target ranked below "
            f"the hybrid threshold: rank={hybrid_rank}, "
            f"maximum={hybrid_max_rank}."
        )

    return {
        "test_id": test_case["test_id"],
        "query": query,
        "query_token_count": token_count,
        "expected_page": (
            test_case["expected_page"]
        ),
        "expected_modality": (
            test_case["expected_modality"]
        ),
        "required_terms": (
            test_case["required_terms"]
        ),
        "candidate_modality": (
            candidate_modality
        ),
        "vector": {
            "maximum_accepted_rank": (
                vector_max_rank
            ),
            "actual_rank": vector_rank,
            "matched_record_id": (
                vector_record_id
            ),
            "score": vector_score,
            "passed": not vector_errors,
            "errors": vector_errors,
            "matched_source_page_numbers": (
                vector_source.get(
                    "source_page_numbers",
                    [],
                )
                if isinstance(
                    vector_source,
                    dict,
                )
                else []
            ),
            "top_results": raw_top_results(
                vector_hits
            ),
        },
        "bm25_diagnostic": {
            "actual_rank": bm25_rank,
            "matched_record_id": (
                bm25_record_id
            ),
            "score": bm25_score,
            "matched_source_page_numbers": (
                bm25_source.get(
                    "source_page_numbers",
                    [],
                )
                if isinstance(
                    bm25_source,
                    dict,
                )
                else []
            ),
        },
        "hybrid": {
            "maximum_accepted_rank": (
                hybrid_max_rank
            ),
            "actual_rank": hybrid_rank,
            "matched_record_id": (
                hybrid_record_id
            ),
            "score": hybrid_score,
            "passed": not hybrid_errors,
            "errors": hybrid_errors,
            "matched_source_page_numbers": (
                hybrid_source.get(
                    "source_page_numbers",
                    [],
                )
                if isinstance(
                    hybrid_source,
                    dict,
                )
                else []
            ),
            "top_results": fused_top_results(
                fused_results
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate full-book vector and hybrid "
            "retrieval without invocation-specific IDs."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=vector_eval.REGION,
    )

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(
            connect=10.0,
            read=120.0,
        )
    )

    print("=" * 72)
    print("FULL-BOOK OPENSEARCH RETRIEVAL REGRESSION")
    print("=" * 72)
    print(
        f"Index:   {vector_eval.INDEX_NAME}"
    )
    print(
        f"Tests:   {len(TEST_CASES)}"
    )
    print(
        "Criteria: page + modality + content anchors"
    )
    print()

    results: list[dict[str, Any]] = []

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        print(
            f"[{test_number}/{len(TEST_CASES)}] "
            f"{test_case['test_id']}"
        )

        result = run_test(
            bedrock_client=bedrock_client,
            http=http,
            test_case=test_case,
        )

        results.append(result)

        vector_result = result["vector"]
        hybrid_result = result["hybrid"]

        print(
            "    Vector: "
            f"{'PASS' if vector_result['passed'] else 'FAIL'} "
            f"| rank={vector_result['actual_rank']} "
            f"| max={vector_result['maximum_accepted_rank']}"
        )

        print(
            "    Hybrid: "
            f"{'PASS' if hybrid_result['passed'] else 'FAIL'} "
            f"| rank={hybrid_result['actual_rank']} "
            f"| max={hybrid_result['maximum_accepted_rank']}"
        )

    vector_pass_count = sum(
        1
        for result in results
        if result["vector"]["passed"]
    )

    hybrid_pass_count = sum(
        1
        for result in results
        if result["hybrid"]["passed"]
    )

    vector_top_1_count = sum(
        1
        for result in results
        if result["vector"]["actual_rank"] == 1
    )

    hybrid_top_1_count = sum(
        1
        for result in results
        if result["hybrid"]["actual_rank"] == 1
    )

    vector_mrr = sum(
        reciprocal_rank(
            result["vector"]["actual_rank"]
        )
        for result in results
    ) / len(results)

    hybrid_mrr = sum(
        reciprocal_rank(
            result["hybrid"]["actual_rank"]
        )
        for result in results
    ) / len(results)

    all_tests_passed = (
        vector_pass_count == len(results)
        and hybrid_pass_count == len(results)
    )

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": (
            "PASSED"
            if all_tests_passed
            else "FAILED"
        ),
        "evaluation_scope": "full-book",
        "matching_method": (
            "expected page, modality and "
            "content-anchor matching"
        ),
        "region": vector_eval.REGION,
        "collection_endpoint": (
            vector_eval.COLLECTION_ENDPOINT
        ),
        "index_name": vector_eval.INDEX_NAME,
        "embedding_model_id": (
            vector_eval.MODEL_ID
        ),
        "embedding_dimensions": (
            vector_eval.DIMENSIONS
        ),
        "fusion_method": (
            "weighted_reciprocal_rank_fusion"
        ),
        "test_count": len(results),
        "vector": {
            "threshold_pass_count": (
                vector_pass_count
            ),
            "threshold_fail_count": (
                len(results)
                - vector_pass_count
            ),
            "top_1_count": (
                vector_top_1_count
            ),
            "top_1_accuracy": (
                vector_top_1_count
                / len(results)
            ),
            "mean_reciprocal_rank": (
                vector_mrr
            ),
        },
        "hybrid": {
            "threshold_pass_count": (
                hybrid_pass_count
            ),
            "threshold_fail_count": (
                len(results)
                - hybrid_pass_count
            ),
            "top_1_count": (
                hybrid_top_1_count
            ),
            "top_1_accuracy": (
                hybrid_top_1_count
                / len(results)
            ),
            "mean_reciprocal_rank": (
                hybrid_mrr
            ),
        },
        "all_tests_passed": (
            all_tests_passed
        ),
        "tests": results,
        "titan_embedding_calls": (
            len(results)
        ),
        "opensearch_read_requests": (
            len(results) * 2
        ),
        "opensearch_write": False,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("FULL-BOOK RETRIEVAL RESULT")
    print("=" * 72)
    print(
        f"Vector threshold pass: "
        f"{vector_pass_count}/{len(results)}"
    )
    print(
        f"Vector top-1 accuracy: "
        f"{report['vector']['top_1_accuracy']:.3f}"
    )
    print(
        f"Vector MRR:            "
        f"{vector_mrr:.3f}"
    )
    print(
        f"Hybrid threshold pass: "
        f"{hybrid_pass_count}/{len(results)}"
    )
    print(
        f"Hybrid top-1 accuracy: "
        f"{report['hybrid']['top_1_accuracy']:.3f}"
    )
    print(
        f"Hybrid MRR:            "
        f"{hybrid_mrr:.3f}"
    )
    print(
        "Result:                "
        f"{report['status']}"
    )
    print(
        f"Report:                "
        f"{args.output}"
    )
    print("OpenSearch write:      False")

    return 0 if all_tests_passed else 1


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
            f"Evaluation failed: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
