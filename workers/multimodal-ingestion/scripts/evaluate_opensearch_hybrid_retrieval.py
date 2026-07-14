from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import urllib3

import evaluate_opensearch_vector_retrieval as vector_eval


CANDIDATE_LIMIT = 20
RESULT_LIMIT = 5
RRF_CONSTANT = 60

VECTOR_WEIGHT = 6.0
BM25_WEIGHT = 1.0

OUTPUT_PATH = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/"
    "opensearch-serverless/"
    "hybrid-retrieval-evaluation-report.json"
)

SOURCE_FIELDS = [
    "record_id",
    "book_id",
    "book_version",
    "element_type",
    "element_sub_type",
    "modality",
    "source_page_numbers",
    "citation_label",
    "embedding_text",
    "asset_s3_uris",
]


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def execute_search(
    http: urllib3.PoolManager,
    body: dict[str, Any],
) -> list[dict[str, Any]]:
    status, response = vector_eval.signed_search(
        http=http,
        body=body,
    )

    if status < 200 or status >= 300:
        raise RuntimeError(
            f"Search failed with HTTP {status}:\n"
            + json.dumps(
                response,
                indent=2,
                default=str,
            )
        )

    hits = (
        response.get("hits", {})
        .get("hits", [])
    )

    if not isinstance(hits, list):
        raise RuntimeError(
            "Search response contains no hits list."
        )

    return hits


def vector_search(
    http: urllib3.PoolManager,
    query_vector: list[float],
    candidate_modality: str | None,
) -> list[dict[str, Any]]:
    candidate_limit = (
        10
        if candidate_modality == "figure"
        else CANDIDATE_LIMIT
    )

    parameters: dict[str, Any] = {
        "vector": query_vector,
        "k": candidate_limit,
    }

    if candidate_modality:
        parameters["filter"] = {
            "term": {
                "modality": candidate_modality
            }
        }

    body = {
        "size": candidate_limit,
        "_source": SOURCE_FIELDS,
        "query": {
            "knn": {
                "embedding": parameters
            }
        },
    }

    return execute_search(
        http=http,
        body=body,
    )


def bm25_search(
    http: urllib3.PoolManager,
    query: str,
    candidate_modality: str | None,
) -> list[dict[str, Any]]:
    text_query = {
        "multi_match": {
            "query": query,
            "fields": [
                "embedding_text^4",
                "citation_label",
            ],
            "type": "best_fields",
        }
    }

    if candidate_modality:
        query_body: dict[str, Any] = {
            "bool": {
                "must": [
                    text_query
                ],
                "filter": [
                    {
                        "term": {
                            "modality": (
                                candidate_modality
                            )
                        }
                    }
                ],
            }
        }

        candidate_limit = 10

    else:
        query_body = text_query
        candidate_limit = CANDIDATE_LIMIT

    body = {
        "size": candidate_limit,
        "_source": SOURCE_FIELDS,
        "query": query_body,
    }

    return execute_search(
        http=http,
        body=body,
    )


def fuse_results(
    vector_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    for channel, weight, hits in (
        (
            "vector",
            VECTOR_WEIGHT,
            vector_hits,
        ),
        (
            "bm25",
            BM25_WEIGHT,
            bm25_hits,
        ),
    ):
        for rank, hit in enumerate(
            hits,
            start=1,
        ):
            document_id = str(
                hit.get("_id", "")
            )

            if not document_id:
                continue

            source = hit.get(
                "_source",
                {},
            )

            if not isinstance(source, dict):
                source = {}

            if document_id not in fused:
                fused[document_id] = {
                    "record_id": document_id,
                    "source": source,
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "vector_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                }

            item = fused[document_id]

            item["rrf_score"] += (
                weight
                / (
                    RRF_CONSTANT
                    + rank
                )
            )

            item[f"{channel}_rank"] = rank

            score = hit.get("_score")

            if isinstance(
                score,
                (int, float),
            ):
                item[f"{channel}_score"] = (
                    float(score)
                )

    def best_channel_rank(
        item: dict[str, Any],
    ) -> int:
        ranks = [
            rank
            for rank in (
                item.get("vector_rank"),
                item.get("bm25_rank"),
            )
            if isinstance(rank, int)
        ]

        return min(ranks) if ranks else 999999

    return sorted(
        fused.values(),
        key=lambda item: (
            -float(item["rrf_score"]),
            best_channel_rank(item),
            str(item["record_id"]),
        ),
    )


def find_rank(
    hits: list[dict[str, Any]],
    expected_record_id: str,
) -> int | None:
    for rank, hit in enumerate(
        hits,
        start=1,
    ):
        if hit.get("_id") == expected_record_id:
            return rank

    return None


def run_test(
    bedrock_client: Any,
    http: urllib3.PoolManager,
    test_case: dict[str, Any],
) -> dict[str, Any]:
    query = str(test_case["query"])

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

    vector_hits = vector_search(
        http=http,
        query_vector=query_vector,
        candidate_modality=(
            candidate_modality
        ),
    )

    bm25_hits = bm25_search(
        http=http,
        query=query,
        candidate_modality=(
            candidate_modality
        ),
    )

    fused_results = fuse_results(
        vector_hits=vector_hits,
        bm25_hits=bm25_hits,
    )

    expected_record_id = str(
        test_case["expected_record_id"]
    )

    vector_rank = find_rank(
        vector_hits,
        expected_record_id,
    )

    bm25_rank = find_rank(
        bm25_hits,
        expected_record_id,
    )

    hybrid_rank: int | None = None
    expected_source: dict[str, Any] | None = None

    for rank, result in enumerate(
        fused_results,
        start=1,
    ):
        if (
            result["record_id"]
            == expected_record_id
        ):
            hybrid_rank = rank
            expected_source = result.get(
                "source"
            )
            break

    errors: list[str] = []

    if hybrid_rank != 1:
        errors.append(
            "Expected record did not rank first "
            f"after fusion: rank={hybrid_rank}"
        )

    if expected_source is None:
        errors.append(
            "Expected record is absent from "
            "hybrid candidates."
        )

    else:
        pages = expected_source.get(
            "source_page_numbers",
            [],
        )

        if (
            test_case["expected_page"]
            not in pages
        ):
            errors.append(
                "Expected source page is missing."
            )

        if (
            expected_source.get("modality")
            != test_case[
                "expected_modality"
            ]
        ):
            errors.append(
                "Expected modality does not match."
            )

    top_results: list[dict[str, Any]] = []

    for rank, result in enumerate(
        fused_results[:RESULT_LIMIT],
        start=1,
    ):
        source = result.get(
            "source",
            {},
        )

        top_results.append(
            {
                "rank": rank,
                "record_id": result[
                    "record_id"
                ],
                "rrf_score": result[
                    "rrf_score"
                ],
                "vector_rank": result[
                    "vector_rank"
                ],
                "vector_score": result[
                    "vector_score"
                ],
                "bm25_rank": result[
                    "bm25_rank"
                ],
                "bm25_score": result[
                    "bm25_score"
                ],
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

    return {
        "test_id": test_case["test_id"],
        "query": query,
        "query_token_count": token_count,
        "expected_record_id": (
            expected_record_id
        ),
        "expected_page": (
            test_case["expected_page"]
        ),
        "expected_modality": (
            test_case["expected_modality"]
        ),
        "candidate_modality": (
            candidate_modality
        ),
        "vector_rank": vector_rank,
        "bm25_rank": bm25_rank,
        "hybrid_rank": hybrid_rank,
        "passed": not errors,
        "errors": errors,
        "top_results": top_results,
    }


def main() -> int:
    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=vector_eval.REGION,
    )

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(
            connect=15.0,
            read=180.0,
        ),
        retries=False,
    )

    results: list[dict[str, Any]] = []

    print("============================================")
    print("OPENSEARCH HYBRID RETRIEVAL REGRESSION")
    print("============================================")
    print(
        f"Endpoint: {vector_eval.COLLECTION_ENDPOINT}"
    )
    print(f"Index:    {vector_eval.INDEX_NAME}")
    print("Fusion:   Weighted Reciprocal Rank Fusion")
    print(f"RRF k:    {RRF_CONSTANT}")
    print(f"Vector weight: {VECTOR_WEIGHT}")
    print(f"BM25 weight:   {BM25_WEIGHT}")
    print(
        f"Tests:    {len(vector_eval.TEST_CASES)}"
    )
    print()

    for test_number, test_case in enumerate(
        vector_eval.TEST_CASES,
        start=1,
    ):
        print(
            f"[{test_number}/"
            f"{len(vector_eval.TEST_CASES)}] "
            f"{test_case['test_id']}"
        )

        result = run_test(
            bedrock_client=bedrock_client,
            http=http,
            test_case=test_case,
        )

        results.append(result)

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"    {status} | "
            f"vector={result['vector_rank']} | "
            f"bm25={result['bm25_rank']} | "
            f"hybrid={result['hybrid_rank']}"
        )

    passed_count = sum(
        1
        for result in results
        if result["passed"]
    )

    reciprocal_ranks = [
        (
            1.0 / result["hybrid_rank"]
            if isinstance(
                result["hybrid_rank"],
                int,
            )
            and result["hybrid_rank"] > 0
            else 0.0
        )
        for result in results
    ]

    top_1_accuracy = (
        sum(
            1
            for result in results
            if result["hybrid_rank"] == 1
        )
        / len(results)
    )

    mean_reciprocal_rank = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    all_tests_passed = (
        passed_count == len(results)
    )

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": (
            "PASSED"
            if all_tests_passed
            else "FAILED"
        ),
        "region": vector_eval.REGION,
        "collection_endpoint": (
            vector_eval.COLLECTION_ENDPOINT
        ),
        "index_name": vector_eval.INDEX_NAME,
        "fusion_method": (
            "weighted_reciprocal_rank_fusion"
        ),
        "rrf_constant": RRF_CONSTANT,
        "vector_weight": VECTOR_WEIGHT,
        "bm25_weight": BM25_WEIGHT,
        "candidate_limit": CANDIDATE_LIMIT,
        "result_limit": RESULT_LIMIT,
        "embedding_model_id": (
            vector_eval.MODEL_ID
        ),
        "embedding_dimensions": (
            vector_eval.DIMENSIONS
        ),
        "test_count": len(results),
        "passed_test_count": passed_count,
        "failed_test_count": (
            len(results) - passed_count
        ),
        "top_1_accuracy": top_1_accuracy,
        "mean_reciprocal_rank": (
            mean_reciprocal_rank
        ),
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

    print()
    print("============================================")
    print("HYBRID RETRIEVAL RESULT")
    print("============================================")
    print(
        f"Passed:               "
        f"{passed_count}/{len(results)}"
    )
    print(
        f"Top-1 accuracy:       "
        f"{top_1_accuracy:.3f}"
    )
    print(
        f"Mean reciprocal rank: "
        f"{mean_reciprocal_rank:.3f}"
    )
    print(
        "Result:               "
        + (
            "PASSED"
            if all_tests_passed
            else "FAILED"
        )
    )
    print(f"Report:               {OUTPUT_PATH}")

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as exc:
        print(
            f"Hybrid retrieval evaluation failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
