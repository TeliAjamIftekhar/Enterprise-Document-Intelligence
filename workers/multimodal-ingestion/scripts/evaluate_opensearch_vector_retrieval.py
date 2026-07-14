from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import urllib3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


REGION = "us-east-1"
SERVICE = "aoss"

COLLECTION_ENDPOINT = (
    "https://"
    "kqjqddn0b5gmcfvgsd2e."
    "aoss.us-east-1.on.aws"
)

INDEX_NAME = "grade-9-english-kaveri-v1"

MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 1024
TOP_K = 5

OUTPUT_PATH = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/"
    "opensearch-serverless/"
    "vector-retrieval-evaluation-report.json"
)


TEST_CASES = [
    {
        "test_id": "mirror-work-table",
        "query": (
            "Which state makes mirror work hand fans "
            "and what materials are used?"
        ),
        "expected_record_id": (
            "grade-9-english-kaveri:v1:bda:"
            "2a8b15b2-f74a-406c-9bc2-206ef52e7cbc:"
            "chunk-0001"
        ),
        "expected_page": 93,
        "expected_modality": "table",
    },
    {
        "test_id": "indigenous-vocabulary",
        "query": (
            "What does the word indigenous mean?"
        ),
        "expected_record_id": (
            "grade-9-english-kaveri:v1:bda:"
            "53d8e6ce-4883-49da-8c64-3ad39fb0c57c:"
            "chunk-0001"
        ),
        "expected_page": 90,
        "expected_modality": "table",
    },
    {
        "test_id": "ancient-pankhi-evidence",
        "query": (
            "Where can evidence of ancient pankhi "
            "fans be found?"
        ),
        "expected_record_id": (
            "grade-9-english-kaveri:v1:bda:"
            "665c08c3-7ca1-4e1a-b1fe-795fd4c27da9:"
            "chunk-0001"
        ),
        "expected_page": 90,
        "expected_modality": "paragraph",
    },
    {
        "test_id": "zardozi-visual",
        "query": (
            "Show the postage stamp depicting a "
            "Zardozi hand fan from Rajasthan."
        ),
        "expected_record_id": (
            "grade-9-english-kaveri:v1:bda:"
            "ebe87479-5522-4dc1-9928-446cf81e3963:"
            "chunk-0001"
        ),
        "expected_page": 91,
        "expected_modality": "figure",
        "candidate_modality": "figure",
    },
]


RETRYABLE_HTTP_STATUS = {
    403,
    408,
    429,
    500,
    502,
    503,
    504,
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def vector_norm(
    vector: list[float],
) -> float:
    return math.sqrt(
        sum(
            value * value
            for value in vector
        )
    )


def get_credentials() -> Any:
    credentials = (
        boto3.Session()
        .get_credentials()
    )

    if credentials is None:
        raise RuntimeError(
            "No AWS credentials were resolved."
        )

    return credentials.get_frozen_credentials()


def create_query_embedding(
    client: Any,
    query: str,
) -> tuple[list[float], int | None]:
    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {
                "inputText": query,
                "dimensions": DIMENSIONS,
                "normalize": True,
            }
        ),
    )

    result = json.loads(
        response["body"].read()
    )

    embedding = result.get(
        "embedding"
    )

    if not isinstance(embedding, list):
        embeddings_by_type = result.get(
            "embeddingsByType",
            {},
        )

        if isinstance(
            embeddings_by_type,
            dict,
        ):
            embedding = embeddings_by_type.get(
                "float"
            )

    if not isinstance(embedding, list):
        raise RuntimeError(
            "Titan returned no float embedding."
        )

    vector = [
        float(value)
        for value in embedding
    ]

    if len(vector) != DIMENSIONS:
        raise RuntimeError(
            "Unexpected query-vector dimensions: "
            f"{len(vector)}"
        )

    if not all(
        math.isfinite(value)
        for value in vector
    ):
        raise RuntimeError(
            "Query vector contains non-finite values."
        )

    norm = vector_norm(vector)

    if not 0.98 <= norm <= 1.02:
        raise RuntimeError(
            f"Unexpected query-vector norm: {norm}"
        )

    token_count = result.get(
        "inputTextTokenCount"
    )

    if not isinstance(token_count, int):
        token_count = None

    return vector, token_count


def signed_search(
    http: urllib3.PoolManager,
    body: dict[str, Any],
    maximum_attempts: int = 12,
) -> tuple[int, dict[str, Any]]:
    url = (
        COLLECTION_ENDPOINT.rstrip("/")
        + "/"
        + INDEX_NAME
        + "/_search"
    )

    body_bytes = json.dumps(
        body,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    payload_hash = __import__(
        "hashlib"
    ).sha256(
        body_bytes
    ).hexdigest()

    for attempt in range(
        1,
        maximum_attempts + 1,
    ):
        request = AWSRequest(
            method="POST",
            url=url,
            data=body_bytes,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-amz-content-sha256": (
                    payload_hash
                ),
            },
        )

        SigV4Auth(
            get_credentials(),
            SERVICE,
            REGION,
        ).add_auth(request)

        prepared = request.prepare()

        response = http.request(
            method="POST",
            url=url,
            body=body_bytes,
            headers=dict(
                prepared.headers.items()
            ),
            preload_content=True,
        )

        text = response.data.decode(
            "utf-8",
            errors="replace",
        )

        try:
            result = (
                json.loads(text)
                if text
                else {}
            )
        except json.JSONDecodeError:
            result = {
                "raw_response": text,
            }

        if (
            response.status
            in RETRYABLE_HTTP_STATUS
            and attempt < maximum_attempts
        ):
            delay = min(
                attempt * 2,
                20,
            )

            print(
                f"    HTTP {response.status}; "
                f"retrying in {delay}s."
            )

            time.sleep(delay)
            continue

        return int(response.status), result

    raise RuntimeError(
        "Search retry loop ended unexpectedly."
    )


def run_test(
    bedrock_client: Any,
    http: urllib3.PoolManager,
    test_case: dict[str, Any],
) -> dict[str, Any]:
    query_vector, token_count = (
        create_query_embedding(
            bedrock_client,
            str(test_case["query"]),
        )
    )

    knn_parameters: dict[str, Any] = {
        "vector": query_vector,
        "k": TOP_K,
    }

    candidate_modality = test_case.get(
        "candidate_modality"
    )

    if isinstance(
        candidate_modality,
        str,
    ) and candidate_modality:
        knn_parameters["filter"] = {
            "term": {
                "modality": candidate_modality
            }
        }

    request_body = {
        "size": TOP_K,
        "_source": [
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
        ],
        "query": {
            "knn": {
                "embedding": knn_parameters
            }
        },
    }

    http_status, response = signed_search(
        http=http,
        body=request_body,
    )

    if not 200 <= http_status < 300:
        raise RuntimeError(
            f"Search failed with HTTP {http_status}:\n"
            + json.dumps(
                response,
                indent=2,
            )
        )

    hits_wrapper = response.get(
        "hits",
        {}
    )

    hits = hits_wrapper.get(
        "hits",
        []
    )

    if not isinstance(hits, list):
        raise RuntimeError(
            "Search response has no hits list."
        )

    expected_record_id = str(
        test_case["expected_record_id"]
    )

    actual_rank: int | None = None
    actual_score: float | None = None
    expected_source: dict[str, Any] | None = None

    top_results: list[
        dict[str, Any]
    ] = []

    for rank, hit in enumerate(
        hits,
        start=1,
    ):
        source = hit.get(
            "_source",
            {},
        )

        document_id = str(
            hit.get("_id", "")
        )

        result = {
            "rank": rank,
            "score": hit.get("_score"),
            "record_id": document_id,
            "modality": source.get(
                "modality"
            ),
            "element_type": source.get(
                "element_type"
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

        top_results.append(result)

        if document_id == expected_record_id:
            actual_rank = rank

            score = hit.get("_score")

            if isinstance(
                score,
                (int, float),
            ):
                actual_score = float(score)

            if isinstance(source, dict):
                expected_source = source

    errors: list[str] = []

    if actual_rank != 1:
        errors.append(
            "Expected document did not rank first: "
            f"actual_rank={actual_rank}"
        )

    if expected_source is not None:
        pages = expected_source.get(
            "source_page_numbers",
            [],
        )

        if (
            test_case["expected_page"]
            not in pages
        ):
            errors.append(
                "Expected page is missing."
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

    return {
        "test_id": test_case["test_id"],
        "query": test_case["query"],
        "query_token_count": token_count,
        "candidate_modality": (
            candidate_modality
        ),
        "http_status": http_status,
        "expected_record_id": (
            expected_record_id
        ),
        "expected_page": (
            test_case["expected_page"]
        ),
        "expected_modality": (
            test_case[
                "expected_modality"
            ]
        ),
        "actual_rank": actual_rank,
        "actual_score": actual_score,
        "passed": not errors,
        "errors": errors,
        "top_results": top_results,
    }


def main() -> int:
    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
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
    print("OPENSEARCH VECTOR RETRIEVAL REGRESSION")
    print("============================================")
    print(f"Endpoint: {COLLECTION_ENDPOINT}")
    print(f"Index:    {INDEX_NAME}")
    print(f"Tests:    {len(TEST_CASES)}")
    print()

    for index, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        print(
            f"[{index}/{len(TEST_CASES)}] "
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
            f"rank={result['actual_rank']} | "
            f"score={result['actual_score']}"
        )

    passed_count = sum(
        1
        for result in results
        if result["passed"]
    )

    reciprocal_ranks = [
        (
            1.0 / result["actual_rank"]
            if isinstance(
                result["actual_rank"],
                int,
            )
            and result["actual_rank"] > 0
            else 0.0
        )
        for result in results
    ]

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": (
            "PASSED"
            if passed_count == len(results)
            else "FAILED"
        ),
        "region": REGION,
        "collection_endpoint": (
            COLLECTION_ENDPOINT
        ),
        "index_name": INDEX_NAME,
        "embedding_model_id": MODEL_ID,
        "embedding_dimensions": DIMENSIONS,
        "test_count": len(results),
        "passed_test_count": passed_count,
        "failed_test_count": (
            len(results) - passed_count
        ),
        "top_1_accuracy": (
            sum(
                1
                for result in results
                if result["actual_rank"] == 1
            )
            / len(results)
        ),
        "mean_reciprocal_rank": (
            sum(reciprocal_ranks)
            / len(reciprocal_ranks)
        ),
        "all_tests_passed": (
            passed_count == len(results)
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
    print("VECTOR RETRIEVAL RESULT")
    print("============================================")
    print(
        f"Passed:               "
        f"{passed_count}/{len(results)}"
    )
    print(
        f"Top-1 accuracy:       "
        f"{report['top_1_accuracy']:.3f}"
    )
    print(
        f"Mean reciprocal rank: "
        f"{report['mean_reciprocal_rank']:.3f}"
    )
    print(
        "Result:               "
        + (
            "PASSED"
            if report["all_tests_passed"]
            else "FAILED"
        )
    )
    print(f"Report:               {OUTPUT_PATH}")

    return (
        0
        if report["all_tests_passed"]
        else 1
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as exc:
        print(
            f"Vector retrieval evaluation failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
