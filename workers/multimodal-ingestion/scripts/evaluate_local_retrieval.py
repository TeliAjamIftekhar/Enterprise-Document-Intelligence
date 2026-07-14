from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3


REGION = "us-east-1"
MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 1024


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
        "maximum_rank": 1,
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
        "maximum_rank": 1,
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
        "maximum_rank": 1,
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
        "maximum_rank": 1,
    },
]


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at line "
                    f"{line_number}."
                )

            records.append(record)

    if not records:
        raise RuntimeError(
            "No embedding records were found."
        )

    return records


def vector_norm(
    vector: list[float],
) -> float:
    return math.sqrt(
        sum(
            value * value
            for value in vector
        )
    )


def cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    if len(left) != len(right):
        raise ValueError(
            "Vector dimensions differ: "
            f"{len(left)} != {len(right)}"
        )

    left_norm = vector_norm(left)
    right_norm = vector_norm(right)

    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot_product = sum(
        left_value * right_value
        for left_value, right_value
        in zip(left, right)
    )

    return (
        dot_product
        / (left_norm * right_norm)
    )


def extract_embedding(
    response_body: dict[str, Any],
) -> list[float]:
    embedding = response_body.get(
        "embedding"
    )

    if not isinstance(embedding, list):
        embeddings_by_type = response_body.get(
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
            "Titan response contains no embedding."
        )

    vector = [
        float(value)
        for value in embedding
    ]

    if len(vector) != DIMENSIONS:
        raise RuntimeError(
            "Unexpected query embedding size: "
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

    return vector


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

    response_body = json.loads(
        response["body"].read()
    )

    if not isinstance(response_body, dict):
        raise RuntimeError(
            "Titan response is not a JSON object."
        )

    vector = extract_embedding(
        response_body
    )

    token_count = response_body.get(
        "inputTextTokenCount"
    )

    if not isinstance(token_count, int):
        token_count = None

    return vector, token_count


def validate_document_vectors(
    records: list[dict[str, Any]],
) -> None:
    record_ids: set[str] = set()

    for record in records:
        record_id = record.get("record_id")

        if not isinstance(
            record_id,
            str,
        ) or not record_id:
            raise RuntimeError(
                "A document record has no record_id."
            )

        if record_id in record_ids:
            raise RuntimeError(
                f"Duplicate record_id: {record_id}"
            )

        record_ids.add(record_id)

        vector = record.get("embedding")

        if not isinstance(vector, list):
            raise RuntimeError(
                f"Missing vector: {record_id}"
            )

        if len(vector) != DIMENSIONS:
            raise RuntimeError(
                f"Incorrect vector length for "
                f"{record_id}: {len(vector)}"
            )


def run_test(
    client: Any,
    all_records: list[dict[str, Any]],
    test_case: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    query = str(test_case["query"])

    candidate_modality = test_case.get(
        "candidate_modality"
    )

    candidates = all_records

    if isinstance(
        candidate_modality,
        str,
    ) and candidate_modality:
        candidates = [
            record
            for record in all_records
            if str(
                record.get("modality", "")
            ).lower()
            == candidate_modality.lower()
        ]

    if not candidates:
        raise RuntimeError(
            f"No candidates for test "
            f"{test_case['test_id']}."
        )

    query_vector, token_count = (
        create_query_embedding(
            client=client,
            query=query,
        )
    )

    scored: list[
        tuple[float, dict[str, Any]]
    ] = []

    for record in candidates:
        vector = [
            float(value)
            for value in record["embedding"]
        ]

        score = cosine_similarity(
            query_vector,
            vector,
        )

        scored.append(
            (score, record)
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    expected_record_id = str(
        test_case["expected_record_id"]
    )

    expected_rank: int | None = None
    expected_score: float | None = None
    expected_record: dict[str, Any] | None = None

    for rank, (score, record) in enumerate(
        scored,
        start=1,
    ):
        if (
            record.get("record_id")
            == expected_record_id
        ):
            expected_rank = rank
            expected_score = score
            expected_record = record
            break

    errors: list[str] = []

    if expected_rank is None:
        errors.append(
            "Expected record was not found."
        )

    elif expected_rank > int(
        test_case["maximum_rank"]
    ):
        errors.append(
            "Expected record rank exceeded "
            f"{test_case['maximum_rank']}: "
            f"actual={expected_rank}"
        )

    if expected_record is not None:
        pages = expected_record.get(
            "source_page_numbers",
            [],
        )

        if (
            test_case["expected_page"]
            not in pages
        ):
            errors.append(
                "Expected page is missing: "
                f"{test_case['expected_page']}"
            )

        actual_modality = str(
            expected_record.get(
                "modality",
                "",
            )
        ).lower()

        if (
            actual_modality
            != str(
                test_case[
                    "expected_modality"
                ]
            ).lower()
        ):
            errors.append(
                "Modality mismatch: "
                f"expected="
                f"{test_case['expected_modality']}, "
                f"actual={actual_modality}"
            )

    top_results = []

    for rank, (score, record) in enumerate(
        scored[:top_k],
        start=1,
    ):
        top_results.append(
            {
                "rank": rank,
                "score": score,
                "record_id": record.get(
                    "record_id"
                ),
                "element_type": record.get(
                    "element_type"
                ),
                "element_sub_type": (
                    record.get(
                        "element_sub_type"
                    )
                ),
                "modality": record.get(
                    "modality"
                ),
                "source_page_numbers": (
                    record.get(
                        "source_page_numbers",
                        [],
                    )
                ),
                "citation_label": (
                    record.get(
                        "citation_label"
                    )
                ),
                "text_preview": " ".join(
                    str(
                        record.get(
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
        "candidate_count": len(candidates),
        "candidate_modality": (
            candidate_modality
        ),
        "expected_record_id": (
            expected_record_id
        ),
        "expected_page": (
            test_case["expected_page"]
        ),
        "expected_modality": (
            test_case["expected_modality"]
        ),
        "maximum_rank": (
            test_case["maximum_rank"]
        ),
        "actual_rank": expected_rank,
        "actual_score": expected_score,
        "passed": not errors,
        "errors": errors,
        "top_results": top_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate local semantic retrieval "
            "using fixed textbook regression tests."
        )
    )

    parser.add_argument(
        "embeddings_jsonl",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.top_k < 1:
        raise ValueError(
            "top-k must be at least 1."
        )

    records = load_jsonl(
        args.embeddings_jsonl
    )

    validate_document_vectors(records)

    record_ids = {
        str(record["record_id"])
        for record in records
    }

    for test_case in TEST_CASES:
        expected_id = str(
            test_case["expected_record_id"]
        )

        if expected_id not in record_ids:
            raise RuntimeError(
                "Benchmark expected record is "
                f"missing: {expected_id}"
            )

    client = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
    )

    results: list[dict[str, Any]] = []

    print("============================================")
    print("LOCAL RETRIEVAL REGRESSION")
    print("============================================")
    print(f"Records: {len(records)}")
    print(f"Tests:   {len(TEST_CASES)}")
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
            client=client,
            all_records=records,
            test_case=test_case,
            top_k=args.top_k,
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
        "region": REGION,
        "model_id": MODEL_ID,
        "dimensions": DIMENSIONS,
        "embeddings_jsonl": str(
            args.embeddings_jsonl
        ),
        "embedding_record_count": len(
            records
        ),
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

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("============================================")
    print("RETRIEVAL EVALUATION COMPLETED")
    print("============================================")
    print(
        f"Passed:              "
        f"{passed_count}/{len(results)}"
    )
    print(
        f"Top-1 accuracy:      "
        f"{report['top_1_accuracy']:.3f}"
    )
    print(
        f"Mean reciprocal rank:"
        f" {report['mean_reciprocal_rank']:.3f}"
    )
    print(
        "Result:              "
        + (
            "PASSED"
            if report["all_tests_passed"]
            else "FAILED"
        )
    )
    print(f"Report:              {args.output}")

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
            f"Retrieval evaluation failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
