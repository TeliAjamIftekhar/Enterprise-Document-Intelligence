from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import boto3


DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_DIMENSIONS = 1024


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
                    f"Invalid JSON at line {line_number}: "
                    f"{exc}"
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


def extract_embedding(
    response_body: dict[str, Any],
) -> list[float]:
    embedding = response_body.get("embedding")

    if not isinstance(embedding, list):
        embeddings_by_type = response_body.get(
            "embeddingsByType",
            {},
        )

        if isinstance(embeddings_by_type, dict):
            embedding = embeddings_by_type.get(
                "float"
            )

    if not isinstance(embedding, list):
        raise RuntimeError(
            "Titan response contains no float embedding."
        )

    vector = [
        float(value)
        for value in embedding
    ]

    if not all(
        math.isfinite(value)
        for value in vector
    ):
        raise RuntimeError(
            "Query embedding contains non-finite values."
        )

    return vector


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
            "Vector dimensions do not match: "
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


def create_query_embedding(
    query: str,
    region: str,
    model_id: str,
    dimensions: int,
) -> tuple[list[float], int | None]:
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
    )

    request_body = {
        "inputText": query,
        "dimensions": dimensions,
        "normalize": True,
    }

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(request_body),
    )

    response_body = json.loads(
        response["body"].read()
    )

    vector = extract_embedding(
        response_body
    )

    if len(vector) != dimensions:
        raise RuntimeError(
            "Unexpected query-vector dimensions. "
            f"Expected={dimensions}, "
            f"actual={len(vector)}"
        )

    norm = vector_norm(vector)

    if not 0.98 <= norm <= 1.02:
        raise RuntimeError(
            f"Unexpected query-vector norm: {norm}"
        )

    token_count = response_body.get(
        "inputTextTokenCount"
    )

    if not isinstance(token_count, int):
        token_count = None

    return vector, token_count


def text_preview(
    value: Any,
    limit: int = 420,
) -> str:
    if not isinstance(value, str):
        return ""

    cleaned = " ".join(value.split())

    if len(cleaned) <= limit:
        return cleaned

    return cleaned[:limit].rstrip() + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local cosine-similarity retrieval "
            "against Titan embeddings."
        )
    )

    parser.add_argument(
        "embeddings_jsonl",
        type=Path,
    )

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
    )

    parser.add_argument(
        "--modality",
        default=None,
        help=(
            "Optional modality filter, such as "
            "table, figure, or paragraph."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.top_k < 1:
        raise ValueError(
            "top-k must be at least 1."
        )

    query = args.query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    records = load_jsonl(
        args.embeddings_jsonl
    )

    if args.modality:
        requested_modality = (
            args.modality.strip().lower()
        )

        records = [
            record
            for record in records
            if str(
                record.get("modality", "")
            ).lower() == requested_modality
        ]

        if not records:
            raise RuntimeError(
                "No records matched modality filter: "
                f"{requested_modality}"
            )

    print("Generating query embedding...")

    query_vector, token_count = (
        create_query_embedding(
            query=query,
            region=args.region,
            model_id=args.model_id,
            dimensions=args.dimensions,
        )
    )

    scored_records: list[
        tuple[float, dict[str, Any]]
    ] = []

    for record in records:
        vector = record.get("embedding")

        if not isinstance(vector, list):
            raise RuntimeError(
                "Record contains no embedding: "
                f"{record.get('record_id')}"
            )

        numeric_vector = [
            float(value)
            for value in vector
        ]

        score = cosine_similarity(
            query_vector,
            numeric_vector,
        )

        scored_records.append(
            (score, record)
        )

    scored_records.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    top_results = scored_records[
        :min(args.top_k, len(scored_records))
    ]

    print()
    print("============================================")
    print("LOCAL SEMANTIC RETRIEVAL")
    print("============================================")
    print(f"Query:        {query}")
    print(f"Query tokens: {token_count}")
    print(f"Candidates:   {len(records)}")
    print(f"Top K:        {len(top_results)}")

    if args.modality:
        print(f"Modality:     {args.modality}")

    for rank, (score, record) in enumerate(
        top_results,
        start=1,
    ):
        print()
        print("-" * 78)
        print(
            f"Rank {rank} | "
            f"score={score:.6f}"
        )
        print(
            f"Record:   {record['record_id']}"
        )
        print(
            f"Type:     "
            f"{record.get('element_type')} / "
            f"{record.get('element_sub_type')}"
        )
        print(
            f"Modality: {record.get('modality')}"
        )
        print(
            f"Pages:    "
            f"{record.get('source_page_numbers')}"
        )
        print(
            f"Citation: "
            f"{record.get('citation_label')}"
        )
        print(
            "Text:     "
            + text_preview(
                record.get("embedding_text")
            )
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as exc:
        print(
            f"Local retrieval failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
