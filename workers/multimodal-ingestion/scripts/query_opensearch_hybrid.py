from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import urllib3

import evaluate_opensearch_hybrid_retrieval as hybrid
import evaluate_opensearch_vector_retrieval as vector


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run weighted hybrid retrieval against "
            "the textbook OpenSearch index."
        )
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Question or search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--modality",
        choices=[
            "heading",
            "paragraph",
            "list",
            "table",
            "figure",
        ],
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    query = args.query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    if args.top_k < 1:
        raise ValueError(
            "top-k must be at least 1."
        )

    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=vector.REGION,
    )

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(
            connect=15.0,
            read=180.0,
        ),
        retries=False,
    )

    print("Generating query embedding...")

    query_vector, token_count = (
        vector.create_query_embedding(
            client=bedrock_client,
            query=query,
        )
    )

    vector_hits = hybrid.vector_search(
        http=http,
        query_vector=query_vector,
        candidate_modality=args.modality,
    )

    bm25_hits = hybrid.bm25_search(
        http=http,
        query=query,
        candidate_modality=args.modality,
    )

    fused_results = hybrid.fuse_results(
        vector_hits=vector_hits,
        bm25_hits=bm25_hits,
    )

    top_results: list[
        dict[str, Any]
    ] = []

    for rank, result in enumerate(
        fused_results[:args.top_k],
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
                "record_id": result[
                    "record_id"
                ],
                "hybrid_score": result[
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
                "element_type": source.get(
                    "element_type"
                ),
                "element_sub_type": (
                    source.get(
                        "element_sub_type"
                    )
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
                "embedding_text": source.get(
                    "embedding_text",
                    "",
                ),
                "asset_s3_uris": source.get(
                    "asset_s3_uris",
                    [],
                ),
            }
        )

    result_document = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "query": query,
        "query_token_count": token_count,
        "index_name": vector.INDEX_NAME,
        "fusion": {
            "method": (
                "weighted_reciprocal_rank_fusion"
            ),
            "rrf_constant": (
                hybrid.RRF_CONSTANT
            ),
            "vector_weight": (
                hybrid.VECTOR_WEIGHT
            ),
            "bm25_weight": (
                hybrid.BM25_WEIGHT
            ),
        },
        "modality_filter": args.modality,
        "result_count": len(
            top_results
        ),
        "results": top_results,
    }

    print()
    print("============================================")
    print("HYBRID TEXTBOOK SEARCH")
    print("============================================")
    print(f"Query:        {query}")
    print(f"Query tokens: {token_count}")
    print(f"Results:      {len(top_results)}")

    if args.modality:
        print(
            f"Modality:     {args.modality}"
        )

    for result in top_results:
        text = " ".join(
            str(
                result["embedding_text"]
            ).split()
        )

        if len(text) > 450:
            text = (
                text[:450].rstrip()
                + "..."
            )

        print()
        print("-" * 78)
        print(
            f"Rank {result['rank']} | "
            f"hybrid="
            f"{result['hybrid_score']:.8f}"
        )
        print(
            f"Vector rank: "
            f"{result['vector_rank']} | "
            f"BM25 rank: "
            f"{result['bm25_rank']}"
        )
        print(
            f"Modality: "
            f"{result['modality']}"
        )
        print(
            f"Pages:    "
            f"{result['source_page_numbers']}"
        )
        print(
            f"Citation: "
            f"{result['citation_label']}"
        )
        print(
            f"Record:   "
            f"{result['record_id']}"
        )
        print(f"Text:     {text}")

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(
                result_document,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        print()
        print(
            f"Saved: {args.output}"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as exc:
        print(
            f"Hybrid query failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
