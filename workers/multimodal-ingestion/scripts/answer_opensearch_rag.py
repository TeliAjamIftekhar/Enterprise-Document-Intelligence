from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import urllib3

import evaluate_opensearch_hybrid_retrieval as hybrid
import evaluate_opensearch_vector_retrieval as vector
from src.book_config import load_book_config


GENERATION_MODEL_ID = "amazon.nova-lite-v1:0"

DEFAULT_TOP_K = 5
MAX_CONTEXT_CHARACTERS = 12000

INSUFFICIENT_MESSAGE = (
    "The provided textbook sources do not "
    "contain enough information to answer this."
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def configure_runtime_from_config(
    config_path: Path | None,
) -> Any | None:
    """Apply book configuration to retrieval and RAG."""

    global GENERATION_MODEL_ID

    if config_path is None:
        return None

    config = load_book_config(
        config_path
    )

    vector.REGION = config.aws.region
    vector.COLLECTION_ENDPOINT = (
        config.opensearch.collection_endpoint
    )
    vector.INDEX_NAME = (
        config.opensearch.index_name
    )
    vector.MODEL_ID = (
        config.models.embedding.model_id
    )
    vector.DIMENSIONS = (
        config.models.embedding.dimensions
    )

    hybrid.VECTOR_CANDIDATE_LIMIT = (
        config.retrieval.vector_candidate_limit
    )
    hybrid.BM25_CANDIDATE_LIMIT = (
        config.retrieval.bm25_candidate_limit
    )
    hybrid.CANDIDATE_LIMIT = max(
        hybrid.VECTOR_CANDIDATE_LIMIT,
        hybrid.BM25_CANDIDATE_LIMIT,
    )
    hybrid.RESULT_LIMIT = (
        config.retrieval.result_limit
    )
    hybrid.RRF_CONSTANT = (
        config.retrieval.rrf_constant
    )
    hybrid.VECTOR_WEIGHT = (
        config.retrieval.vector_weight
    )
    hybrid.BM25_WEIGHT = (
        config.retrieval.bm25_weight
    )

    GENERATION_MODEL_ID = (
        config.models.generation.model_id
    )

    return config


def resolve_top_k(
    requested_top_k: int | None,
    config: Any | None,
) -> int:
    if requested_top_k is not None:
        return requested_top_k

    if config is not None:
        return int(
            config.retrieval.result_limit
        )

    return DEFAULT_TOP_K


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve textbook passages from "
            "OpenSearch and generate a grounded "
            "answer with page citations."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional book configuration JSON. "
            "When omitted, legacy Kaveri defaults "
            "are preserved."
        ),
    )

    parser.add_argument(
        "--query",
        required=True,
        help="User's textbook question.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Number of RAG context results. "
            "Defaults to the configured result limit "
            "or 5 in legacy mode."
        ),
    )

    parser.add_argument(
        "--modality",
        choices=[
            "heading",
            "paragraph",
            "list",
            "table",
            "figure",
            "diagram",
        ],
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def generation_exclusion_reasons(
    modality: Any,
    text: str,
) -> list[str]:
    reasons: list[str] = []

    if modality != "table":
        return reasons

    marker = "Structured CSV:"

    if marker not in text:
        return reasons

    csv_blob = text.split(
        marker,
        1,
    )[1].strip()

    row_strings = re.split(
        r"\s+(?=[0-9]+,)",
        csv_blob,
    )

    data_rows = [
        row
        for row in row_strings
        if re.match(
            r"^[0-9]+,",
            row,
        )
    ]

    if len(data_rows) < 2:
        return reasons

    sparse_row_count = 0

    for row in data_rows:
        try:
            cells = next(
                csv.reader([row])
            )
        except (
            csv.Error,
            StopIteration,
        ):
            reasons.append(
                "unparseable_structured_csv"
            )
            return reasons

        # First value is the generated CSV row index.
        values = cells[1:]

        empty_count = sum(
            1
            for value in values
            if not value.strip()
        )

        if empty_count > 0:
            sparse_row_count += 1

    sparse_ratio = (
        sparse_row_count
        / len(data_rows)
    )

    if sparse_ratio >= 0.30:
        reasons.append(
            "sparse_table_alignment"
        )

    return reasons


def select_chapter_consistent_results(
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Conservatively remove cross-chapter noise.

    Filtering is applied only when the top-ranked
    chapter has a strict majority among the initial
    top-k candidates and at least two supporting
    candidates. If chapter metadata is unavailable
    or the result set is mixed, normal retrieval
    behaviour is preserved.
    """

    baseline = candidates[:top_k]

    if not baseline:
        return []

    top_chapter_id = baseline[0].get(
        "chapter_id"
    )

    top_chapter_title = baseline[0].get(
        "chapter_title"
    )

    top_chapter_status = baseline[0].get(
        "chapter_context_status"
    )

    comparable = [
        candidate
        for candidate in baseline
        if (
            candidate.get(
                "chapter_context_status"
            )
            == "single"
            and isinstance(
                candidate.get("chapter_id"),
                str,
            )
            and candidate.get("chapter_id")
        )
    ]

    should_filter = False

    if (
        top_chapter_status == "single"
        and isinstance(
            top_chapter_id,
            str,
        )
        and top_chapter_id
    ):
        top_chapter_count = sum(
            1
            for candidate in comparable
            if candidate.get("chapter_id")
            == top_chapter_id
        )

        should_filter = (
            top_chapter_count >= 2
            and top_chapter_count * 2
            > len(comparable)
        )

    selected = baseline

    if should_filter:
        same_chapter = [
            candidate
            for candidate in candidates
            if (
                candidate.get(
                    "chapter_context_status"
                )
                == "single"
                and candidate.get(
                    "chapter_id"
                )
                == top_chapter_id
            )
        ][:top_k]

        if len(same_chapter) >= 2:
            selected = same_chapter
        else:
            should_filter = False

    finalized: list[
        dict[str, Any]
    ] = []

    for context_rank, candidate in enumerate(
        selected,
        start=1,
    ):
        finalized.append({
            **candidate,
            "rank": context_rank,
            "chapter_filter_applied": (
                should_filter
            ),
            "selected_chapter_id": (
                top_chapter_id
                if should_filter
                else None
            ),
            "selected_chapter_title": (
                top_chapter_title
                if should_filter
                else None
            ),
        })

    return finalized


def retrieve_context(
    query: str,
    top_k: int,
    modality: str | None,
) -> list[dict[str, Any]]:
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

    query_vector, _ = (
        vector.create_query_embedding(
            client=bedrock_client,
            query=query,
        )
    )

    vector_hits = hybrid.vector_search(
        http=http,
        query_vector=query_vector,
        candidate_modality=modality,
    )

    bm25_hits = hybrid.bm25_search(
        http=http,
        query=query,
        candidate_modality=modality,
    )

    fused_results = hybrid.fuse_results(
        vector_hits=vector_hits,
        bm25_hits=bm25_hits,
    )

    results: list[dict[str, Any]] = []
    excluded_candidates: list[
        dict[str, Any]
    ] = []

    for hybrid_rank, result in enumerate(
        fused_results,
        start=1,
    ):
        source = result.get(
            "source",
            {},
        )

        if not isinstance(source, dict):
            source = {}

        pages = source.get(
            "source_page_numbers",
            [],
        )

        if not isinstance(pages, list):
            pages = []

        text = str(
            source.get(
                "embedding_text",
                "",
            )
        ).strip()

        if not text:
            continue

        source_modality = source.get(
            "modality"
        )

        exclusion_reasons = (
            generation_exclusion_reasons(
                modality=source_modality,
                text=text,
            )
        )

        if exclusion_reasons:
            excluded_candidates.append(
                {
                    "hybrid_rank": (
                        hybrid_rank
                    ),
                    "record_id": result[
                        "record_id"
                    ],
                    "modality": (
                        source_modality
                    ),
                    "source_page_numbers": (
                        pages
                    ),
                    "reasons": (
                        exclusion_reasons
                    ),
                }
            )
            continue

        results.append(
            {
                "hybrid_rank": hybrid_rank,
                "record_id": result[
                    "record_id"
                ],
                "hybrid_score": result[
                    "rrf_score"
                ],
                "vector_rank": result[
                    "vector_rank"
                ],
                "bm25_rank": result[
                    "bm25_rank"
                ],
                "document_type": source.get(
                    "document_type"
                ),
                "modality": source_modality,
                "source_page_numbers": pages,
                "page_context_status": source.get(
                    "page_context_status"
                ),
                "chapter_id": source.get(
                    "chapter_id"
                ),
                "chapter_title": source.get(
                    "chapter_title"
                ),
                "chapter_context_status": (
                    source.get(
                        "chapter_context_status"
                    )
                ),
                "chapter_ids": source.get(
                    "chapter_ids",
                    [],
                ),
                "chapter_titles": source.get(
                    "chapter_titles",
                    [],
                ),
                "citation_label": (
                    source.get(
                        "context_citation_label"
                    )
                    or source.get(
                        "citation_label"
                    )
                ),
                "embedding_text": text,
                "asset_s3_uris": source.get(
                    "asset_s3_uris",
                    [],
                ),
                "generation_exclusion_reasons": [],
            }
        )

    results = select_chapter_consistent_results(
        candidates=results,
        top_k=top_k,
    )

    if not results:
        raise RuntimeError(
            "Hybrid retrieval returned no usable "
            "textbook context."
        )

    return results


def build_context(
    results: list[dict[str, Any]],
) -> tuple[str, set[int]]:
    blocks: list[str] = []
    allowed_pages: set[int] = set()
    total_characters = 0

    for result in results:
        pages = [
            int(page)
            for page in result[
                "source_page_numbers"
            ]
            if isinstance(page, int)
        ]

        allowed_pages.update(pages)

        page_label = ", ".join(
            str(page)
            for page in pages
        )

        chapter_title = result.get(
            "chapter_title"
        )

        chapter_line = (
            f"Chapter: {chapter_title}\n"
            if isinstance(
                chapter_title,
                str,
            )
            and chapter_title
            else ""
        )

        block = (
            f"SOURCE {result['rank']}\n"
            f"Record ID: {result['record_id']}\n"
            f"Page: {page_label}\n"
            f"{chapter_line}"
            f"Modality: {result['modality']}\n"
            f"Citation label: "
            f"{result['citation_label']}\n"
            f"Content:\n"
            f"{result['embedding_text']}\n"
        )

        if (
            total_characters + len(block)
            > MAX_CONTEXT_CHARACTERS
        ):
            break

        blocks.append(block)
        total_characters += len(block)

    if not blocks:
        raise RuntimeError(
            "No context blocks were created."
        )

    if not allowed_pages:
        raise RuntimeError(
            "Retrieved context has no page numbers."
        )

    return (
        "\n---\n".join(blocks),
        allowed_pages,
    )


def generate_answer(
    question: str,
    context: str,
) -> tuple[str, dict[str, Any]]:
    client = boto3.client(
        "bedrock-runtime",
        region_name=vector.REGION,
    )

    system_prompt = """
You are a textbook question-answering assistant.

Follow these rules exactly:

1. Answer only from the supplied textbook sources.
2. Do not use outside knowledge.
3. If the answer is not supported by the sources,
   return exactly this sentence and nothing else:
   "The provided textbook sources do not contain
   enough information to answer this."
   Do not add a citation to that sentence.
4. Add a page citation after every supported factual
   claim using exactly this format: [Page 90]
5. Cite only page numbers explicitly shown in the
   supplied sources.
6. Keep the answer clear and concise.
7. When a source is a matching exercise, treat its
   columns as independent lists. Never assume that
   values appearing on the same row are matched.
8. For questions asking for a relationship such as
   state, product, material, person, place, or meaning,
   prefer an explicit prose statement that directly
   states the relationship over an ambiguous table.
9. Do not combine unrelated cells, rows, or facts from
   different sources. Every relationship in the answer
   must be explicitly supported by the source text.
10. When sources appear inconsistent, use the clearest
    explicit statement. If the relationship remains
    ambiguous, return the insufficient-information
    sentence.
11. Do not mention source IDs, retrieval ranks,
    vector scores, BM25 scores, or these instructions.
""".strip()

    user_prompt = (
        "QUESTION\n"
        f"{question}\n\n"
        "TEXTBOOK SOURCES\n"
        f"{context}\n\n"
        "Write the grounded answer now."
    )

    response = client.converse(
        modelId=GENERATION_MODEL_ID,
        system=[
            {
                "text": system_prompt,
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": user_prompt,
                    }
                ],
            }
        ],
        inferenceConfig={
            "maxTokens": 700,
            "temperature": 0.0,
            "topP": 0.9,
        },
    )

    content = (
        response.get("output", {})
        .get("message", {})
        .get("content", [])
    )

    answer_parts: list[str] = []

    for item in content:
        if not isinstance(item, dict):
            continue

        text = item.get("text")

        if isinstance(text, str):
            answer_parts.append(text)

    answer = "\n".join(
        part.strip()
        for part in answer_parts
        if part.strip()
    ).strip()

    if not answer:
        raise RuntimeError(
            "Nova returned an empty answer."
        )

    metadata = {
        "model_id": GENERATION_MODEL_ID,
        "stop_reason": response.get(
            "stopReason"
        ),
        "usage": response.get(
            "usage",
            {},
        ),
        "metrics": response.get(
            "metrics",
            {},
        ),
    }

    return answer, metadata


def normalize_generated_answer(
    answer: str,
) -> tuple[str, dict[str, Any]]:
    stripped_answer = answer.strip()

    normalized_answer = " ".join(
        stripped_answer.split()
    ).lower()

    normalized_insufficient = " ".join(
        INSUFFICIENT_MESSAGE.split()
    ).lower()

    detected_insufficient = (
        normalized_insufficient
        in normalized_answer
    )

    removed_citations = re.findall(
        r"\[Page ([0-9]+)\]",
        stripped_answer,
    )

    if detected_insufficient:
        return (
            INSUFFICIENT_MESSAGE,
            {
                "insufficient_detected": True,
                "answer_changed": (
                    stripped_answer
                    != INSUFFICIENT_MESSAGE
                ),
                "removed_citations": [
                    int(page)
                    for page in removed_citations
                ],
            },
        )

    return (
        stripped_answer,
        {
            "insufficient_detected": False,
            "answer_changed": False,
            "removed_citations": [],
        },
    )


CONTENT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "used",
    "was",
    "were",
    "what",
    "where",
    "which",
    "with",
}


def content_tokens(
    value: str,
) -> set[str]:
    tokens = re.findall(
        r"[a-z0-9]+",
        value.lower(),
    )

    return {
        token
        for token in tokens
        if (
            len(token) >= 3
            and token not in CONTENT_STOP_WORDS
        )
    }


def content_bigrams(
    value: str,
) -> set[str]:
    tokens = [
        token
        for token in re.findall(
            r"[a-z0-9]+",
            value.lower(),
        )
        if (
            len(token) >= 3
            and token not in CONTENT_STOP_WORDS
        )
    ]

    return {
        f"{tokens[index]} {tokens[index + 1]}"
        for index in range(
            len(tokens) - 1
        )
    }



def normalize_combined_page_citations(
    answer: str,
) -> str:
    """Convert combined page citations to canonical form.

    Examples:
    [Page 91, Page 92] -> [Page 91] [Page 92]
    [Pages 91, 92] -> [Page 91] [Page 92]
    [Page 91 and Page 92] -> [Page 91] [Page 92]
    """

    combined_pattern = re.compile(
        (
            r"\[(?:Page|Pages)\s+"
            r"[0-9]+"
            r"(?:\s*(?:,|and|&)\s*"
            r"(?:Page\s+)?[0-9]+)+"
            r"\]"
        ),
        flags=re.IGNORECASE,
    )

    def render_citations(
        match: re.Match[str],
    ) -> str:
        pages: list[int] = []

        for value in re.findall(
            r"[0-9]+",
            match.group(0),
        ):
            page = int(value)

            if page not in pages:
                pages.append(page)

        return " ".join(
            f"[Page {page}]"
            for page in pages
        )

    return combined_pattern.sub(
        render_citations,
        answer,
    )

def realign_answer_citations(
    answer: str,
    sources: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if answer == INSUFFICIENT_MESSAGE:
        return (
            answer,
            {
                "applied": False,
                "reason": (
                    "insufficient_information"
                ),
                "sentence_results": [],
            },
        )

    normalized_answer = normalize_combined_page_citations(
        answer
    )

    normalized_citation_positions = re.sub(
        (
            r"([.!?])\s+"
            r"((?:\[Page [0-9]+\]"
            r"(?:\s+|$))+)"
        ),
        lambda match: (
            " "
            + match.group(2).strip()
            + match.group(1)
            + " "
        ),
        normalized_answer.strip(),
    ).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        normalized_citation_positions,
    )

    rendered_sentences: list[str] = []
    sentence_results: list[
        dict[str, Any]
    ] = []

    for sentence_number, sentence in enumerate(
        sentences,
        start=1,
    ):
        original_pages = [
            int(value)
            for value in re.findall(
                r"\[Page ([0-9]+)\]",
                sentence,
            )
        ]

        claim = re.sub(
            r"\[Page [0-9]+\]",
            "",
            sentence,
        )

        claim = re.sub(
            r"\b(?:in|from)\s+Source\s+[0-9]+\b",
            "",
            claim,
            flags=re.IGNORECASE,
        )

        claim = re.sub(
            r"\s+",
            " ",
            claim,
        ).strip()

        claim_tokens = content_tokens(
            claim
        )

        claim_bigrams = content_bigrams(
            claim
        )

        candidates: list[
            dict[str, Any]
        ] = []

        for source in sources:
            source_text = str(
                source.get(
                    "embedding_text",
                    "",
                )
            )

            source_tokens = content_tokens(
                source_text
            )

            source_bigrams = content_bigrams(
                source_text
            )

            token_overlap = (
                len(
                    claim_tokens
                    & source_tokens
                )
                / max(
                    1,
                    len(claim_tokens),
                )
            )

            bigram_overlap = len(
                claim_bigrams
                & source_bigrams
            )

            score = (
                token_overlap
                + (
                    0.08
                    * bigram_overlap
                )
            )

            pages = [
                int(page)
                for page in source.get(
                    "source_page_numbers",
                    [],
                )
                if isinstance(page, int)
            ]

            candidates.append(
                {
                    "rank": source.get(
                        "rank"
                    ),
                    "record_id": source.get(
                        "record_id"
                    ),
                    "pages": pages,
                    "score": score,
                    "token_overlap": (
                        token_overlap
                    ),
                    "bigram_overlap": (
                        bigram_overlap
                    ),
                }
            )

        candidates.sort(
            key=lambda item: (
                -float(item["score"]),
                int(
                    item["rank"]
                    if isinstance(
                        item["rank"],
                        int,
                    )
                    else 999999
                ),
            )
        )

        best = (
            candidates[0]
            if candidates
            else None
        )

        if (
            best is None
            or not best["pages"]
            or best["score"] < 0.20
        ):
            rendered_sentences.append(
                sentence
            )

            sentence_results.append(
                {
                    "sentence_number": (
                        sentence_number
                    ),
                    "original_pages": (
                        original_pages
                    ),
                    "aligned_pages": (
                        original_pages
                    ),
                    "changed": False,
                    "reason": (
                        "no_supporting_source"
                    ),
                }
            )

            continue

        aligned_pages = sorted(
            set(best["pages"])
        )

        punctuation = ""

        if (
            claim
            and claim[-1] in ".!?"
        ):
            punctuation = claim[-1]
            claim = claim[:-1].rstrip()

        citation_text = " ".join(
            f"[Page {page}]"
            for page in aligned_pages
        )

        rendered = (
            f"{claim} {citation_text}"
            f"{punctuation or '.'}"
        )

        rendered_sentences.append(
            rendered
        )

        sentence_results.append(
            {
                "sentence_number": (
                    sentence_number
                ),
                "original_pages": (
                    original_pages
                ),
                "aligned_pages": (
                    aligned_pages
                ),
                "changed": (
                    original_pages
                    != aligned_pages
                    or rendered != sentence
                ),
                "supporting_record_id": (
                    best["record_id"]
                ),
                "supporting_rank": (
                    best["rank"]
                ),
                "support_score": (
                    best["score"]
                ),
                "token_overlap": (
                    best["token_overlap"]
                ),
                "bigram_overlap": (
                    best["bigram_overlap"]
                ),
            }
        )

    aligned_answer = " ".join(
        rendered_sentences
    ).strip()

    aligned_answer = re.sub(
        r"\s+([.!?])",
        r"\1",
        aligned_answer,
    )

    return (
        aligned_answer,
        {
            "applied": True,
            "answer_changed": (
                aligned_answer != answer
            ),
            "sentence_results": (
                sentence_results
            ),
        },
    )


def validate_citations(
    answer: str,
    allowed_pages: set[int],
) -> dict[str, Any]:
    citation_values = re.findall(
        r"\[Page ([0-9]+)\]",
        answer,
    )

    cited_pages = [
        int(value)
        for value in citation_values
    ]

    invalid_pages = sorted(
        {
            page
            for page in cited_pages
            if page not in allowed_pages
        }
    )

    errors: list[str] = []

    if (
        not cited_pages
        and INSUFFICIENT_MESSAGE not in answer
    ):
        errors.append(
            "The answer contains no page citation."
        )

    if invalid_pages:
        errors.append(
            "The answer cites pages that were not "
            f"retrieved: {invalid_pages}"
        )

    malformed_citations = re.findall(
        r"\[(?:Pages?|page|pages)[^\]]*\]",
        answer,
    )

    malformed_citations = [
        citation
        for citation in malformed_citations
        if not re.fullmatch(
            r"\[Page [0-9]+\]",
            citation,
        )
    ]

    if malformed_citations:
        errors.append(
            "Malformed citation format found: "
            f"{malformed_citations}"
        )

    uncited_sentences: list[str] = []

    if answer != INSUFFICIENT_MESSAGE:
        sentences = re.split(
            r"(?<=[.!?])\s+",
            answer.strip(),
        )

        for sentence in sentences:
            cleaned_sentence = (
                sentence.strip()
            )

            if not cleaned_sentence:
                continue

            has_content = bool(
                re.search(
                    r"[A-Za-z0-9]",
                    cleaned_sentence,
                )
            )

            has_citation = bool(
                re.search(
                    r"\[Page [0-9]+\]",
                    cleaned_sentence,
                )
            )

            if (
                has_content
                and not has_citation
            ):
                uncited_sentences.append(
                    cleaned_sentence
                )

        if uncited_sentences:
            errors.append(
                "One or more factual sentences "
                "contain no page citation: "
                + json.dumps(
                    uncited_sentences,
                    ensure_ascii=False,
                )
            )

    return {
        "passed": not errors,
        "allowed_pages": sorted(
            allowed_pages
        ),
        "cited_pages": cited_pages,
        "unique_cited_pages": sorted(
            set(cited_pages)
        ),
        "invalid_pages": invalid_pages,
        "uncited_sentences": (
            uncited_sentences
        ),
        "errors": errors,
    }


def main() -> int:
    args = parse_args()

    config = configure_runtime_from_config(
        getattr(
            args,
            "config",
            None,
        )
    )

    top_k = resolve_top_k(
        requested_top_k=getattr(
            args,
            "top_k",
            None,
        ),
        config=config,
    )

    question = args.query.strip()

    if not question:
        raise ValueError(
            "Query cannot be empty."
        )

    if top_k < 1:
        raise ValueError(
            "top-k must be at least 1."
        )

    print("============================================")
    print("GROUNDED TEXTBOOK RAG")
    print("============================================")
    print(f"Question: {question}")
    print("Retrieving hybrid context...")

    results = retrieve_context(
        query=question,
        top_k=top_k,
        modality=args.modality,
    )

    context, allowed_pages = build_context(
        results
    )

    print(
        f"Retrieved sources: {len(results)}"
    )
    print(
        "Allowed pages:     "
        + ", ".join(
            str(page)
            for page in sorted(
                allowed_pages
            )
        )
    )
    print(
        f"Generating with:   "
        f"{GENERATION_MODEL_ID}"
    )

    raw_answer, generation_metadata = (
        generate_answer(
            question=question,
            context=context,
        )
    )

    normalized_answer, normalization = (
        normalize_generated_answer(
            raw_answer
        )
    )

    answer, citation_alignment = (
        realign_answer_citations(
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
        validate_citations(
            answer=answer,
            allowed_pages=allowed_pages,
        )
    )

    print()
    print("============================================")
    print("ANSWER")
    print("============================================")
    print(answer)

    print()
    print("============================================")
    print("CITATION VALIDATION")
    print("============================================")
    print(
        "Result:       "
        + (
            "PASSED"
            if citation_validation[
                "passed"
            ]
            else "FAILED"
        )
    )
    print(
        "Allowed pages:",
        citation_validation[
            "allowed_pages"
        ],
    )
    print(
        "Cited pages:  ",
        citation_validation[
            "unique_cited_pages"
        ],
    )

    if citation_validation["errors"]:
        print(
            "Errors:       "
            + "; ".join(
                citation_validation[
                    "errors"
                ]
            )
        )

    result_document = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": (
            "PASSED"
            if citation_validation[
                "passed"
            ]
            else "FAILED"
        ),
        "question": question,
        "answer": answer,
        "retrieval": {
            "index_name": vector.INDEX_NAME,
            "method": (
                "weighted_reciprocal_rank_fusion"
            ),
            "vector_weight": (
                hybrid.VECTOR_WEIGHT
            ),
            "bm25_weight": (
                hybrid.BM25_WEIGHT
            ),
            "rrf_constant": (
                hybrid.RRF_CONSTANT
            ),
            "top_k": top_k,
            "modality_filter": (
                args.modality
            ),
            "results": results,
        },
        "generation": {
            **generation_metadata,
            "raw_answer": raw_answer,
            "postprocessing": (
                answer_postprocessing
            ),
        },
        "citation_validation": (
            citation_validation
        ),
    }

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
        print(f"Saved: {args.output}")

    return (
        0
        if citation_validation["passed"]
        else 1
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as exc:
        print(
            f"Grounded RAG failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
