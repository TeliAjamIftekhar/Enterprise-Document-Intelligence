from __future__ import annotations

import answer_opensearch_rag as rag
import evaluate_opensearch_hybrid_retrieval as hybrid


def candidate(
    rank: int,
    chapter_id: str | None,
    *,
    status: str | None = "single",
) -> dict:
    return {
        "hybrid_rank": rank,
        "record_id": f"record-{rank}",
        "chapter_id": chapter_id,
        "chapter_title": (
            f"Chapter {chapter_id}"
            if chapter_id
            else None
        ),
        "chapter_context_status": status,
    }


def test_filters_cross_chapter_noise_when_dominant():
    candidates = [
        candidate(1, "chapter-8"),
        candidate(2, "chapter-8"),
        candidate(3, "chapter-8"),
        candidate(4, "chapter-8"),
        candidate(5, "chapter-2"),
    ]

    results = (
        rag.select_chapter_consistent_results(
            candidates=candidates,
            top_k=5,
        )
    )

    assert len(results) == 4
    assert [
        result["rank"]
        for result in results
    ] == [1, 2, 3, 4]

    assert all(
        result["chapter_id"] == "chapter-8"
        for result in results
    )

    assert all(
        result["chapter_filter_applied"] is True
        for result in results
    )

    assert all(
        result["selected_chapter_id"]
        == "chapter-8"
        for result in results
    )


def test_preserves_mixed_chapter_results():
    candidates = [
        candidate(1, "chapter-1"),
        candidate(2, "chapter-2"),
        candidate(3, "chapter-3"),
    ]

    results = (
        rag.select_chapter_consistent_results(
            candidates=candidates,
            top_k=3,
        )
    )

    assert len(results) == 3

    assert [
        result["chapter_id"]
        for result in results
    ] == [
        "chapter-1",
        "chapter-2",
        "chapter-3",
    ]

    assert all(
        result["chapter_filter_applied"] is False
        for result in results
    )


def test_preserves_results_without_metadata():
    candidates = [
        candidate(
            1,
            None,
            status=None,
        ),
        candidate(
            2,
            None,
            status=None,
        ),
    ]

    results = (
        rag.select_chapter_consistent_results(
            candidates=candidates,
            top_k=2,
        )
    )

    assert len(results) == 2

    assert all(
        result["chapter_filter_applied"] is False
        for result in results
    )


def test_requires_two_same_chapter_candidates():
    candidates = [
        candidate(1, "chapter-8"),
        candidate(2, "chapter-2"),
    ]

    results = (
        rag.select_chapter_consistent_results(
            candidates=candidates,
            top_k=2,
        )
    )

    assert len(results) == 2

    assert all(
        result["chapter_filter_applied"] is False
        for result in results
    )


def test_hybrid_retrieval_requests_chapter_fields():
    required_fields = {
        "document_type",
        "page_context_status",
        "chapter_id",
        "chapter_title",
        "chapter_context_status",
        "chapter_ids",
        "chapter_titles",
        "context_citation_label",
    }

    assert required_fields.issubset(
        set(hybrid.SOURCE_FIELDS)
    )
