from resolve_chapter_structure import (
    assess_toc_scope,
    build_document_boundary_fallback,
)


def lesson_documents(count: int):
    return [
        {
            "order": index,
            "document_id": f"unit-{index}",
            "document_type": "unit",
            "source_filename": (
                f"chapter-{index:02d}.pdf"
            ),
            "canonical_start_page": (
                10 + (index - 1) * 5
            ),
            "canonical_end_page": (
                14 + (index - 1) * 5
            ),
            "page_count": 5,
        }
        for index in range(1, count + 1)
    ]


def test_rejects_lesson_internal_index():
    documents = lesson_documents(16)

    toc_entries = [
        {
            "toc_title": f"Entry {index}"
        }
        for index in range(1, 16)
    ]

    resolved = [
        {
            "document_id": "unit-1"
        },
        {
            "document_id": "unit-1"
        },
    ]

    unresolved = toc_entries[2:]

    result = assess_toc_scope(
        toc_entries=toc_entries,
        resolved=resolved,
        unresolved=unresolved,
        documents=documents,
    )

    assert result["status"] == "REJECTED"
    assert (
        result["resolved_document_count"]
        == 1
    )


def test_accepts_full_book_toc():
    documents = lesson_documents(16)

    toc_entries = [
        {
            "toc_title": f"Lesson {index}"
        }
        for index in range(1, 17)
    ]

    resolved = [
        {
            "document_id": f"unit-{index}"
        }
        for index in range(1, 17)
    ]

    result = assess_toc_scope(
        toc_entries=toc_entries,
        resolved=resolved,
        unresolved=[],
        documents=documents,
    )

    assert result["status"] == "ACCEPTED"
    assert result[
        "document_coverage_ratio"
    ] == 1.0


def test_document_boundary_fallback():
    documents = lesson_documents(3)

    internal_candidates = [
        {
            "document_id": "unit-1",
            "source_filename": (
                "chapter-01.pdf"
            ),
            "title": "प्रथमः पाठः",
            "canonical_page": 10,
            "source_page": 1,
            "font_size": 30.0,
        },
        {
            "document_id": "unit-2",
            "source_filename": (
                "chapter-02.pdf"
            ),
            "title": "द्वितीयः पाठः",
            "canonical_page": 15,
            "source_page": 1,
            "font_size": 30.0,
        },
    ]

    chapters = (
        build_document_boundary_fallback(
            documents=documents,
            internal_candidates=(
                internal_candidates
            ),
        )
    )

    assert len(chapters) == 3

    assert chapters[0][
        "lesson_number"
    ] == 1

    assert chapters[0][
        "source_start_page"
    ] == 1

    assert chapters[0][
        "canonical_start_page"
    ] == 10

    assert chapters[0][
        "resolution_evidence"
    ] == "document-boundary-fallback"

    assert chapters[2][
        "toc_title"
    ] == "Chapter 3"
