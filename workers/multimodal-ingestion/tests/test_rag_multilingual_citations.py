from __future__ import annotations

import answer_opensearch_rag as rag


def test_unicode_content_tokens_preserve_devanagari() -> None:
    tokens = rag.content_tokens(
        "संयुक्त-व्यञ्जनानि द्वयोः "
        "व्यञ्जन-वर्णानां मेलनेन भवन्ति"
    )

    assert "संयुक्त" in tokens
    assert "व्यञ्जनानि" in tokens
    assert "वर्णानां" in tokens
    assert "मेलनेन" in tokens


def test_verbose_source_citation_becomes_canonical() -> None:
    answer = (
        "संयुक्त-व्यञ्जनानि भवन्ति। "
        "[द्वितीयः पाठः "
        "(grade-6-sanskrit-deepakam, page 41)]"
    )

    normalized = (
        rag.normalize_verbose_page_citations(
            answer
        )
    )

    assert normalized == (
        "संयुक्त-व्यञ्जनानि भवन्ति। [Page 41]"
    )
    assert "grade-6-sanskrit-deepakam" not in normalized


def test_sanskrit_answer_realigns_and_validates() -> None:
    answer = (
        "संयुक्त-व्यञ्जनानि द्वयोः बहूनां वा "
        "व्यञ्जन-वर्णानां मेलनेन भवन्ति। "
        "क्ष इत्येतत् क् + ष् + अ इत्येतेषां "
        "मेलनेन भवति। "
        "[द्वितीयः पाठः "
        "(grade-6-sanskrit-deepakam, page 41)]"
    )

    sources = [
        {
            "rank": 1,
            "record_id": "deepakam-page-41",
            "source_page_numbers": [41],
            "embedding_text": (
                "संयुक्त-व्यञ्जनानि द्वयोः बहूनां वा "
                "व्यञ्जन-वर्णानां मेलनेन भवन्ति। "
                "क्ष = क् + ष् + अ।"
            ),
        }
    ]

    aligned, metadata = (
        rag.realign_answer_citations(
            answer=answer,
            sources=sources,
        )
    )

    validation = rag.validate_citations(
        answer=aligned,
        allowed_pages={41},
    )

    assert metadata["applied"] is True
    assert "[Page 41]" in aligned
    assert validation["passed"] is True
    assert validation["cited_pages"] == [41]
