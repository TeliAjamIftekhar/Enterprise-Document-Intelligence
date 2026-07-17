from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import answer_opensearch_rag as rag


def fake_config() -> SimpleNamespace:
    return SimpleNamespace(
        aws=SimpleNamespace(
            region="ap-south-1",
        ),
        opensearch=SimpleNamespace(
            collection_endpoint=(
                "https://example.aoss.amazonaws.com"
            ),
            index_name="test-book-v2",
        ),
        models=SimpleNamespace(
            embedding=SimpleNamespace(
                model_id="test-embedding-model",
                dimensions=768,
            ),
            generation=SimpleNamespace(
                model_id="test-generation-model",
            ),
        ),
        retrieval=SimpleNamespace(
            vector_candidate_limit=30,
            bm25_candidate_limit=25,
            result_limit=7,
            rrf_constant=55,
            vector_weight=4.0,
            bm25_weight=2.0,
        ),
    )


def test_parse_args_accepts_config_and_diagram(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "answer_opensearch_rag.py",
            "--config",
            "book.json",
            "--query",
            "Explain the diagram.",
            "--modality",
            "diagram",
        ],
    )

    args = rag.parse_args()

    assert args.config == Path("book.json")
    assert args.query == "Explain the diagram."
    assert args.modality == "diagram"
    assert args.top_k is None


def test_resolve_top_k_uses_requested_value():
    config = fake_config()

    assert rag.resolve_top_k(
        requested_top_k=3,
        config=config,
    ) == 3


def test_resolve_top_k_uses_config_then_legacy():
    config = fake_config()

    assert rag.resolve_top_k(
        requested_top_k=None,
        config=config,
    ) == 7

    assert rag.resolve_top_k(
        requested_top_k=None,
        config=None,
    ) == rag.DEFAULT_TOP_K


def test_configure_runtime_from_config(
    monkeypatch,
):
    config = fake_config()

    monkeypatch.setattr(
        rag,
        "load_book_config",
        lambda path: config,
    )

    resolved = (
        rag.configure_runtime_from_config(
            Path("book.json")
        )
    )

    assert resolved is config

    assert rag.vector.REGION == "ap-south-1"
    assert (
        rag.vector.COLLECTION_ENDPOINT
        == "https://example.aoss.amazonaws.com"
    )
    assert (
        rag.vector.INDEX_NAME
        == "test-book-v2"
    )
    assert (
        rag.vector.MODEL_ID
        == "test-embedding-model"
    )
    assert rag.vector.DIMENSIONS == 768

    assert (
        rag.hybrid.VECTOR_CANDIDATE_LIMIT
        == 30
    )
    assert (
        rag.hybrid.BM25_CANDIDATE_LIMIT
        == 25
    )
    assert rag.hybrid.CANDIDATE_LIMIT == 30
    assert rag.hybrid.RESULT_LIMIT == 7
    assert rag.hybrid.RRF_CONSTANT == 55
    assert rag.hybrid.VECTOR_WEIGHT == 4.0
    assert rag.hybrid.BM25_WEIGHT == 2.0

    assert (
        rag.GENERATION_MODEL_ID
        == "test-generation-model"
    )


def test_none_config_preserves_runtime():
    assert (
        rag.configure_runtime_from_config(None)
        is None
    )
