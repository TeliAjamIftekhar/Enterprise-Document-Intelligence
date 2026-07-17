from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import evaluate_opensearch_hybrid_retrieval as hybrid


def fake_config() -> SimpleNamespace:
    return SimpleNamespace(
        retrieval=SimpleNamespace(
            vector_candidate_limit=30,
            bm25_candidate_limit=24,
            result_limit=7,
            rrf_constant=55,
            vector_weight=4.0,
            bm25_weight=2.0,
        ),
        local_root=Path(
            "data/multimodal-output/test-book/v2"
        ),
    )


def preserve_runtime_globals(
    monkeypatch,
) -> None:
    for name in (
        "CANDIDATE_LIMIT",
        "VECTOR_CANDIDATE_LIMIT",
        "BM25_CANDIDATE_LIMIT",
        "RESULT_LIMIT",
        "RRF_CONSTANT",
        "VECTOR_WEIGHT",
        "BM25_WEIGHT",
    ):
        monkeypatch.setattr(
            hybrid,
            name,
            getattr(hybrid, name),
        )


def test_parse_args_accepts_runtime_paths(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_opensearch_hybrid_retrieval.py",
            "--config",
            "book.json",
            "--test-cases",
            "tests.json",
            "--output",
            "report.json",
        ],
    )

    args = hybrid.parse_args()

    assert args.config == Path("book.json")
    assert args.test_cases == Path("tests.json")
    assert args.output == Path("report.json")


def test_configure_runtime_from_config(
    monkeypatch,
):
    preserve_runtime_globals(
        monkeypatch
    )

    config = fake_config()

    monkeypatch.setattr(
        hybrid.vector_eval,
        "configure_runtime_from_config",
        lambda path: config,
    )

    resolved = (
        hybrid.configure_runtime_from_config(
            Path("book.json")
        )
    )

    assert resolved is config
    assert hybrid.VECTOR_CANDIDATE_LIMIT == 30
    assert hybrid.BM25_CANDIDATE_LIMIT == 24
    assert hybrid.CANDIDATE_LIMIT == 30
    assert hybrid.RESULT_LIMIT == 7
    assert hybrid.RRF_CONSTANT == 55
    assert hybrid.VECTOR_WEIGHT == 4.0
    assert hybrid.BM25_WEIGHT == 2.0


def test_none_config_preserves_runtime(
    monkeypatch,
):
    preserve_runtime_globals(
        monkeypatch
    )

    monkeypatch.setattr(
        hybrid.vector_eval,
        "configure_runtime_from_config",
        lambda path: None,
    )

    assert (
        hybrid.configure_runtime_from_config(None)
        is None
    )


def test_resolve_output_path_from_config():
    output = hybrid.resolve_output_path(
        requested_output=None,
        config=fake_config(),
    )

    assert output == Path(
        "data/multimodal-output/test-book/v2/"
        "opensearch-serverless/"
        "hybrid-retrieval-evaluation-report.json"
    )


def test_vector_search_uses_configured_limit_and_field(
    monkeypatch,
):
    preserve_runtime_globals(
        monkeypatch
    )

    monkeypatch.setattr(
        hybrid,
        "VECTOR_CANDIDATE_LIMIT",
        17,
    )

    monkeypatch.setattr(
        hybrid.vector_eval,
        "VECTOR_FIELD",
        "custom-vector",
    )

    captured = {}

    def fake_execute_search(
        http,
        body,
    ):
        captured.update(body)
        return []

    monkeypatch.setattr(
        hybrid,
        "execute_search",
        fake_execute_search,
    )

    hybrid.vector_search(
        http=object(),
        query_vector=[0.1, 0.2],
        candidate_modality=None,
    )

    assert captured["size"] == 17

    knn = captured["query"]["knn"]

    assert "custom-vector" in knn
    assert "embedding" not in knn
    assert (
        knn["custom-vector"]["k"]
        == 17
    )


def test_bm25_search_uses_configured_limit(
    monkeypatch,
):
    preserve_runtime_globals(
        monkeypatch
    )

    monkeypatch.setattr(
        hybrid,
        "BM25_CANDIDATE_LIMIT",
        13,
    )

    captured = {}

    def fake_execute_search(
        http,
        body,
    ):
        captured.update(body)
        return []

    monkeypatch.setattr(
        hybrid,
        "execute_search",
        fake_execute_search,
    )

    hybrid.bm25_search(
        http=object(),
        query="test query",
        candidate_modality=None,
    )

    assert captured["size"] == 13
