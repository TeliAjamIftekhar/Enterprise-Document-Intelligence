from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import evaluate_opensearch_vector_retrieval as vector


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
            vector_field="custom-vector",
        ),
        models=SimpleNamespace(
            embedding=SimpleNamespace(
                model_id="test-embedding-model",
                dimensions=768,
            ),
        ),
        retrieval=SimpleNamespace(
            result_limit=7,
        ),
        local_root=Path(
            "data/multimodal-output/test-book/v2"
        ),
    )


def valid_test_case() -> dict:
    return {
        "test_id": "test-record",
        "query": "What is the answer?",
        "expected_record_id": "record-1",
        "expected_page": 12,
        "expected_modality": "paragraph",
    }


def preserve_runtime_globals(
    monkeypatch,
) -> None:
    for name in (
        "REGION",
        "COLLECTION_ENDPOINT",
        "INDEX_NAME",
        "MODEL_ID",
        "DIMENSIONS",
        "VECTOR_FIELD",
        "TOP_K",
    ):
        monkeypatch.setattr(
            vector,
            name,
            getattr(vector, name),
        )


def test_parse_args_accepts_config_test_cases_output(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_opensearch_vector_retrieval.py",
            "--config",
            "book.json",
            "--test-cases",
            "tests.json",
            "--output",
            "report.json",
        ],
    )

    args = vector.parse_args()

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
        vector,
        "load_book_config",
        lambda path: config,
    )

    resolved = (
        vector.configure_runtime_from_config(
            Path("book.json")
        )
    )

    assert resolved is config
    assert vector.REGION == "ap-south-1"
    assert (
        vector.COLLECTION_ENDPOINT
        == "https://example.aoss.amazonaws.com"
    )
    assert vector.INDEX_NAME == "test-book-v2"
    assert (
        vector.MODEL_ID
        == "test-embedding-model"
    )
    assert vector.DIMENSIONS == 768
    assert vector.VECTOR_FIELD == "custom-vector"
    assert vector.TOP_K == 7


def test_load_test_cases_from_array(
    tmp_path,
):
    path = tmp_path / "tests.json"

    path.write_text(
        json.dumps(
            [valid_test_case()]
        ),
        encoding="utf-8",
    )

    results = vector.load_test_cases(
        path
    )

    assert results == [valid_test_case()]


def test_load_test_cases_from_object(
    tmp_path,
):
    path = tmp_path / "tests.json"

    path.write_text(
        json.dumps({
            "tests": [
                valid_test_case()
            ]
        }),
        encoding="utf-8",
    )

    results = vector.load_test_cases(
        path
    )

    assert results == [valid_test_case()]


def test_resolve_output_path_from_config():
    config = fake_config()

    output = vector.resolve_output_path(
        requested_output=None,
        config=config,
    )

    assert output == Path(
        "data/multimodal-output/test-book/v2/"
        "opensearch-serverless/"
        "vector-retrieval-evaluation-report.json"
    )


def test_run_test_uses_configured_vector_field(
    monkeypatch,
):
    preserve_runtime_globals(
        monkeypatch
    )

    monkeypatch.setattr(
        vector,
        "VECTOR_FIELD",
        "custom-vector",
    )

    monkeypatch.setattr(
        vector,
        "TOP_K",
        5,
    )

    monkeypatch.setattr(
        vector,
        "create_query_embedding",
        lambda client, query: (
            [0.0] * vector.DIMENSIONS,
            4,
        ),
    )

    captured_body = {}

    def fake_signed_search(
        http,
        body,
        maximum_attempts=12,
    ):
        captured_body.update(body)

        return (
            200,
            {
                "hits": {
                    "hits": [
                        {
                            "_id": "record-1",
                            "_score": 1.0,
                            "_source": {
                                "record_id": (
                                    "record-1"
                                ),
                                "modality": (
                                    "paragraph"
                                ),
                                "source_page_numbers": [
                                    12
                                ],
                                "embedding_text": (
                                    "Supported answer."
                                ),
                            },
                        }
                    ]
                }
            },
        )

    monkeypatch.setattr(
        vector,
        "signed_search",
        fake_signed_search,
    )

    result = vector.run_test(
        bedrock_client=object(),
        http=object(),
        test_case=valid_test_case(),
    )

    knn_query = captured_body[
        "query"
    ]["knn"]

    assert "custom-vector" in knn_query
    assert "embedding" not in knn_query
    assert result["passed"] is True
