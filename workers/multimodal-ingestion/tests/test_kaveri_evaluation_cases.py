from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(
    "workers/multimodal-ingestion"
)

VECTOR_PATH = (
    ROOT
    / "config/retrieval-tests/"
    "grade-9-english-kaveri-v1-vector.json"
)

HYBRID_PATH = (
    ROOT
    / "config/retrieval-tests/"
    "grade-9-english-kaveri-v1-hybrid.json"
)

RAG_PATH = (
    ROOT
    / "config/rag-tests/"
    "grade-9-english-kaveri-v1-rag.json"
)


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def extract_cases(
    payload: Any,
) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        cases = payload

    elif isinstance(payload, dict):
        cases = None

        for key in (
            "tests",
            "test_cases",
            "cases",
        ):
            if isinstance(
                payload.get(key),
                list,
            ):
                cases = payload[key]
                break

        assert cases is not None

    else:
        raise AssertionError(
            "Evaluation payload must be "
            "an object or list."
        )

    assert cases
    assert all(
        isinstance(case, dict)
        for case in cases
    )

    return cases


def load_runner():
    runner_path = (
        ROOT
        / "scripts/run_all_textbooks.py"
    )

    spec = (
        importlib.util.spec_from_file_location(
            "run_all_textbooks",
            runner_path,
        )
    )

    assert spec is not None
    assert spec.loader is not None

    runner = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(runner)

    return runner


def test_kaveri_explicit_case_files_are_ready() -> None:
    runner = load_runner()

    paths = (
        runner.evaluation_test_case_paths(
            book_id=(
                "grade-9-english-kaveri"
            ),
            version="v1",
        )
    )

    assert paths == {
        "vector": VECTOR_PATH,
        "hybrid": HYBRID_PATH,
        "rag": RAG_PATH,
    }

    assert all(
        path.is_file()
        for path in paths.values()
    )

    vector_cases = extract_cases(
        load_json(VECTOR_PATH)
    )

    hybrid_cases = extract_cases(
        load_json(HYBRID_PATH)
    )

    rag_cases = extract_cases(
        load_json(RAG_PATH)
    )

    assert vector_cases
    assert hybrid_cases
    assert rag_cases


def test_kaveri_rag_cases_match_generic_schema() -> None:
    payload = load_json(
        RAG_PATH
    )

    assert payload[
        "schema_version"
    ] == "1.0"

    assert payload[
        "book_id"
    ] == "grade-9-english-kaveri"

    assert payload[
        "book_version"
    ] == "v1"

    cases = extract_cases(payload)

    assert any(
        case.get(
            "expect_insufficient"
        )
        is True
        for case in cases
    )

    for case in cases:
        assert str(
            case.get(
                "test_id",
                "",
            )
        ).strip()

        assert str(
            case.get(
                "question",
                "",
            )
        ).strip()

        assert isinstance(
            case.get(
                "expected_citation_pages",
                [],
            ),
            list,
        )

        assert isinstance(
            case.get(
                "required_term_groups",
                [],
            ),
            list,
        )
