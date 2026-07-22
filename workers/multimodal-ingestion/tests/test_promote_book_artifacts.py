from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(
    "workers/multimodal-ingestion/scripts/"
    "promote_book_artifacts.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "promote_book_artifacts",
        SCRIPT_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


def write_jsonl(
    path: Path,
    values: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "".join(
            json.dumps(
                value,
                ensure_ascii=False,
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def create_record(
    record_id: str,
    page: int,
) -> dict:
    return {
        "record_id": record_id,
        "book_id": "test-book",
        "book_version": "v1-source",
        "text": f"Text from page {page}",
        "citation_label": (
            f"test-book, page {page}"
        ),
        "context_citation_label": (
            f"Chapter (test-book, page {page})"
        ),
        "source_page_numbers": [page],
        "locations": [
            {
                "source_page_number": page,
            }
        ],
        "chapter_id": "chapter-one",
        "chapter_ids": ["chapter-one"],
        "chapter_title": "Chapter One",
        "chapter_titles": ["Chapter One"],
    }


def create_embedding(
    record_id: str,
    value: float,
) -> dict:
    return {
        "record_id": record_id,
        "book_id": "test-book",
        "book_version": "v1-source",
        "embedding": [value] * 1024,
    }


def create_source(
    tmp_path: Path,
) -> Path:
    source = tmp_path / "source-version"

    first_id = (
        "test-book:v1-source:bda:first"
    )

    second_id = (
        "test-book:v1-source:bda:second"
    )

    batch_one = (
        source
        / "full-book/bda-results/"
        "batch-0001/job/normalized/"
        "embedding-ready"
    )

    batch_two = (
        source
        / "full-book/bda-results/"
        "batch-0002/job/normalized/"
        "embedding-ready"
    )

    write_jsonl(
        batch_one / "embedding-records.jsonl",
        [
            create_record(
                first_id,
                1,
            )
        ],
    )

    write_jsonl(
        batch_two / "embedding-records.jsonl",
        [
            create_record(
                second_id,
                2,
            )
        ],
    )

    write_jsonl(
        (
            batch_one
            / "titan-text-v2/embeddings.jsonl"
        ),
        [
            create_embedding(
                first_id,
                0.1,
            ),
            create_embedding(
                second_id,
                0.2,
            ),
        ],
    )

    write_jsonl(
        (
            batch_two
            / "titan-text-v2/embeddings.jsonl"
        ),
        [
            create_embedding(
                second_id,
                0.2,
            )
        ],
    )

    source_directory = source / "source"

    source_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        source_directory
        / "chapter-page-map.json"
    ).write_text(
        json.dumps(
            {
                "book_id": "test-book",
                "book_version": "v1-source",
                "canonical_page_count": 2,
                "pages": [
                    {
                        "canonical_page": 1,
                    },
                    {
                        "canonical_page": 2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    (
        source_directory
        / "chapter-merge-report.json"
    ).write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "book_version": "v1-source",
                "page_count": 2,
            }
        ),
        encoding="utf-8",
    )

    return source


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def test_promotes_and_deduplicates_artifacts(
    tmp_path: Path,
) -> None:
    module = load_module()

    source = create_source(tmp_path)
    output = tmp_path / "promoted"

    report = module.promote(
        source_root=source,
        output_root=output,
        book_id="test-book",
        source_version="v1-source",
        target_version="v1",
        expected_count=2,
        expected_dimension=1024,
    )

    assert report["status"] == "COMPLETED"
    assert report["unique_record_count"] == 2
    assert report["unique_embedding_count"] == 2

    assert (
        report[
            "duplicate_embedding_occurrences"
        ]
        == 1
    )

    records = read_jsonl(
        output / "embedding-records.jsonl"
    )

    embeddings = read_jsonl(
        output / "embeddings.jsonl"
    )

    assert len(records) == 2
    assert len(embeddings) == 2

    assert all(
        "v1-source"
        not in json.dumps(value)
        for value in records + embeddings
    )

    assert all(
        value["book_version"] == "v1"
        for value in records + embeddings
    )

    assert {
        value["citation_label"]
        for value in records
    } == {
        "test-book, page 1",
        "test-book, page 2",
    }

    second_report = module.promote(
        source_root=source,
        output_root=output,
        book_id="test-book",
        source_version="v1-source",
        target_version="v1",
        expected_count=2,
        expected_dimension=1024,
    )

    assert (
        second_report["record_sha256"]
        == report["record_sha256"]
    )


def test_rejects_conflicting_duplicate_vectors(
    tmp_path: Path,
) -> None:
    module = load_module()

    source = create_source(tmp_path)

    conflict_path = (
        source
        / "full-book/bda-results/"
        "batch-0003/job/normalized/"
        "embedding-ready/titan-text-v2/"
        "embeddings.jsonl"
    )

    write_jsonl(
        conflict_path,
        [
            create_embedding(
                "test-book:v1-source:bda:second",
                0.9,
            )
        ],
    )

    with pytest.raises(
        module.PromotionError,
        match="Conflicting vectors",
    ):
        module.promote(
            source_root=source,
            output_root=tmp_path / "promoted",
            book_id="test-book",
            source_version="v1-source",
            target_version="v1",
            expected_count=2,
            expected_dimension=1024,
        )
