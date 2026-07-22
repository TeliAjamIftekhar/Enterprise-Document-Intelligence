from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(
    "workers/multimodal-ingestion/scripts/"
    "reconstruct_embedding_bridge.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "reconstruct_embedding_bridge",
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
            json.dumps(value) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def test_reconstructs_idempotent_bridge(
    tmp_path: Path,
) -> None:
    module = load_module()

    records_path = (
        tmp_path / "embedding-records.jsonl"
    )

    embeddings_path = (
        tmp_path / "embeddings.jsonl"
    )

    report_path = (
        tmp_path
        / "embedding-preparation-report.json"
    )

    manifest_path = (
        tmp_path / "embedding-manifest.json"
    )

    records = [
        {
            "record_id": "book:v1:one",
            "book_id": "book",
            "book_version": "v1",
            "text": "First record",
            "citation_label": "book, page 1",
            "modality": "text",
        },
        {
            "record_id": "book:v1:two",
            "book_id": "book",
            "book_version": "v1",
            "text": "Second record",
            "citation_label": "book, page 2",
            "modality": "figure",
        },
    ]

    embeddings = [
        {
            "record_id": "book:v1:one",
            "embedding": [0.5] * 4,
        },
        {
            "record_id": "book:v1:two",
            "embedding": [0.25] * 4,
        },
    ]

    write_jsonl(records_path, records)
    write_jsonl(
        embeddings_path,
        embeddings,
    )

    first = module.reconstruct_bridge(
        records_path=records_path,
        embeddings_path=embeddings_path,
        embedding_report_path=report_path,
        titan_manifest_path=manifest_path,
        book_id="book",
        version="v1",
        expected_count=2,
        expected_dimension=4,
        model_id="test-model",
    )

    second = module.reconstruct_bridge(
        records_path=records_path,
        embeddings_path=embeddings_path,
        embedding_report_path=report_path,
        titan_manifest_path=manifest_path,
        book_id="book",
        version="v1",
        expected_count=2,
        expected_dimension=4,
        model_id="test-model",
    )

    assert first[
        "embedding_report_status"
    ] == "WRITTEN"

    assert first[
        "titan_manifest_status"
    ] == "WRITTEN"

    assert second[
        "embedding_report_status"
    ] == "UNCHANGED"

    assert second[
        "titan_manifest_status"
    ] == "UNCHANGED"

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        report["embedding_record_count"]
        == 2
    )

    assert (
        report["validation"][
            "record_embedding_ids_equal"
        ]
        is True
    )

    assert manifest["status"] == "COMPLETED"
    assert manifest[
        "input_record_count"
    ] == 2

    assert manifest[
        "completed_record_count"
    ] == 2


def test_rejects_mismatched_ids(
    tmp_path: Path,
) -> None:
    module = load_module()

    records_path = (
        tmp_path / "embedding-records.jsonl"
    )

    embeddings_path = (
        tmp_path / "embeddings.jsonl"
    )

    write_jsonl(
        records_path,
        [
            {
                "record_id": "book:v1:one",
                "text": "Record",
                "citation_label": (
                    "book, page 1"
                ),
            }
        ],
    )

    write_jsonl(
        embeddings_path,
        [
            {
                "record_id": "book:v1:other",
                "embedding": [0.5] * 4,
            }
        ],
    )

    with pytest.raises(
        module.BridgeError,
        match=(
            "Record and embedding ID sets differ"
        ),
    ):
        module.reconstruct_bridge(
            records_path=records_path,
            embeddings_path=embeddings_path,
            embedding_report_path=(
                tmp_path / "report.json"
            ),
            titan_manifest_path=(
                tmp_path / "manifest.json"
            ),
            book_id="book",
            version="v1",
            expected_count=1,
            expected_dimension=4,
            model_id="test-model",
        )
