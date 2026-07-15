import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.book_config import (
    BookConfig,
    load_book_config,
)


CONFIG_PATH = Path(
    "workers/multimodal-ingestion/"
    "config/books/"
    "grade-9-english-kaveri-v1.json"
)


def test_load_kaveri_book_config():
    config = load_book_config(
        CONFIG_PATH
    )

    assert (
        config.book.book_id
        == "grade-9-english-kaveri"
    )
    assert config.book.version == "v1"
    assert config.book.page_count == 300
    assert (
        config.processing.page_batch_size
        == 20
    )
    assert (
        config.models.embedding.dimensions
        == 1024
    )


def test_derived_book_paths():
    config = load_book_config(
        CONFIG_PATH
    )

    assert config.source_s3_uri == (
        "s3://edi-documents-ajam-2026/"
        "source-documents/grade-9/"
        "grade-9-english-kaveri/"
        "versions/v1/textbook.pdf"
    )

    assert config.source_pdf_path == Path(
        "data/multimodal-output/"
        "grade-9-english-kaveri/v1/"
        "source/textbook.pdf"
    )

    assert config.batch_manifest_path == Path(
        "data/multimodal-output/"
        "grade-9-english-kaveri/v1/"
        "full-book/"
        "full-book-batch-manifest.json"
    )


def test_rejects_wrong_index_name():
    raw_config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    raw_config["opensearch"][
        "index_name"
    ] = "wrong-index-v1"

    with pytest.raises(
        ValidationError,
        match="OpenSearch index mismatch",
    ):
        BookConfig.model_validate(
            raw_config
        )


def test_rejects_unknown_fields():
    raw_config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    raw_config["unknown_setting"] = True

    with pytest.raises(ValidationError):
        BookConfig.model_validate(
            raw_config
        )


def test_accepts_chapter_folder_mode():
    raw_config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    raw_config["book"]["book_id"] = (
        "grade-9-science"
    )
    raw_config["book"]["title"] = (
        "Grade 9 Science Textbook"
    )
    raw_config["book"]["subject"] = "science"
    raw_config["book"]["page_count"] = 250
    raw_config["book"]["status"] = "draft"

    raw_config["opensearch"]["index_name"] = (
        "grade-9-science-v1"
    )

    raw_config["source"] = {
        "mode": "chapter_folder",
        "chapter_directory": (
            "input/grade-9/science/chapters"
        ),
        "chapter_order": "filename",
        "chapter_manifest": None,
        "merged_pdf_name": "textbook.pdf",
    }

    raw_config["storage"]["source_s3_key"] = (
        "source-documents/grade-9/"
        "grade-9-science/versions/v1/"
        "textbook.pdf"
    )
    raw_config["storage"]["derived_prefix"] = (
        "derived-artifacts/grade-9/"
        "grade-9-science/v1"
    )
    raw_config["storage"]["bda_input_prefix"] = (
        "bda-input/grade-9/"
        "grade-9-science/v1"
    )
    raw_config["storage"]["local_root"] = (
        "data/multimodal-output/"
        "grade-9-science/v1"
    )

    config = BookConfig.model_validate(
        raw_config
    )

    assert config.source.mode == (
        "chapter_folder"
    )
    assert config.chapter_directory_path == Path(
        "input/grade-9/science/chapters"
    )
    assert config.source_pdf_path == Path(
        "data/multimodal-output/"
        "grade-9-science/v1/"
        "source/textbook.pdf"
    )


def test_chapter_folder_requires_directory():
    raw_config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    raw_config["source"] = {
        "mode": "chapter_folder",
        "chapter_directory": None,
        "chapter_order": "filename",
        "chapter_manifest": None,
        "merged_pdf_name": "textbook.pdf",
    }

    with pytest.raises(
        ValidationError,
        match="chapter_directory",
    ):
        BookConfig.model_validate(
            raw_config
        )


def test_manifest_order_requires_manifest_file():
    raw_config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    raw_config["source"] = {
        "mode": "chapter_folder",
        "chapter_directory": (
            "input/grade-9/science/chapters"
        ),
        "chapter_order": "manifest",
        "chapter_manifest": None,
        "merged_pdf_name": "textbook.pdf",
    }

    with pytest.raises(
        ValidationError,
        match="chapter_manifest",
    ):
        BookConfig.model_validate(
            raw_config
        )
