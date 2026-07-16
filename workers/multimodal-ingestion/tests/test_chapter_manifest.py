import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.book_config import load_book_config
from src.chapter_manifest import (
    ChapterManifest,
    load_chapter_manifest,
    validate_manifest_for_book_config,
)


MANIFEST_PATH = Path(
    "workers/multimodal-ingestion/"
    "config/books/manifests/"
    "grade-9-english-kaveri-"
    "v1-chapters.json"
)

CHAPTER_TEST_CONFIG_PATH = Path(
    "workers/multimodal-ingestion/"
    "config/books/"
    "grade-9-english-kaveri-"
    "v1-chapter-test.json"
)

ORIGINAL_CONFIG_PATH = Path(
    "workers/multimodal-ingestion/"
    "config/books/"
    "grade-9-english-kaveri-v1.json"
)


def load_raw_manifest() -> dict:
    return json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_load_kaveri_chapter_manifest():
    manifest = load_chapter_manifest(
        MANIFEST_PATH
    )

    assert manifest.book_id == (
        "grade-9-english-kaveri"
    )
    assert manifest.book_version == (
        "v1-chapter-test"
    )
    assert len(manifest.documents) == 10
    assert manifest.chapter_count == 16

    assert (
        manifest.canonical_layout
        .source_document_pages
        == 296
    )
    assert (
        manifest.canonical_layout
        .canonical_page_count
        == 300
    )

    assert (
        manifest.documents[0]
        .source_filename
        == "iebe1ps.pdf"
    )
    assert (
        manifest.documents[-1]
        .source_filename
        == "iebe1a1.pdf"
    )


def test_manifest_matches_chapter_config():
    manifest = load_chapter_manifest(
        MANIFEST_PATH
    )
    config = load_book_config(
        CHAPTER_TEST_CONFIG_PATH
    )

    validate_manifest_for_book_config(
        manifest,
        config,
    )


def test_page_lookup_returns_document_and_chapter():
    manifest = load_chapter_manifest(
        MANIFEST_PATH
    )

    document = (
        manifest.document_for_canonical_page(
            21
        )
    )
    chapter = (
        manifest.chapter_for_canonical_page(
            21
        )
    )

    assert document is not None
    assert document.document_id == "unit-1"

    assert chapter is not None
    assert chapter.chapter_id == (
        "how-i-taught-my-grandmother-"
        "to-read"
    )

    second_chapter = (
        manifest.chapter_for_canonical_page(
            43
        )
    )

    assert second_chapter is not None
    assert second_chapter.chapter_id == (
        "bharat-our-land"
    )

    appendix_document = (
        manifest.document_for_canonical_page(
            279
        )
    )

    assert appendix_document is not None
    assert (
        appendix_document.document_type
        == "appendix"
    )
    assert (
        manifest.chapter_for_canonical_page(
            279
        )
        is None
    )

    assert (
        manifest.document_for_canonical_page(
            1
        )
        is None
    )
    assert (
        manifest.document_for_canonical_page(
            300
        )
        is None
    )


def test_rejects_invalid_document_order():
    raw_manifest = load_raw_manifest()

    raw_manifest["documents"][1][
        "order"
    ] = 99

    with pytest.raises(
        ValidationError,
        match="Document order",
    ):
        ChapterManifest.model_validate(
            raw_manifest
        )


def test_rejects_gap_between_chapters():
    raw_manifest = load_raw_manifest()

    second_chapter = (
        raw_manifest["documents"][1]
        ["chapters"][1]
    )

    second_chapter[
        "source_start_page"
    ] = 24
    second_chapter[
        "canonical_start_page"
    ] = 44

    with pytest.raises(
        ValidationError,
        match="Chapter source ranges",
    ):
        ChapterManifest.model_validate(
            raw_manifest
        )


def test_rejects_wrong_canonical_page_count():
    raw_manifest = load_raw_manifest()

    raw_manifest["canonical_layout"][
        "canonical_page_count"
    ] = 301

    with pytest.raises(
        ValidationError,
        match=(
            "Canonical layout page count "
            "mismatch"
        ),
    ):
        ChapterManifest.model_validate(
            raw_manifest
        )


def test_rejects_original_book_version():
    manifest = load_chapter_manifest(
        MANIFEST_PATH
    )
    original_config = load_book_config(
        ORIGINAL_CONFIG_PATH
    )

    with pytest.raises(
        ValueError,
        match="book version mismatch",
    ):
        validate_manifest_for_book_config(
            manifest,
            original_config,
        )
