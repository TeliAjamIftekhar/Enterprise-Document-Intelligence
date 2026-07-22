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


def test_rejects_overlapping_chapter_ranges():
    raw_manifest = load_raw_manifest()

    document = next(
        document
        for document in raw_manifest["documents"]
        if len(document.get("chapters", [])) >= 2
    )

    first_chapter = document["chapters"][0]
    second_chapter = document["chapters"][1]

    # Force a one-page overlap while keeping
    # source/canonical range lengths equal.
    second_chapter[
        "source_start_page"
    ] = first_chapter[
        "source_end_page"
    ]

    second_chapter[
        "canonical_start_page"
    ] = first_chapter[
        "canonical_end_page"
    ]

    with pytest.raises(
        ValidationError,
        match=(
            "Chapter source ranges must "
            "be ordered and non-overlapping"
        ),
    ):
        ChapterManifest.model_validate(
            raw_manifest
        )


def test_allows_unassigned_pages_between_chapters():
    raw_manifest = load_raw_manifest()

    document = next(
        document
        for document in raw_manifest["documents"]
        if (
            len(document.get("chapters", [])) >= 2
            and document["chapters"][1][
                "source_start_page"
            ]
            < document["chapters"][1][
                "source_end_page"
            ]
        )
    )

    second_chapter = document[
        "chapters"
    ][1]

    # Leave one valid, unassigned page between
    # the two chapter ranges.
    second_chapter[
        "source_start_page"
    ] += 1

    second_chapter[
        "canonical_start_page"
    ] += 1

    manifest = (
        ChapterManifest.model_validate(
            raw_manifest
        )
    )

    assert manifest.book_id == (
        raw_manifest["book_id"]
    )


def test_allows_unassigned_document_edge_pages():
    raw_manifest = load_raw_manifest()

    document = next(
        document
        for document in raw_manifest["documents"]
        if (
            len(document.get("chapters", [])) >= 2
            and document["chapters"][0][
                "source_start_page"
            ]
            < document["chapters"][0][
                "source_end_page"
            ]
            and document["chapters"][-1][
                "source_start_page"
            ]
            < document["chapters"][-1][
                "source_end_page"
            ]
        )
    )

    first_chapter = document[
        "chapters"
    ][0]

    last_chapter = document[
        "chapters"
    ][-1]

    # Page 1 remains valid document/unit
    # metadata but is not assigned to a chapter.
    first_chapter[
        "source_start_page"
    ] += 1

    first_chapter[
        "canonical_start_page"
    ] += 1

    # The final document page may likewise be an
    # activity or unit-summary page.
    last_chapter[
        "source_end_page"
    ] -= 1

    last_chapter[
        "canonical_end_page"
    ] -= 1

    manifest = (
        ChapterManifest.model_validate(
            raw_manifest
        )
    )

    assert manifest.chapter_count > 0


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
