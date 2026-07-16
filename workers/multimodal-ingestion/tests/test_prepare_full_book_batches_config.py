from argparse import Namespace
from pathlib import Path

import pytest

from scripts.prepare_full_book_batches import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EXPECTED_PAGES,
    resolve_runtime_args,
)


CONFIG_PATH = Path(
    "workers/multimodal-ingestion/"
    "config/books/"
    "grade-9-english-kaveri-"
    "v1-chapter-test.json"
)


def build_args(
    **overrides,
) -> Namespace:
    values = {
        "source_pdf": None,
        "config": None,
        "output_dir": None,
        "manifest": None,
        "batch_size": None,
        "expected_pages": None,
        "s3_prefix": None,
    }

    values.update(overrides)

    return Namespace(**values)


def test_resolves_chapter_test_config():
    args = resolve_runtime_args(
        build_args(
            config=CONFIG_PATH
        )
    )

    assert args.book_id == (
        "grade-9-english-kaveri"
    )
    assert args.book_version == (
        "v1-chapter-test"
    )
    assert args.grade == 9
    assert args.expected_pages == 300
    assert args.batch_size == 20

    assert args.source_pdf == Path(
        "data/multimodal-output/"
        "grade-9-english-kaveri/"
        "v1-chapter-test/source/"
        "textbook.pdf"
    )

    assert args.output_dir == Path(
        "data/multimodal-output/"
        "grade-9-english-kaveri/"
        "v1-chapter-test/full-book/"
        "batches"
    )

    assert args.manifest == Path(
        "data/multimodal-output/"
        "grade-9-english-kaveri/"
        "v1-chapter-test/full-book/"
        "full-book-batch-manifest.json"
    )

    assert args.s3_prefix == (
        "bda-input/grade-9/"
        "grade-9-english-kaveri/"
        "v1-chapter-test/"
        "full-book/batches"
    )

    assert args.bucket == (
        "edi-documents-ajam-2026"
    )
    assert (
        args.configuration_mode
        == "book_config"
    )


def test_rejects_config_with_legacy_args():
    with pytest.raises(
        ValueError,
        match="cannot be combined",
    ):
        resolve_runtime_args(
            build_args(
                config=CONFIG_PATH,
                source_pdf=Path(
                    "legacy.pdf"
                ),
            )
        )


def test_legacy_mode_retains_defaults():
    args = resolve_runtime_args(
        build_args(
            source_pdf=Path(
                "legacy.pdf"
            ),
            output_dir=Path(
                "legacy-batches"
            ),
            manifest=Path(
                "legacy-manifest.json"
            ),
        )
    )

    assert (
        args.batch_size
        == DEFAULT_BATCH_SIZE
    )
    assert (
        args.expected_pages
        == DEFAULT_EXPECTED_PAGES
    )
    assert (
        args.configuration_mode
        == "legacy"
    )


def test_legacy_mode_requires_paths():
    with pytest.raises(
        ValueError,
        match="Legacy mode requires",
    ):
        resolve_runtime_args(
            build_args()
        )
