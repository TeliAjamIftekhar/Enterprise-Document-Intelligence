import pytest

from upload_full_book_batches import (
    derive_manifest_s3_key,
)


def test_derives_key_from_s3_uri_prefix():
    manifest = {
        "batching": {
            "s3_prefix": (
                "s3://edi-documents-ajam-2026/"
                "bda-input/grade-1/"
                "grade-1-urdu-shahnai/v1/"
                "full-book/batches/"
            )
        }
    }

    assert derive_manifest_s3_key(
        manifest
    ) == (
        "bda-input/grade-1/"
        "grade-1-urdu-shahnai/v1/"
        "full-book/"
        "full-book-batch-manifest.json"
    )


def test_derives_key_from_plain_prefix():
    manifest = {
        "batching": {
            "s3_prefix": (
                "bda-input/grade-6/"
                "grade-6-sanskrit-deepakam/v1/"
                "full-book/batches/"
            )
        }
    }

    assert derive_manifest_s3_key(
        manifest
    ) == (
        "bda-input/grade-6/"
        "grade-6-sanskrit-deepakam/v1/"
        "full-book/"
        "full-book-batch-manifest.json"
    )


def test_falls_back_to_batch_key():
    manifest = {
        "batches": [
            {
                "s3_key": (
                    "bda-input/grade-9/"
                    "grade-9-english-kaveri/v1/"
                    "full-book/batches/"
                    "batch-0001.pdf"
                )
            }
        ]
    }

    assert derive_manifest_s3_key(
        manifest
    ) == (
        "bda-input/grade-9/"
        "grade-9-english-kaveri/v1/"
        "full-book/"
        "full-book-batch-manifest.json"
    )


def test_rejects_manifest_without_s3_identity():
    with pytest.raises(ValueError):
        derive_manifest_s3_key({
            "batching": {},
            "batches": [],
        })
