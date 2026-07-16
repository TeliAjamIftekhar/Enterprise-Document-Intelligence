import sys
from pathlib import Path

import pytest

from scripts import (
    create_textbook_index as index_script,
)


CONFIG_PATH = Path(
    "workers/multimodal-ingestion/"
    "config/books/"
    "grade-9-english-kaveri-"
    "v1-chapter-test.json"
)


def reset_legacy_runtime() -> None:
    index_script.configure_runtime(
        None
    )


def test_configure_runtime_uses_chapter_index():
    try:
        plan = (
            index_script.configure_runtime(
                CONFIG_PATH
            )
        )

        assert index_script.INDEX_NAME == (
            "grade-9-english-kaveri-"
            "v1-chapter-test"
        )

        assert (
            index_script.VECTOR_DIMENSIONS
            == 1024
        )

        assert len(
            index_script.INDEX_SCHEMA[
                "mappings"
            ][
                "properties"
            ]
        ) == 46

        assert (
            index_script.OUTPUT_PATH
            == Path(
                "data/multimodal-output/"
                "grade-9-english-kaveri/"
                "v1-chapter-test/"
                "opensearch-serverless/"
                "index-provisioning-report.json"
            )
        )

        assert plan["runtime"][
            "include_context_fields"
        ] is True

    finally:
        reset_legacy_runtime()


def test_configure_runtime_preserves_legacy():
    plan = (
        index_script.configure_runtime(
            None
        )
    )

    assert index_script.INDEX_NAME == (
        "grade-9-english-kaveri-v1"
    )

    assert len(
        index_script.INDEX_SCHEMA[
            "mappings"
        ][
            "properties"
        ]
    ) == 27

    assert plan["runtime"][
        "include_context_fields"
    ] is False


def test_local_only_never_creates_aws_client(
    monkeypatch,
):
    captured: dict = {}

    def fail_client(*args, **kwargs):
        raise AssertionError(
            "boto3.client must not be "
            "called in local-only mode."
        )

    monkeypatch.setattr(
        index_script.boto3,
        "client",
        fail_client,
    )

    monkeypatch.setattr(
        index_script,
        "write_report",
        lambda report: captured.update(
            report
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_textbook_index.py",
            "--config",
            str(CONFIG_PATH),
            "--local-only",
        ],
    )

    try:
        assert index_script.main() == 0

        assert captured["status"] == (
            "LOCAL_VALIDATED"
        )

        assert captured["aws_calls"] == 0

        assert captured["index"][
            "name"
        ] == (
            "grade-9-english-kaveri-"
            "v1-chapter-test"
        )

        assert captured["index"][
            "field_count"
        ] == 46

        assert captured[
            "resources_created"
        ] is False

    finally:
        reset_legacy_runtime()


def test_rejects_local_only_with_create(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_textbook_index.py",
            "--config",
            str(CONFIG_PATH),
            "--local-only",
            "--create",
        ],
    )

    try:
        with pytest.raises(
            ValueError,
            match="cannot be combined",
        ):
            index_script.main()

    finally:
        reset_legacy_runtime()
