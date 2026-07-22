from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


HELPER_FILE = Path(__file__).with_name(
    "test_run_all_textbooks_ocr_planning.py"
)

SPEC = importlib.util.spec_from_file_location(
    "opensearch_execution_helpers",
    HELPER_FILE,
)

assert SPEC is not None
assert SPEC.loader is not None

helpers = importlib.util.module_from_spec(
    SPEC
)

SPEC.loader.exec_module(
    helpers
)


def prepare_embedded_book(
    runner,
    paths: dict[str, Path],
) -> None:
    helpers.prepare_book_artifacts(
        paths,
        language="English",
        subject="English",
        page_count=1,
        normalized_text=(
            "This is a complete English textbook "
            "paragraph that passes the quality gate."
        ),
    )

    helpers.write_json(
        paths["ocr_plan"],
        {
            "classification": "BDA_ACCEPTED",
            "fallback_pages": [],
            "accepted_bda_pages": [1],
            "assessments": [],
        },
    )

    titan = runner.titan_artifact_paths(
        paths
    )

    titan["root"].mkdir(
        parents=True,
        exist_ok=True,
    )

    titan["embeddings"].write_text(
        '{"record_id":"record-1",'
        '"embedding":[0.1,0.2]}\n',
        encoding="utf-8",
    )

    helpers.write_json(
        titan["manifest"],
        {
            "status": "COMPLETED",
            "input_record_count": 1,
            "completed_record_count": 1,
        },
    )


def test_embedded_book_runs_opensearch_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = helpers.load_runner()
    paths = helpers.build_paths(tmp_path)

    prepare_embedded_book(
        runner,
        paths,
    )

    artifacts = (
        runner.opensearch_artifact_paths(
            paths
        )
    )

    monkeypatch.setattr(
        runner,
        "paths_for_book",
        lambda book_id, version: paths,
    )

    commands: list[list[str]] = []

    def fake_run_command(
        command: list[str],
        *,
        log_path: Path,
        maximum_retries: int,
        dry_run: bool,
    ) -> None:
        del log_path
        del maximum_retries

        assert dry_run is False
        commands.append(command)

        if (
            "prepare_opensearch_bulk.py"
            in command[1]
        ):
            artifacts["bulk_root"].mkdir(
                parents=True,
                exist_ok=True,
            )

            artifacts[
                "bulk_payload"
            ].write_text(
                '{"index":{"_id":"record-1"}}\n'
                '{"record_id":"record-1"}\n',
                encoding="utf-8",
            )

            helpers.write_json(
                artifacts[
                    "bulk_preparation_report"
                ],
                {
                    "status": "PREPARED",
                    "validation": {
                        "document_count": 1,
                        "errors": [],
                    },
                },
            )

            return

        if (
            "create_textbook_index.py"
            in command[1]
        ):
            artifacts["root"].mkdir(
                parents=True,
                exist_ok=True,
            )

            helpers.write_json(
                artifacts["index_report"],
                {
                    "status": "PROVISIONED",
                    "action": "matching",
                },
            )

            return

        if (
            "upload_opensearch_bulk.py"
            in command[1]
        ):
            helpers.write_json(
                artifacts["upload_report"],
                {
                    "status": "COMPLETED",
                    "prepared_document_count": 1,
                },
            )

            return

        raise AssertionError(
            f"Unexpected command: {command}"
        )

    monkeypatch.setattr(
        runner,
        "run_command",
        fake_run_command,
    )

    state_path = tmp_path / "state.json"

    state = helpers.create_state(
        runner,
        state_path,
    )

    result = runner.process_book(
        {
            "book_id": "grade-1-search-test",
            "grade": 1,
            "title": "Grade 1 Search Test",
            "language": "English",
            "subject": "English",
            "source_bucket": "test-bucket",
            "source_zip_key": (
                "NCERT I-X/Standard-I/"
                "Search-Test.zip"
            ),
        },
        registry_path=tmp_path / "registry.json",
        state_path=state_path,
        state=state,
        maximum_retries=0,
        dry_run=False,
    )

    assert result == "INDEXED"
    assert len(commands) == 3

    assert (
        "prepare_opensearch_bulk.py"
        in commands[0][1]
    )

    assert (
        "create_textbook_index.py"
        in commands[1][1]
    )

    assert (
        "upload_opensearch_bulk.py"
        in commands[2][1]
    )

    assert (
        state["books"][
            "grade-1-search-test"
        ]["status"]
        == "INDEXED"
    )


def test_indexed_book_is_resume_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = helpers.load_runner()
    paths = helpers.build_paths(tmp_path)

    prepare_embedded_book(
        runner,
        paths,
    )

    artifacts = (
        runner.opensearch_artifact_paths(
            paths
        )
    )

    artifacts["root"].mkdir(
        parents=True,
        exist_ok=True,
    )

    helpers.write_json(
        artifacts["upload_report"],
        {
            "status": "COMPLETED",
            "prepared_document_count": 1,
        },
    )

    monkeypatch.setattr(
        runner,
        "paths_for_book",
        lambda book_id, version: paths,
    )

    def unexpected_command(
        *args,
        **kwargs,
    ) -> None:
        raise AssertionError(
            "Indexed book must not rerun "
            "OpenSearch commands."
        )

    monkeypatch.setattr(
        runner,
        "run_command",
        unexpected_command,
    )

    state_path = tmp_path / "state.json"

    state = helpers.create_state(
        runner,
        state_path,
    )

    result = runner.process_book(
        {
            "book_id": "grade-1-index-resume",
            "grade": 1,
            "title": "Grade 1 Index Resume",
            "language": "English",
            "subject": "English",
            "source_bucket": "test-bucket",
            "source_zip_key": (
                "NCERT I-X/Standard-I/"
                "Index-Resume.zip"
            ),
        },
        registry_path=tmp_path / "registry.json",
        state_path=state_path,
        state=state,
        maximum_retries=0,
        dry_run=False,
    )

    assert result == "INDEXED"
