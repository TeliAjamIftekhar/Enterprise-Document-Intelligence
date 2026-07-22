from __future__ import annotations

import importlib.util
from pathlib import Path


HELPER_FILE = Path(__file__).with_name(
    "test_run_all_textbooks_ocr_planning.py"
)

SPEC = importlib.util.spec_from_file_location(
    "opensearch_stage_helpers",
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


def test_completed_upload_is_indexed(
    tmp_path: Path,
) -> None:
    runner = helpers.load_runner()
    paths = helpers.build_paths(tmp_path)

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

    assert runner.opensearch_upload_is_valid(
        paths
    )

    assert (
        runner.discover_current_stage(
            "grade-1-indexed-test",
            paths,
        )
        == "INDEXED"
    )


def test_bulk_preparation_contract(
    tmp_path: Path,
) -> None:
    runner = helpers.load_runner()
    paths = helpers.build_paths(tmp_path)

    artifacts = (
        runner.opensearch_artifact_paths(
            paths
        )
    )

    artifacts["bulk_root"].mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts["bulk_payload"].write_text(
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

    assert runner.opensearch_bulk_is_valid(
        paths
    )


def test_index_provisioning_contract(
    tmp_path: Path,
) -> None:
    runner = helpers.load_runner()
    paths = helpers.build_paths(tmp_path)

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
        artifacts["index_report"],
        {
            "status": "PROVISIONED",
            "action": "matching",
        },
    )

    assert runner.opensearch_index_is_valid(
        paths
    )


def test_opensearch_commands_are_generic(
    tmp_path: Path,
) -> None:
    runner = helpers.load_runner()
    paths = helpers.build_paths(tmp_path)

    titan = runner.titan_artifact_paths(
        paths
    )

    artifacts = (
        runner.opensearch_artifact_paths(
            paths
        )
    )

    bulk_command = (
        runner.build_opensearch_bulk_command(
            paths
        )
    )

    index_command = (
        runner.build_textbook_index_command(
            paths
        )
    )

    upload_command = (
        runner.build_opensearch_upload_command(
            paths
        )
    )

    assert (
        "prepare_opensearch_bulk.py"
        in bulk_command[1]
    )

    assert str(
        titan["embeddings"]
    ) in bulk_command

    assert (
        "create_textbook_index.py"
        in index_command[1]
    )

    assert "--create" in index_command
    assert "--config" in index_command
    assert str(paths["config"]) in index_command

    assert (
        "upload_opensearch_bulk.py"
        in upload_command[1]
    )

    assert str(
        artifacts["bulk_payload"]
    ) in upload_command

    assert (
        str(
            artifacts[
                "bulk_preparation_report"
            ]
        )
        in upload_command
    )
