from __future__ import annotations

import importlib.util
from pathlib import Path


HELPER_FILE = Path(__file__).with_name(
    "test_run_all_textbooks_ocr_planning.py"
)

SPEC = importlib.util.spec_from_file_location(
    "ocr_planning_helpers",
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


def test_completed_titan_artifacts_are_embedded(
    tmp_path: Path,
) -> None:
    runner = helpers.load_runner()

    paths = helpers.build_paths(
        tmp_path
    )

    artifacts = runner.titan_artifact_paths(
        paths
    )

    artifacts["root"].mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts["embeddings"].write_text(
        '{"record_id":"record-1"}\n',
        encoding="utf-8",
    )

    helpers.write_json(
        artifacts["manifest"],
        {
            "status": "COMPLETED",
            "input_record_count": 1,
            "completed_record_count": 1,
        },
    )

    assert runner.titan_embeddings_are_valid(
        paths
    )

    assert (
        runner.discover_current_stage(
            "grade-1-titan-stage-test",
            paths,
        )
        == "EMBEDDED"
    )


def test_titan_command_uses_prepared_records(
    tmp_path: Path,
) -> None:
    runner = helpers.load_runner()

    paths = helpers.build_paths(
        tmp_path
    )

    downstream = (
        runner.downstream_artifact_paths(
            paths
        )
    )

    artifacts = runner.titan_artifact_paths(
        paths
    )

    command = (
        runner.build_titan_embedding_command(
            paths
        )
    )

    assert (
        "embed_records_titan_v2.py"
        in command[1]
    )

    assert str(
        downstream[
            "embedding_records"
        ]
    ) in command

    assert command[-2:] == [
        "--output-dir",
        str(artifacts["root"]),
    ]
