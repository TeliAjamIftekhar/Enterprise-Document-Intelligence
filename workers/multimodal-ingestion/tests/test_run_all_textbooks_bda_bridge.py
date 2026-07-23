import importlib.util
import json
from pathlib import Path
from types import ModuleType


RUNNER_PATH = Path(
    "workers/multimodal-ingestion/"
    "scripts/run_all_textbooks.py"
)


def load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_all_textbooks_bda_bridge_test",
        RUNNER_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


def test_paths_include_bda_bridge_report():
    runner = load_runner()

    paths = runner.paths_for_book(
        "grade-6-test-book",
        "v1",
    )

    assert paths["bda_bridge_report"] == Path(
        "data/multimodal-output/"
        "grade-6-test-book/v1/full-book/"
        "bda-bridge-report.json"
    )


def test_builds_paid_resume_safe_bridge_command():
    runner = load_runner()

    paths = runner.paths_for_book(
        "grade-6-test-book",
        "v1",
    )

    command = runner.build_bda_bridge_command(
        paths
    )

    assert command[-4:] == [
        str(
            runner.SCRIPTS_ROOT
            / "run_full_book_bda_bridge.py"
        ),
        "--config",
        str(paths["config"]),
        "--execute",
    ]


def test_valid_bridge_requires_report_and_records(
    tmp_path: Path,
):
    runner = load_runner()

    paths = {
        "bda_bridge_report": (
            tmp_path / "bda-bridge-report.json"
        ),
        "bda_normalized": (
            tmp_path / "normalized"
        ),
    }

    assert (
        runner.bda_bridge_is_valid(paths)
        is False
    )

    paths["bda_bridge_report"].write_text(
        json.dumps({
            "status": "BDA_NORMALIZED",
        }),
        encoding="utf-8",
    )

    normalized = (
        paths["bda_normalized"]
        / "batch-0001"
        / "content-units.jsonl"
    )
    normalized.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    normalized.write_text(
        '{"canonical_page":1}\n',
        encoding="utf-8",
    )

    assert (
        runner.bda_bridge_is_valid(paths)
        is True
    )
