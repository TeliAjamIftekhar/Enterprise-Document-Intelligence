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
        "run_all_textbooks_ocr_stage_test",
        RUNNER_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


def write_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_expanded_stage_order() -> None:
    runner = load_runner()

    assert (
        runner.STAGES.index("MERGED")
        < runner.STAGES.index(
            "BDA_NORMALIZED"
        )
        < runner.STAGES.index(
            "OCR_QUALITY_CHECKED"
        )
        < runner.STAGES.index(
            "OCR_FALLBACK_REQUIRED"
        )
        < runner.STAGES.index(
            "OCR_FALLBACK_VERIFIED"
        )
        < runner.STAGES.index("EMBEDDED")
    )


def test_paths_include_ocr_pipeline_artifacts() -> None:
    runner = load_runner()

    paths = runner.paths_for_book(
        "grade-1-urdu-test",
        "v1",
    )

    required = {
        "bda_normalized",
        "ocr_plan",
        "ocr_fallback_root",
        "ocr_fallback_state",
        "ocr_fallback_report",
        "ocr_fallback_marker",
        "ocr_approval",
    }

    assert required.issubset(paths)

    assert (
        paths["ocr_approval"]
        == runner.SURYA_APPROVAL_DEFAULT
    )


def test_approval_does_not_authorize_full_book(
    tmp_path: Path,
) -> None:
    runner = load_runner()

    approval = tmp_path / "approval.json"

    write_json(
        approval,
        {
            "ocr_engine": "surya-ocr",
            "model": "datalab-to/surya-ocr-2",
            "approved_for_pipeline_integration": True,
            "full_book_run_authorized": False,
        },
    )

    status = runner.load_ocr_authorization(
        approval
    )

    assert status["integration_approved"] is True
    assert (
        status["full_book_run_authorized"]
        is False
    )


def base_paths(tmp_path: Path) -> dict[str, Path]:
    runner = load_runner()

    paths = runner.paths_for_book(
        "grade-1-urdu-stage-test",
        "v1",
    )

    return {
        key: (
            tmp_path
            / value.name
            if key
            not in {
                "bda_normalized",
                "ocr_fallback_root",
            }
            else tmp_path / key
        )
        for key, value in paths.items()
    }


def make_valid_merge(
    paths: dict[str, Path],
) -> None:
    paths["canonical_pdf"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths["canonical_pdf"].write_bytes(
        b"%PDF-test"
    )

    write_json(
        paths["page_map"],
        {"pages": []},
    )

    write_json(
        paths["merge_report"],
        {"status": "VALID"},
    )


def test_discover_merged_stage(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    paths = base_paths(tmp_path)

    make_valid_merge(paths)

    assert runner.discover_current_stage(
        "grade-1-urdu-stage-test",
        paths,
    ) == "MERGED"


def test_discover_bda_normalized_stage(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    paths = base_paths(tmp_path)

    make_valid_merge(paths)

    normalized = (
        paths["bda_normalized"]
        / "pages.jsonl"
    )

    normalized.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized.write_text(
        '{"canonical_page":1}\n',
        encoding="utf-8",
    )

    assert runner.discover_current_stage(
        "grade-1-urdu-stage-test",
        paths,
    ) == "BDA_NORMALIZED"


def test_discover_quality_checked_stage(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    paths = base_paths(tmp_path)

    make_valid_merge(paths)

    write_json(
        paths["ocr_plan"],
        {
            "classification": "BDA_ACCEPTED",
            "fallback_pages": [],
            "accepted_bda_pages": [1],
        },
    )

    assert runner.discover_current_stage(
        "grade-1-urdu-stage-test",
        paths,
    ) == "OCR_QUALITY_CHECKED"


def test_discover_fallback_required_stage(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    paths = base_paths(tmp_path)

    make_valid_merge(paths)

    write_json(
        paths["ocr_plan"],
        {
            "classification": (
                "OCR_FALLBACK_REQUIRED"
            ),
            "fallback_pages": [1],
            "accepted_bda_pages": [],
        },
    )

    assert runner.discover_current_stage(
        "grade-1-urdu-stage-test",
        paths,
    ) == "OCR_FALLBACK_REQUIRED"


def test_discover_fallback_verified_stage(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    paths = base_paths(tmp_path)

    make_valid_merge(paths)

    write_json(
        paths["ocr_fallback_report"],
        {
            "classification": "PASS",
            "accepted_for_pipeline": True,
        },
    )

    paths[
        "ocr_fallback_marker"
    ].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths[
        "ocr_fallback_marker"
    ].write_text(
        "PASS\n",
        encoding="utf-8",
    )

    assert runner.discover_current_stage(
        "grade-1-urdu-stage-test",
        paths,
    ) == "OCR_FALLBACK_VERIFIED"
