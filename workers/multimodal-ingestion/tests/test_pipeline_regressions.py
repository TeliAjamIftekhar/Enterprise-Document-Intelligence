from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from src.ocr_fallback_planner import (
    extract_page_numbers,
    load_normalized_records,
    records_from_payload,
    select_page_candidates,
)


def normalized_record(
    page: int,
    *,
    text: str = "Sanskrit textbook content",
) -> dict:
    return {
        "schema_version": "1.0",
        "unit_id": f"unit-{page}",
        "book_id": "test-book",
        "book_version": "v1",
        "modality": "text",
        "raw_text": text,
        "search_text": text,
        "source_page_numbers": [page],
        "sample_page_indices": [0],
        "locations": [
            {
                "sample_page_index": 0,
                "sample_page_number": 1,
                "source_page_number": page,
            }
        ],
        # Regression: the digit 3 in this key must
        # never be interpreted as canonical page 3.
        "asset_s3_uris": [],
    }


def test_extract_page_numbers_prefers_source_pages() -> None:
    record = normalized_record(21)

    assert extract_page_numbers(record) == (21,)


def test_normalized_record_is_not_reinterpreted_as_page_three() -> None:
    record = normalized_record(21)

    converted = records_from_payload(record)

    assert converted == [record]
    assert extract_page_numbers(converted[0]) == (21,)


def test_directory_loader_ignores_reports_and_markers(
    tmp_path: Path,
) -> None:
    batch_root = tmp_path / "batch-0001"
    batch_root.mkdir(parents=True)

    content_path = batch_root / "content-units.jsonl"
    content_path.write_text(
        json.dumps(normalized_record(7)) + "\n",
        encoding="utf-8",
    )

    # These numbers must not become discovered pages.
    (batch_root / "normalization-report.json").write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "record_count": 256,
                "status": "COMPLETED",
            }
        ),
        encoding="utf-8",
    )

    (batch_root / "validation-passed.json").write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "validated": True,
            }
        ),
        encoding="utf-8",
    )

    records = load_normalized_records(tmp_path)

    assert len(records) == 1
    assert extract_page_numbers(records[0]) == (7,)


def test_full_book_discovers_all_184_pages(
    tmp_path: Path,
) -> None:
    batch_root = tmp_path / "batch-0001"
    batch_root.mkdir(parents=True)

    content_path = batch_root / "content-units.jsonl"

    with content_path.open("w", encoding="utf-8") as handle:
        for page in range(1, 185):
            handle.write(
                json.dumps(
                    normalized_record(page),
                    ensure_ascii=False,
                )
                + "\n"
            )

    records = load_normalized_records(tmp_path)
    candidates = select_page_candidates(records)

    discovered = tuple(
        candidate.canonical_page
        for candidate in candidates
    )

    assert discovered == tuple(range(1, 185))


def load_sequential_runner():
    script_path = Path(
        "workers/multimodal-ingestion/scripts/"
        "run_full_book_batches_sequentially.py"
    )

    spec = importlib.util.spec_from_file_location(
        "sequential_runner_regression",
        script_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load sequential runner."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_partial_final_batch_download_is_valid(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "download-report.json"

    report_path.write_text(
        json.dumps(
            {
                "status": "VALIDATED",
                "validated": True,
                "source_page_start": 181,
                "source_page_end": 184,
                "result_page_count": 4,
            }
        ),
        encoding="utf-8",
    )

    runner = load_sequential_runner()
    report = runner.successful_download(report_path)

    assert report is not None
    assert report["result_page_count"] == 4



def test_surya_scope_accepts_exact_pilot_pages() -> None:
    from src.surya_ocr_fallback import (
        validate_approval_scope,
    )

    approval = {
        "book_id": "grade-6-sanskrit-deepakam",
        "version": "v1",
        "approved_for_pipeline_integration": True,
        "full_book_run_authorized": False,
        "representative_pages": [
            1,
            16,
            27,
            38,
            43,
            184,
        ],
    }

    validate_approval_scope(
        approval,
        book_id="grade-6-sanskrit-deepakam",
        version="v1",
        selected_pages=(
            1,
            16,
            27,
            38,
            43,
            184,
        ),
    )


def test_surya_scope_rejects_wrong_book() -> None:
    import pytest

    from src.surya_ocr_fallback import (
        validate_approval_scope,
    )

    approval = {
        "book_id": "grade-1-urdu-shahnai",
        "version": "v1",
        "approved_for_pipeline_integration": True,
        "full_book_run_authorized": False,
        "representative_pages": [5, 17],
    }

    with pytest.raises(
        ValueError,
        match="book mismatch",
    ):
        validate_approval_scope(
            approval,
            book_id="grade-6-sanskrit-deepakam",
            version="v1",
            selected_pages=(5, 17),
        )


def test_surya_scope_rejects_unapproved_pages() -> None:
    import pytest

    from src.surya_ocr_fallback import (
        validate_approval_scope,
    )

    approval = {
        "book_id": "grade-6-sanskrit-deepakam",
        "version": "v1",
        "approved_for_pipeline_integration": True,
        "full_book_run_authorized": False,
        "representative_pages": [1, 16, 27],
    }

    with pytest.raises(
        ValueError,
        match="outside the approved pilot scope",
    ):
        validate_approval_scope(
            approval,
            book_id="grade-6-sanskrit-deepakam",
            version="v1",
            selected_pages=(1, 16, 27, 184),
        )
