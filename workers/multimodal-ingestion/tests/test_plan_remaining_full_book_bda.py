from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import json

import pytest

import scripts.plan_remaining_full_book_bda as planner


def write_json(
    path: Path,
    value: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_manifest(
    expected_pages: int,
    batch_count: int,
) -> dict:
    assert expected_pages % batch_count == 0

    pages_per_batch = (
        expected_pages // batch_count
    )

    batches = []
    next_page = 1

    for batch_number in range(
        1,
        batch_count + 1,
    ):
        batch_id = (
            f"batch-{batch_number:04d}"
        )

        page_start = next_page
        page_end = (
            page_start
            + pages_per_batch
            - 1
        )

        batches.append({
            "batch_id": batch_id,
            "source_page_start": page_start,
            "source_page_end": page_end,
            "page_count": pages_per_batch,
            "s3_uri": (
                "s3://example-bucket/"
                f"{batch_id}.pdf"
            ),
        })

        next_page = page_end + 1

    return {
        "validation": {
            "expected_pages": expected_pages,
            "expected_batch_count": (
                batch_count
            ),
            "actual_batch_count": (
                batch_count
            ),
        },
        "batches": batches,
    }


def configure_args(
    monkeypatch: pytest.MonkeyPatch,
    manifest_path: Path,
    jobs_dir: Path,
    output_path: Path,
) -> None:
    monkeypatch.setattr(
        planner,
        "parse_args",
        lambda: Namespace(
            manifest=manifest_path,
            jobs_dir=jobs_dir,
            output=output_path,
        ),
    )


@pytest.mark.parametrize(
    (
        "expected_pages",
        "batch_count",
        "completed_count",
    ),
    [
        (220, 11, 2),
        (300, 15, 1),
    ],
)
def test_planner_supports_dynamic_books(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_pages: int,
    batch_count: int,
    completed_count: int,
) -> None:
    manifest_path = (
        tmp_path / "manifest.json"
    )

    jobs_dir = tmp_path / "jobs"
    output_path = tmp_path / "plan.json"

    write_json(
        manifest_path,
        build_manifest(
            expected_pages,
            batch_count,
        ),
    )

    for batch_number in range(
        1,
        completed_count + 1,
    ):
        batch_id = (
            f"batch-{batch_number:04d}"
        )

        write_json(
            jobs_dir / f"{batch_id}.json",
            {
                "latest_status": (
                    "Success"
                    if batch_number % 2
                    else "COMPLETED"
                ),
                "invocation_arn": (
                    "arn:aws:bedrock:"
                    f"test:{batch_id}"
                ),
            },
        )

    configure_args(
        monkeypatch,
        manifest_path,
        jobs_dir,
        output_path,
    )

    planner.main()

    report = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    expected_completed = [
        f"batch-{number:04d}"
        for number in range(
            1,
            completed_count + 1,
        )
    ]

    expected_remaining = [
        f"batch-{number:04d}"
        for number in range(
            completed_count + 1,
            batch_count + 1,
        )
    ]

    assert report["status"] == (
        "DRY_RUN_PASSED"
    )

    assert report[
        "total_batch_count"
    ] == batch_count

    assert report[
        "completed_batch_count"
    ] == completed_count

    assert report[
        "remaining_batch_count"
    ] == (
        batch_count - completed_count
    )

    assert report[
        "blocked_batch_count"
    ] == 0

    assert report[
        "completed_batch_ids"
    ] == expected_completed

    assert report[
        "remaining_batch_ids"
    ] == expected_remaining

    assert report["aws_calls"] == 0
    assert report["bda_invocations"] == 0
    assert report["s3_writes"] == 0


def test_planner_rejects_manifest_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_manifest(
        expected_pages=220,
        batch_count=11,
    )

    manifest["validation"][
        "actual_batch_count"
    ] = 10

    manifest_path = (
        tmp_path / "manifest.json"
    )

    jobs_dir = tmp_path / "jobs"
    output_path = tmp_path / "plan.json"

    write_json(
        manifest_path,
        manifest,
    )

    configure_args(
        monkeypatch,
        manifest_path,
        jobs_dir,
        output_path,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "actual batch count differs "
            "from the batches list"
        ),
    ):
        planner.main()


def test_planner_blocks_existing_non_success_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = (
        tmp_path / "manifest.json"
    )

    jobs_dir = tmp_path / "jobs"
    output_path = tmp_path / "plan.json"

    write_json(
        manifest_path,
        build_manifest(
            expected_pages=40,
            batch_count=2,
        ),
    )

    write_json(
        jobs_dir / "batch-0001.json",
        {
            "latest_status": "InProgress",
            "invocation_arn": (
                "arn:aws:bedrock:test:"
                "batch-0001"
            ),
        },
    )

    configure_args(
        monkeypatch,
        manifest_path,
        jobs_dir,
        output_path,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "existing non-success "
            "job records"
        ),
    ):
        planner.main()

    assert not output_path.exists()
