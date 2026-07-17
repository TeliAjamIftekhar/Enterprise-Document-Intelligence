from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/"
    "full-book/full-book-batch-manifest.json"
)

DEFAULT_JOBS_DIR = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/"
    "full-book/bda-jobs"
)

SUCCESS_STATUSES = {
    "Success",
    "COMPLETED",
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a local dry-run plan for "
            "remaining full-book BDA batches."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=DEFAULT_JOBS_DIR,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest = load_json_object(
        args.manifest
    )

    batches = manifest.get(
        "batches",
        []
    )

    if not isinstance(batches, list):
        raise RuntimeError(
            "Manifest batches field is not a list."
        )

    validation = manifest.get(
        "validation"
    )

    if not isinstance(validation, dict):
        raise RuntimeError(
            "Manifest validation field is not "
            "an object."
        )

    expected_pages = validation.get(
        "expected_pages"
    )

    expected_batch_count = validation.get(
        "expected_batch_count"
    )

    actual_batch_count = validation.get(
        "actual_batch_count"
    )

    for field_name, value in (
        ("expected_pages", expected_pages),
        (
            "expected_batch_count",
            expected_batch_count,
        ),
        (
            "actual_batch_count",
            actual_batch_count,
        ),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            raise RuntimeError(
                "Manifest validation field must "
                f"be a positive integer: "
                f"{field_name}={value!r}"
            )

    if actual_batch_count != len(batches):
        raise RuntimeError(
            "Manifest actual batch count differs "
            "from the batches list: "
            f"validation={actual_batch_count}, "
            f"actual={len(batches)}"
        )

    if len(batches) != expected_batch_count:
        raise RuntimeError(
            "Manifest batch count differs from "
            "the expected batch count: "
            f"expected={expected_batch_count}, "
            f"actual={len(batches)}"
        )

    expected_start_page = 1
    plan_batches: list[
        dict[str, Any]
    ] = []

    for expected_number, batch in enumerate(
        batches,
        start=1,
    ):
        if not isinstance(batch, dict):
            raise RuntimeError(
                "Manifest contains an invalid batch."
            )

        expected_batch_id = (
            f"batch-{expected_number:04d}"
        )

        batch_id = str(
            batch.get("batch_id", "")
        )

        if batch_id != expected_batch_id:
            raise RuntimeError(
                "Unexpected batch sequence: "
                f"expected={expected_batch_id}, "
                f"actual={batch_id}"
            )

        page_start = int(
            batch["source_page_start"]
        )

        page_end = int(
            batch["source_page_end"]
        )

        page_count = int(
            batch["page_count"]
        )

        if page_start != expected_start_page:
            raise RuntimeError(
                f"Page gap or overlap at {batch_id}: "
                f"expected start={expected_start_page}, "
                f"actual start={page_start}"
            )

        calculated_count = (
            page_end - page_start + 1
        )

        if calculated_count != page_count:
            raise RuntimeError(
                f"Page-count mismatch for {batch_id}."
            )

        expected_start_page = (
            page_end + 1
        )

        job_path = (
            args.jobs_dir
            / f"{batch_id}.json"
        )

        job_status = None
        invocation_arn = None
        action = "PREFLIGHT_AND_INVOKE"

        if job_path.is_file():
            job = load_json_object(
                job_path
            )

            job_status = job.get(
                "latest_status",
                job.get("status"),
            )

            invocation_arn = job.get(
                "invocation_arn"
            )

            if job_status in SUCCESS_STATUSES:
                action = "SKIP_COMPLETED"
            else:
                action = (
                    "STOP_EXISTING_NON_SUCCESS_JOB"
                )

        plan_batches.append(
            {
                "batch_id": batch_id,
                "source_page_start": page_start,
                "source_page_end": page_end,
                "page_count": page_count,
                "input_s3_uri": batch.get(
                    "s3_uri",
                    batch.get("input_s3_uri"),
                ),
                "job_record_path": str(
                    job_path
                ),
                "job_record_exists": (
                    job_path.is_file()
                ),
                "job_status": job_status,
                "invocation_arn": invocation_arn,
                "planned_action": action,
            }
        )

    if expected_start_page != expected_pages + 1:
        raise RuntimeError(
            "Batch manifest does not end at the "
            "expected final page: "
            f"expected={expected_pages}, "
            f"actual={expected_start_page - 1}"
        )

    completed = [
        batch
        for batch in plan_batches
        if batch["planned_action"]
        == "SKIP_COMPLETED"
    ]

    remaining = [
        batch
        for batch in plan_batches
        if batch["planned_action"]
        == "PREFLIGHT_AND_INVOKE"
    ]

    blocked = [
        batch
        for batch in plan_batches
        if batch["planned_action"]
        == "STOP_EXISTING_NON_SUCCESS_JOB"
    ]

    if blocked:
        raise RuntimeError(
            "One or more batches have existing "
            "non-success job records:\n"
            + "\n".join(
                (
                    f"- {batch['batch_id']}: "
                    f"{batch['job_status']}"
                )
                for batch in blocked
            )
        )

    completed_ids = {
        str(batch["batch_id"])
        for batch in completed
    }

    remaining_ids = {
        str(batch["batch_id"])
        for batch in remaining
    }

    blocked_ids = {
        str(batch["batch_id"])
        for batch in blocked
    }

    all_batch_ids = {
        str(batch["batch_id"])
        for batch in plan_batches
    }

    expected_batch_ids = {
        f"batch-{number:04d}"
        for number in range(
            1,
            expected_batch_count + 1,
        )
    }

    if all_batch_ids != expected_batch_ids:
        raise RuntimeError(
            "Planned batch IDs differ from the "
            "manifest-driven expected sequence: "
            f"expected={sorted(expected_batch_ids)}, "
            f"actual={sorted(all_batch_ids)}"
        )

    if (
        completed_ids & remaining_ids
        or completed_ids & blocked_ids
        or remaining_ids & blocked_ids
    ):
        raise RuntimeError(
            "Completed, remaining and blocked "
            "batch sets overlap."
        )

    classified_ids = (
        completed_ids
        | remaining_ids
        | blocked_ids
    )

    if classified_ids != expected_batch_ids:
        raise RuntimeError(
            "Batch classification is incomplete: "
            f"expected={sorted(expected_batch_ids)}, "
            f"actual={sorted(classified_ids)}"
        )

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": "DRY_RUN_PASSED",
        "manifest_path": str(
            args.manifest
        ),
        "jobs_directory": str(
            args.jobs_dir
        ),
        "total_batch_count": len(
            plan_batches
        ),
        "completed_batch_count": len(
            completed
        ),
        "remaining_batch_count": len(
            remaining
        ),
        "blocked_batch_count": len(
            blocked
        ),
        "completed_batch_ids": [
            batch["batch_id"]
            for batch in completed
        ],
        "remaining_batch_ids": [
            batch["batch_id"]
            for batch in remaining
        ],
        "batches": plan_batches,
        "aws_calls": 0,
        "bda_invocations": 0,
        "s3_writes": 0,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 56)
    print("FULL-BOOK BDA REMAINING-BATCH DRY RUN")
    print("=" * 56)
    print(
        f"Total batches:     "
        f"{len(plan_batches)}"
    )
    print(
        f"Completed batches: "
        f"{len(completed)}"
    )
    print(
        f"Remaining batches: "
        f"{len(remaining)}"
    )
    print(
        f"Blocked batches:   "
        f"{len(blocked)}"
    )
    print()

    for batch in plan_batches:
        print(
            f"{batch['batch_id']} | "
            f"pages "
            f"{batch['source_page_start']}-"
            f"{batch['source_page_end']} | "
            f"{batch['planned_action']}"
        )

    print()
    print("Status:          DRY_RUN_PASSED")
    print("AWS calls:       0")
    print("BDA invocations: 0")
    print("S3 writes:       0")
    print(f"Plan:            {args.output}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print(
            "Planning interrupted.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except Exception as error:
        print(
            f"Planning failed: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
