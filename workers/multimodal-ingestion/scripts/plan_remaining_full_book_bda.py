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

    if len(batches) != 15:
        raise RuntimeError(
            f"Expected 15 batches, found "
            f"{len(batches)}."
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

    if expected_start_page != 301:
        raise RuntimeError(
            "Batch manifest does not end at page 300."
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

    expected_completed_ids = {
        "batch-0001",
    }

    completed_ids = {
        str(batch["batch_id"])
        for batch in completed
    }

    if completed_ids != expected_completed_ids:
        raise RuntimeError(
            "Unexpected completed batch set: "
            f"{sorted(completed_ids)}"
        )

    expected_remaining_ids = {
        f"batch-{number:04d}"
        for number in range(
            2,
            16,
        )
    }

    remaining_ids = {
        str(batch["batch_id"])
        for batch in remaining
    }

    if remaining_ids != expected_remaining_ids:
        raise RuntimeError(
            "Unexpected remaining batch set: "
            f"{sorted(remaining_ids)}"
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
