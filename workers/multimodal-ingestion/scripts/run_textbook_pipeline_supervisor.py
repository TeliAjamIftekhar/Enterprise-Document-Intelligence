from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_ROOT.parents[2]

BASE_RUNNER = (
    SCRIPTS_ROOT / "run_all_textbooks.py"
)

BDA_BRIDGE = (
    SCRIPTS_ROOT / "run_full_book_bda_bridge.py"
)

CONFIG_ROOT = (
    REPO_ROOT
    / "workers"
    / "multimodal-ingestion"
    / "config"
    / "books"
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def parse_grades(value: str) -> set[int]:
    grades: set[int] = set()

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        if "-" in item:
            start_text, end_text = item.split(
                "-",
                1,
            )

            start = int(start_text)
            end = int(end_text)

            if start > end:
                raise ValueError(
                    f"Invalid grade range: {item}"
                )

            grades.update(
                range(start, end + 1)
            )

        else:
            grades.add(int(item))

    if not grades:
        raise ValueError(
            "At least one grade is required."
        )

    return grades


def grade_from_book_id(
    book_id: str,
) -> int | None:
    match = re.match(
        r"^grade-(\d+)-",
        book_id,
    )

    if match is None:
        return None

    return int(match.group(1))


def config_for_book(
    book_id: str,
) -> Path:
    return CONFIG_ROOT / f"{book_id}-v1.json"


def discover_configs(
    *,
    selected_grades: set[int],
    selected_book_id: str | None,
) -> list[Path]:
    if selected_book_id:
        path = config_for_book(
            selected_book_id
        )

        if not path.is_file():
            raise RuntimeError(
                "Book config was not created by "
                f"the base runner: {path}"
            )

        return [path]

    configs: list[Path] = []

    for path in sorted(
        CONFIG_ROOT.glob("*-v1.json")
    ):
        book_id = path.name.removesuffix(
            "-v1.json"
        )

        grade = grade_from_book_id(
            book_id
        )

        if grade in selected_grades:
            configs.append(path)

    if not configs:
        raise RuntimeError(
            "No production v1 book configs "
            "were discovered."
        )

    return configs


def identity_from_config(
    config_path: Path,
) -> tuple[str, str]:
    suffix = "-v1.json"

    if not config_path.name.endswith(
        suffix
    ):
        raise RuntimeError(
            "Only production v1 configs are "
            f"supported: {config_path}"
        )

    return (
        config_path.name.removesuffix(
            suffix
        ),
        "v1",
    )


def book_root(
    book_id: str,
    version: str,
) -> Path:
    return (
        REPO_ROOT
        / "data"
        / "multimodal-output"
        / book_id
        / version
    )


def normalized_ready(
    root: Path,
) -> bool:
    normalized_root = (
        root
        / "pipeline"
        / "bda"
        / "normalized"
    )

    return any(
        path.is_file()
        and path.stat().st_size > 0
        for path in normalized_root.rglob(
            "content-units.jsonl"
        )
    )


def merge_ready(
    root: Path,
) -> bool:
    source_root = root / "source"

    canonical_pdf = (
        source_root / "textbook.pdf"
    )

    page_map = (
        source_root
        / "chapter-page-map.json"
    )

    merge_report = (
        source_root
        / "chapter-merge-report.json"
    )

    if not (
        canonical_pdf.is_file()
        and page_map.is_file()
        and merge_report.is_file()
    ):
        return False

    try:
        report = json.loads(
            merge_report.read_text(
                encoding="utf-8"
            )
        )
    except (
        json.JSONDecodeError,
        OSError,
    ):
        return False

    return (
        isinstance(report, dict)
        and report.get("status") == "VALID"
    )


def verified_ready(
    root: Path,
) -> bool:
    marker = (
        root
        / "unified-normalized"
        / "opensearch-serverless"
        / "verification"
        / "PIPELINE_VERIFIED"
    )

    return marker.is_file()


def run_command(
    command: list[str],
) -> None:
    print()
    print("$", shlex.join(command))
    print()

    environment = os.environ.copy()

    required_pythonpath = (
        "workers/multimodal-ingestion:"
        "workers/multimodal-ingestion/scripts"
    )

    existing_pythonpath = environment.get(
        "PYTHONPATH",
        "",
    )

    environment["PYTHONPATH"] = (
        required_pythonpath
        if not existing_pythonpath
        else (
            required_pythonpath
            + ":"
            + existing_pythonpath
        )
    )

    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )


def base_runner_command(
    *,
    bucket: str,
    prefix: str,
    grades: str,
    book_id: str | None,
    resume: bool,
    dry_run: bool,
    maximum_retries: int,
) -> list[str]:
    command = [
        sys.executable,
        str(BASE_RUNNER),
        "--bucket",
        bucket,
        "--prefix",
        prefix,
        "--grades",
        grades,
        "--max-retries",
        str(maximum_retries),
    ]

    if book_id:
        command.extend(
            [
                "--book-id",
                book_id,
            ]
        )

    if resume:
        command.append("--resume")

    if dry_run:
        command.append("--dry-run")

    return command


def write_report(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        f".{path.name}.tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Supervise the complete resume-safe "
            "textbook pipeline from source ZIP "
            "through BDA and downstream stages."
        )
    )

    parser.add_argument(
        "--bucket",
        required=True,
    )

    parser.add_argument(
        "--prefix",
        required=True,
    )

    parser.add_argument(
        "--grades",
        default="1-10",
    )

    parser.add_argument(
        "--book-id",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/textbook-automation/"
            "pipeline-supervisor-report.json"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    selected_grades = parse_grades(
        args.grades
    )

    lock_path = (
        REPO_ROOT
        / "data"
        / "textbook-automation"
        / ".pipeline-supervisor.lock"
    )

    lock_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        args.report
        if args.report.is_absolute()
        else REPO_ROOT / args.report
    )

    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX
                | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise RuntimeError(
                "Another pipeline supervisor "
                "is already running."
            ) from error

        print("=" * 80)
        print("AUTOMATIC TEXTBOOK PIPELINE SUPERVISOR")
        print("=" * 80)
        print("Bucket:  ", args.bucket)
        print("Prefix:  ", args.prefix)
        print("Grades:  ", sorted(selected_grades))
        print("Book ID: ", args.book_id or "ALL")
        print("Dry run: ", args.dry_run)
        print("Resume:  ", args.resume)

        # Initial pass creates or resumes source,
        # extraction, config and merge artifacts.
        run_command(
            base_runner_command(
                bucket=args.bucket,
                prefix=args.prefix,
                grades=args.grades,
                book_id=args.book_id,
                resume=args.resume,
                dry_run=args.dry_run,
                maximum_retries=(
                    args.max_retries
                ),
            )
        )

        configs = discover_configs(
            selected_grades=selected_grades,
            selected_book_id=args.book_id,
        )

        results: list[dict[str, Any]] = []
        failed = 0

        for config_path in configs:
            book_id, version = (
                identity_from_config(
                    config_path
                )
            )

            root = book_root(
                book_id,
                version,
            )

            print()
            print("#" * 80)
            print(book_id)
            print("#" * 80)

            result: dict[str, Any] = {
                "book_id": book_id,
                "book_version": version,
                "config_path": str(
                    config_path
                ),
                "status": "STARTED",
            }

            try:
                if verified_ready(root):
                    result["status"] = (
                        "SKIPPED_VERIFIED"
                    )

                    print(
                        "SKIP: already VERIFIED"
                    )

                    results.append(result)
                    continue

                if not merge_ready(root):
                    raise RuntimeError(
                        "Book is not ready for BDA; "
                        "valid canonical merge "
                        "artifacts are missing."
                    )

                if not normalized_ready(root):
                    bridge_command = [
                        sys.executable,
                        str(BDA_BRIDGE),
                        "--config",
                        str(config_path),
                    ]

                    if not args.dry_run:
                        bridge_command.append(
                            "--execute"
                        )

                    run_command(
                        bridge_command
                    )

                    result["bda_bridge"] = (
                        "DRY_RUN_PASSED"
                        if args.dry_run
                        else "BDA_NORMALIZED"
                    )

                else:
                    result["bda_bridge"] = (
                        "SKIPPED_EXISTING"
                    )

                    print(
                        "SKIP: normalized BDA "
                        "records already exist"
                    )

                if not args.dry_run:
                    if not normalized_ready(
                        root
                    ):
                        raise RuntimeError(
                            "BDA bridge completed "
                            "without normalized "
                            "content records."
                        )

                    # Second pass continues OCR,
                    # unified records, Titan,
                    # OpenSearch and verification.
                    run_command(
                        base_runner_command(
                            bucket=args.bucket,
                            prefix=args.prefix,
                            grades=args.grades,
                            book_id=book_id,
                            resume=True,
                            dry_run=False,
                            maximum_retries=(
                                args.max_retries
                            ),
                        )
                    )

                    result["status"] = (
                        "VERIFIED"
                        if verified_ready(root)
                        else "DOWNSTREAM_COMPLETED"
                    )

                else:
                    result["status"] = (
                        "DRY_RUN_PASSED"
                    )

            except Exception as error:
                failed += 1

                result["status"] = "FAILED"
                result["error"] = str(error)

                print(
                    "BOOK FAILED:",
                    book_id,
                    error,
                )

            results.append(result)

            write_report(
                report_path,
                {
                    "schema_version": "1.0",
                    "updated_at": utc_now(),
                    "dry_run": args.dry_run,
                    "selected_book_count": len(
                        configs
                    ),
                    "failed_book_count": failed,
                    "books": results,
                },
            )

        completed = len(results) - failed

        final_report = {
            "schema_version": "1.0",
            "completed_at": utc_now(),
            "status": (
                "COMPLETED"
                if failed == 0
                else "COMPLETED_WITH_FAILURES"
            ),
            "dry_run": args.dry_run,
            "selected_book_count": len(
                configs
            ),
            "completed_book_count": completed,
            "failed_book_count": failed,
            "books": results,
        }

        write_report(
            report_path,
            final_report,
        )

        print()
        print("=" * 80)
        print("SUPERVISOR SUMMARY")
        print("=" * 80)
        print("Books:    ", len(configs))
        print("Completed:", completed)
        print("Failed:   ", failed)
        print("Report:   ", report_path)

        return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
