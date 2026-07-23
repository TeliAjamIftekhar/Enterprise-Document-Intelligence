#!/usr/bin/env python3
"""Run the reusable Surya OCR fallback for selected textbook pages."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.surya_ocr_fallback import (
    SuryaRuntimeConfig,
    build_surya_command,
    build_surya_environment,
    load_approval_record,
    validate_approval_scope,
    locate_results_json,
    parse_surya_results,
    prepare_runtime_directories,
    render_pdf_pages,
    validate_runtime_config,
    write_fallback_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RunPaths:
    root: Path
    input_dir: Path
    raw_output_dir: Path
    verified_dir: Path
    state_file: Path
    render_manifest: Path
    runner_log: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )

        handle.write("\n")
        temporary_path = Path(handle.name)

    temporary_path.replace(path)


def parse_page_spec(value: str) -> tuple[int, ...]:
    """Parse page expressions such as ``1,5,17,20-25``."""

    pages: set[int] = set()

    for raw_token in value.split(","):
        token = raw_token.strip()

        if not token:
            continue

        if "-" in token:
            parts = token.split("-")

            if len(parts) != 2:
                raise argparse.ArgumentTypeError(
                    f"Invalid page range: {token}"
                )

            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"Invalid page range: {token}"
                ) from error

            if start <= 0 or end <= 0:
                raise argparse.ArgumentTypeError(
                    "Page numbers must be positive"
                )

            if end < start:
                raise argparse.ArgumentTypeError(
                    f"Descending page range is invalid: {token}"
                )

            pages.update(
                range(start, end + 1)
            )

        else:
            try:
                page = int(token)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"Invalid page number: {token}"
                ) from error

            if page <= 0:
                raise argparse.ArgumentTypeError(
                    "Page numbers must be positive"
                )

            pages.add(page)

    if not pages:
        raise argparse.ArgumentTypeError(
            "At least one page must be provided"
        )

    return tuple(sorted(pages))


def paths_for_run(output_root: Path) -> RunPaths:
    return RunPaths(
        root=output_root,
        input_dir=output_root / "input",
        raw_output_dir=output_root / "raw",
        verified_dir=output_root / "verified",
        state_file=output_root / "run-state.json",
        render_manifest=(
            output_root
            / "render-manifest.json"
        ),
        runner_log=output_root / "runner.log",
    )


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return payload


def resume_result_is_valid(
    paths: RunPaths,
    *,
    expected_pages: tuple[int, ...],
) -> bool:
    report_path = (
        paths.verified_dir
        / "surya-fallback-report.json"
    )

    marker_path = (
        paths.verified_dir
        / "SURYA_OCR_FALLBACK_VERIFIED"
    )

    if not report_path.is_file():
        return False

    if not marker_path.is_file():
        return False

    try:
        report = load_json_object(report_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False

    if report.get("classification") != "PASS":
        return False

    if not report.get(
        "accepted_for_pipeline",
        False,
    ):
        return False

    recorded_pages = tuple(
        sorted(
            int(page)
            for page in report.get(
                "expected_pages",
                [],
            )
        )
    )

    return recorded_pages == expected_pages


def write_state(
    paths: RunPaths,
    *,
    status: str,
    book_id: str,
    version: str,
    expected_language: str,
    expected_pages: tuple[int, ...],
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "book_id": book_id,
        "version": version,
        "status": status,
        "expected_language": (
            expected_language
        ),
        "expected_pages": list(
            expected_pages
        ),
        "updated_at": utc_now(),
        "paths": {
            "root": str(paths.root),
            "input_dir": str(
                paths.input_dir
            ),
            "raw_output_dir": str(
                paths.raw_output_dir
            ),
            "verified_dir": str(
                paths.verified_dir
            ),
            "render_manifest": str(
                paths.render_manifest
            ),
            "runner_log": str(
                paths.runner_log
            ),
        },
    }

    if extra:
        payload.update(extra)

    atomic_write_json(
        paths.state_file,
        payload,
    )


def write_render_manifest(
    paths: RunPaths,
    *,
    book_id: str,
    version: str,
    pdf_path: Path,
    expected_language: str,
    pages: tuple[int, ...],
    rendered_pages: Sequence[Any],
) -> None:
    atomic_write_json(
        paths.render_manifest,
        {
            "book_id": book_id,
            "version": version,
            "canonical_pdf": str(
                pdf_path
            ),
            "expected_language": (
                expected_language
            ),
            "selected_pages": list(
                pages
            ),
            "rendered_pages": [
                page.to_dict()
                for page in rendered_pages
            ],
            "created_at": utc_now(),
        },
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render selected textbook pages, run "
            "Surya OCR and validate the results."
        )
    )

    parser.add_argument(
        "--book-id",
        required=True,
    )

    parser.add_argument(
        "--version",
        required=True,
    )

    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--expected-language",
        required=True,
    )

    parser.add_argument(
        "--pages",
        type=parse_page_spec,
        required=True,
        help=(
            "One-based pages, for example: "
            "1,5,17,20-25"
        ),
    )

    parser.add_argument(
        "--approval-record",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )

    parser.add_argument(
        "--surya-executable",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--render-dpi",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    project_root = (
        args.project_root
        .expanduser()
        .resolve()
    )

    pdf_path = (
        args.pdf
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    approval_record = (
        args.approval_record
        .expanduser()
        .resolve()
    )

    executable = (
        args.surya_executable
        if args.surya_executable
        is not None
        else (
            project_root
            / "workers/multimodal-ingestion/"
            ".venv-surya/bin/surya_ocr"
        )
    )

    executable = (
        executable
        .expanduser()
        .resolve()
    )

    expected_pages: tuple[int, ...] = (
        args.pages
    )

    paths = paths_for_run(
        output_root
    )

    paths.root.mkdir(
        parents=True,
        exist_ok=True,
    )

    approval_payload = load_approval_record(
        approval_record,
        require_approved=True,
    )

    validate_approval_scope(
        approval_payload,
        book_id=args.book_id,
        version=args.version,
        selected_pages=expected_pages,
    )

    if (
        args.resume
        and resume_result_is_valid(
            paths,
            expected_pages=expected_pages,
        )
    ):
        print(
            "SURYA_OCR_FALLBACK_ALREADY_VERIFIED"
        )
        print(
            "Report:",
            (
                paths.verified_dir
                / "surya-fallback-report.json"
            ),
        )

        return 0

    runtime = SuryaRuntimeConfig(
        executable=executable,
        project_root=project_root,
        render_dpi=args.render_dpi,
    )

    validate_runtime_config(
        runtime,
        require_executable=(
            not args.dry_run
        ),
    )

    write_state(
        paths,
        status="RENDERING",
        book_id=args.book_id,
        version=args.version,
        expected_language=(
            args.expected_language
        ),
        expected_pages=expected_pages,
    )

    if not args.resume:
        for directory in (
            paths.input_dir,
            paths.raw_output_dir,
            paths.verified_dir,
        ):
            if directory.exists():
                shutil.rmtree(directory)

    paths.input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rendered_pages = render_pdf_pages(
        pdf_path,
        paths.input_dir,
        page_numbers=expected_pages,
        dpi=args.render_dpi,
    )

    write_render_manifest(
        paths,
        book_id=args.book_id,
        version=args.version,
        pdf_path=pdf_path,
        expected_language=(
            args.expected_language
        ),
        pages=expected_pages,
        rendered_pages=rendered_pages,
    )

    command = build_surya_command(
        runtime,
        input_path=paths.input_dir,
        output_dir=paths.raw_output_dir,
    )

    environment = build_surya_environment(
        runtime,
        base_environment=os.environ,
    )

    if args.dry_run:
        write_state(
            paths,
            status="DRY_RUN_READY",
            book_id=args.book_id,
            version=args.version,
            expected_language=(
                args.expected_language
            ),
            expected_pages=expected_pages,
            extra={
                "command": command,
                "rendered_page_count": len(
                    rendered_pages
                ),
                "aws_calls": 0,
            },
        )

        print(
            "SURYA_OCR_FALLBACK_DRY_RUN_READY"
        )
        print(
            "Pages:",
            ", ".join(
                str(page)
                for page in expected_pages
            ),
        )
        print(
            "Command:",
            " ".join(command),
        )
        print(
            "AWS API calls: 0"
        )

        return 0

    prepare_runtime_directories(
        runtime
    )

    paths.raw_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_state(
        paths,
        status="OCR_FALLBACK_PROCESSING",
        book_id=args.book_id,
        version=args.version,
        expected_language=(
            args.expected_language
        ),
        expected_pages=expected_pages,
        extra={
            "command": command,
            "started_at": utc_now(),
        },
    )

    with paths.runner_log.open(
        "w",
        encoding="utf-8",
    ) as log_handle:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if completed.returncode != 0:
        write_state(
            paths,
            status="FAILED",
            book_id=args.book_id,
            version=args.version,
            expected_language=(
                args.expected_language
            ),
            expected_pages=expected_pages,
            extra={
                "failure_stage": (
                    "OCR_FALLBACK_PROCESSING"
                ),
                "exit_code": (
                    completed.returncode
                ),
                "runner_log": str(
                    paths.runner_log
                ),
            },
        )

        print(
            "SURYA_OCR_FALLBACK_COMMAND_FAILED",
            file=sys.stderr,
        )
        print(
            f"Exit code: {completed.returncode}",
            file=sys.stderr,
        )
        print(
            f"Log: {paths.runner_log}",
            file=sys.stderr,
        )

        return completed.returncode or 1

    results_json = locate_results_json(
        paths.raw_output_dir
    )

    report = parse_surya_results(
        results_json,
        expected_language=(
            args.expected_language
        ),
        expected_pages=expected_pages,
        input_dir=paths.input_dir,
        canonical_pdf_path=pdf_path,
    )

    artifacts = write_fallback_artifacts(
        report,
        paths.verified_dir,
    )

    if report.classification == "PASS":
        status = "OCR_FALLBACK_VERIFIED"
        return_code = 0

    elif report.classification == "REVIEW":
        status = "OCR_FALLBACK_REVIEW"
        return_code = 2

    else:
        status = "OCR_FALLBACK_FAILED"
        return_code = 3

    write_state(
        paths,
        status=status,
        book_id=args.book_id,
        version=args.version,
        expected_language=(
            args.expected_language
        ),
        expected_pages=expected_pages,
        extra={
            "completed_at": utc_now(),
            "classification": (
                report.classification
            ),
            "accepted_for_pipeline": (
                report.accepted_for_pipeline
            ),
            "passed": report.passed,
            "review": report.review,
            "failed": report.failed,
            "missing_pages": list(
                report.missing_pages
            ),
            "results_json": str(
                results_json
            ),
            "report": str(
                artifacts["report"]
            ),
            "marker": str(
                artifacts["marker"]
            ),
        },
    )

    print("=" * 80)
    print("SURYA OCR FALLBACK RESULT")
    print("=" * 80)
    print(
        "Book:",
        args.book_id,
    )
    print(
        "Pages:",
        ", ".join(
            str(page)
            for page in expected_pages
        ),
    )
    print(
        "Classification:",
        report.classification,
    )
    print(
        "Passed:",
        report.passed,
    )
    print(
        "Review:",
        report.review,
    )
    print(
        "Failed:",
        report.failed,
    )
    print(
        "Missing:",
        list(report.missing_pages),
    )
    print(
        "Accepted:",
        report.accepted_for_pipeline,
    )
    print(
        "Report:",
        artifacts["report"],
    )
    print(
        "AWS API calls: 0"
    )

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
