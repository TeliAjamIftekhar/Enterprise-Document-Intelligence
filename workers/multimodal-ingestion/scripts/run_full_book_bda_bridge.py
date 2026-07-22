from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_ROOT.parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected JSON object: {path}"
        )

    return value


def parse_identity(
    config_path: Path,
) -> tuple[str, str]:
    stem = config_path.stem

    match = re.fullmatch(
        r"(?P<book_id>.+)-(?P<version>v\d+"
        r"(?:[-._][A-Za-z0-9]+)*)",
        stem,
    )

    if match is None:
        raise RuntimeError(
            "Cannot derive book identity from "
            f"config filename: {config_path.name}"
        )

    return (
        match.group("book_id"),
        match.group("version"),
    )


def run_command(
    command: list[str],
) -> None:
    print()
    print("$", shlex.join(command))
    print()

    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
    )


def expected_batch_ids(
    manifest: dict[str, Any],
) -> list[str]:
    batches = manifest.get("batches")

    if not isinstance(batches, list):
        raise RuntimeError(
            "Manifest batches field is invalid."
        )

    batch_ids: list[str] = []

    for index, batch in enumerate(
        batches,
        start=1,
    ):
        if not isinstance(batch, dict):
            raise RuntimeError(
                f"Invalid batch at position {index}."
            )

        expected_id = f"batch-{index:04d}"
        batch_id = str(
            batch.get("batch_id", "")
        )

        if batch_id != expected_id:
            raise RuntimeError(
                "Unexpected batch order: "
                f"expected={expected_id}, "
                f"actual={batch_id}"
            )

        batch_ids.append(batch_id)

    if not batch_ids:
        raise RuntimeError(
            "Manifest contains no batches."
        )

    return batch_ids


def select_normalized_root(
    results_root: Path,
    batch_id: str,
) -> Path:
    batch_root = results_root / batch_id

    candidates = [
        path.parent
        for path in batch_root.rglob(
            "content-units.jsonl"
        )
        if path.is_file()
    ]

    if not candidates:
        raise RuntimeError(
            "No normalized content found for "
            f"{batch_id}: {batch_root}"
        )

    return max(
        candidates,
        key=lambda path: (
            path
            / "content-units.jsonl"
        ).stat().st_mtime,
    )


def create_normalized_bridge(
    *,
    selected_roots: dict[str, Path],
    bridge_root: Path,
) -> None:
    bridge_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for batch_id, source_root in sorted(
        selected_roots.items()
    ):
        target_root = bridge_root / batch_id

        if target_root.exists() or (
            target_root.is_symlink()
        ):
            if target_root.is_dir() and not (
                target_root.is_symlink()
            ):
                shutil.rmtree(target_root)
            else:
                target_root.unlink()

        target_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        for source_item in source_root.iterdir():
            target_item = (
                target_root / source_item.name
            )

            target_item.symlink_to(
                source_item.resolve(),
                target_is_directory=(
                    source_item.is_dir()
                ),
            )


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
            "Prepare, upload, execute, download "
            "and expose full-book BDA batches "
            "using resume-safe existing scripts."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run S3 uploads and paid BDA "
            "processing. Without this flag, "
            "perform local preparation and "
            "sequential dry-run only."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = args.config.resolve()

    if not config_path.is_file():
        raise RuntimeError(
            f"Config missing: {config_path}"
        )

    book_id, version = parse_identity(
        config_path
    )

    book_root = (
        REPO_ROOT
        / "data"
        / "multimodal-output"
        / book_id
        / version
    )

    full_book_root = (
        book_root / "full-book"
    )

    manifest_path = (
        full_book_root
        / "full-book-batch-manifest.json"
    )

    jobs_root = (
        full_book_root / "bda-jobs"
    )

    results_root = (
        full_book_root / "bda-results"
    )

    upload_report = (
        full_book_root
        / "full-book-s3-upload-report.json"
    )

    sequential_report = (
        full_book_root
        / (
            "sequential-execution-report.json"
            if args.execute
            else "sequential-dry-run-report.json"
        )
    )

    bridge_report = (
        full_book_root
        / "bda-bridge-report.json"
    )

    normalized_bridge = (
        book_root
        / "pipeline"
        / "bda"
        / "normalized"
    )

    full_book_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock_path = (
        full_book_root / ".bda-bridge.lock"
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
                "Another BDA bridge process is "
                f"already running for {book_id}."
            ) from error

        print("=" * 80)
        print("FULL-BOOK BDA BRIDGE")
        print("=" * 80)
        print("Book ID:       ", book_id)
        print("Version:       ", version)
        print("Config:        ", config_path)
        print("Execute:       ", args.execute)
        print("Full-book root:", full_book_root)
        print()

        prepare_command = [
            sys.executable,
            str(
                SCRIPTS_ROOT
                / "prepare_full_book_batches.py"
            ),
            "--config",
            str(config_path),
        ]

        run_command(prepare_command)

        if not manifest_path.is_file():
            raise RuntimeError(
                "Batch preparation did not create "
                f"the expected manifest: "
                f"{manifest_path}"
            )

        manifest = load_json(
            manifest_path
        )

        batch_ids = expected_batch_ids(
            manifest
        )

        print()
        print(
            "Prepared batch count:",
            len(batch_ids),
        )
        print(
            "First batch:",
            batch_ids[0],
        )
        print(
            "Last batch:",
            batch_ids[-1],
        )

        sequential_command = [
            sys.executable,
            str(
                SCRIPTS_ROOT
                / (
                    "run_full_book_batches_"
                    "sequentially.py"
                )
            ),
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--jobs-dir",
            str(jobs_root),
            "--results-root",
            str(results_root),
            "--report",
            str(sequential_report),
            "--through-stage",
            "bda-normalized",
        ]

        if not args.execute:
            run_command(
                sequential_command
            )

            write_report(
                bridge_report,
                {
                    "schema_version": "1.0",
                    "generated_at": utc_now(),
                    "status": "BDA_BATCHED",
                    "execution_mode": "dry_run",
                    "book_id": book_id,
                    "book_version": version,
                    "config_path": str(
                        config_path
                    ),
                    "manifest_path": str(
                        manifest_path
                    ),
                    "batch_count": len(
                        batch_ids
                    ),
                    "batch_ids": batch_ids,
                    "sequential_report": str(
                        sequential_report
                    ),
                    "aws_calls": 0,
                    "paid_bda_calls": 0,
                },
            )

            print()
            print("=" * 80)
            print("BDA BRIDGE DRY-RUN PASSED")
            print("=" * 80)
            print("Manifest:", manifest_path)
            print("Batches: ", len(batch_ids))
            print("Report:  ", bridge_report)
            print("AWS calls: 0")
            return

        upload_command = [
            sys.executable,
            str(
                SCRIPTS_ROOT
                / "upload_full_book_batches.py"
            ),
            str(manifest_path),
            "--report",
            str(upload_report),
            "--upload",
        ]

        # BDA_UPLOAD_RESUME_GUARD
        #
        # The bridge uploads every batch before the
        # sequential BDA runner starts. Therefore, if
        # even one durable invocation record exists,
        # the upload stage was already completed in an
        # earlier run and must not be repeated.
        existing_invocation_records = []

        for batch_id in batch_ids:
            job_path = (
                jobs_root / f"{batch_id}.json"
            )

            if not job_path.is_file():
                continue

            try:
                job_record = load_json(
                    job_path
                )
            except Exception:
                continue

            invocation_arn = str(
                job_record.get(
                    "invocation_arn",
                    job_record.get(
                        "invocationArn",
                        "",
                    ),
                )
            ).strip()

            if invocation_arn:
                existing_invocation_records.append(
                    batch_id
                )

        if existing_invocation_records:
            print()
            print(
                "SKIP: S3 batch upload already "
                "completed before existing BDA "
                "invocations."
            )
            print(
                "Existing invocation records:",
                len(existing_invocation_records),
                "/",
                len(batch_ids),
            )

        else:
            run_command(
                upload_command
            )

        sequential_command.append(
            "--execute"
        )

        run_command(
            sequential_command
        )

        selected_roots = {
            batch_id: select_normalized_root(
                results_root,
                batch_id,
            )
            for batch_id in batch_ids
        }

        create_normalized_bridge(
            selected_roots=selected_roots,
            bridge_root=normalized_bridge,
        )

        bridged_records = list(
            normalized_bridge.rglob(
                "content-units.jsonl"
            )
        )

        if len(bridged_records) != len(
            batch_ids
        ):
            raise RuntimeError(
                "Normalized bridge validation "
                "failed: "
                f"expected={len(batch_ids)}, "
                f"actual={len(bridged_records)}"
            )

        write_report(
            bridge_report,
            {
                "schema_version": "1.0",
                "generated_at": utc_now(),
                "status": "BDA_NORMALIZED",
                "execution_mode": "execute",
                "book_id": book_id,
                "book_version": version,
                "config_path": str(
                    config_path
                ),
                "manifest_path": str(
                    manifest_path
                ),
                "batch_count": len(
                    batch_ids
                ),
                "normalized_batch_count": len(
                    selected_roots
                ),
                "selected_normalized_roots": {
                    key: str(value)
                    for key, value
                    in selected_roots.items()
                },
                "normalized_bridge": str(
                    normalized_bridge
                ),
                "upload_report": str(
                    upload_report
                ),
                "sequential_report": str(
                    sequential_report
                ),
            },
        )

        print()
        print("=" * 80)
        print("BDA BRIDGE COMPLETED")
        print("=" * 80)
        print("Normalized batches:", len(
            selected_roots
        ))
        print("Bridge:", normalized_bridge)
        print("Report:", bridge_report)


if __name__ == "__main__":
    main()
