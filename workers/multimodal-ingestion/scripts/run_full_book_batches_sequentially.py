from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.book_config import load_book_config


FULL_BOOK_ROOT = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/full-book"
)

DEFAULT_MANIFEST = (
    FULL_BOOK_ROOT
    / "full-book-batch-manifest.json"
)

DEFAULT_JOBS_DIR = (
    FULL_BOOK_ROOT
    / "bda-jobs"
)

DEFAULT_RESULTS_ROOT = (
    FULL_BOOK_ROOT
    / "bda-results"
)

DEFAULT_REPORT = (
    FULL_BOOK_ROOT
    / "full-book-sequential-run-report.json"
)

SCRIPTS_DIR = Path(
    "workers/multimodal-ingestion/scripts"
)

PYTHONPATH_ENTRIES = [
    str(SCRIPTS_DIR),
    "workers/multimodal-ingestion",
]



def resolve_runner_runtime(
    config_path: Path | None,
) -> dict[str, Any]:
    if config_path is None:
        return {
            "mode": "legacy",
            "config_path": None,
            "book_id": (
                "grade-9-english-kaveri"
            ),
            "book_version": "v1",
            "vector_dimensions": 1024,
            "manifest_path": str(
                DEFAULT_MANIFEST
            ),
            "jobs_dir": str(
                DEFAULT_JOBS_DIR
            ),
            "results_root": str(
                DEFAULT_RESULTS_ROOT
            ),
            "report_path": str(
                DEFAULT_REPORT
            ),
            "start_batch": "batch-0003",
            "end_batch": "batch-0015",
        }

    config = load_book_config(
        config_path
    )

    full_book_root = (
        Path(config.storage.local_root)
        / "full-book"
    )

    return {
        "mode": "book_config",
        "config_path": str(config_path),
        "book_id": config.book.book_id,
        "book_version": (
            config.book.version
        ),
        "vector_dimensions": int(
            config.models.embedding.dimensions
        ),
        "manifest_path": str(
            full_book_root
            / "full-book-batch-manifest.json"
        ),
        "jobs_dir": str(
            full_book_root
            / "bda-jobs"
        ),
        "results_root": str(
            full_book_root
            / "bda-results"
        ),
        "report_path": str(
            full_book_root
            / "full-book-sequential-run-report.json"
        ),
        "start_batch": None,
        "end_batch": None,
    }


def validate_manifest_identity(
    manifest: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    if runtime["mode"] == "legacy":
        return

    expected_book_id = str(
        runtime["book_id"]
    )

    expected_book_version = str(
        runtime["book_version"]
    )

    actual_book_id = manifest.get(
        "book_id"
    )

    actual_book_version = manifest.get(
        "book_version"
    )

    if actual_book_id != expected_book_id:
        raise RuntimeError(
            "Manifest book_id does not "
            "match configuration: "
            f"expected={expected_book_id!r}, "
            f"actual={actual_book_id!r}"
        )

    if (
        actual_book_version
        != expected_book_version
    ):
        raise RuntimeError(
            "Manifest book_version does not "
            "match configuration: "
            f"expected="
            f"{expected_book_version!r}, "
            f"actual={actual_book_version!r}"
        )


def manifest_batch_bounds(
    manifest: dict[str, Any],
) -> tuple[str, str]:
    batches = manifest.get(
        "batches",
        [],
    )

    if not isinstance(batches, list):
        raise RuntimeError(
            "Manifest batches field is invalid."
        )

    batch_ids = [
        str(batch["batch_id"])
        for batch in batches
        if isinstance(batch, dict)
        and batch.get("batch_id")
    ]

    if not batch_ids:
        raise RuntimeError(
            "Manifest contains no batches."
        )

    ordered = sorted(
        batch_ids,
        key=batch_number,
    )

    return ordered[0], ordered[-1]


def config_cli_args(
    config_path: Path | None,
) -> list[str]:
    if config_path is None:
        return []

    return [
        "--config",
        str(config_path),
    ]


def build_bda_preflight_command(
    manifest_path: Path,
    batch_id: str,
    preflight_path: Path,
    config_path: Path | None,
) -> list[str]:
    return [
        sys.executable,
        str(
            SCRIPTS_DIR
            / "preflight_full_book_bda_pilot.py"
        ),
        "--manifest",
        str(manifest_path),
        "--batch-id",
        batch_id,
        "--report",
        str(preflight_path),
        *config_cli_args(config_path),
    ]


def build_titan_command(
    records_path: Path,
    titan_dir: Path,
    vector_dimensions: int,
) -> list[str]:
    return [
        sys.executable,
        str(
            SCRIPTS_DIR
            / "embed_records_titan_v2.py"
        ),
        str(records_path),
        "--output-dir",
        str(titan_dir),
        "--dimensions",
        str(vector_dimensions),
    ]


def build_bulk_command(
    embeddings_path: Path,
    bulk_dir: Path,
    config_path: Path | None,
) -> list[str]:
    return [
        sys.executable,
        str(
            SCRIPTS_DIR
            / "prepare_opensearch_bulk.py"
        ),
        str(embeddings_path),
        "--output-dir",
        str(bulk_dir),
        *config_cli_args(config_path),
    ]


def build_upload_command(
    bulk_dir: Path,
    upload_dir: Path,
    config_path: Path | None,
) -> list[str]:
    return [
        sys.executable,
        str(
            SCRIPTS_DIR
            / "upload_opensearch_bulk.py"
        ),
        str(
            bulk_dir
            / "bulk-index.ndjson"
        ),
        "--preparation-report",
        str(
            bulk_dir
            / "bulk-preparation-report.json"
        ),
        "--output-dir",
        str(upload_dir),
        *config_cli_args(config_path),
    ]


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file missing: {path}"
        )

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected JSON object: {path}"
        )

    return value


def atomic_write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        path,
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def count_jsonl_records(
    path: Path,
) -> int:
    if not path.is_file():
        return 0

    return sum(
        1
        for line in path.open(
            encoding="utf-8"
        )
        if line.strip()
    )


def build_environment() -> dict[str, str]:
    environment = os.environ.copy()

    existing_pythonpath = environment.get(
        "PYTHONPATH",
        "",
    )

    entries = list(
        PYTHONPATH_ENTRIES
    )

    if existing_pythonpath:
        entries.append(
            existing_pythonpath
        )

    environment["PYTHONPATH"] = (
        os.pathsep.join(entries)
    )

    return environment


def run_command(
    label: str,
    command: list[str],
) -> None:
    print()
    print("=" * 72)
    print(label)
    print("=" * 72)
    print(
        shlex.join(command)
    )
    print()

    subprocess.run(
        command,
        check=True,
        env=build_environment(),
    )


def batch_number(
    batch_id: str,
) -> int:
    prefix = "batch-"

    if not batch_id.startswith(prefix):
        raise ValueError(
            f"Invalid batch ID: {batch_id}"
        )

    return int(
        batch_id[len(prefix):]
    )


def select_batches(
    manifest: dict[str, Any],
    start_batch: str,
    end_batch: str,
) -> list[dict[str, Any]]:
    batches = manifest.get(
        "batches",
        [],
    )

    if not isinstance(batches, list):
        raise RuntimeError(
            "Manifest batches field is invalid."
        )

    start_number = batch_number(
        start_batch
    )

    end_number = batch_number(
        end_batch
    )

    if start_number > end_number:
        raise RuntimeError(
            "Start batch is after end batch."
        )

    selected: list[
        dict[str, Any]
    ] = []

    for batch in batches:
        if not isinstance(batch, dict):
            continue

        batch_id = str(
            batch.get("batch_id", "")
        )

        number = batch_number(
            batch_id
        )

        if (
            start_number
            <= number
            <= end_number
        ):
            selected.append(batch)

    expected_count = (
        end_number
        - start_number
        + 1
    )

    if len(selected) != expected_count:
        raise RuntimeError(
            "Manifest does not contain all "
            "requested batches."
        )

    return selected


def successful_job(
    path: Path,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    job = load_json_object(
        path
    )

    if (
        job.get("status") == "COMPLETED"
        and job.get("latest_status")
        == "Success"
        and job.get("completed") is True
    ):
        return job

    raise RuntimeError(
        "Existing BDA job is not successfully "
        "completed.\n"
        f"Path: {path}\n"
        f"status={job.get('status')}\n"
        f"latest_status="
        f"{job.get('latest_status')}"
    )


def successful_preflight(
    path: Path,
    config_path: Path | None = None,
) -> bool:
    if not path.is_file():
        return False

    report = load_json_object(path)

    valid = (
        report.get("status")
        == "PREFLIGHT_PASSED"
        and report.get("bda_invoked")
        is False
        and report.get(
            "invocation_submitted"
        )
        is False
    )

    if not valid:
        return False

    if config_path is None:
        return True

    runtime = resolve_runner_runtime(
        config_path
    )

    configuration = report.get(
        "configuration"
    )

    if not isinstance(
        configuration,
        dict,
    ):
        return False

    return (
        report.get("book_id")
        == runtime["book_id"]
        and report.get("book_version")
        == runtime["book_version"]
        and configuration.get("mode")
        == "book_config"
        and configuration.get(
            "config_path"
        )
        == str(config_path)
    )


def successful_download(
    path: Path,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    report = load_json_object(
        path
    )

    if (
        report.get("status")
        == "VALIDATED"
        and report.get("validated")
        is True
        and int(
            report.get(
                "result_page_count",
                0,
            )
        )
        == 20
    ):
        return report

    raise RuntimeError(
        f"Invalid download report: {path}"
    )


def successful_normalization(
    normalized_dir: Path,
) -> bool:
    report_path = (
        normalized_dir
        / "normalization-report.json"
    )

    content_units = (
        normalized_dir
        / "content-units.jsonl"
    )

    if not report_path.is_file():
        return False

    report = load_json_object(
        report_path
    )

    expected_count = int(
        report.get(
            "normalized_content_unit_count",
            0,
        )
    )

    actual_count = count_jsonl_records(
        content_units
    )

    if (
        expected_count > 0
        and actual_count == expected_count
        and int(
            report.get(
                "missing_page_reference_count",
                0,
            )
        )
        == 0
        and int(
            report.get(
                "missing_asset_count",
                0,
            )
        )
        == 0
    ):
        return True

    raise RuntimeError(
        "Existing normalized output is "
        "incomplete or invalid:\n"
        f"{normalized_dir}"
    )


def validation_files(
    normalized_dir: Path,
) -> list[Path]:
    return [
        normalized_dir
        / "content-units.jsonl",
        normalized_dir
        / "figures.jsonl",
        normalized_dir
        / "tables.jsonl",
    ]


def current_validation_hashes(
    normalized_dir: Path,
) -> dict[str, str]:
    hashes: dict[str, str] = {}

    for path in validation_files(
        normalized_dir
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Normalized file missing: {path}"
            )

        hashes[path.name] = sha256_file(
            path
        )

    return hashes


def validation_marker_valid(
    normalized_dir: Path,
) -> bool:
    marker_path = (
        normalized_dir
        / "validation-passed.json"
    )

    if not marker_path.is_file():
        return False

    marker = load_json_object(
        marker_path
    )

    if marker.get("status") != "PASSED":
        return False

    return (
        marker.get("file_sha256")
        == current_validation_hashes(
            normalized_dir
        )
    )


def write_validation_marker(
    normalized_dir: Path,
) -> None:
    marker_path = (
        normalized_dir
        / "validation-passed.json"
    )

    atomic_write_json(
        marker_path,
        {
            "schema_version": "1.0",
            "generated_at": utc_now(),
            "status": "PASSED",
            "normalized_directory": str(
                normalized_dir
            ),
            "file_sha256": (
                current_validation_hashes(
                    normalized_dir
                )
            ),
        },
    )


def successful_embedding_preparation(
    embedding_ready_dir: Path,
) -> dict[str, Any] | None:
    report_path = (
        embedding_ready_dir
        / "embedding-preparation-report.json"
    )

    records_path = (
        embedding_ready_dir
        / "embedding-records.jsonl"
    )

    if not report_path.is_file():
        return None

    report = load_json_object(
        report_path
    )

    expected_count = int(
        report.get(
            "embedding_record_count",
            0,
        )
    )

    actual_count = count_jsonl_records(
        records_path
    )

    if (
        expected_count > 0
        and actual_count == expected_count
    ):
        return report

    raise RuntimeError(
        "Existing embedding preparation is "
        "incomplete or invalid:\n"
        f"{embedding_ready_dir}"
    )


def successful_titan_embeddings(
    titan_dir: Path,
) -> dict[str, Any] | None:
    manifest_path = (
        titan_dir
        / "embedding-manifest.json"
    )

    embeddings_path = (
        titan_dir
        / "embeddings.jsonl"
    )

    if not manifest_path.is_file():
        return None

    manifest = load_json_object(
        manifest_path
    )

    input_count = int(
        manifest.get(
            "input_record_count",
            0,
        )
    )

    completed_count = int(
        manifest.get(
            "completed_record_count",
            0,
        )
    )

    output_count = count_jsonl_records(
        embeddings_path
    )

    if (
        manifest.get("status")
        == "COMPLETED"
        and input_count > 0
        and completed_count == input_count
        and output_count == completed_count
    ):
        return manifest

    return None


def successful_bulk_preparation(
    bulk_dir: Path,
) -> dict[str, Any] | None:
    report_path = (
        bulk_dir
        / "bulk-preparation-report.json"
    )

    payload_path = (
        bulk_dir
        / "bulk-index.ndjson"
    )

    if not report_path.is_file():
        return None

    report = load_json_object(
        report_path
    )

    validation = report.get(
        "validation",
        {},
    )

    document_count = int(
        validation.get(
            "document_count",
            0,
        )
    )

    unique_count = int(
        validation.get(
            "unique_document_ids",
            0,
        )
    )

    dimensions = int(
        validation.get(
            "vector_dimensions",
            0,
        )
    )

    if (
        report.get("status") == "PREPARED"
        and document_count > 0
        and unique_count == document_count
        and dimensions == 1024
        and payload_path.is_file()
    ):
        return report

    raise RuntimeError(
        "Existing OpenSearch bulk preparation "
        "is invalid:\n"
        f"{bulk_dir}"
    )


def successful_upload(
    upload_dir: Path,
) -> dict[str, Any] | None:
    report_path = (
        upload_dir
        / "bulk-upload-report.json"
    )

    if not report_path.is_file():
        return None

    report = load_json_object(
        report_path
    )

    bulk_result = report.get(
        "bulk_result",
        {},
    )

    if (
        report.get("status")
        == "COMPLETED"
        and report.get("uploaded")
        is True
        and int(
            bulk_result.get(
                "failure_count",
                -1,
            )
        )
        == 0
    ):
        return report

    raise RuntimeError(
        "Existing OpenSearch upload report "
        "is invalid:\n"
        f"{upload_dir}"
    )


def process_batch(
    batch: dict[str, Any],
    *,
    execute: bool,
    manifest_path: Path,
    jobs_dir: Path,
    results_root: Path,
    config_path: Path | None,
    vector_dimensions: int,
) -> dict[str, Any]:
    batch_id = str(
        batch["batch_id"]
    )

    page_start = int(
        batch["source_page_start"]
    )

    page_end = int(
        batch["source_page_end"]
    )

    preflight_path = (
        jobs_dir
        / f"{batch_id}-preflight.json"
    )

    job_path = (
        jobs_dir
        / f"{batch_id}.json"
    )

    result: dict[str, Any] = {
        "batch_id": batch_id,
        "source_page_start": page_start,
        "source_page_end": page_end,
        "status": "PLANNED",
        "stages": {},
    }

    print()
    print("#" * 72)
    print(
        f"{batch_id} | pages "
        f"{page_start}-{page_end}"
    )
    print("#" * 72)

    existing_job = successful_job(
        job_path
    )

    # Hard safety boundary: dry-run mode must return
    # before any subprocess, AWS, model, or OpenSearch call.
    if not execute:
        if existing_job is None:
            result["stages"]["preflight"] = "RUN"
            result["stages"]["bda"] = "RUN"
            downstream_status = "RUN_AFTER_BDA"

            print(
                "PLAN: preflight → BDA → download "
                "→ normalize → validate → embeddings "
                "→ OpenSearch"
            )

        else:
            result["stages"]["preflight"] = (
                "SKIPPED_COMPLETED"
            )
            result["stages"]["bda"] = (
                "SKIPPED_COMPLETED"
            )
            downstream_status = "CHECK_OR_RESUME"

            print(
                "PLAN: existing BDA job is completed; "
                "downstream stages would be checked "
                "or resumed in execute mode"
            )

        for stage in (
            "download",
            "normalize",
            "validate",
            "prepare_embeddings",
            "titan_embeddings",
            "prepare_bulk",
            "upload_opensearch",
        ):
            result["stages"][stage] = (
                downstream_status
            )

        result["status"] = "PLANNED"
        return result

    if existing_job is None:
        result["stages"]["preflight"] = (
            "RUN"
        )
        result["stages"]["bda"] = "RUN"

        if not execute:
            for stage in (
                "download",
                "normalize",
                "validate",
                "prepare_embeddings",
                "titan_embeddings",
                "prepare_bulk",
                "upload_opensearch",
            ):
                result["stages"][stage] = (
                    "RUN_AFTER_BDA"
                )

            print(
                "PLAN: preflight → BDA → download "
                "→ normalize → validate → embeddings "
                "→ OpenSearch"
            )

            return result

        if preflight_path.exists():
            if not successful_preflight(
                preflight_path,
                config_path=config_path,
            ):
                raise RuntimeError(
                    "Existing preflight report is "
                    "not valid:\n"
                    f"{preflight_path}"
                )

            print(
                "SKIP: valid preflight report exists"
            )

        else:
            run_command(
                f"{batch_id}: BDA PREFLIGHT",
                build_bda_preflight_command(
                    manifest_path=manifest_path,
                    batch_id=batch_id,
                    preflight_path=preflight_path,
                    config_path=config_path,
                ),
            )

        run_command(
            f"{batch_id}: BDA INVOCATION",
            [
                sys.executable,
                str(
                    SCRIPTS_DIR
                    / "invoke_full_book_bda_pilot.py"
                ),
                "--preflight-report",
                str(preflight_path),
                "--job-record",
                str(job_path),
            ],
        )

        existing_job = successful_job(
            job_path
        )

    else:
        result["stages"]["preflight"] = (
            "SKIPPED_COMPLETED"
        )
        result["stages"]["bda"] = (
            "SKIPPED_COMPLETED"
        )
        print(
            "SKIP: BDA job already completed"
        )

    invocation_arn = str(
        existing_job["invocation_arn"]
    )

    invocation_id = invocation_arn.rsplit(
        "/",
        1,
    )[-1]

    invocation_root = (
        results_root
        / batch_id
        / invocation_id
    )

    download_report_path = (
        invocation_root
        / "download-report.json"
    )

    download_report = successful_download(
        download_report_path
    )

    if download_report is None:
        result["stages"]["download"] = "RUN"

        run_command(
            f"{batch_id}: DOWNLOAD BDA OUTPUT",
            [
                sys.executable,
                str(
                    SCRIPTS_DIR
                    / "download_full_book_bda_batch.py"
                ),
                "--manifest",
                str(manifest_path),
                "--job-record",
                str(job_path),
                "--batch-id",
                batch_id,
                "--output-root",
                str(results_root),
            ],
        )

        download_report = successful_download(
            download_report_path
        )

        if download_report is None:
            raise RuntimeError(
                "Download completed without a "
                "valid report."
            )

    else:
        result["stages"]["download"] = (
            "SKIPPED_COMPLETED"
        )
        print(
            "SKIP: BDA output already downloaded"
        )

    result_json = Path(
        str(
            download_report[
                "result_json_path"
            ]
        )
    )

    metadata_path = Path(
        str(
            download_report[
                "normalizer_metadata_path"
            ]
        )
    )

    normalized_dir = (
        invocation_root
        / "normalized"
    )

    if successful_normalization(
        normalized_dir
    ):
        result["stages"]["normalize"] = (
            "SKIPPED_COMPLETED"
        )
        print(
            "SKIP: normalization already complete"
        )

    else:
        result["stages"]["normalize"] = "RUN"

        if normalized_dir.exists():
            raise RuntimeError(
                "Normalized directory exists but "
                "has no valid completion report:\n"
                f"{normalized_dir}"
            )

        run_command(
            f"{batch_id}: NORMALIZE",
            [
                sys.executable,
                str(
                    SCRIPTS_DIR
                    / "normalize_bda_result.py"
                ),
                str(result_json),
                "--sample-metadata",
                str(metadata_path),
                "--output-dir",
                str(normalized_dir),
            ],
        )

        if not successful_normalization(
            normalized_dir
        ):
            raise RuntimeError(
                "Normalization verification failed."
            )

    if validation_marker_valid(
        normalized_dir
    ):
        result["stages"]["validate"] = (
            "SKIPPED_COMPLETED"
        )
        print(
            "SKIP: normalized output already validated"
        )

    else:
        result["stages"]["validate"] = "RUN"

        run_command(
            f"{batch_id}: VALIDATE NORMALIZED OUTPUT",
            [
                sys.executable,
                str(
                    SCRIPTS_DIR
                    / "validate_normalized_output.py"
                ),
                str(normalized_dir),
            ],
        )

        write_validation_marker(
            normalized_dir
        )

    embedding_ready_dir = (
        normalized_dir
        / "embedding-ready"
    )

    preparation_report = (
        successful_embedding_preparation(
            embedding_ready_dir
        )
    )

    if preparation_report is None:
        result[
            "stages"
        ][
            "prepare_embeddings"
        ] = "RUN"

        if embedding_ready_dir.exists():
            raise RuntimeError(
                "Embedding-ready directory exists "
                "without a valid report:\n"
                f"{embedding_ready_dir}"
            )

        run_command(
            f"{batch_id}: PREPARE EMBEDDING RECORDS",
            [
                sys.executable,
                str(
                    SCRIPTS_DIR
                    / "prepare_embedding_records.py"
                ),
                str(normalized_dir),
                "--output-dir",
                str(embedding_ready_dir),
            ],
        )

        preparation_report = (
            successful_embedding_preparation(
                embedding_ready_dir
            )
        )

        if preparation_report is None:
            raise RuntimeError(
                "Embedding preparation verification "
                "failed."
            )

    else:
        result[
            "stages"
        ][
            "prepare_embeddings"
        ] = "SKIPPED_COMPLETED"

        print(
            "SKIP: embedding records already prepared"
        )

    records_path = (
        embedding_ready_dir
        / "embedding-records.jsonl"
    )

    titan_dir = (
        embedding_ready_dir
        / "titan-text-v2"
    )

    titan_manifest = (
        successful_titan_embeddings(
            titan_dir
        )
    )

    if titan_manifest is None:
        result[
            "stages"
        ][
            "titan_embeddings"
        ] = "RUN"

        run_command(
            f"{batch_id}: TITAN EMBEDDINGS",
            build_titan_command(
                records_path=records_path,
                titan_dir=titan_dir,
                vector_dimensions=(
                    vector_dimensions
                ),
            ),
        )

        titan_manifest = (
            successful_titan_embeddings(
                titan_dir
            )
        )

        if titan_manifest is None:
            raise RuntimeError(
                "Titan embedding verification failed."
            )

    else:
        result[
            "stages"
        ][
            "titan_embeddings"
        ] = "SKIPPED_COMPLETED"

        print(
            "SKIP: Titan embeddings already complete"
        )

    embeddings_path = (
        titan_dir
        / "embeddings.jsonl"
    )

    bulk_dir = (
        normalized_dir
        / "opensearch-serverless"
        / "bulk"
    )

    bulk_report = (
        successful_bulk_preparation(
            bulk_dir
        )
    )

    if bulk_report is None:
        result[
            "stages"
        ][
            "prepare_bulk"
        ] = "RUN"

        if bulk_dir.exists():
            raise RuntimeError(
                "Bulk directory exists without a "
                "valid preparation report:\n"
                f"{bulk_dir}"
            )

        run_command(
            f"{batch_id}: PREPARE OPENSEARCH BULK",
            build_bulk_command(
                embeddings_path=embeddings_path,
                bulk_dir=bulk_dir,
                config_path=config_path,
            ),
        )

        bulk_report = (
            successful_bulk_preparation(
                bulk_dir
            )
        )

        if bulk_report is None:
            raise RuntimeError(
                "Bulk preparation verification "
                "failed."
            )

    else:
        result[
            "stages"
        ][
            "prepare_bulk"
        ] = "SKIPPED_COMPLETED"

        print(
            "SKIP: OpenSearch bulk already prepared"
        )

    upload_dir = (
        bulk_dir
        / "upload"
    )

    upload_report = successful_upload(
        upload_dir
    )

    if upload_report is None:
        result[
            "stages"
        ][
            "upload_opensearch"
        ] = "RUN"

        if upload_dir.exists():
            raise RuntimeError(
                "Upload directory exists without a "
                "valid completion report:\n"
                f"{upload_dir}"
            )

        run_command(
            f"{batch_id}: UPLOAD TO OPENSEARCH",
            build_upload_command(
                bulk_dir=bulk_dir,
                upload_dir=upload_dir,
                config_path=config_path,
            ),
        )

        upload_report = successful_upload(
            upload_dir
        )

        if upload_report is None:
            raise RuntimeError(
                "OpenSearch upload verification "
                "failed."
            )

    else:
        result[
            "stages"
        ][
            "upload_opensearch"
        ] = "SKIPPED_COMPLETED"

        print(
            "SKIP: OpenSearch upload already complete"
        )

    result.update(
        {
            "status": "COMPLETED",
            "invocation_id": invocation_id,
            "invocation_root": str(
                invocation_root
            ),
            "embedding_record_count": int(
                preparation_report[
                    "embedding_record_count"
                ]
            ),
            "final_opensearch_count": (
                upload_report.get(
                    "final_count"
                )
            ),
            "completed_at": utc_now(),
        }
    )

    print()
    print(
        f"COMPLETED: {batch_id} | "
        f"pages {page_start}-{page_end} | "
        f"records "
        f"{result['embedding_record_count']} | "
        f"index count "
        f"{result['final_opensearch_count']}"
    )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially process full-book BDA "
            "batches with resume-safe stage checks."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional book configuration JSON. "
            "Paths, identity and embedding "
            "dimensions are derived from BookConfig."
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--start-batch",
        default=None,
    )

    parser.add_argument(
        "--end-batch",
        default=None,
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually run AWS/model/OpenSearch "
            "operations. Without this flag, only "
            "a local dry-run plan is created."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    runtime = resolve_runner_runtime(
        args.config
    )

    manifest_path = (
        args.manifest
        or Path(runtime["manifest_path"])
    )

    jobs_dir = (
        args.jobs_dir
        or Path(runtime["jobs_dir"])
    )

    results_root = (
        args.results_root
        or Path(runtime["results_root"])
    )

    report_path = (
        args.report
        or Path(runtime["report_path"])
    )

    manifest = load_json_object(
        manifest_path
    )

    validate_manifest_identity(
        manifest,
        runtime,
    )

    first_batch, last_batch = (
        manifest_batch_bounds(manifest)
    )

    start_batch = (
        args.start_batch
        or runtime.get("start_batch")
        or first_batch
    )

    end_batch = (
        args.end_batch
        or runtime.get("end_batch")
        or last_batch
    )

    selected_batches = select_batches(
        manifest=manifest,
        start_batch=start_batch,
        end_batch=end_batch,
    )

    mode = (
        "EXECUTE"
        if args.execute
        else "DRY_RUN"
    )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "RUNNING",
        "mode": mode,
        "configuration": runtime,
        "manifest_path": str(
            manifest_path
        ),
        "jobs_dir": str(jobs_dir),
        "results_root": str(
            results_root
        ),
        "start_batch": start_batch,
        "end_batch": end_batch,
        "selected_batch_count": len(
            selected_batches
        ),
        "batches": [],
    }

    atomic_write_json(
        report_path,
        report,
    )

    print("=" * 72)
    print("FULL-BOOK SEQUENTIAL BATCH RUNNER")
    print("=" * 72)
    print(f"Mode:          {mode}")
    print(
        f"Config mode:   "
        f"{runtime['mode']}"
    )
    print(
        f"Book version:  "
        f"{runtime['book_version']}"
    )
    print(
        f"Dimensions:    "
        f"{runtime['vector_dimensions']}"
    )
    print(
        f"Batch range:   "
        f"{start_batch} → {end_batch}"
    )
    print(
        f"Batch count:   "
        f"{len(selected_batches)}"
    )
    print(f"Manifest:      {manifest_path}")
    print(f"Results root:  {results_root}")
    print(f"Report:        {report_path}")

    try:
        for batch in selected_batches:
            batch_result = process_batch(
                batch,
                execute=args.execute,
                manifest_path=manifest_path,
                jobs_dir=jobs_dir,
                results_root=results_root,
                config_path=args.config,
                vector_dimensions=int(
                    runtime[
                        "vector_dimensions"
                    ]
                ),
            )

            report["batches"].append(
                batch_result
            )

            report["updated_at"] = utc_now()

            atomic_write_json(
                report_path,
                report,
            )

    except Exception as error:
        report["status"] = "FAILED"
        report["updated_at"] = utc_now()
        report["error"] = str(error)

        atomic_write_json(
            report_path,
            report,
        )

        raise

    completed_count = sum(
        1
        for batch in report["batches"]
        if batch.get("status")
        == "COMPLETED"
    )

    report["status"] = (
        "COMPLETED"
        if args.execute
        else "DRY_RUN_PASSED"
    )

    report["completed_batch_count"] = (
        completed_count
    )

    report["updated_at"] = utc_now()
    report["completed_at"] = utc_now()

    atomic_write_json(
        report_path,
        report,
    )

    print()
    print("=" * 72)
    print("SEQUENTIAL RUN RESULT")
    print("=" * 72)
    print(
        f"Status:            "
        f"{report['status']}"
    )
    print(
        f"Selected batches:  "
        f"{len(selected_batches)}"
    )

    if args.execute:
        print(
            f"Completed batches: "
            f"{completed_count}"
        )
    else:
        print("AWS calls:         0")
        print("BDA invocations:   0")
        print("Titan calls:       0")
        print("OpenSearch writes: 0")

    print(f"Report:            {report_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print(
            "Sequential processing interrupted.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except subprocess.CalledProcessError as error:
        print(
            "A pipeline command failed with "
            f"exit code {error.returncode}.",
            file=sys.stderr,
        )
        raise SystemExit(
            error.returncode
        )

    except Exception as error:
        print(
            f"Sequential processing failed: "
            f"{error}",
            file=sys.stderr,
        )
        raise SystemExit(1)