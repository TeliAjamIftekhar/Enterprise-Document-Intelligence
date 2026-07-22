from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_DEFAULT = Path(
    "data/textbook-automation/"
    "ncert-i-x-book-registry.json"
)

STATE_DEFAULT = Path(
    "data/textbook-automation/"
    "ncert-i-x-auto-pipeline-state.json"
)

LOG_ROOT = Path(
    "data/textbook-automation/logs"
)

CONFIG_ROOT = Path(
    "workers/multimodal-ingestion/config/books"
)

MANIFEST_ROOT = CONFIG_ROOT / "manifests"

SCRIPTS_ROOT = Path(
    "workers/multimodal-ingestion/scripts"
)

SURYA_APPROVAL_DEFAULT = Path(
    "data/textbook-automation/"
    "ocr-approvals/surya-urdu-v1.json"
)

KNOWN_VERIFIED_BOOKS = {
    "grade-9-mathematics-ganita-manjari",
}

STAGES = [
    "DISCOVERED",
    "DOWNLOADED",
    "CONFIG_GENERATED",
    "EXTRACTED",
    "MERGED",
    "BDA_BATCHES_PREPARED",
    "BDA_PROCESSING",
    "BDA_DOWNLOADED",
    "BDA_NORMALIZED",
    "OCR_QUALITY_CHECKED",
    "OCR_FALLBACK_REQUIRED",
    "OCR_FALLBACK_PROCESSING",
    "OCR_FALLBACK_VERIFIED",
    "UNIFIED_RECORDS_PREPARED",
    "EMBEDDING_RECORDS_PREPARED",
    "EMBEDDED",
    "INDEXED",
    "VERIFIED",
    "FAILED",
]


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return value


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def parse_grades(value: str) -> set[int]:
    selected: set[int] = set()

    for component in value.split(","):
        component = component.strip()

        if not component:
            continue

        if "-" in component:
            start_raw, end_raw = (
                component.split("-", 1)
            )

            start = int(start_raw)
            end = int(end_raw)

            if end < start:
                raise ValueError(
                    "Grade range end cannot be "
                    "smaller than start."
                )

            selected.update(
                range(start, end + 1)
            )

        else:
            selected.add(int(component))

    invalid = sorted(
        grade
        for grade in selected
        if not 1 <= grade <= 10
    )

    if invalid:
        raise ValueError(
            f"Invalid grades: {invalid}"
        )

    if not selected:
        raise ValueError(
            "At least one grade is required."
        )

    return selected


def valid_report(
    path: Path,
    *,
    status_field: str,
    accepted_statuses: set[str],
) -> bool:
    if not path.is_file():
        return False

    try:
        report = load_json_object(path)
    except Exception:
        return False

    return str(
        report.get(status_field, "")
    ) in accepted_statuses


def paths_for_book(
    book_id: str,
    version: str,
) -> dict[str, Path]:
    archive_root = (
        Path("data/source-archives")
        / book_id
        / version
    )

    output_root = (
        Path("data/multimodal-output")
        / book_id
        / version
    )

    pipeline_root = (
        output_root / "pipeline"
    )

    return {
        "archive": (
            archive_root / "source.zip"
        ),
        "inspection": (
            Path(
                "data/textbook-automation/"
                "archive-inspections"
            )
            / f"{book_id}-{version}.json"
        ),
        "config": (
            CONFIG_ROOT
            / f"{book_id}-{version}.json"
        ),
        "manifest": (
            MANIFEST_ROOT
            / (
                f"{book_id}-{version}"
                "-chapters.json"
            )
        ),
        "generation_report": (
            Path(
                "data/textbook-automation/"
                "config-generation"
            )
            / f"{book_id}-{version}.json"
        ),
        "extracted": (
            archive_root / "extracted"
        ),
        "extraction_report": (
            Path(
                "data/textbook-automation/"
                "extractions"
            )
            / f"{book_id}-{version}.json"
        ),
        "canonical_pdf": (
            output_root
            / "source"
            / "textbook.pdf"
        ),
        "page_map": (
            output_root
            / "source"
            / "chapter-page-map.json"
        ),
        "merge_report": (
            output_root
            / "source"
            / "chapter-merge-report.json"
        ),
        "bda_normalized": (
            pipeline_root
            / "bda"
            / "normalized"
        ),
        "ocr_plan": (
            pipeline_root
            / "ocr"
            / "fallback-plan.json"
        ),
        "ocr_fallback_root": (
            pipeline_root
            / "ocr"
            / "surya"
        ),
        "ocr_fallback_state": (
            pipeline_root
            / "ocr"
            / "surya"
            / "run-state.json"
        ),
        "ocr_fallback_report": (
            pipeline_root
            / "ocr"
            / "surya"
            / "verified"
            / "surya-fallback-report.json"
        ),
        "ocr_fallback_marker": (
            pipeline_root
            / "ocr"
            / "surya"
            / "verified"
            / "SURYA_OCR_FALLBACK_VERIFIED"
        ),
        "ocr_approval": (
            SURYA_APPROVAL_DEFAULT
        ),
        "log": (
            LOG_ROOT
            / f"{book_id}-{version}.log"
        ),
    }


def directory_has_normalized_records(
    path: Path,
) -> bool:
    """Return whether normalized BDA records exist."""

    if not path.is_dir():
        return False

    accepted_suffixes = {
        ".json",
        ".jsonl",
        ".ndjson",
    }

    return any(
        candidate.is_file()
        and candidate.suffix.casefold()
        in accepted_suffixes
        for candidate in path.rglob("*")
    )


def ocr_fallback_plan_classification(
    path: Path,
) -> str | None:
    """Read a valid OCR fallback plan classification."""

    if not path.is_file():
        return None

    try:
        payload = load_json_object(path)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None

    classification = str(
        payload.get(
            "classification",
            "",
        )
    )

    if classification not in {
        "BDA_ACCEPTED",
        "OCR_FALLBACK_REQUIRED",
    }:
        return None

    if not isinstance(
        payload.get("fallback_pages"),
        list,
    ):
        return None

    if not isinstance(
        payload.get("accepted_bda_pages"),
        list,
    ):
        return None

    return classification


def ocr_fallback_runtime_status(
    path: Path,
) -> str | None:
    """Read active Surya fallback state."""

    if not path.is_file():
        return None

    try:
        payload = load_json_object(path)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None

    status = str(
        payload.get("status", "")
    )

    if status in {
        "RENDERING",
        "OCR_FALLBACK_PROCESSING",
    }:
        return status

    return None


def load_ocr_authorization(
    path: Path,
) -> dict[str, Any]:
    """Load OCR integration and full-book authorization flags."""

    result: dict[str, Any] = {
        "record_exists": False,
        "integration_approved": False,
        "full_book_run_authorized": False,
        "ocr_engine": None,
        "model": None,
    }

    if not path.is_file():
        return result

    try:
        payload = load_json_object(path)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
    ):
        return result

    result.update(
        {
            "record_exists": True,
            "integration_approved": bool(
                payload.get(
                    "approved_for_pipeline_integration",
                    False,
                )
            ),
            "full_book_run_authorized": bool(
                payload.get(
                    "full_book_run_authorized",
                    False,
                )
            ),
            "ocr_engine": payload.get(
                "ocr_engine"
            ),
            "model": payload.get(
                "model"
            ),
        }
    )

    return result


# TITAN_EMBEDDING_STAGE_HELPERS


def titan_artifact_paths(
    paths: dict[str, Path],
) -> dict[str, Path]:
    downstream = downstream_artifact_paths(
        paths
    )

    root = (
        downstream["embedding_root"].parent
        / "titan-embeddings"
    )

    return {
        "root": root,
        "manifest": (
            root
            / "embedding-manifest.json"
        ),
        "embeddings": (
            root
            / "embeddings.jsonl"
        ),
    }


def titan_embeddings_are_valid(
    paths: dict[str, Path],
) -> bool:
    artifacts = titan_artifact_paths(
        paths
    )

    manifest_path = artifacts[
        "manifest"
    ]

    if not manifest_path.is_file():
        return False

    try:
        manifest = load_json_object(
            manifest_path
        )
    except (
        OSError,
        TypeError,
        ValueError,
    ):
        return False

    if not isinstance(manifest, dict):
        return False

    if manifest.get("status") != "COMPLETED":
        return False

    try:
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

    except (
        TypeError,
        ValueError,
    ):
        return False

    return (
        input_count > 0
        and completed_count == input_count
        and jsonl_has_records(
            artifacts["embeddings"]
        )
    )


def build_titan_embedding_command(
    paths: dict[str, Path],
) -> list[str]:
    downstream = downstream_artifact_paths(
        paths
    )

    artifacts = titan_artifact_paths(
        paths
    )

    return [
        sys.executable,
        str(
            SCRIPTS_ROOT
            / "embed_records_titan_v2.py"
        ),
        str(
            downstream[
                "embedding_records"
            ]
        ),
        "--output-dir",
        str(artifacts["root"]),
    ]


# OPENSEARCH_STAGE_HELPERS


def opensearch_artifact_paths(
    paths: dict[str, Path],
) -> dict[str, Path]:
    titan = titan_artifact_paths(
        paths
    )

    root = (
        titan["root"].parent
        / "opensearch-serverless"
    )

    bulk_root = (
        root
        / "bulk"
    )

    return {
        "root": root,
        "bulk_root": bulk_root,
        "bulk_payload": (
            bulk_root
            / "bulk-index.ndjson"
        ),
        "bulk_preparation_report": (
            bulk_root
            / "bulk-preparation-report.json"
        ),
        "index_report": (
            root
            / "index-provisioning-report.json"
        ),
        "upload_report": (
            root
            / "bulk-upload-report.json"
        ),
    }


def positive_record_count(
    value: object,
) -> bool:
    try:
        return int(value) > 0
    except (
        TypeError,
        ValueError,
    ):
        return False


def opensearch_bulk_is_valid(
    paths: dict[str, Path],
) -> bool:
    artifacts = opensearch_artifact_paths(
        paths
    )

    report_path = artifacts[
        "bulk_preparation_report"
    ]

    bulk_path = artifacts[
        "bulk_payload"
    ]

    if not (
        report_path.is_file()
        and bulk_path.is_file()
        and bulk_path.stat().st_size > 0
    ):
        return False

    try:
        report = load_json_object(
            report_path
        )
    except (
        OSError,
        TypeError,
        ValueError,
    ):
        return False

    if not isinstance(report, dict):
        return False

    if report.get("status") != "PREPARED":
        return False

    validation = report.get(
        "validation"
    )

    if not isinstance(validation, dict):
        return False

    if validation.get("errors") != []:
        return False

    if not positive_record_count(
        validation.get(
            "document_count"
        )
    ):
        return False

    try:
        return bulk_path.read_bytes().endswith(
            b"\n"
        )
    except OSError:
        return False


def opensearch_index_is_valid(
    paths: dict[str, Path],
) -> bool:
    report_path = (
        opensearch_artifact_paths(
            paths
        )["index_report"]
    )

    if not report_path.is_file():
        return False

    try:
        report = load_json_object(
            report_path
        )
    except (
        OSError,
        TypeError,
        ValueError,
    ):
        return False

    if not isinstance(report, dict):
        return False

    return (
        report.get("status")
        == "PROVISIONED"
        and report.get("action")
        in {
            "matching",
            "created",
        }
    )


def opensearch_upload_is_valid(
    paths: dict[str, Path],
) -> bool:
    report_path = (
        opensearch_artifact_paths(
            paths
        )["upload_report"]
    )

    if not report_path.is_file():
        return False

    try:
        report = load_json_object(
            report_path
        )
    except (
        OSError,
        TypeError,
        ValueError,
    ):
        return False

    if not isinstance(report, dict):
        return False

    return (
        report.get("status")
        == "COMPLETED"
        and positive_record_count(
            report.get(
                "prepared_document_count"
            )
        )
    )


def build_opensearch_bulk_command(
    paths: dict[str, Path],
) -> list[str]:
    titan = titan_artifact_paths(
        paths
    )

    artifacts = opensearch_artifact_paths(
        paths
    )

    return [
        sys.executable,
        str(
            SCRIPTS_ROOT
            / "prepare_opensearch_bulk.py"
        ),
        str(titan["embeddings"]),
        "--output-dir",
        str(artifacts["bulk_root"]),
        "--config",
        str(paths["config"]),
    ]


def build_textbook_index_command(
    paths: dict[str, Path],
) -> list[str]:
    return [
        sys.executable,
        str(
            SCRIPTS_ROOT
            / "create_textbook_index.py"
        ),
        "--create",
        "--config",
        str(paths["config"]),
    ]


def build_opensearch_upload_command(
    paths: dict[str, Path],
) -> list[str]:
    artifacts = opensearch_artifact_paths(
        paths
    )

    return [
        sys.executable,
        str(
            SCRIPTS_ROOT
            / "upload_opensearch_bulk.py"
        ),
        str(artifacts["bulk_payload"]),
        "--preparation-report",
        str(
            artifacts[
                "bulk_preparation_report"
            ]
        ),
        "--output-dir",
        str(artifacts["root"]),
        "--config",
        str(paths["config"]),
    ]


def discover_current_stage(
    book_id: str,
    paths: dict[str, Path],
) -> str:
    if book_id in KNOWN_VERIFIED_BOOKS:
        return "VERIFIED"

    if final_verification_is_valid(
        paths
    ):
        return "VERIFIED"

    # Downstream artifacts must be checked before
    # earlier OCR/BDA stage returns.
    if opensearch_upload_is_valid(
        paths
    ):
        return "INDEXED"

    if titan_embeddings_are_valid(
        paths
    ):
        return "EMBEDDED"

    if embedding_records_are_valid(
        paths
    ):
        return "EMBEDDING_RECORDS_PREPARED"

    if unified_records_are_valid(
        paths
    ):
        return "UNIFIED_RECORDS_PREPARED"

    ocr_fallback_verified = (
        paths["ocr_fallback_marker"].is_file()
        and valid_report(
            paths["ocr_fallback_report"],
            status_field="classification",
            accepted_statuses={"PASS"},
        )
    )

    if ocr_fallback_verified:
        return "OCR_FALLBACK_VERIFIED"

    runtime_status = (
        ocr_fallback_runtime_status(
            paths["ocr_fallback_state"]
        )
    )

    if runtime_status is not None:
        return "OCR_FALLBACK_PROCESSING"

    plan_classification = (
        ocr_fallback_plan_classification(
            paths["ocr_plan"]
        )
    )

    if plan_classification == "BDA_ACCEPTED":
        return "OCR_QUALITY_CHECKED"

    if (
        plan_classification
        == "OCR_FALLBACK_REQUIRED"
    ):
        return "OCR_FALLBACK_REQUIRED"

    if directory_has_normalized_records(
        paths["bda_normalized"]
    ):
        return "BDA_NORMALIZED"

    merge_valid = (
        paths["canonical_pdf"].is_file()
        and paths["page_map"].is_file()
        and valid_report(
            paths["merge_report"],
            status_field="status",
            accepted_statuses={"VALID"},
        )
    )

    if merge_valid:
        return "MERGED"

    extraction_valid = (
        paths["extracted"].is_dir()
        and valid_report(
            paths["extraction_report"],
            status_field="status",
            accepted_statuses={"VALID"},
        )
    )

    if extraction_valid:
        return "EXTRACTED"

    if (
        paths["config"].is_file()
        and paths["manifest"].is_file()
    ):
        return "CONFIG_GENERATED"

    inspection_valid = valid_report(
        paths["inspection"],
        status_field="inspection_status",
        accepted_statuses={"PASSED"},
    )

    if (
        paths["archive"].is_file()
        and inspection_valid
    ):
        return "DOWNLOADED"

    return "DISCOVERED"


def build_environment() -> dict[str, str]:
    environment = dict(os.environ)

    pythonpath_parts = [
        "workers/multimodal-ingestion",
        (
            "workers/multimodal-ingestion/"
            "scripts"
        ),
    ]

    existing = environment.get(
        "PYTHONPATH"
    )

    if existing:
        pythonpath_parts.append(existing)

    environment["PYTHONPATH"] = (
        os.pathsep.join(pythonpath_parts)
    )

    return environment


def append_log(
    path: Path,
    message: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(message)

        if not message.endswith("\n"):
            handle.write("\n")


def run_command(
    command: list[str],
    *,
    log_path: Path,
    maximum_retries: int,
    dry_run: bool,
) -> None:
    command_text = shlex.join(command)

    append_log(
        log_path,
        (
            f"\n[{utc_now()}]\n"
            f"$ {command_text}\n"
        ),
    )

    if dry_run:
        print(
            "DRY RUN:",
            command_text,
        )
        return

    environment = build_environment()
    attempts = maximum_retries + 1

    for attempt in range(
        1,
        attempts + 1,
    ):
        print(
            f"Attempt {attempt}/{attempts}:",
            command_text,
        )

        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        append_log(
            log_path,
            completed.stdout,
        )

        append_log(
            log_path,
            completed.stderr,
        )

        if completed.returncode == 0:
            if completed.stdout.strip():
                print(completed.stdout)

            return

        if completed.stderr.strip():
            print(
                completed.stderr,
                file=sys.stderr,
            )

        if attempt < attempts:
            delay = min(
                5 * attempt,
                15,
            )

            print(
                f"Retrying after {delay} "
                "seconds..."
            )

            time.sleep(delay)

    raise RuntimeError(
        "Command failed after retries: "
        f"{command_text}"
    )


def load_registry_books(
    registry_path: Path,
    *,
    bucket: str,
    prefix: str,
    grades: set[int],
    book_ids: set[str],
) -> list[dict[str, Any]]:
    registry = load_json_object(
        registry_path
    )

    raw_books = registry.get("books")

    if not isinstance(raw_books, list):
        raise ValueError(
            "Registry field 'books' must "
            "be a list."
        )

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    normalized_prefix = prefix.strip("/")

    for raw_book in raw_books:
        if not isinstance(raw_book, dict):
            raise ValueError(
                "Registry book must be "
                "a JSON object."
            )

        book_id = str(
            raw_book.get("book_id", "")
        ).strip()

        if not book_id:
            raise ValueError(
                "Registry book has no book_id."
            )

        if book_id in seen_ids:
            raise ValueError(
                f"Duplicate book ID: {book_id}"
            )

        seen_ids.add(book_id)

        grade = raw_book.get("grade")

        if grade not in grades:
            continue

        if (
            book_ids
            and book_id not in book_ids
        ):
            continue

        source_bucket = str(
            raw_book.get(
                "source_bucket",
                "",
            )
        )

        source_key = str(
            raw_book.get(
                "source_zip_key",
                "",
            )
        )

        if source_bucket != bucket:
            raise ValueError(
                f"{book_id}: registry bucket "
                f"{source_bucket!r} differs from "
                f"requested bucket {bucket!r}."
            )

        if not source_key.startswith(
            normalized_prefix + "/"
        ):
            raise ValueError(
                f"{book_id}: S3 key is outside "
                f"requested prefix: {source_key}"
            )

        selected.append(dict(raw_book))

    selected.sort(
        key=lambda item: (
            int(item["grade"]),
            str(item["subject"]),
            str(item["title"]).casefold(),
        )
    )

    if book_ids:
        found = {
            str(book["book_id"])
            for book in selected
        }

        missing = sorted(
            book_ids - found
        )

        if missing:
            raise ValueError(
                "Requested book IDs not found "
                "in selected grades/prefix: "
                + ", ".join(missing)
            )

    return selected


def load_or_create_state(
    state_path: Path,
    *,
    resume: bool,
    bucket: str,
    prefix: str,
    grades: set[int],
) -> dict[str, Any]:
    if resume and state_path.exists():
        state = load_json_object(
            state_path
        )

        if not isinstance(
            state.get("books"),
            dict,
        ):
            state["books"] = {}

        return state

    return {
        "schema_version": "1.0",
        "pipeline": "run_all_textbooks",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "bucket": bucket,
        "prefix": prefix,
        "grades": sorted(grades),
        "books": {},
        "summary": {},
    }


def save_state(
    state_path: Path,
    state: dict[str, Any],
) -> None:
    books = state.get("books", {})

    statuses = Counter(
        str(record.get("status"))
        for record in books.values()
        if isinstance(record, dict)
    )

    state["updated_at"] = utc_now()
    state["summary"] = {
        "book_count": len(books),
        "status_counts": dict(
            sorted(statuses.items())
        ),
    }

    atomic_write_json(
        state_path,
        state,
    )


def update_book_state(
    state_path: Path,
    state: dict[str, Any],
    *,
    book: dict[str, Any],
    status: str,
    paths: dict[str, Path],
    error: str | None = None,
) -> None:
    book_id = str(book["book_id"])

    existing = state["books"].get(
        book_id,
        {},
    )

    history = existing.get(
        "history",
        [],
    )

    if not isinstance(history, list):
        history = []

    if (
        not history
        or history[-1].get("status")
        != status
    ):
        history.append({
            "status": status,
            "timestamp": utc_now(),
        })

    state["books"][book_id] = {
        "book_id": book_id,
        "grade": book["grade"],
        "title": book["title"],
        "subject": book["subject"],
        "language": book["language"],
        "source_bucket": (
            book["source_bucket"]
        ),
        "source_zip_key": (
            book["source_zip_key"]
        ),
        "version": "v1",
        "status": status,
        "error": error,
        "updated_at": utc_now(),
        "config_path": str(
            paths["config"]
        ),
        "manifest_path": str(
            paths["manifest"]
        ),
        "archive_path": str(
            paths["archive"]
        ),
        "canonical_pdf": str(
            paths["canonical_pdf"]
        ),
        "log_path": str(
            paths["log"]
        ),
        "history": history,
    }

    save_state(
        state_path,
        state,
    )


def normalize_fallback_pages(
    value: Any,
) -> tuple[int, ...]:
    """Validate and normalize planned fallback pages."""

    if not isinstance(value, list):
        raise ValueError(
            "OCR fallback_pages must be a JSON list."
        )

    pages: set[int] = set()

    for page in value:
        if (
            isinstance(page, bool)
            or not isinstance(page, int)
            or page <= 0
        ):
            raise ValueError(
                "OCR fallback page numbers must "
                f"be positive integers: {page!r}"
            )

        pages.add(page)

    return tuple(sorted(pages))


def format_fallback_page_spec(
    pages: tuple[int, ...],
) -> str:
    """Format page numbers for the Surya CLI."""

    if not pages:
        raise ValueError(
            "At least one OCR fallback page is required."
        )

    return ",".join(
        str(page)
        for page in pages
    )


def build_surya_fallback_command(
    *,
    book_id: str,
    version: str,
    canonical_pdf: Path,
    output_root: Path,
    expected_language: str,
    fallback_pages: tuple[int, ...],
    approval_record: Path,
) -> list[str]:
    """Build the resume-safe Surya fallback command."""

    return [
        sys.executable,
        str(
            SCRIPTS_ROOT
            / "run_surya_ocr_fallback.py"
        ),
        "--book-id",
        book_id,
        "--version",
        version,
        "--pdf",
        str(canonical_pdf),
        "--output-root",
        str(output_root),
        "--expected-language",
        expected_language,
        "--pages",
        format_fallback_page_spec(
            fallback_pages
        ),
        "--approval-record",
        str(approval_record),
        "--resume",
    ]



def downstream_artifact_paths(
    paths: dict[str, Path],
) -> dict[str, Path]:
    """Derive immutable unified and embedding-ready paths."""

    version_root = (
        paths["canonical_pdf"].parent.parent
    )

    unified_root = (
        version_root
        / "unified-normalized"
    )

    embedding_root = (
        unified_root
        / "embedding-ready"
    )

    return {
        "page_map": (
            paths["canonical_pdf"].parent
            / "chapter-page-map.json"
        ),
        "unified_root": unified_root,
        "unified_content_units": (
            unified_root
            / "content-units.jsonl"
        ),
        "unified_figures": (
            unified_root
            / "figures.jsonl"
        ),
        "unified_tables": (
            unified_root
            / "tables.jsonl"
        ),
        "unified_report": (
            unified_root
            / "bda-surya-merge-report.json"
        ),
        "unified_marker": (
            unified_root
            / "BDA_SURYA_MERGE_VALID"
        ),
        "embedding_root": embedding_root,
        "embedding_records": (
            embedding_root
            / "embedding-records.jsonl"
        ),
        "embedding_skipped": (
            embedding_root
            / "skipped-records.jsonl"
        ),
        "embedding_report": (
            embedding_root
            / "embedding-preparation-report.json"
        ),
    }


def jsonl_has_records(
    path: Path,
) -> bool:
    """Return True when a JSONL file has a non-empty record."""

    if not path.is_file():
        return False

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            return any(
                line.strip()
                for line in handle
            )
    except OSError:
        return False


def readable_json_object(
    path: Path,
) -> bool:
    """Return True for a readable JSON object."""

    if not path.is_file():
        return False

    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return False

    return isinstance(value, dict)


def discover_normalized_record_roots(
    root: Path,
) -> tuple[Path, ...]:
    """Find normalized directories containing content units."""

    if not root.exists():
        return ()

    discovered: set[Path] = set()

    candidates = []

    if root.is_file():
        candidates = [root]
    else:
        direct = root / "content-units.jsonl"

        if direct.is_file():
            candidates.append(direct)

        candidates.extend(
            root.rglob(
                "content-units.jsonl"
            )
        )

    for candidate in candidates:
        lowered_parts = {
            part.casefold()
            for part in candidate.parts
        }

        if (
            "embedding-ready"
            in lowered_parts
            or "unified-normalized"
            in lowered_parts
        ):
            continue

        discovered.add(
            candidate.parent.resolve()
        )

    return tuple(
        sorted(
            discovered,
            key=str,
        )
    )


def unified_records_are_valid(
    paths: dict[str, Path],
) -> bool:
    artifacts = downstream_artifact_paths(
        paths
    )

    return (
        artifacts[
            "unified_marker"
        ].is_file()
        and jsonl_has_records(
            artifacts[
                "unified_content_units"
            ]
        )
        and valid_report(
            artifacts[
                "unified_report"
            ],
            status_field="status",
            accepted_statuses={
                "VALID",
            },
        )
    )


def embedding_records_are_valid(
    paths: dict[str, Path],
) -> bool:
    artifacts = downstream_artifact_paths(
        paths
    )

    return (
        jsonl_has_records(
            artifacts[
                "embedding_records"
            ]
        )
        and readable_json_object(
            artifacts[
                "embedding_report"
            ]
        )
    )


def later_pipeline_stage(
    current_stage: str,
    candidate_stage: str,
) -> str:
    """Return whichever pipeline stage is later."""

    if (
        STAGES.index(candidate_stage)
        > STAGES.index(current_stage)
    ):
        return candidate_stage

    return current_stage


def build_unified_merge_command(
    *,
    paths: dict[str, Path],
    normalized_roots: tuple[Path, ...],
) -> list[str]:
    """Build the BDA + Surya unified merge command."""

    if not normalized_roots:
        raise ValueError(
            "At least one normalized BDA "
            "record root is required."
        )

    artifacts = downstream_artifact_paths(
        paths
    )

    command = [
        sys.executable,
        str(
            SCRIPTS_ROOT
            / "merge_bda_surya_records.py"
        ),
    ]

    for normalized_root in normalized_roots:
        command.extend(
            [
                "--normalized-root",
                str(normalized_root),
            ]
        )

    command.extend(
        [
            "--ocr-plan",
            str(paths["ocr_plan"]),
            "--surya-report",
            str(
                paths[
                    "ocr_fallback_report"
                ]
            ),
            "--page-map",
            str(artifacts["page_map"]),
            "--output-dir",
            str(
                artifacts[
                    "unified_root"
                ]
            ),
            "--source-pdf",
            str(paths["canonical_pdf"]),
        ]
    )

    return command


def build_embedding_preparation_command(
    *,
    paths: dict[str, Path],
) -> list[str]:
    """Build embedding-record preparation command."""

    artifacts = downstream_artifact_paths(
        paths
    )

    return [
        sys.executable,
        str(
            SCRIPTS_ROOT
            / "prepare_embedding_records.py"
        ),
        str(
            artifacts[
                "unified_root"
            ]
        ),
        "--output-dir",
        str(
            artifacts[
                "embedding_root"
            ]
        ),
    ]


def load_book_processing_metadata(
    book: dict[str, Any],
    config_path: Path,
) -> tuple[str, int]:
    """Resolve quality-gate language and expected page count."""

    config = load_json_object(
        config_path
    )

    config_book = config.get("book")

    if not isinstance(config_book, dict):
        raise ValueError(
            "Book config is missing the 'book' object: "
            f"{config_path}"
        )

    raw_subject = str(
        config_book.get(
            "subject",
            book.get("subject", ""),
        )
    ).strip()

    raw_language = str(
        config_book.get(
            "language",
            book.get("language", ""),
        )
    ).strip()

    if raw_subject.casefold() in {
        "mathematics",
        "math",
        "maths",
    }:
        expected_language = "Mathematics"
    else:
        expected_language = raw_language

    if not expected_language:
        raise ValueError(
            "Expected textbook language is missing: "
            f"{config_path}"
        )

    page_count = config_book.get(
        "page_count"
    )

    if (
        isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or page_count <= 0
    ):
        raise ValueError(
            "Book config has an invalid page_count: "
            f"{page_count!r}"
        )

    return expected_language, page_count


def final_verification_artifact_paths(
    paths: dict[str, Path],
) -> dict[str, Path]:
    unified_root = (
        downstream_artifact_paths(
            paths
        )["unified_root"]
    )

    search_root = (
        unified_root
        / "opensearch-serverless"
    )

    evaluation_root = (
        search_root / "evaluation"
    )

    verification_root = (
        search_root / "verification"
    )

    return {
        "search_root": search_root,
        "bulk_upload_report": (
            opensearch_artifact_paths(
                paths
            )["upload_report"]
        ),
        "vector_report": (
            evaluation_root
            / (
                "vector-retrieval-"
                "evaluation-report.json"
            )
        ),
        "hybrid_report": (
            evaluation_root
            / (
                "hybrid-retrieval-"
                "evaluation-report.json"
            )
        ),
        "rag_report": (
            evaluation_root
            / "rag-evaluation-report.json"
        ),
        "verification_root": (
            verification_root
        ),
        "verification_report": (
            verification_root
            / "final-verification-report.json"
        ),
        "verification_marker": (
            verification_root
            / "PIPELINE_VERIFIED"
        ),
    }


def final_verification_is_valid(
    paths: dict[str, Path],
) -> bool:
    artifacts = (
        final_verification_artifact_paths(
            paths
        )
    )

    if not artifacts[
        "verification_marker"
    ].is_file():
        return False

    report = load_json_object(
        artifacts[
            "verification_report"
        ]
    )

    return (
        report.get("status")
        == "VERIFIED"
        and report.get(
            "all_checks_passed"
        )
        is True
    )


def build_final_verification_command(
    *,
    book_id: str,
    version: str,
    paths: dict[str, Path],
) -> list[str]:
    artifacts = (
        final_verification_artifact_paths(
            paths
        )
    )

    return [
        sys.executable,
        str(
            SCRIPTS_ROOT
            / "verify_indexed_textbook.py"
        ),
        "--book-id",
        book_id,
        "--book-version",
        version,
        "--bulk-upload-report",
        str(
            artifacts[
                "bulk_upload_report"
            ]
        ),
        "--vector-report",
        str(
            artifacts["vector_report"]
        ),
        "--hybrid-report",
        str(
            artifacts["hybrid_report"]
        ),
        "--rag-report",
        str(
            artifacts["rag_report"]
        ),
        "--output-dir",
        str(
            artifacts[
                "verification_root"
            ]
        ),
    ]


def _first_evaluation_case_path(
    candidates: list[Path],
) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return candidates[0]


def evaluation_test_case_paths(
    *,
    book_id: str,
    version: str,
) -> dict[str, Path]:
    config_root = (
        SCRIPTS_ROOT.parent / "config"
    )

    retrieval_root = (
        config_root / "retrieval-tests"
    )

    rag_root = (
        config_root / "rag-tests"
    )

    vector_candidates = [
        retrieval_root
        / f"{book_id}-{version}-vector.json",
        retrieval_root
        / f"{book_id}-vector.json",
    ]

    hybrid_candidates = [
        retrieval_root
        / f"{book_id}-{version}-hybrid.json",
        retrieval_root
        / f"{book_id}-hybrid.json",
    ]

    rag_candidates = [
        rag_root
        / f"{book_id}-{version}-rag.json",
        rag_root
        / f"{book_id}-rag.json",
        retrieval_root
        / f"{book_id}-{version}-rag.json",
        retrieval_root
        / f"{book_id}-rag.json",
    ]

    return {
        "vector": (
            _first_evaluation_case_path(
                vector_candidates
            )
        ),
        "hybrid": (
            _first_evaluation_case_path(
                hybrid_candidates
            )
        ),
        "rag": (
            _first_evaluation_case_path(
                rag_candidates
            )
        ),
    }


def evaluation_report_is_valid(
    report_path: Path,
) -> bool:
    if not report_path.is_file():
        return False

    report = load_json_object(
        report_path
    )

    test_count = report.get(
        "test_count"
    )

    return (
        report.get("status")
        == "PASSED"
        and report.get(
            "all_tests_passed"
        )
        is True
        and isinstance(
            test_count,
            int,
        )
        and not isinstance(
            test_count,
            bool,
        )
        and test_count > 0
        and report.get(
            "failed_test_count"
        )
        == 0
    )


def build_vector_evaluation_command(
    *,
    config_path: Path,
    test_cases_path: Path,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(
            SCRIPTS_ROOT
            / (
                "evaluate_opensearch_"
                "vector_retrieval.py"
            )
        ),
        "--config",
        str(config_path),
        "--test-cases",
        str(test_cases_path),
        "--output",
        str(output_path),
    ]


def build_hybrid_evaluation_command(
    *,
    config_path: Path,
    test_cases_path: Path,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(
            SCRIPTS_ROOT
            / (
                "evaluate_opensearch_"
                "hybrid_retrieval.py"
            )
        ),
        "--config",
        str(config_path),
        "--test-cases",
        str(test_cases_path),
        "--output",
        str(output_path),
    ]


def build_rag_evaluation_command(
    *,
    config_path: Path,
    test_cases_path: Path,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(
            SCRIPTS_ROOT
            / "evaluate_book_rag.py"
        ),
        "--config",
        str(config_path),
        "--test-cases",
        str(test_cases_path),
        "--output",
        str(output_path),
    ]


def process_book(
    book: dict[str, Any],
    *,
    registry_path: Path,
    state_path: Path,
    state: dict[str, Any],
    maximum_retries: int,
    dry_run: bool,
) -> str:
    book_id = str(book["book_id"])
    version = "v1"

    paths = paths_for_book(
        book_id,
        version,
    )

    current_stage = discover_current_stage(
        book_id,
        paths,
    )

    print()
    print("=" * 80)
    print(
        f"{book_id} | "
        f"Grade {book['grade']} | "
        f"{book['language']}"
    )
    print("=" * 80)
    print("Current stage:", current_stage)

    update_book_state(
        state_path,
        state,
        book=book,
        status=current_stage,
        paths=paths,
    )

    if current_stage == "VERIFIED":
        print(
            "Already verified. Skipping."
        )
        return "VERIFIED"

    inspection_ready = (
        paths["archive"].is_file()
        and valid_report(
            paths["inspection"],
            status_field=(
                "inspection_status"
            ),
            accepted_statuses={"PASSED"},
        )
    )

    if not inspection_ready:
        command = [
            sys.executable,
            str(
                SCRIPTS_ROOT
                / (
                    "inspect_textbook_"
                    "source_archive.py"
                )
            ),
            "--registry",
            str(registry_path),
            "--book-id",
            book_id,
        ]

        run_command(
            command,
            log_path=paths["log"],
            maximum_retries=(
                maximum_retries
            ),
            dry_run=dry_run,
        )

        if not dry_run:
            if not (
                paths["archive"].is_file()
                and valid_report(
                    paths["inspection"],
                    status_field=(
                        "inspection_status"
                    ),
                    accepted_statuses={
                        "PASSED"
                    },
                )
            ):
                raise RuntimeError(
                    "Archive inspection did not "
                    "produce valid artifacts."
                )

            update_book_state(
                state_path,
                state,
                book=book,
                status="DOWNLOADED",
                paths=paths,
            )

    config_ready = (
        paths["config"].is_file()
        and paths["manifest"].is_file()
        and valid_report(
            paths["generation_report"],
            status_field="status",
            accepted_statuses={"READY"},
        )
    )

    if not config_ready:
        command = [
            sys.executable,
            str(
                SCRIPTS_ROOT
                / (
                    "generate_book_config_"
                    "from_inspection.py"
                )
            ),
            "--registry",
            str(registry_path),
            "--inspection",
            str(paths["inspection"]),
            "--book-id",
            book_id,
            "--version",
            version,
            "--write",
        ]

        if any(
            path.exists()
            for path in (
                paths["config"],
                paths["manifest"],
                paths[
                    "generation_report"
                ],
            )
        ):
            command.append(
                "--replace"
            )

        run_command(
            command,
            log_path=paths["log"],
            maximum_retries=0,
            dry_run=dry_run,
        )

        if not dry_run:
            if not (
                paths["config"].is_file()
                and paths["manifest"].is_file()
            ):
                raise RuntimeError(
                    "Config generation did not "
                    "produce required files."
                )

            if not valid_report(
                paths["generation_report"],
                status_field="status",
                accepted_statuses={"READY"},
            ):
                raise RuntimeError(
                    "Config generation requires "
                    "chapter-title review. See: "
                    f"{paths['generation_report']}"
                )

            update_book_state(
                state_path,
                state,
                book=book,
                status="CONFIG_GENERATED",
                paths=paths,
            )

    extraction_ready = (
        paths["extracted"].is_dir()
        and valid_report(
            paths["extraction_report"],
            status_field="status",
            accepted_statuses={"VALID"},
        )
    )

    if not extraction_ready:
        if (
            not dry_run
            and paths["extracted"].exists()
        ):
            raise RuntimeError(
                "Extraction directory exists "
                "without a valid report. Review "
                "it before retrying: "
                f"{paths['extracted']}"
            )

        command = [
            sys.executable,
            str(
                SCRIPTS_ROOT
                / "extract_chapter_archive.py"
            ),
            "--config",
            str(paths["config"]),
            "--archive",
            str(paths["archive"]),
            "--report",
            str(
                paths["extraction_report"]
            ),
        ]

        run_command(
            command,
            log_path=paths["log"],
            maximum_retries=0,
            dry_run=dry_run,
        )

        if not dry_run:
            if not (
                paths["extracted"].is_dir()
                and valid_report(
                    paths[
                        "extraction_report"
                    ],
                    status_field="status",
                    accepted_statuses={
                        "VALID"
                    },
                )
            ):
                raise RuntimeError(
                    "Archive extraction did not "
                    "produce valid artifacts."
                )

            update_book_state(
                state_path,
                state,
                book=book,
                status="EXTRACTED",
                paths=paths,
            )

    merge_ready = (
        paths["canonical_pdf"].is_file()
        and paths["page_map"].is_file()
        and valid_report(
            paths["merge_report"],
            status_field="status",
            accepted_statuses={"VALID"},
        )
    )

    if not merge_ready:
        existing_merge_outputs = [
            path
            for path in (
                paths["canonical_pdf"],
                paths["page_map"],
                paths["merge_report"],
            )
            if path.exists()
        ]

        if (
            not dry_run
            and existing_merge_outputs
        ):
            raise RuntimeError(
                "Incomplete merge artifacts "
                "already exist. Review before "
                "retrying: "
                + ", ".join(
                    str(path)
                    for path
                    in existing_merge_outputs
                )
            )

        command = [
            sys.executable,
            str(
                SCRIPTS_ROOT
                / "build_chapter_textbook.py"
            ),
            "--config",
            str(paths["config"]),
        ]

        run_command(
            command,
            log_path=paths["log"],
            maximum_retries=0,
            dry_run=dry_run,
        )

        if not dry_run:
            if not (
                paths[
                    "canonical_pdf"
                ].is_file()
                and paths[
                    "page_map"
                ].is_file()
                and valid_report(
                    paths["merge_report"],
                    status_field="status",
                    accepted_statuses={
                        "VALID"
                    },
                )
            ):
                raise RuntimeError(
                    "Canonical merge did not "
                    "produce valid artifacts."
                )

            update_book_state(
                state_path,
                state,
                book=book,
                status="MERGED",
                paths=paths,
            )

    normalized_ready = (
        directory_has_normalized_records(
            paths["bda_normalized"]
        )
    )

    if normalized_ready:
        (
            expected_language,
            expected_page_count,
        ) = load_book_processing_metadata(
            book,
            paths["config"],
        )

        plan_classification = (
            ocr_fallback_plan_classification(
                paths["ocr_plan"]
            )
        )

        if (
            paths["ocr_plan"].exists()
            and plan_classification is None
            and not dry_run
        ):
            raise RuntimeError(
                "OCR fallback plan exists but is "
                "invalid or incomplete. Review it "
                "before retrying: "
                f"{paths['ocr_plan']}"
            )

        if plan_classification is None:
            command = [
                sys.executable,
                str(
                    SCRIPTS_ROOT
                    / "plan_ocr_fallback_pages.py"
                ),
                "--input",
                str(paths["bda_normalized"]),
                "--expected-language",
                expected_language,
                "--expected-pages",
                f"1-{expected_page_count}",
                "--output",
                str(paths["ocr_plan"]),
            ]

            run_command(
                command,
                log_path=paths["log"],
                maximum_retries=0,
                dry_run=dry_run,
            )

            if not dry_run:
                plan_classification = (
                    ocr_fallback_plan_classification(
                        paths["ocr_plan"]
                    )
                )

                if plan_classification is None:
                    raise RuntimeError(
                        "OCR planning did not produce "
                        "a valid fallback plan: "
                        f"{paths['ocr_plan']}"
                    )

        if not dry_run:
            if (
                plan_classification
                == "BDA_ACCEPTED"
            ):
                update_book_state(
                    state_path,
                    state,
                    book=book,
                    status="OCR_QUALITY_CHECKED",
                    paths=paths,
                )

                print(
                    "BDA text passed the expected-"
                    "language quality gate."
                )

            elif (
                plan_classification
                == "OCR_FALLBACK_REQUIRED"
            ):
                authorization = (
                    load_ocr_authorization(
                        paths["ocr_approval"]
                    )
                )

                if not authorization[
                    "integration_approved"
                ]:
                    raise RuntimeError(
                        "OCR fallback is required, but "
                        "the Surya integration approval "
                        "record is missing or invalid: "
                        f"{paths['ocr_approval']}"
                    )

                update_book_state(
                    state_path,
                    state,
                    book=book,
                    status="OCR_FALLBACK_REQUIRED",
                    paths=paths,
                )

                plan = load_json_object(
                    paths["ocr_plan"]
                )

                fallback_pages = (
                    normalize_fallback_pages(
                        plan.get(
                            "fallback_pages",
                            [],
                        )
                    )
                )

                if not fallback_pages:
                    raise RuntimeError(
                        "OCR plan classification is "
                        "OCR_FALLBACK_REQUIRED but no "
                        "fallback pages were provided: "
                        f"{paths['ocr_plan']}"
                    )

                print(
                    "OCR fallback required for pages:",
                    list(fallback_pages),
                )

                if authorization[
                    "full_book_run_authorized"
                ]:
                    command = (
                        build_surya_fallback_command(
                            book_id=book_id,
                            version=version,
                            canonical_pdf=(
                                paths["canonical_pdf"]
                            ),
                            output_root=(
                                paths[
                                    "ocr_fallback_root"
                                ]
                            ),
                            expected_language=(
                                expected_language
                            ),
                            fallback_pages=(
                                fallback_pages
                            ),
                            approval_record=(
                                paths["ocr_approval"]
                            ),
                        )
                    )

                    update_book_state(
                        state_path,
                        state,
                        book=book,
                        status=(
                            "OCR_FALLBACK_PROCESSING"
                        ),
                        paths=paths,
                    )

                    run_command(
                        command,
                        log_path=paths["log"],
                        maximum_retries=0,
                        dry_run=False,
                    )

                    fallback_verified = (
                        paths[
                            "ocr_fallback_marker"
                        ].is_file()
                        and valid_report(
                            paths[
                                "ocr_fallback_report"
                            ],
                            status_field=(
                                "classification"
                            ),
                            accepted_statuses={
                                "PASS"
                            },
                        )
                    )

                    if not fallback_verified:
                        raise RuntimeError(
                            "Surya OCR completed without "
                            "verified fallback artifacts. "
                            "Review the OCR report and log: "
                            f"{paths['ocr_fallback_report']}"
                        )

                    update_book_state(
                        state_path,
                        state,
                        book=book,
                        status=(
                            "OCR_FALLBACK_VERIFIED"
                        ),
                        paths=paths,
                    )

                    print(
                        "Surya OCR fallback verified "
                        "for pages:",
                        list(fallback_pages),
                    )

                else:
                    print(
                        "Surya execution blocked by "
                        "authorization guard: "
                        "full_book_run_authorized=false"
                    )

    if dry_run:
        print(
            "Dry-run plan completed for book."
        )

        return current_stage


    downstream_stage = discover_current_stage(
        book_id,
        paths,
    )

    normalized_record_roots = (
        discover_normalized_record_roots(
            paths["bda_normalized"]
        )
    )

    downstream_artifacts = (
        downstream_artifact_paths(
            paths
        )
    )

    downstream_eligible = (
        downstream_stage
        in {
            "OCR_QUALITY_CHECKED",
            "OCR_FALLBACK_VERIFIED",
            "UNIFIED_RECORDS_PREPARED",
            "EMBEDDING_RECORDS_PREPARED",
        }
    )

    if (
        downstream_eligible
        and normalized_record_roots
        and downstream_artifacts[
            "page_map"
        ].is_file()
    ):
        if not unified_records_are_valid(
            paths
        ):
            merge_command = (
                build_unified_merge_command(
                    paths=paths,
                    normalized_roots=(
                        normalized_record_roots
                    ),
                )
            )

            run_command(
                merge_command,
                log_path=paths["log"],
                maximum_retries=(
                    maximum_retries
                ),
                dry_run=dry_run,
            )

            if not dry_run:
                if not unified_records_are_valid(
                    paths
                ):
                    raise RuntimeError(
                        "Unified BDA + Surya merge "
                        "did not create valid "
                        "artifacts: "
                        f"{downstream_artifacts['unified_root']}"
                    )

                update_book_state(
                    state_path,
                    state,
                    book=book,
                    status=(
                        "UNIFIED_RECORDS_PREPARED"
                    ),
                    paths=paths,
                )

                print(
                    "Unified normalized records "
                    "prepared:",
                    downstream_artifacts[
                        "unified_root"
                    ],
                )

        downstream_stage = (
            discover_current_stage(
                book_id,
                paths,
            )
        )

        if (
            downstream_stage
            == "UNIFIED_RECORDS_PREPARED"
            and not embedding_records_are_valid(
                paths
            )
        ):
            embedding_command = (
                build_embedding_preparation_command(
                    paths=paths
                )
            )

            run_command(
                embedding_command,
                log_path=paths["log"],
                maximum_retries=(
                    maximum_retries
                ),
                dry_run=dry_run,
            )

            if not dry_run:
                if not embedding_records_are_valid(
                    paths
                ):
                    raise RuntimeError(
                        "Embedding preparation did "
                        "not create valid records: "
                        f"{downstream_artifacts['embedding_root']}"
                    )

                update_book_state(
                    state_path,
                    state,
                    book=book,
                    status=(
                        "EMBEDDING_RECORDS_PREPARED"
                    ),
                    paths=paths,
                )

                print(
                    "Embedding-ready records "
                    "prepared:",
                    downstream_artifacts[
                        "embedding_records"
                    ],
                )

    elif (
        downstream_eligible
        and normalized_record_roots
        and not downstream_artifacts[
            "page_map"
        ].is_file()
    ):
        print(
            "Downstream preparation blocked: "
            "chapter-page-map.json missing:",
            downstream_artifacts[
                "page_map"
            ],
        )

    elif (
        downstream_eligible
        and not normalized_record_roots
    ):
        print(
            "No normalized content-units.jsonl "
            "records found; downstream "
            "preparation skipped."
        )

    # TITAN_EMBEDDING_EXECUTION_BRIDGE
    titan_stage = discover_current_stage(
        book_id,
        paths,
    )

    if (
        titan_stage
        == "EMBEDDING_RECORDS_PREPARED"
    ):
        titan_artifacts = titan_artifact_paths(
            paths
        )

        run_command(
            build_titan_embedding_command(
                paths
            ),
            log_path=paths["log"],
            maximum_retries=maximum_retries,
            dry_run=False,
        )

        if not titan_embeddings_are_valid(
            paths
        ):
            raise RuntimeError(
                "Titan embedding execution did "
                "not produce valid COMPLETED "
                "artifacts: "
                f"{titan_artifacts['root']}"
            )

        update_book_state(
            state_path,
            state,
            book=book,
            status="EMBEDDED",
            paths=paths,
        )

        print(
            "Titan embedding records prepared:",
            titan_artifacts["embeddings"],
        )

    # OPENSEARCH_EXECUTION_BRIDGE
    search_stage = discover_current_stage(
        book_id,
        paths,
    )

    if search_stage == "EMBEDDED":
        search_artifacts = (
            opensearch_artifact_paths(
                paths
            )
        )

        if not opensearch_bulk_is_valid(
            paths
        ):
            run_command(
                build_opensearch_bulk_command(
                    paths
                ),
                log_path=paths["log"],
                maximum_retries=0,
                dry_run=False,
            )

            if not opensearch_bulk_is_valid(
                paths
            ):
                raise RuntimeError(
                    "OpenSearch bulk preparation "
                    "did not produce valid PREPARED "
                    "artifacts: "
                    f"{search_artifacts['bulk_root']}"
                )

            print(
                "OpenSearch bulk payload prepared:",
                search_artifacts[
                    "bulk_payload"
                ],
            )

        if not opensearch_index_is_valid(
            paths
        ):
            run_command(
                build_textbook_index_command(
                    paths
                ),
                log_path=paths["log"],
                maximum_retries=(
                    maximum_retries
                ),
                dry_run=False,
            )

            if not opensearch_index_is_valid(
                paths
            ):
                raise RuntimeError(
                    "Textbook index provisioning "
                    "did not produce a valid "
                    "PROVISIONED report: "
                    f"{search_artifacts['index_report']}"
                )

            print(
                "OpenSearch textbook index "
                "provisioned:",
                search_artifacts[
                    "index_report"
                ],
            )

        if not opensearch_upload_is_valid(
            paths
        ):
            run_command(
                build_opensearch_upload_command(
                    paths
                ),
                log_path=paths["log"],
                maximum_retries=(
                    maximum_retries
                ),
                dry_run=False,
            )

            if not opensearch_upload_is_valid(
                paths
            ):
                raise RuntimeError(
                    "OpenSearch upload did not "
                    "produce a valid COMPLETED "
                    "report: "
                    f"{search_artifacts['upload_report']}"
                )

            update_book_state(
                state_path,
                state,
                book=book,
                status="INDEXED",
                paths=paths,
            )

            print(
                "OpenSearch indexing completed:",
                search_artifacts[
                    "upload_report"
                ],
            )

    evaluation_stage = (
        discover_current_stage(
            book_id,
            paths,
        )
    )

    if evaluation_stage == "INDEXED":
        evaluation_artifacts = (
            final_verification_artifact_paths(
                paths
            )
        )

        evaluation_cases = (
            evaluation_test_case_paths(
                book_id=book_id,
                version=version,
            )
        )

        evaluation_jobs = [
            (
                "vector",
                evaluation_cases["vector"],
                evaluation_artifacts[
                    "vector_report"
                ],
                build_vector_evaluation_command(
                    config_path=paths["config"],
                    test_cases_path=(
                        evaluation_cases[
                            "vector"
                        ]
                    ),
                    output_path=(
                        evaluation_artifacts[
                            "vector_report"
                        ]
                    ),
                ),
            ),
            (
                "hybrid",
                evaluation_cases["hybrid"],
                evaluation_artifacts[
                    "hybrid_report"
                ],
                build_hybrid_evaluation_command(
                    config_path=paths["config"],
                    test_cases_path=(
                        evaluation_cases[
                            "hybrid"
                        ]
                    ),
                    output_path=(
                        evaluation_artifacts[
                            "hybrid_report"
                        ]
                    ),
                ),
            ),
            (
                "rag",
                evaluation_cases["rag"],
                evaluation_artifacts[
                    "rag_report"
                ],
                build_rag_evaluation_command(
                    config_path=paths["config"],
                    test_cases_path=(
                        evaluation_cases[
                            "rag"
                        ]
                    ),
                    output_path=(
                        evaluation_artifacts[
                            "rag_report"
                        ]
                    ),
                ),
            ),
        ]

        for (
            evaluation_name,
            test_cases_path,
            report_path,
            evaluation_command,
        ) in evaluation_jobs:
            if evaluation_report_is_valid(
                report_path
            ):
                continue

            if not test_cases_path.is_file():
                print(
                    "Evaluation blocked; "
                    f"{evaluation_name} test "
                    "cases are missing:",
                    test_cases_path,
                )
                continue

            run_command(
                evaluation_command,
                log_path=paths["log"],
                maximum_retries=0,
                dry_run=False,
            )

            if not evaluation_report_is_valid(
                report_path
            ):
                raise RuntimeError(
                    f"{evaluation_name} "
                    "evaluation did not produce "
                    "a passing report: "
                    f"{report_path}"
                )

            print(
                f"{evaluation_name.title()} "
                "evaluation passed:",
                report_path,
            )

    verification_stage = (
        discover_current_stage(
            book_id,
            paths,
        )
    )

    if verification_stage == "INDEXED":
        verification_artifacts = (
            final_verification_artifact_paths(
                paths
            )
        )

        required_verification_reports = [
            verification_artifacts[
                "bulk_upload_report"
            ],
            verification_artifacts[
                "vector_report"
            ],
            verification_artifacts[
                "hybrid_report"
            ],
            verification_artifacts[
                "rag_report"
            ],
        ]

        missing_verification_reports = [
            report_path
            for report_path
            in required_verification_reports
            if not report_path.is_file()
        ]

        if missing_verification_reports:
            print(
                "Final verification blocked; "
                "required evaluation reports "
                "are missing:",
                ", ".join(
                    str(report_path)
                    for report_path
                    in missing_verification_reports
                ),
            )
        else:
            verification_command = (
                build_final_verification_command(
                    book_id=book_id,
                    version=version,
                    paths=paths,
                )
            )

            run_command(
                verification_command,
                log_path=paths["log"],
                maximum_retries=0,
                dry_run=False,
            )

            if not final_verification_is_valid(
                paths
            ):
                raise RuntimeError(
                    "Generic final verification "
                    "did not produce valid "
                    "VERIFIED artifacts: "
                    f"{verification_artifacts['verification_report']}"
                )

            update_book_state(
                state_path,
                state,
                book=book,
                status="VERIFIED",
                paths=paths,
            )

            print(
                "Generic final verification "
                "completed:",
                verification_artifacts[
                    "verification_report"
                ],
            )

    final_stage = discover_current_stage(
        book_id,
        paths,
    )

    print(
        "Completed through stage:",
        final_stage,
    )

    return final_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically process all textbook "
            "ZIP archives from S3 with resume, "
            "retries and failure isolation."
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
        help=(
            "Grade range such as 1-10 or "
            "comma-separated values."
        ),
    )

    parser.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY_DEFAULT,
    )

    parser.add_argument(
        "--state",
        type=Path,
        default=STATE_DEFAULT,
    )

    parser.add_argument(
        "--book-id",
        action="append",
        default=[],
        help=(
            "Process only this book ID. "
            "May be provided multiple times."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print planned operations without "
            "executing commands."
        ),
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_retries < 0:
        raise ValueError(
            "--max-retries cannot be negative."
        )

    grades = parse_grades(
        args.grades
    )

    requested_book_ids = {
        value.strip()
        for value in args.book_id
        if value.strip()
    }

    books = load_registry_books(
        args.registry,
        bucket=args.bucket,
        prefix=args.prefix,
        grades=grades,
        book_ids=requested_book_ids,
    )

    state = load_or_create_state(
        args.state,
        resume=args.resume,
        bucket=args.bucket,
        prefix=args.prefix,
        grades=grades,
    )

    print("=" * 80)
    print("AUTOMATIC TEXTBOOK PIPELINE")
    print("=" * 80)
    print("Bucket:       ", args.bucket)
    print("Prefix:       ", args.prefix)
    print("Grades:       ", sorted(grades))
    print("Books:        ", len(books))
    print("Resume:       ", args.resume)
    print("Dry run:      ", args.dry_run)
    print(
        "Through stage: OCR planning "
        "(when normalized BDA output exists)"
    )
    print("State:        ", args.state)

    completed = 0
    failed = 0

    for book in books:
        book_id = str(book["book_id"])
        paths = paths_for_book(
            book_id,
            "v1",
        )

        try:
            result = process_book(
                book,
                registry_path=args.registry,
                state_path=args.state,
                state=state,
                maximum_retries=(
                    args.max_retries
                ),
                dry_run=args.dry_run,
            )

            if (
                result in STAGES
                and result != "FAILED"
                and STAGES.index(result)
                >= STAGES.index("MERGED")
            ):
                completed += 1

        except Exception as error:
            failed += 1

            update_book_state(
                args.state,
                state,
                book=book,
                status="FAILED",
                paths=paths,
                error=str(error),
            )

            append_log(
                paths["log"],
                (
                    f"\n[{utc_now()}] "
                    f"FAILED: {error}\n"
                ),
            )

            print(
                f"FAILED: {book_id}: {error}",
                file=sys.stderr,
            )

            # Failure isolation:
            # continue with the next textbook.
            continue

    save_state(
        args.state,
        state,
    )

    print()
    print("=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print("Selected books:", len(books))
    print("Completed:     ", completed)
    print("Failed:        ", failed)
    print("State report:  ", args.state)
    print("Failure isolation: enabled")
    print("Automatic retries: enabled")
    print("Resume support: enabled")

    if args.dry_run:
        print("Execution mode: dry-run")
        return 0

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
