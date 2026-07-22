#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONFIGS_DIR = Path(
    "workers/multimodal-ingestion/config/books"
)

DEFAULT_STATE_PATH = Path(
    "data/textbook-automation/"
    "ncert-i-x-collection-state.json"
)

KNOWN_VERIFIED_BOOK_IDS = {
    "grade-9-english-kaveri",
    "grade-9-mathematics-ganita-manjari",
}

FINAL_STATUSES = {
    "VERIFIED",
    "SKIPPED_VERIFIED",
}


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
            f"Expected a JSON object: {path}"
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_registry(
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    books = registry.get("books")

    if not isinstance(books, list):
        raise ValueError(
            "Registry field 'books' must be a list."
        )

    if not books:
        raise ValueError(
            "Registry contains no books."
        )

    required_fields = {
        "book_id",
        "grade",
        "title",
        "subject",
        "language",
        "source_bucket",
        "source_zip_key",
    }

    seen_book_ids: set[str] = set()
    validated: list[dict[str, Any]] = []

    for number, raw_book in enumerate(
        books,
        start=1,
    ):
        if not isinstance(raw_book, dict):
            raise ValueError(
                f"Book #{number} is not an object."
            )

        missing = sorted(
            field
            for field in required_fields
            if raw_book.get(field) in (
                None,
                "",
            )
        )

        if missing:
            raise ValueError(
                f"Book #{number} is missing: "
                f"{', '.join(missing)}"
            )

        book_id = str(
            raw_book["book_id"]
        ).strip()

        if book_id in seen_book_ids:
            raise ValueError(
                f"Duplicate book_id: {book_id}"
            )

        seen_book_ids.add(book_id)

        grade = raw_book["grade"]

        if (
            not isinstance(grade, int)
            or not 1 <= grade <= 10
        ):
            raise ValueError(
                f"{book_id}: invalid grade {grade!r}"
            )

        validated.append(dict(raw_book))

    return validated


def discover_book_configs(
    configs_dir: Path,
    book_id: str,
) -> list[Path]:
    if not configs_dir.exists():
        return []

    candidates: list[Path] = []

    for path in configs_dir.glob(
        f"{book_id}-*.json"
    ):
        if not path.is_file():
            continue

        if "manifest" in path.parts:
            continue

        candidates.append(path)

    exact_path = configs_dir / (
        f"{book_id}.json"
    )

    if (
        exact_path.exists()
        and exact_path.is_file()
        and exact_path not in candidates
    ):
        candidates.append(exact_path)

    return sorted(
        candidates,
        key=lambda item: item.name,
    )


def select_preferred_config(
    candidates: list[Path],
) -> Path | None:
    if not candidates:
        return None

    chapter_test = [
        path
        for path in candidates
        if "chapter-test" in path.stem
    ]

    if chapter_test:
        return chapter_test[-1]

    return candidates[-1]


def load_previous_books(
    state_path: Path,
    resume: bool,
) -> dict[str, dict[str, Any]]:
    if not resume or not state_path.exists():
        return {}

    previous = load_json_object(state_path)
    records = previous.get("books", [])

    if not isinstance(records, list):
        return {}

    result: dict[str, dict[str, Any]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        book_id = str(
            record.get("book_id", "")
        ).strip()

        if book_id:
            result[book_id] = record

    return result


def build_book_plan(
    book: dict[str, Any],
    configs_dir: Path,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    book_id = str(book["book_id"])

    candidates = discover_book_configs(
        configs_dir,
        book_id,
    )

    selected_config = select_preferred_config(
        candidates
    )

    if (
        previous
        and previous.get("status")
        in FINAL_STATUSES
    ):
        status = str(previous["status"])
        reason = "preserved_from_resume_state"

    elif book_id in KNOWN_VERIFIED_BOOK_IDS:
        status = "SKIPPED_VERIFIED"
        reason = (
            "book_already_processed_and_verified"
        )

    elif selected_config is not None:
        status = "CONFIG_READY"
        reason = (
            "existing_book_config_detected"
        )

    else:
        status = "CONFIG_REQUIRED"
        reason = (
            "book_config_and_chapter_manifest_"
            "must_be_generated"
        )

    return {
        "book_id": book_id,
        "grade": book["grade"],
        "title": book["title"],
        "subject": book["subject"],
        "language": book["language"],
        "processing_profile": book.get(
            "processing_profile"
        ),
        "source_bucket": book["source_bucket"],
        "source_zip_key": book[
            "source_zip_key"
        ],
        "status": status,
        "status_reason": reason,
        "config_path": (
            str(selected_config)
            if selected_config is not None
            else None
        ),
        "config_candidates": [
            str(path)
            for path in candidates
        ],
        "updated_at": utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan and track processing of a "
            "textbook collection registry."
        )
    )

    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help=(
            "Path to the finalized textbook "
            "collection registry JSON."
        ),
    )

    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=DEFAULT_CONFIGS_DIR,
        help=(
            "Directory containing per-book "
            "configuration JSON files."
        ),
    )

    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help=(
            "Collection progress/state report."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Preserve completed statuses from "
            "an existing collection state file."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    registry = load_json_object(
        args.registry
    )

    books = validate_registry(registry)

    previous_books = load_previous_books(
        args.state,
        args.resume,
    )

    plans: list[dict[str, Any]] = []

    for book in books:
        plans.append(
            build_book_plan(
                book=book,
                configs_dir=args.configs_dir,
                previous=previous_books.get(
                    str(book["book_id"])
                ),
            )
        )

    plans.sort(
        key=lambda item: (
            item["grade"],
            item["subject"],
            item["title"].casefold(),
        )
    )

    status_counts = Counter(
        plan["status"]
        for plan in plans
    )

    grade_status_counts: dict[
        int,
        Counter[str],
    ] = defaultdict(Counter)

    for plan in plans:
        grade_status_counts[
            int(plan["grade"])
        ][str(plan["status"])] += 1

    state = {
        "schema_version": "1.0",
        "mode": "dry-run-plan",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "registry_path": str(args.registry),
        "registry_sha256": sha256_file(
            args.registry
        ),
        "configs_dir": str(
            args.configs_dir
        ),
        "resume_enabled": bool(
            args.resume
        ),
        "summary": {
            "total_books": len(plans),
            "status_counts": dict(
                sorted(status_counts.items())
            ),
            "aws_calls": 0,
            "s3_writes": 0,
            "bedrock_calls": 0,
            "opensearch_calls": 0,
        },
        "books": plans,
    }

    atomic_write_json(
        args.state,
        state,
    )

    print("=" * 76)
    print("TEXTBOOK COLLECTION DRY-RUN PLAN")
    print("=" * 76)
    print(
        "Registry:     ",
        args.registry,
    )
    print(
        "Configs dir:  ",
        args.configs_dir,
    )
    print(
        "State report: ",
        args.state,
    )
    print(
        "Total books:  ",
        len(plans),
    )

    print()
    print("STATUS SUMMARY")
    print("-" * 76)

    for status, count in sorted(
        status_counts.items()
    ):
        print(f"{status:28} {count}")

    print()
    print("GRADE SUMMARY")
    print("-" * 76)

    for grade in range(1, 11):
        counts = grade_status_counts.get(
            grade,
            Counter(),
        )

        total = sum(counts.values())

        if not total:
            print(
                f"Grade {grade:2}: "
                "0 books"
            )
            continue

        detail = ", ".join(
            f"{status}={count}"
            for status, count in sorted(
                counts.items()
            )
        )

        print(
            f"Grade {grade:2}: "
            f"{total} books | {detail}"
        )

    print()
    print("VERIFIED / SKIPPED BOOKS")
    print("-" * 76)

    skipped = [
        plan
        for plan in plans
        if plan["status"]
        == "SKIPPED_VERIFIED"
    ]

    if not skipped:
        print("None")
    else:
        for plan in skipped:
            print(
                f"- {plan['book_id']}"
            )
            print(
                f"  Config: "
                f"{plan['config_path']}"
            )

    print()
    print("FIRST PENDING BOOKS")
    print("-" * 76)

    pending = [
        plan
        for plan in plans
        if plan["status"]
        in {
            "CONFIG_REQUIRED",
            "CONFIG_READY",
        }
    ]

    for plan in pending[:15]:
        print(
            f"- Grade {plan['grade']} | "
            f"{plan['book_id']} | "
            f"{plan['status']}"
        )

    if len(pending) > 15:
        print(
            f"... and "
            f"{len(pending) - 15} more"
        )

    print()
    print("SAFETY")
    print("-" * 76)
    print("AWS calls:        0")
    print("S3 writes:        0")
    print("Bedrock calls:    0")
    print("OpenSearch calls: 0")
    print()
    print(
        "Dry-run collection plan completed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
