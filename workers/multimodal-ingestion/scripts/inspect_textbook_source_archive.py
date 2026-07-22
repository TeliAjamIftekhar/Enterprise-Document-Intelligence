#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

import boto3
import fitz


DEFAULT_DOWNLOAD_ROOT = Path(
    "data/source-archives"
)

DEFAULT_REPORT_ROOT = Path(
    "data/textbook-automation/"
    "archive-inspections"
)

SUPPORTED_ASSET_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".svg",
}

EXPECTED_SCRIPT_BY_LANGUAGE = {
    "english": "latin",
    "hindi": "devanagari",
    "marathi": "devanagari",
    "sanskrit": "devanagari",
    "urdu": "arabic",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def find_registry_book(
    registry: dict[str, Any],
    book_id: str,
) -> dict[str, Any]:
    books = registry.get("books")

    if not isinstance(books, list):
        raise ValueError(
            "Registry field 'books' must be a list."
        )

    matches = [
        book
        for book in books
        if isinstance(book, dict)
        and book.get("book_id") == book_id
    ]

    if not matches:
        raise ValueError(
            f"Book not found in registry: {book_id}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Duplicate registry book: {book_id}"
        )

    return dict(matches[0])


def validate_archive_path(
    raw_name: str,
) -> str:
    normalized = raw_name.replace(
        "\\",
        "/",
    )

    if normalized.startswith("/"):
        raise ValueError(
            f"Absolute ZIP entry rejected: {raw_name}"
        )

    if re.match(
        r"^[A-Za-z]:",
        normalized,
    ):
        raise ValueError(
            f"Drive path rejected: {raw_name}"
        )

    path = PurePosixPath(normalized)

    if ".." in path.parts:
        raise ValueError(
            f"Traversal path rejected: {raw_name}"
        )

    if not path.parts:
        raise ValueError(
            f"Empty ZIP entry rejected: {raw_name}"
        )

    return path.as_posix()


def natural_key(value: str) -> tuple[Any, ...]:
    normalized = unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()

    parts = re.split(
        r"(\d+)",
        normalized,
    )

    result: list[Any] = []

    for part in parts:
        if part.isdigit():
            result.append(
                (0, int(part))
            )
        else:
            result.append(
                (1, part)
            )

    return tuple(result)


def detect_archive_root(
    names: list[str],
) -> str | None:
    if not names:
        return None

    paths = [
        PurePosixPath(name)
        for name in names
    ]

    if any(
        len(path.parts) < 2
        for path in paths
    ):
        return None

    first_parts = {
        path.parts[0]
        for path in paths
    }

    if len(first_parts) == 1:
        return next(iter(first_parts))

    return None


def relative_to_archive_root(
    name: str,
    archive_root: str | None,
) -> str:
    path = PurePosixPath(name)

    if (
        archive_root
        and path.parts
        and path.parts[0] == archive_root
    ):
        return PurePosixPath(
            *path.parts[1:]
        ).as_posix()

    return path.as_posix()


def count_scripts(
    text: str,
) -> Counter[str]:
    counts: Counter[str] = Counter()

    for character in text:
        codepoint = ord(character)

        if (
            "A" <= character <= "Z"
            or "a" <= character <= "z"
            or 0x00C0 <= codepoint <= 0x024F
        ):
            counts["latin"] += 1

        elif 0x0900 <= codepoint <= 0x097F:
            counts["devanagari"] += 1

        elif (
            0x0600 <= codepoint <= 0x06FF
            or 0x0750 <= codepoint <= 0x077F
            or 0x08A0 <= codepoint <= 0x08FF
        ):
            counts["arabic"] += 1

        elif character.isalpha():
            counts["other"] += 1

    return counts


def dominant_script(
    counts: Counter[str],
) -> str:
    meaningful = {
        key: value
        for key, value in counts.items()
        if value > 0
    }

    if not meaningful:
        return "unknown"

    return max(
        meaningful,
        key=meaningful.get,
    )


def inspect_pdf_entry(
    archive: ZipFile,
    zip_name: str,
    relative_name: str,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
    ) as temporary:
        with archive.open(
            zip_name,
            "r",
        ) as source:
            shutil.copyfileobj(
                source,
                temporary,
                length=1024 * 1024,
            )

        temporary.flush()

        document = fitz.open(
            temporary.name
        )

        try:
            page_count = document.page_count
            extracted_parts: list[str] = []

            sample_pages = sorted({
                0,
                min(1, page_count - 1),
                min(2, page_count - 1),
                max(0, page_count - 1),
            })

            for page_number in sample_pages:
                if not (
                    0 <= page_number < page_count
                ):
                    continue

                page = document.load_page(
                    page_number
                )

                extracted_parts.append(
                    page.get_text("text")
                )

            sample_text = "\n".join(
                extracted_parts
            )[:50000]

            script_counts = count_scripts(
                sample_text
            )

            return {
                "source_filename": (
                    PurePosixPath(
                        relative_name
                    ).name
                ),
                "relative_archive_path": (
                    relative_name
                ),
                "archive_path": zip_name,
                "source_page_count": (
                    page_count
                ),
                "sample_text_characters": (
                    len(sample_text)
                ),
                "script_counts": dict(
                    sorted(
                        script_counts.items()
                    )
                ),
                "dominant_script": (
                    dominant_script(
                        script_counts
                    )
                ),
                "pdf_status": "valid",
            }

        finally:
            document.close()


def assess_script_evidence(
    *,
    expected_script: str | None,
    script_counts: Counter[str],
    total_pages: int,
) -> dict[str, Any]:
    """
    Decide whether extracted PDF text provides
    enough evidence to assign a document script.

    Tiny metadata, URLs, page labels, or Latin
    copyright text must not override the registry
    language of an otherwise scanned textbook.
    """

    normalized_counts = Counter({
        str(script): max(
            0,
            int(count),
        )
        for script, count
        in script_counts.items()
    })

    raw_detected_script = dominant_script(
        normalized_counts
    )

    meaningful_characters = sum(
        normalized_counts.values()
    )

    page_count = max(
        1,
        int(total_pages),
    )

    characters_per_page = (
        meaningful_characters / page_count
    )

    # Require at least a small absolute amount of
    # text and approximately two script characters
    # per source page before trusting a conflicting
    # script result.
    minimum_required_characters = max(
        40,
        page_count * 2,
    )

    text_evidence_status = (
        "sufficient"
        if meaningful_characters
        >= minimum_required_characters
        else "insufficient"
    )

    expected_character_count = (
        int(
            normalized_counts.get(
                expected_script,
                0,
            )
        )
        if expected_script
        else 0
    )

    mismatch = (
        expected_script is not None
        and raw_detected_script
        not in {
            "unknown",
            expected_script,
        }
    )

    weak_mismatch = (
        mismatch
        and text_evidence_status
        == "insufficient"
        and expected_character_count == 0
    )

    if (
        raw_detected_script == "unknown"
        or weak_mismatch
    ):
        detected_script = "unknown"
        script_verification = "unverified"
        extraction_route = "visual-bda"

        reason = (
            "Extracted text contains insufficient "
            "meaningful script evidence. Visual "
            "document extraction is required."
        )

    elif mismatch:
        detected_script = raw_detected_script
        script_verification = "mismatch"
        extraction_route = "manual-review"

        reason = (
            "Extracted text provides substantial "
            "evidence for a script that differs "
            "from the registry language."
        )

    else:
        detected_script = raw_detected_script

        script_verification = (
            "verified"
            if expected_script
            and detected_script
            == expected_script
            else "detected"
        )

        extraction_route = "text-layout"
        reason = None

    return {
        "raw_detected_script": (
            raw_detected_script
        ),
        "detected_script": detected_script,
        "expected_script": expected_script,
        "script_verification": (
            script_verification
        ),
        "text_evidence_status": (
            text_evidence_status
        ),
        "extraction_route": (
            extraction_route
        ),
        "meaningful_script_characters": (
            meaningful_characters
        ),
        "script_characters_per_page": round(
            characters_per_page,
            4,
        ),
        "minimum_required_characters": (
            minimum_required_characters
        ),
        "expected_script_characters": (
            expected_character_count
        ),
        "reason": reason,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and safely inspect one "
            "chapter-wise textbook ZIP from S3."
        )
    )

    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--book-id",
        required=True,
    )

    parser.add_argument(
        "--region",
        default="us-east-1",
    )

    parser.add_argument(
        "--download-root",
        type=Path,
        default=DEFAULT_DOWNLOAD_ROOT,
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    registry = load_json_object(
        args.registry
    )

    book = find_registry_book(
        registry,
        args.book_id,
    )

    bucket = str(
        book["source_bucket"]
    )

    source_key = str(
        book["source_zip_key"]
    )

    version = str(
        book.get(
            "proposed_version",
            "v1",
        )
    )

    local_directory = (
        args.download_root
        / args.book_id
        / version
    )

    local_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_path = (
        local_directory
        / "source.zip"
    )

    if args.output:
        report_path = args.output
    else:
        report_path = (
            DEFAULT_REPORT_ROOT
            / f"{args.book_id}-{version}.json"
        )

    s3 = boto3.client(
        "s3",
        region_name=args.region,
    )

    head = s3.head_object(
        Bucket=bucket,
        Key=source_key,
    )

    expected_size = int(
        head["ContentLength"]
    )

    download_status = "downloaded"

    if (
        archive_path.exists()
        and archive_path.stat().st_size
        == expected_size
        and not args.force_download
    ):
        download_status = (
            "reused-existing-download"
        )

    else:
        temporary_download = (
            archive_path.with_suffix(
                ".zip.part"
            )
        )

        if temporary_download.exists():
            temporary_download.unlink()

        s3.download_file(
            bucket,
            source_key,
            str(temporary_download),
        )

        actual_size = (
            temporary_download.stat().st_size
        )

        if actual_size != expected_size:
            temporary_download.unlink(
                missing_ok=True
            )

            raise RuntimeError(
                "Downloaded ZIP size mismatch: "
                f"expected={expected_size}, "
                f"actual={actual_size}"
            )

        temporary_download.replace(
            archive_path
        )

    warnings: list[str] = []
    pdf_documents: list[
        dict[str, Any]
    ] = []
    supplementary_assets: list[str] = []
    unsupported_files: list[str] = []

    try:
        with ZipFile(
            archive_path,
            "r",
        ) as archive:
            bad_member = (
                archive.testzip()
            )

            if bad_member is not None:
                raise RuntimeError(
                    "ZIP CRC validation failed: "
                    f"{bad_member}"
                )

            file_infos = [
                info
                for info in archive.infolist()
                if not info.is_dir()
            ]

            normalized_names: list[str] = []

            for info in file_infos:
                normalized_name = (
                    validate_archive_path(
                        info.filename
                    )
                )

                normalized_names.append(
                    normalized_name
                )

                if info.flag_bits & 0x1:
                    raise RuntimeError(
                        "Encrypted ZIP entry is "
                        "not supported: "
                        f"{normalized_name}"
                    )

            if len(
                set(normalized_names)
            ) != len(normalized_names):
                raise RuntimeError(
                    "Duplicate ZIP paths detected."
                )

            archive_root = (
                detect_archive_root(
                    normalized_names
                )
            )

            normalized_to_info = {
                validate_archive_path(
                    info.filename
                ): info
                for info in file_infos
            }

            pdf_names = sorted(
                [
                    name
                    for name in normalized_names
                    if PurePosixPath(
                        name
                    ).suffix.casefold()
                    == ".pdf"
                ],
                key=natural_key,
            )

            if not pdf_names:
                raise RuntimeError(
                    "No PDF files found in ZIP."
                )

            for order, name in enumerate(
                pdf_names,
                start=1,
            ):
                relative_name = (
                    relative_to_archive_root(
                        name,
                        archive_root,
                    )
                )

                info = normalized_to_info[
                    name
                ]

                try:
                    document = (
                        inspect_pdf_entry(
                            archive=archive,
                            zip_name=(
                                info.filename
                            ),
                            relative_name=(
                                relative_name
                            ),
                        )
                    )

                    document.update({
                        "candidate_order": (
                            order
                        ),
                        "candidate_document_id": (
                            f"document-{order:03d}"
                        ),
                    })

                    pdf_documents.append(
                        document
                    )

                except Exception as error:
                    pdf_documents.append({
                        "candidate_order": (
                            order
                        ),
                        "candidate_document_id": (
                            f"document-{order:03d}"
                        ),
                        "source_filename": (
                            PurePosixPath(
                                relative_name
                            ).name
                        ),
                        "relative_archive_path": (
                            relative_name
                        ),
                        "archive_path": name,
                        "source_page_count": (
                            None
                        ),
                        "dominant_script": (
                            "unknown"
                        ),
                        "pdf_status": "invalid",
                        "error": str(error),
                    })

            for name in sorted(
                normalized_names,
                key=natural_key,
            ):
                suffix = PurePosixPath(
                    name
                ).suffix.casefold()

                relative_name = (
                    relative_to_archive_root(
                        name,
                        archive_root,
                    )
                )

                if suffix in (
                    SUPPORTED_ASSET_SUFFIXES
                ):
                    supplementary_assets.append(
                        relative_name
                    )

                elif suffix != ".pdf":
                    unsupported_files.append(
                        relative_name
                    )

            total_uncompressed = sum(
                info.file_size
                for info in file_infos
            )

            total_compressed = sum(
                info.compress_size
                for info in file_infos
            )

    except BadZipFile as error:
        raise RuntimeError(
            f"Invalid ZIP archive: {error}"
        ) from error

    invalid_pdfs = [
        item
        for item in pdf_documents
        if item["pdf_status"] != "valid"
    ]

    if invalid_pdfs:
        warnings.append(
            f"{len(invalid_pdfs)} invalid PDF "
            "file(s) detected"
        )

    total_pages = sum(
        int(item["source_page_count"])
        for item in pdf_documents
        if isinstance(
            item.get("source_page_count"),
            int,
        )
    )

    combined_script_counts: Counter[
        str
    ] = Counter()

    for document in pdf_documents:
        combined_script_counts.update(
            document.get(
                "script_counts",
                {},
            )
        )

    expected_script = (
        EXPECTED_SCRIPT_BY_LANGUAGE.get(
            str(book.get("language", ""))
        )
    )

    script_evidence = assess_script_evidence(
        expected_script=expected_script,
        script_counts=combined_script_counts,
        total_pages=total_pages,
    )

    detected_script = script_evidence[
        "detected_script"
    ]

    if (
        script_evidence[
            "script_verification"
        ] == "unverified"
    ):
        warnings.append(
            "PDF text layer does not provide "
            "enough meaningful script evidence; "
            "visual/BDA extraction required. "
            f"raw_detected="
            f"{script_evidence['raw_detected_script']}, "
            f"meaningful_characters="
            f"{script_evidence['meaningful_script_characters']}, "
            f"pages={total_pages}"
        )

    elif (
        script_evidence[
            "script_verification"
        ] == "mismatch"
    ):
        warnings.append(
            "Registry language/script differs "
            "from sampled PDF text: "
            f"expected={expected_script}, "
            f"detected={detected_script}"
        )

    if unsupported_files:
        warnings.append(
            f"{len(unsupported_files)} "
            "unsupported supplementary file(s)"
        )

    report = {
        "schema_version": "1.0",
        "inspection_status": (
            "NEEDS_REVIEW"
            if invalid_pdfs
            else (
                "PASSED_WITH_WARNING"
                if warnings
                else "PASSED"
            )
        ),
        "inspected_at": utc_now(),
        "book": {
            "book_id": args.book_id,
            "title": book["title"],
            "grade": book["grade"],
            "subject": book["subject"],
            "language": book["language"],
            "expected_script": (
                expected_script
            ),
            "detected_script": (
                detected_script
            ),
            "raw_detected_script": (
                script_evidence[
                    "raw_detected_script"
                ]
            ),
            "script_verification": (
                script_evidence[
                    "script_verification"
                ]
            ),
            "extraction_route": (
                script_evidence[
                    "extraction_route"
                ]
            ),
        },
        "source_archive": {
            "bucket": bucket,
            "key": source_key,
            "local_path": str(
                archive_path
            ),
            "download_status": (
                download_status
            ),
            "size_bytes": (
                archive_path.stat().st_size
            ),
            "etag": str(
                head.get("ETag", "")
            ).strip('"'),
            "last_modified": (
                head["LastModified"].isoformat()
            ),
            "sha256": sha256_file(
                archive_path
            ),
            "archive_root": archive_root,
        },
        "archive_inventory": {
            "file_count": (
                len(pdf_documents)
                + len(supplementary_assets)
                + len(unsupported_files)
            ),
            "pdf_count": len(
                pdf_documents
            ),
            "supplementary_asset_count": (
                len(supplementary_assets)
            ),
            "unsupported_file_count": (
                len(unsupported_files)
            ),
            "total_uncompressed_bytes": (
                total_uncompressed
            ),
            "total_compressed_bytes": (
                total_compressed
            ),
        },
        "canonical_candidate": {
            "ordering_strategy": (
                "natural-filename"
            ),
            "source_document_pages": (
                total_pages
            ),
            "leading_blank_pages": 0,
            "trailing_blank_pages": 0,
            "canonical_page_count": (
                total_pages
            ),
        },
        "script_counts": dict(
            sorted(
                combined_script_counts.items()
            )
        ),
        "script_evidence": (
            script_evidence
        ),
        "documents": pdf_documents,
        "supplementary_assets": (
            supplementary_assets
        ),
        "unsupported_files": (
            unsupported_files
        ),
        "warnings": warnings,
        "safety": {
            "s3_reads": True,
            "s3_writes": 0,
            "bedrock_calls": 0,
            "opensearch_calls": 0,
        },
    }

    atomic_write_json(
        report_path,
        report,
    )

    print("=" * 80)
    print("TEXTBOOK SOURCE ARCHIVE INSPECTION")
    print("=" * 80)
    print("Book ID:          ", args.book_id)
    print("Language:         ", book["language"])
    print("Expected script:  ", expected_script)
    print("Detected script:  ", detected_script)
    print(
        "Raw script result: ",
        script_evidence[
            "raw_detected_script"
        ],
    )
    print(
        "Script verification:",
        script_evidence[
            "script_verification"
        ],
    )
    print(
        "Extraction route: ",
        script_evidence[
            "extraction_route"
        ],
    )
    print(
        "Text evidence:    ",
        script_evidence[
            "text_evidence_status"
        ],
        "("
        f"{script_evidence['meaningful_script_characters']}"
        " meaningful characters)",
    )
    print("Download status:  ", download_status)
    print("Archive:          ", archive_path)
    print("Archive root:     ", archive_root)
    print("PDF documents:    ", len(pdf_documents))
    print("Total pages:      ", total_pages)
    print(
        "Supplementary:   ",
        len(supplementary_assets),
    )
    print(
        "Unsupported:     ",
        len(unsupported_files),
    )
    print(
        "Inspection:      ",
        report["inspection_status"],
    )

    print()
    print("CANDIDATE PDF ORDER")
    print("-" * 80)

    for document in pdf_documents:
        print(
            f"{document['candidate_order']:02}. "
            f"{document['source_filename']} | "
            f"pages="
            f"{document.get('source_page_count')} | "
            f"script="
            f"{document.get('dominant_script')} | "
            f"status="
            f"{document['pdf_status']}"
        )

    if supplementary_assets:
        print()
        print("SUPPLEMENTARY ASSETS")
        print("-" * 80)

        for name in supplementary_assets:
            print("-", name)

    if warnings:
        print()
        print("WARNINGS")
        print("-" * 80)

        for warning in warnings:
            print("-", warning)

    print()
    print("Report:", report_path)
    print("S3 reads:        yes")
    print("S3 writes:       0")
    print("Bedrock calls:   0")
    print("OpenSearch calls: 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
