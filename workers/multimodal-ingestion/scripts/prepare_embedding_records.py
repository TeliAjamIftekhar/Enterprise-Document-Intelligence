from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"

EXCLUDED_MODALITIES = {
    "logo",
    "icon",
    "qr_code",
}

LOW_INFORMATION_PATTERN = re.compile(
    r"^[\W_]*\d+[\W_]*$"
)


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, "
                    f"line {line_number}: {exc}"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object in {path}, "
                    f"line {line_number}."
                )

            records.append(value)

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")


def normalize_whitespace(
    text: str,
) -> str:
    return " ".join(text.split())


def unique_nonempty(
    values: list[str],
) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()

    for value in values:
        if not isinstance(value, str):
            continue

        cleaned = normalize_whitespace(value)

        if not cleaned or cleaned in seen:
            continue

        seen.add(cleaned)
        results.append(cleaned)

    return results


def is_low_information_text(
    text: str,
) -> bool:
    cleaned = normalize_whitespace(text)

    if not cleaned:
        return True

    if LOW_INFORMATION_PATTERN.fullmatch(cleaned):
        return True

    alphanumeric = re.sub(
        r"[^A-Za-z0-9]+",
        "",
        cleaned,
    )

    if len(alphanumeric) < 3:
        return True

    return False


def build_text_embedding_text(
    content_unit: dict[str, Any],
) -> str:
    return normalize_whitespace(
        str(
            content_unit.get("search_text")
            or content_unit.get("raw_text")
            or ""
        )
    )


def build_figure_embedding_text(
    content_unit: dict[str, Any],
    figure: dict[str, Any],
) -> str:
    title = str(
        figure.get("generated_title")
        or content_unit.get("generated_title")
        or ""
    )

    summary = str(
        figure.get("generated_summary")
        or content_unit.get("generated_summary")
        or ""
    )

    ocr_text = str(
        figure.get("ocr_text")
        or content_unit.get("raw_text")
        or ""
    )

    # Prefer generated visual understanding.
    # Use OCR only when no generated summary exists.
    if summary.strip():
        parts = [
            title,
            summary,
        ]
    else:
        parts = [
            title,
            ocr_text,
        ]

    return "\n\n".join(
        unique_nonempty(parts)
    )


def build_table_embedding_text(
    content_unit: dict[str, Any],
    table: dict[str, Any],
) -> str:
    title = str(
        table.get("generated_title")
        or content_unit.get("generated_title")
        or ""
    )

    headers = table.get("headers", [])

    if not isinstance(headers, list):
        headers = []

    header_text = ", ".join(
        value
        for value in headers
        if isinstance(value, str)
        and value.strip()
    )

    plain_text = str(
        table.get("plain_text")
        or content_unit.get("raw_text")
        or ""
    )

    csv_text = str(
        table.get("csv_text")
        or ""
    )

    parts = []

    if title.strip():
        parts.append(
            f"Table title: {title}"
        )

    if header_text:
        parts.append(
            f"Table columns: {header_text}"
        )

    if plain_text.strip():
        parts.append(
            f"Table content:\n{plain_text}"
        )

    if csv_text.strip():
        parts.append(
            f"Structured CSV:\n{csv_text}"
        )

    return "\n\n".join(
        unique_nonempty(parts)
    )


def split_long_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    cleaned = text.strip()

    if not cleaned:
        return []

    if len(cleaned) <= max_chars:
        return [cleaned]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            cleaned,
        )
        if paragraph.strip()
    ]

    if len(paragraphs) <= 1:
        paragraphs = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                cleaned,
            )
            if sentence.strip()
        ]

    chunks: list[str] = []
    current = ""

    for part in paragraphs:
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            start = 0

            while start < len(part):
                end = min(
                    start + max_chars,
                    len(part),
                )

                chunks.append(
                    part[start:end].strip()
                )

                if end >= len(part):
                    break

                start = max(
                    end - overlap_chars,
                    start + 1,
                )

            continue

        candidate = (
            f"{current}\n\n{part}"
            if current
            else part
        )

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        overlap = ""

        if chunks and overlap_chars > 0:
            overlap = chunks[-1][
                -overlap_chars:
            ].strip()

        current = (
            f"{overlap}\n\n{part}".strip()
            if overlap
            else part
        )

    if current:
        chunks.append(current)

    return [
        chunk
        for chunk in chunks
        if chunk.strip()
    ]


def determine_skip_reason(
    content_unit: dict[str, Any],
    embedding_text: str,
) -> str | None:
    retrieval_priority = str(
        content_unit.get(
            "retrieval_priority",
            "normal",
        )
    ).lower()

    modality = str(
        content_unit.get(
            "modality",
            "",
        )
    ).lower()

    element_sub_type = str(
        content_unit.get(
            "element_sub_type",
            "",
        )
    ).upper()

    if retrieval_priority == "low":
        return "low_retrieval_priority"

    if modality in EXCLUDED_MODALITIES:
        return f"excluded_modality:{modality}"

    if element_sub_type in {
        "FOOTER",
        "HEADER",
    }:
        return (
            f"excluded_subtype:"
            f"{element_sub_type.lower()}"
        )

    if not embedding_text.strip():
        return "empty_embedding_text"

    if is_low_information_text(
        embedding_text
    ):
        return "low_information_text"

    return None


def build_citation_label(
    book_id: str,
    pages: list[int],
) -> str:
    if not pages:
        return book_id

    if len(pages) == 1:
        return (
            f"{book_id}, page {pages[0]}"
        )

    return (
        f"{book_id}, pages "
        f"{min(pages)}-{max(pages)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare normalized multimodal records "
            "for text embedding and retrieval."
        )
    )

    parser.add_argument(
        "normalized_dir",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=1800,
    )

    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    if args.max_chars < 500:
        raise ValueError(
            "max-chars must be at least 500."
        )

    if not (
        0 <= args.overlap_chars
        < args.max_chars
    ):
        raise ValueError(
            "overlap-chars must be between "
            "0 and max-chars - 1."
        )

    normalized_dir = args.normalized_dir

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else normalized_dir / "embedding-ready"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    content_units = load_jsonl(
        normalized_dir
        / "content-units.jsonl"
    )

    figures = load_jsonl(
        normalized_dir
        / "figures.jsonl"
    )

    tables = load_jsonl(
        normalized_dir
        / "tables.jsonl"
    )

    figure_by_id = {
        str(record["figure_id"]): record
        for record in figures
    }

    table_by_id = {
        str(record["table_id"]): record
        for record in tables
    }

    embedding_records: list[
        dict[str, Any]
    ] = []

    skipped_records: list[
        dict[str, Any]
    ] = []

    modality_counter: Counter[str] = (
        Counter()
    )

    skip_counter: Counter[str] = Counter()

    for content_unit in content_units:
        unit_id = str(
            content_unit["unit_id"]
        )

        element_type = str(
            content_unit.get(
                "element_type",
                "UNKNOWN",
            )
        ).upper()

        modality = str(
            content_unit.get(
                "modality",
                "unknown",
            )
        )

        if element_type == "FIGURE":
            figure = figure_by_id.get(
                unit_id,
                {},
            )

            embedding_text = (
                build_figure_embedding_text(
                    content_unit,
                    figure,
                )
            )

        elif element_type == "TABLE":
            table = table_by_id.get(
                unit_id,
                {},
            )

            embedding_text = (
                build_table_embedding_text(
                    content_unit,
                    table,
                )
            )

        else:
            embedding_text = (
                build_text_embedding_text(
                    content_unit
                )
            )

        skip_reason = determine_skip_reason(
            content_unit=content_unit,
            embedding_text=embedding_text,
        )

        if skip_reason:
            skip_counter[skip_reason] += 1

            skipped_records.append(
                {
                    "unit_id": unit_id,
                    "element_index": (
                        content_unit.get(
                            "element_index"
                        )
                    ),
                    "element_type": (
                        element_type
                    ),
                    "element_sub_type": (
                        content_unit.get(
                            "element_sub_type"
                        )
                    ),
                    "modality": modality,
                    "source_page_numbers": (
                        content_unit.get(
                            "source_page_numbers",
                            [],
                        )
                    ),
                    "reason": skip_reason,
                    "text_preview": (
                        normalize_whitespace(
                            embedding_text
                        )[:240]
                    ),
                }
            )

            continue

        chunks = split_long_text(
            text=embedding_text,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )

        pages = content_unit.get(
            "source_page_numbers",
            [],
        )

        if not isinstance(pages, list):
            pages = []

        pages = [
            value
            for value in pages
            if isinstance(value, int)
        ]

        for chunk_index, chunk in enumerate(
            chunks,
            start=1,
        ):
            record_id = (
                f"{unit_id}:chunk-{chunk_index:04d}"
            )

            record = {
                "schema_version": (
                    SCHEMA_VERSION
                ),
                "record_id": record_id,
                "source_unit_id": unit_id,
                "book_id": content_unit.get(
                    "book_id"
                ),
                "book_version": (
                    content_unit.get(
                        "book_version"
                    )
                ),
                "element_index": (
                    content_unit.get(
                        "element_index"
                    )
                ),
                "element_type": element_type,
                "element_sub_type": (
                    content_unit.get(
                        "element_sub_type"
                    )
                ),
                "modality": modality,
                "retrieval_priority": (
                    content_unit.get(
                        "retrieval_priority"
                    )
                ),
                "chunk_index": chunk_index,
                "chunk_count": len(chunks),
                "source_page_numbers": pages,
                "locations": content_unit.get(
                    "locations",
                    [],
                ),
                "citation_label": (
                    build_citation_label(
                        str(
                            content_unit.get(
                                "book_id",
                                "",
                            )
                        ),
                        pages,
                    )
                ),
                "embedding_text": chunk,
                "character_count": len(chunk),
                "asset_s3_uris": (
                    content_unit.get(
                        "asset_s3_uris",
                        [],
                    )
                ),
                "asset_local_paths": (
                    content_unit.get(
                        "asset_local_paths",
                        [],
                    )
                ),
                "quality_flags": (
                    content_unit.get(
                        "quality_flags",
                        [],
                    )
                ),
            }

            embedding_records.append(record)
            modality_counter[modality] += 1

    record_ids = [
        record["record_id"]
        for record in embedding_records
    ]

    if len(record_ids) != len(
        set(record_ids)
    ):
        raise RuntimeError(
            "Duplicate embedding record IDs found."
        )

    output_path = (
        output_dir
        / "embedding-records.jsonl"
    )

    skipped_path = (
        output_dir
        / "skipped-records.jsonl"
    )

    report_path = (
        output_dir
        / "embedding-preparation-report.json"
    )

    write_jsonl(
        output_path,
        embedding_records,
    )

    write_jsonl(
        skipped_path,
        skipped_records,
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "normalized_dir": str(
            normalized_dir
        ),
        "input_content_units": len(
            content_units
        ),
        "input_figures": len(figures),
        "input_tables": len(tables),
        "embedding_record_count": len(
            embedding_records
        ),
        "skipped_unit_count": len(
            skipped_records
        ),
        "records_by_modality": dict(
            sorted(
                modality_counter.items()
            )
        ),
        "skipped_by_reason": dict(
            sorted(skip_counter.items())
        ),
        "max_characters": max(
            (
                record["character_count"]
                for record in embedding_records
            ),
            default=0,
        ),
        "minimum_characters": min(
            (
                record["character_count"]
                for record in embedding_records
            ),
            default=0,
        ),
        "average_characters": round(
            (
                sum(
                    record[
                        "character_count"
                    ]
                    for record
                    in embedding_records
                )
                / len(embedding_records)
            )
            if embedding_records
            else 0.0
        ),
        "chunking": {
            "max_chars": args.max_chars,
            "overlap_chars": (
                args.overlap_chars
            ),
        },
        "policy": {
            "generated_figure_summary_preferred": True,
            "figure_ocr_used_only_without_summary": True,
            "table_csv_preserved": True,
            "low_priority_units_excluded": True,
            "decorative_modalities_excluded": sorted(
                EXCLUDED_MODALITIES
            ),
            "empty_text_not_embedded": True,
            "numeric_page_markers_excluded": True,
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("============================================")
    print("EMBEDDING RECORD PREPARATION")
    print("============================================")
    print(
        f"Input content units: "
        f"{len(content_units)}"
    )
    print(
        f"Embedding records:   "
        f"{len(embedding_records)}"
    )
    print(
        f"Skipped units:       "
        f"{len(skipped_records)}"
    )
    print()

    print("Records by modality:")

    for key, value in sorted(
        modality_counter.items()
    ):
        print(f"- {key}: {value}")

    print()
    print("Skipped by reason:")

    if skip_counter:
        for key, value in sorted(
            skip_counter.items()
        ):
            print(f"- {key}: {value}")
    else:
        print("- None")

    print()
    print(
        f"Maximum characters:  "
        f"{report['max_characters']}"
    )
    print(
        f"Average characters:  "
        f"{report['average_characters']}"
    )
    print()
    print(
        f"Embedding records: {output_path}"
    )
    print(
        f"Skipped records:   {skipped_path}"
    )
    print(
        f"Report:            {report_path}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as exc:
        print(
            f"Embedding preparation failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
