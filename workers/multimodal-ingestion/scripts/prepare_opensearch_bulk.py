from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INDEX_NAME = "grade-9-english-kaveri-v1"
VECTOR_DIMENSIONS = 1024

DOCUMENT_FIELDS = (
    "schema_version",
    "record_id",
    "book_id",
    "book_version",
    "source_unit_id",
    "element_index",
    "element_type",
    "element_sub_type",
    "modality",
    "source_page_numbers",
    "citation_label",
    "embedding_text",
    "asset_s3_uris",
    "quality_flags",
    "retrieval_priority",
    "chunk_index",
    "chunk_count",
    "character_count",
    "input_token_count",
    "input_text_sha256",
    "embedding_model_id",
    "embedding_dimensions",
    "embedding_normalized",
    "vector_length",
    "vector_l2_norm",
    "locations",
    "embedding",
)

LOCAL_ONLY_FIELDS = {
    "asset_local_paths",
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
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
                    f"Invalid JSON at line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object at line "
                    f"{line_number}."
                )

            records.append(value)

    if not records:
        raise RuntimeError(
            "No embedding records were found."
        )

    return records


def vector_norm(
    vector: list[float],
) -> float:
    return math.sqrt(
        sum(
            value * value
            for value in vector
        )
    )


def validate_record(
    record: dict[str, Any],
    line_number: int,
) -> tuple[dict[str, Any], float]:
    record_id = record.get("record_id")

    if not isinstance(
        record_id,
        str,
    ) or not record_id.strip():
        raise ValueError(
            f"Record {line_number} has no record_id."
        )

    record_fields = set(record.keys())

    unknown_fields = (
        record_fields
        - set(DOCUMENT_FIELDS)
        - LOCAL_ONLY_FIELDS
    )

    if unknown_fields:
        raise ValueError(
            f"Record {record_id} contains unmapped "
            f"fields: {sorted(unknown_fields)}"
        )

    missing_fields = (
        set(DOCUMENT_FIELDS)
        - record_fields
    )

    if missing_fields:
        raise ValueError(
            f"Record {record_id} is missing mapped "
            f"fields: {sorted(missing_fields)}"
        )

    vector = record.get("embedding")

    if not isinstance(vector, list):
        raise ValueError(
            f"Record {record_id} has no embedding."
        )

    if len(vector) != VECTOR_DIMENSIONS:
        raise ValueError(
            f"Record {record_id} vector length is "
            f"{len(vector)}, expected "
            f"{VECTOR_DIMENSIONS}."
        )

    numeric_vector: list[float] = []

    for value in vector:
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Record {record_id} contains a "
                "non-numeric vector value."
            )

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise ValueError(
                f"Record {record_id} contains a "
                "non-finite vector value."
            )

        numeric_vector.append(numeric_value)

    norm = vector_norm(numeric_vector)

    if not 0.98 <= norm <= 1.02:
        raise ValueError(
            f"Record {record_id} has unexpected "
            f"vector norm: {norm}"
        )

    if record.get(
        "embedding_dimensions"
    ) != VECTOR_DIMENSIONS:
        raise ValueError(
            f"Record {record_id} has incorrect "
            "embedding_dimensions metadata."
        )

    if record.get(
        "vector_length"
    ) != VECTOR_DIMENSIONS:
        raise ValueError(
            f"Record {record_id} has incorrect "
            "vector_length metadata."
        )

    if record.get(
        "embedding_normalized"
    ) is not True:
        raise ValueError(
            f"Record {record_id} is not marked "
            "as normalized."
        )

    pages = record.get(
        "source_page_numbers"
    )

    if not isinstance(pages, list) or not pages:
        raise ValueError(
            f"Record {record_id} has no source pages."
        )

    if not all(
        isinstance(page, int) and page > 0
        for page in pages
    ):
        raise ValueError(
            f"Record {record_id} has invalid "
            "source page numbers."
        )

    locations = record.get(
        "locations"
    )

    if not isinstance(locations, list):
        raise ValueError(
            f"Record {record_id} locations is not a list."
        )

    normalized_locations: list[
        dict[str, Any]
    ] = []

    for location_number, location in enumerate(
        locations,
        start=1,
    ):
        if not isinstance(location, dict):
            raise ValueError(
                f"Record {record_id} location "
                f"{location_number} is not an object."
            )

        sample_page_index = location.get(
            "sample_page_index"
        )

        if not isinstance(
            sample_page_index,
            int,
        ) or sample_page_index < 0:
            raise ValueError(
                f"Record {record_id} location "
                f"{location_number} has an invalid "
                "sample_page_index."
            )

        source_page_number = location.get(
            "source_page_number"
        )

        if (
            not isinstance(
                source_page_number,
                int,
            )
            or source_page_number not in pages
        ):
            raise ValueError(
                f"Record {record_id} location "
                f"{location_number} source page does "
                "not match source_page_numbers."
            )

        bounding_box = location.get(
            "bounding_box"
        )

        if not isinstance(
            bounding_box,
            dict,
        ):
            raise ValueError(
                f"Record {record_id} location "
                f"{location_number} has no bounding box."
            )

        normalized_box: dict[str, float] = {}

        for coordinate in (
            "left",
            "top",
            "width",
            "height",
        ):
            value = bounding_box.get(
                coordinate
            )

            if not isinstance(
                value,
                (int, float),
            ):
                raise ValueError(
                    f"Record {record_id} location "
                    f"{location_number} has invalid "
                    f"{coordinate}."
                )

            numeric_coordinate = float(
                value
            )

            if not math.isfinite(
                numeric_coordinate
            ):
                raise ValueError(
                    f"Record {record_id} location "
                    f"{location_number} has non-finite "
                    f"{coordinate}."
                )

            normalized_box[
                coordinate
            ] = numeric_coordinate

        normalized_locations.append(
            {
                "page_index": (
                    sample_page_index
                ),
                "bounding_box": (
                    normalized_box
                ),
            }
        )

    document = {
        field: record[field]
        for field in DOCUMENT_FIELDS
    }

    document["locations"] = (
        normalized_locations
    )

    document["embedding"] = (
        numeric_vector
    )

    return document, norm


def atomic_write_text(
    path: Path,
    value: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        value,
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        path,
    )


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
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Titan embedding records and "
            "prepare an OpenSearch bulk NDJSON file."
        )
    )

    parser.add_argument(
        "embeddings_jsonl",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    records = load_jsonl(
        args.embeddings_jsonl
    )

    if not records:
        raise RuntimeError(
            "No embedding records were found."
        )

    embedding_manifest_path = (
        args.embeddings_jsonl.parent
        / "embedding-manifest.json"
    )

    expected_count = len(records)

    if embedding_manifest_path.is_file():
        embedding_manifest = json.loads(
            embedding_manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            embedding_manifest,
            dict,
        ):
            raise RuntimeError(
                "Embedding manifest is not "
                "a JSON object."
            )

        if (
            embedding_manifest.get("status")
            != "COMPLETED"
        ):
            raise RuntimeError(
                "Embedding manifest status is not "
                f"COMPLETED: "
                f"{embedding_manifest.get('status')}"
            )

        expected_count = int(
            embedding_manifest.get(
                "completed_record_count",
                embedding_manifest.get(
                    "input_record_count",
                    -1,
                ),
            )
        )

        if expected_count < 1:
            raise RuntimeError(
                "Embedding manifest has no valid "
                "completed record count."
            )

        if len(records) != expected_count:
            raise RuntimeError(
                "Embedding file and manifest count "
                "do not match: "
                f"manifest={expected_count}, "
                f"records={len(records)}."
            )

    document_ids: set[str] = set()
    output_lines: list[str] = []
    norms: list[float] = []

    removed_local_field_count = 0

    modality_counts: dict[str, int] = {}
    page_counts: dict[int, int] = {}

    for line_number, record in enumerate(
        records,
        start=1,
    ):
        record_id = str(
            record.get("record_id", "")
        )

        if record_id in document_ids:
            raise RuntimeError(
                f"Duplicate record_id: {record_id}"
            )

        document_ids.add(record_id)

        if "asset_local_paths" in record:
            removed_local_field_count += 1

        document, norm = validate_record(
            record=record,
            line_number=line_number,
        )

        norms.append(norm)

        modality = str(
            document["modality"]
        )

        modality_counts[modality] = (
            modality_counts.get(
                modality,
                0,
            )
            + 1
        )

        for page in document[
            "source_page_numbers"
        ]:
            page_counts[page] = (
                page_counts.get(page, 0)
                + 1
            )

        action = {
            "index": {
                "_index": INDEX_NAME,
                "_id": record_id,
            }
        }

        output_lines.append(
            json.dumps(
                action,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )

        output_lines.append(
            json.dumps(
                document,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )

    bulk_text = "\n".join(
        output_lines
    ) + "\n"

    bulk_path = (
        args.output_dir
        / "bulk-index.ndjson"
    )

    report_path = (
        args.output_dir
        / "bulk-preparation-report.json"
    )

    atomic_write_text(
        bulk_path,
        bulk_text,
    )

    if not bulk_path.read_bytes().endswith(
        b"\n"
    ):
        raise RuntimeError(
            "Bulk payload does not end with newline."
        )

    bulk_sha256 = sha256_file(
        bulk_path
    )

    input_sha256 = sha256_file(
        args.embeddings_jsonl
    )

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": "PREPARED",
        "index_name": INDEX_NAME,
        "input": {
            "path": str(
                args.embeddings_jsonl
            ),
            "sha256": input_sha256,
            "record_count": len(records),
        },
        "output": {
            "path": str(bulk_path),
            "sha256": bulk_sha256,
            "size_bytes": (
                bulk_path.stat().st_size
            ),
            "line_count": len(
                output_lines
            ),
            "ends_with_newline": True,
        },
        "validation": {
            "document_count": len(records),
            "unique_document_ids": len(
                document_ids
            ),
            "mapped_field_count": len(
                DOCUMENT_FIELDS
            ),
            "removed_local_field": (
                "asset_local_paths"
            ),
            "removed_local_field_count": (
                removed_local_field_count
            ),
            "vector_dimensions": (
                VECTOR_DIMENSIONS
            ),
            "minimum_vector_norm": min(
                norms
            ),
            "maximum_vector_norm": max(
                norms
            ),
            "modality_counts": (
                modality_counts
            ),
            "page_counts": {
                str(key): page_counts[key]
                for key in sorted(
                    page_counts
                )
            },
            "errors": [],
        },
        "uploaded": False,
    }

    atomic_write_json(
        report_path,
        report,
    )

    print("============================================")
    print("OPENSEARCH BULK PREPARATION")
    print("============================================")
    print(
        f"Input records:       {len(records)}"
    )
    print(
        f"Unique document IDs: {len(document_ids)}"
    )
    print(
        f"Mapped fields:       {len(DOCUMENT_FIELDS)}"
    )
    print(
        "Removed local paths: "
        f"{removed_local_field_count}"
    )
    print(
        f"Vector dimensions:   {VECTOR_DIMENSIONS}"
    )
    print(
        "Vector norm range:   "
        f"{min(norms):.8f} - "
        f"{max(norms):.8f}"
    )
    print(
        f"Bulk lines:          {len(output_lines)}"
    )
    print(
        f"Bulk size:           "
        f"{bulk_path.stat().st_size:,} bytes"
    )
    print(
        f"Ends with newline:   True"
    )
    print(
        f"Bulk SHA256:         {bulk_sha256}"
    )
    print(
        f"Bulk payload:        {bulk_path}"
    )
    print(
        f"Report:              {report_path}"
    )
    print("Uploaded:            False")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as exc:
        print(
            f"Bulk preparation failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
