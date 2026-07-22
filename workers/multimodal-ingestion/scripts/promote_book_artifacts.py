from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


class PromotionError(RuntimeError):
    pass


PRESERVED_REFERENCE_FIELDS = frozenset(
    {
        "asset_local_paths",
        "asset_s3_uris",
    }
)


def extract_id(
    value: dict[str, Any],
) -> str | None:
    for key in (
        "record_id",
        "id",
        "_id",
        "document_id",
        "chunk_id",
    ):
        candidate = value.get(key)

        if (
            isinstance(candidate, str)
            and candidate.strip()
        ):
            return candidate.strip()

    metadata = value.get("metadata")

    if isinstance(metadata, dict):
        for key in (
            "record_id",
            "id",
            "document_id",
            "chunk_id",
        ):
            candidate = metadata.get(key)

            if (
                isinstance(candidate, str)
                and candidate.strip()
            ):
                return candidate.strip()

    return None


def extract_text(
    value: dict[str, Any],
) -> str:
    for key in (
        "text",
        "content",
        "chunk_text",
        "search_text",
        "embedding_text",
    ):
        candidate = value.get(key)

        if isinstance(candidate, str):
            return candidate.strip()

    document = value.get("document")

    if isinstance(document, dict):
        return extract_text(document)

    return ""


def extract_vector(
    value: dict[str, Any],
) -> list[float] | None:
    for key in (
        "embedding",
        "vector",
        "embedding_vector",
        "text_embedding",
    ):
        candidate = value.get(key)

        if (
            isinstance(candidate, list)
            and candidate
            and all(
                isinstance(
                    item,
                    (int, float),
                )
                for item in candidate
            )
        ):
            return [
                float(item)
                for item in candidate
            ]

    for key in (
        "result",
        "data",
        "document",
    ):
        child = value.get(key)

        if isinstance(child, dict):
            vector = extract_vector(child)

            if vector is not None:
                return vector

    return None


def canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rewrite_value(
    value: Any,
    *,
    source_version: str,
    target_version: str,
) -> Any:
    if isinstance(value, dict):
        rewritten: dict[str, Any] = {}

        for key, child in value.items():
            if key in PRESERVED_REFERENCE_FIELDS:
                rewritten[key] = child
                continue

            rewritten[key] = rewrite_value(
                child,
                source_version=source_version,
                target_version=target_version,
            )

        return rewritten

    if isinstance(value, list):
        return [
            rewrite_value(
                child,
                source_version=source_version,
                target_version=target_version,
            )
            for child in value
        ]

    if isinstance(value, str):
        return value.replace(
            source_version,
            target_version,
        )

    return value


def count_unexpected_source_version_strings(
    value: Any,
    *,
    source_version: str,
) -> int:
    if isinstance(value, dict):
        return sum(
            count_unexpected_source_version_strings(
                child,
                source_version=source_version,
            )
            for key, child in value.items()
            if key not in PRESERVED_REFERENCE_FIELDS
        )

    if isinstance(value, list):
        return sum(
            count_unexpected_source_version_strings(
                child,
                source_version=source_version,
            )
            for child in value
        )

    if isinstance(value, str):
        return int(source_version in value)

    return 0


def walk_strings(
    value: Any,
) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from walk_strings(child)

    elif isinstance(value, list):
        for child in value:
            yield from walk_strings(child)

    elif isinstance(value, str):
        yield value


def load_occurrences(
    paths: list[Path],
) -> tuple[
    dict[str, list[tuple[Path, dict[str, Any]]]],
    int,
]:
    occurrences: dict[
        str,
        list[tuple[Path, dict[str, Any]]],
    ] = {}

    raw_line_count = 0

    for path in paths:
        with path.open(
            "r",
            encoding="utf-8",
            errors="strict",
        ) as handle:
            for line_number, raw_line in enumerate(
                handle,
                start=1,
            ):
                raw_line = raw_line.strip()

                if not raw_line:
                    continue

                raw_line_count += 1

                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise PromotionError(
                        f"Invalid JSON in {path}:"
                        f"{line_number}: {exc}"
                    ) from exc

                if not isinstance(value, dict):
                    raise PromotionError(
                        "JSONL entries must be "
                        f"objects: {path}:"
                        f"{line_number}"
                    )

                record_id = extract_id(value)

                if record_id is None:
                    raise PromotionError(
                        "Missing record ID: "
                        f"{path}:{line_number}"
                    )

                occurrences.setdefault(
                    record_id,
                    [],
                ).append(
                    (
                        path,
                        value,
                    )
                )

    return occurrences, raw_line_count


def choose_records(
    occurrences: dict[
        str,
        list[tuple[Path, dict[str, Any]]],
    ],
) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}

    for record_id, items in occurrences.items():
        text_hashes = {
            sha256_bytes(
                extract_text(value).encode(
                    "utf-8"
                )
            )
            for _, value in items
        }

        if len(text_hashes) > 1:
            raise PromotionError(
                "Conflicting text for duplicate "
                f"record ID: {record_id}"
            )

        selected_path, selected_value = min(
            items,
            key=lambda item: str(item[0]),
        )

        del selected_path

        chosen[record_id] = selected_value

    return chosen


def choose_embeddings(
    occurrences: dict[
        str,
        list[tuple[Path, dict[str, Any]]],
    ],
    *,
    expected_dimension: int,
) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}

    for record_id, items in occurrences.items():
        vectors: list[list[float]] = []

        for path, value in items:
            vector = extract_vector(value)

            if vector is None:
                raise PromotionError(
                    "Missing vector for "
                    f"{record_id} in {path}"
                )

            if len(vector) != expected_dimension:
                raise PromotionError(
                    "Unexpected vector dimension "
                    f"for {record_id}: "
                    f"{len(vector)}"
                )

            vectors.append(vector)

        vector_hashes = {
            sha256_bytes(
                canonical_json(vector).encode(
                    "utf-8"
                )
            )
            for vector in vectors
        }

        if len(vector_hashes) > 1:
            raise PromotionError(
                "Conflicting vectors for "
                f"duplicate ID: {record_id}"
            )

        selected_path, selected_value = min(
            items,
            key=lambda item: str(item[0]),
        )

        del selected_path

        chosen[record_id] = selected_value

    return chosen


def render_jsonl(
    values: dict[str, dict[str, Any]],
) -> bytes:
    lines = [
        canonical_json(values[record_id])
        for record_id in sorted(values)
    ]

    return (
        "\n".join(lines) + "\n"
    ).encode("utf-8")


def render_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def write_artifact(
    path: Path,
    data: bytes,
    *,
    overwrite: bool,
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        existing = path.read_bytes()

        if existing == data:
            return "UNCHANGED"

        if not overwrite:
            raise PromotionError(
                "Existing artifact differs: "
                f"{path}. Use --overwrite only "
                "after reviewing the difference."
            )

    temporary = path.with_name(
        f".{path.name}.tmp"
    )

    temporary.write_bytes(data)
    temporary.replace(path)

    return "WRITTEN"


def discover_source_files(
    source_root: Path,
) -> tuple[list[Path], list[Path]]:
    batch_root = (
        source_root
        / "full-book/bda-results"
    )

    if not batch_root.is_dir():
        raise PromotionError(
            "Missing batch root: "
            f"{batch_root}"
        )

    record_files = sorted(
        path
        for path in batch_root.rglob(
            "embedding-records.jsonl"
        )
        if (
            path.is_file()
            and "/normalized/embedding-ready/"
            in path.as_posix()
            and "/smoke"
            not in path.as_posix()
        )
    )

    embedding_files = sorted(
        path
        for path in batch_root.rglob(
            "embeddings.jsonl"
        )
        if (
            path.is_file()
            and "/titan-text-v2/"
            in path.as_posix()
            and "/smoke"
            not in path.as_posix()
        )
    )

    if not record_files:
        raise PromotionError(
            "No embedding-record files found."
        )

    if not embedding_files:
        raise PromotionError(
            "No embedding files found."
        )

    return record_files, embedding_files


def promote(
    *,
    source_root: Path,
    output_root: Path,
    book_id: str,
    source_version: str,
    target_version: str,
    expected_count: int,
    expected_dimension: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    (
        record_files,
        embedding_files,
    ) = discover_source_files(source_root)

    (
        record_occurrences,
        raw_record_lines,
    ) = load_occurrences(record_files)

    (
        embedding_occurrences,
        raw_embedding_lines,
    ) = load_occurrences(
        embedding_files
    )

    records = choose_records(
        record_occurrences
    )

    embeddings = choose_embeddings(
        embedding_occurrences,
        expected_dimension=expected_dimension,
    )

    if len(records) != expected_count:
        raise PromotionError(
            "Unexpected unique record count: "
            f"{len(records)}; expected "
            f"{expected_count}"
        )

    if len(embeddings) != expected_count:
        raise PromotionError(
            "Unexpected unique embedding count: "
            f"{len(embeddings)}; expected "
            f"{expected_count}"
        )

    if set(records) != set(embeddings):
        raise PromotionError(
            "Record and embedding ID sets differ."
        )

    promoted_records: dict[
        str,
        dict[str, Any],
    ] = {}

    promoted_embeddings: dict[
        str,
        dict[str, Any],
    ] = {}

    citation_fields = (
        "citation_label",
        "context_citation_label",
        "source_page_numbers",
        "locations",
        "chapter_id",
        "chapter_ids",
        "chapter_title",
        "chapter_titles",
    )

    citation_change_count = 0

    for source_id, source_value in records.items():
        promoted_value = rewrite_value(
            source_value,
            source_version=source_version,
            target_version=target_version,
        )

        promoted_id = extract_id(
            promoted_value
        )

        if promoted_id is None:
            raise PromotionError(
                "Promoted record has no ID: "
                f"{source_id}"
            )

        if promoted_id in promoted_records:
            raise PromotionError(
                "Promoted record ID collision: "
                f"{promoted_id}"
            )

        for field in citation_fields:
            if (
                source_value.get(field)
                != promoted_value.get(field)
            ):
                citation_change_count += 1

        promoted_records[
            promoted_id
        ] = promoted_value

    for source_id, source_value in embeddings.items():
        promoted_value = rewrite_value(
            source_value,
            source_version=source_version,
            target_version=target_version,
        )

        promoted_id = extract_id(
            promoted_value
        )

        if promoted_id is None:
            raise PromotionError(
                "Promoted embedding has no ID: "
                f"{source_id}"
            )

        if promoted_id in promoted_embeddings:
            raise PromotionError(
                "Promoted embedding ID collision: "
                f"{promoted_id}"
            )

        promoted_embeddings[
            promoted_id
        ] = promoted_value

    if (
        set(promoted_records)
        != set(promoted_embeddings)
    ):
        raise PromotionError(
            "Promoted ID sets differ."
        )

    page_map_source = (
        source_root
        / "source/chapter-page-map.json"
    )

    merge_report_source = (
        source_root
        / "source/chapter-merge-report.json"
    )

    if not page_map_source.is_file():
        raise PromotionError(
            "Missing page map: "
            f"{page_map_source}"
        )

    if not merge_report_source.is_file():
        raise PromotionError(
            "Missing merge report: "
            f"{merge_report_source}"
        )

    page_map = rewrite_value(
        json.loads(
            page_map_source.read_text(
                encoding="utf-8"
            )
        ),
        source_version=source_version,
        target_version=target_version,
    )

    merge_report = rewrite_value(
        json.loads(
            merge_report_source.read_text(
                encoding="utf-8"
            )
        ),
        source_version=source_version,
        target_version=target_version,
    )

    source_version_remaining = sum(
        count_unexpected_source_version_strings(
            value,
            source_version=source_version,
        )
        for value in (
            list(promoted_records.values())
            + list(
                promoted_embeddings.values()
            )
            + [page_map, merge_report]
        )
    )

    preserved_reference_count = sum(
        len(value.get(field, []))
        for value in promoted_records.values()
        for field in PRESERVED_REFERENCE_FIELDS
        if isinstance(value.get(field, []), list)
    )

    if source_version_remaining:
        raise PromotionError(
            "Source-version strings remain "
            "after promotion: "
            f"{source_version_remaining}"
        )

    if citation_change_count:
        raise PromotionError(
            "Citation metadata changed during "
            f"promotion: {citation_change_count}"
        )

    record_data = render_jsonl(
        promoted_records
    )

    embedding_data = render_jsonl(
        promoted_embeddings
    )

    page_map_data = render_json(
        page_map
    )

    merge_report_data = render_json(
        merge_report
    )

    output_paths = {
        "records": (
            output_root
            / "embedding-records.jsonl"
        ),
        "embeddings": (
            output_root
            / "embeddings.jsonl"
        ),
        "page_map": (
            output_root
            / "source/chapter-page-map.json"
        ),
        "merge_report": (
            output_root
            / "source/chapter-merge-report.json"
        ),
        "report": (
            output_root
            / "promotion-report.json"
        ),
    }

    write_status = {
        "records": write_artifact(
            output_paths["records"],
            record_data,
            overwrite=overwrite,
        ),
        "embeddings": write_artifact(
            output_paths["embeddings"],
            embedding_data,
            overwrite=overwrite,
        ),
        "page_map": write_artifact(
            output_paths["page_map"],
            page_map_data,
            overwrite=overwrite,
        ),
        "merge_report": write_artifact(
            output_paths["merge_report"],
            merge_report_data,
            overwrite=overwrite,
        ),
    }

    report = {
        "schema_version": "1.0",
        "status": "COMPLETED",
        "book_id": book_id,
        "source_version": source_version,
        "target_version": target_version,
        "expected_count": expected_count,
        "expected_dimension": expected_dimension,
        "record_source_file_count": (
            len(record_files)
        ),
        "embedding_source_file_count": (
            len(embedding_files)
        ),
        "raw_record_line_count": (
            raw_record_lines
        ),
        "unique_record_count": (
            len(promoted_records)
        ),
        "raw_embedding_line_count": (
            raw_embedding_lines
        ),
        "unique_embedding_count": (
            len(promoted_embeddings)
        ),
        "duplicate_record_occurrences": (
            raw_record_lines
            - len(records)
        ),
        "duplicate_embedding_occurrences": (
            raw_embedding_lines
            - len(embeddings)
        ),
        "citation_metadata_preserved": True,
        "source_version_strings_remaining": 0,
        "preserved_reference_fields": sorted(
            PRESERVED_REFERENCE_FIELDS
        ),
        "preserved_source_reference_count": (
            preserved_reference_count
        ),
        "record_sha256": (
            sha256_bytes(record_data)
        ),
        "embedding_sha256": (
            sha256_bytes(embedding_data)
        ),
        "page_map_sha256": (
            sha256_bytes(page_map_data)
        ),
        "merge_report_sha256": (
            sha256_bytes(
                merge_report_data
            )
        ),
        "output_paths": {
            key: str(path)
            for key, path
            in output_paths.items()
        },
        "write_status": {
            key: "READY"
            for key in sorted(write_status)
        },
    }

    report_data = render_json(report)

    report_write_status = write_artifact(
        output_paths["report"],
        report_data,
        overwrite=overwrite,
    )

    report["runtime_write_status"] = {
        **write_status,
        "report": report_write_status,
    }

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate and promote completed "
            "textbook records and Titan "
            "embeddings between versions."
        )
    )

    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--book-id",
        required=True,
    )

    parser.add_argument(
        "--source-version",
        required=True,
    )

    parser.add_argument(
        "--target-version",
        required=True,
    )

    parser.add_argument(
        "--expected-count",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--expected-dimension",
        default=1024,
        type=int,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    report = promote(
        source_root=args.source_root,
        output_root=args.output_root,
        book_id=args.book_id,
        source_version=args.source_version,
        target_version=args.target_version,
        expected_count=args.expected_count,
        expected_dimension=(
            args.expected_dimension
        ),
        overwrite=args.overwrite,
    )

    print(
        "Promotion status:",
        report["status"],
    )
    print(
        "Unique records:",
        report["unique_record_count"],
    )
    print(
        "Unique embeddings:",
        report["unique_embedding_count"],
    )
    print(
        "Duplicate embedding occurrences:",
        report[
            "duplicate_embedding_occurrences"
        ],
    )
    print(
        "Record SHA256:",
        report["record_sha256"],
    )
    print(
        "Embedding SHA256:",
        report["embedding_sha256"],
    )
    print(
        "Output report:",
        report["output_paths"]["report"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
