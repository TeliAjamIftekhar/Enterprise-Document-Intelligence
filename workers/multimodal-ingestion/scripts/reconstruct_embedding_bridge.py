from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


class BridgeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def extract_id(value: dict[str, Any]) -> str | None:
    for key in (
        "record_id",
        "id",
        "_id",
        "document_id",
        "chunk_id",
    ):
        candidate = value.get(key)

        if isinstance(candidate, str) and candidate.strip():
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

            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

    return None


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
                isinstance(item, (int, float))
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


def extract_text(value: dict[str, Any]) -> str:
    for key in (
        "text",
        "content",
        "chunk_text",
        "search_text",
        "embedding_text",
    ):
        candidate = value.get(key)

        if isinstance(candidate, str):
            return candidate

    document = value.get("document")

    if isinstance(document, dict):
        return extract_text(document)

    return ""


def atomic_write_json(
    path: Path,
    value: dict[str, Any],
    *,
    overwrite: bool,
) -> str:
    data = (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        existing = path.read_bytes()

        if existing == data:
            return "UNCHANGED"

        if not overwrite:
            raise BridgeError(
                "Existing bridge artifact differs: "
                f"{path}. Review it before using "
                "--overwrite."
            )

    temporary = path.with_name(
        f".{path.name}.tmp"
    )

    temporary.write_bytes(data)

    if temporary.read_bytes() != data:
        temporary.unlink(missing_ok=True)

        raise BridgeError(
            f"Temporary write verification failed: "
            f"{path}"
        )

    os.replace(
        temporary,
        path,
    )

    return "WRITTEN"


def inspect_records(
    path: Path,
) -> dict[str, Any]:
    identifiers: set[str] = set()
    modalities: Counter[str] = Counter()

    line_count = 0
    citation_count = 0
    text_lengths: list[int] = []
    local_assets: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):
            raw_line = raw_line.strip()

            if not raw_line:
                continue

            value = json.loads(raw_line)

            if not isinstance(value, dict):
                raise BridgeError(
                    "Embedding record is not an "
                    f"object: {path}:{line_number}"
                )

            record_id = extract_id(value)

            if record_id is None:
                raise BridgeError(
                    "Embedding record ID missing: "
                    f"{path}:{line_number}"
                )

            if record_id in identifiers:
                raise BridgeError(
                    "Duplicate embedding record ID: "
                    f"{record_id}"
                )

            identifiers.add(record_id)
            line_count += 1

            modality = value.get(
                "modality",
                value.get(
                    "record_type",
                    "text",
                ),
            )

            modalities[str(modality)] += 1

            if value.get("citation_label"):
                citation_count += 1

            text_lengths.append(
                len(extract_text(value))
            )

            for asset_path in value.get(
                "asset_local_paths",
                [],
            ):
                local_assets.add(
                    str(asset_path)
                )

    missing_assets = [
        asset_path
        for asset_path in sorted(local_assets)
        if not Path(asset_path).is_file()
    ]

    return {
        "line_count": line_count,
        "ids": identifiers,
        "modalities": dict(
            sorted(modalities.items())
        ),
        "citation_count": citation_count,
        "minimum_characters": (
            min(text_lengths)
            if text_lengths
            else 0
        ),
        "maximum_characters": (
            max(text_lengths)
            if text_lengths
            else 0
        ),
        "average_characters": (
            sum(text_lengths) / len(text_lengths)
            if text_lengths
            else 0.0
        ),
        "unique_local_asset_count": len(
            local_assets
        ),
        "missing_local_assets": (
            missing_assets
        ),
    }


def inspect_embeddings(
    path: Path,
    *,
    expected_dimension: int,
) -> dict[str, Any]:
    identifiers: set[str] = set()

    line_count = 0
    minimum_norm: float | None = None
    maximum_norm: float | None = None
    total_norm = 0.0

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):
            raw_line = raw_line.strip()

            if not raw_line:
                continue

            value = json.loads(raw_line)

            if not isinstance(value, dict):
                raise BridgeError(
                    "Embedding is not an object: "
                    f"{path}:{line_number}"
                )

            record_id = extract_id(value)

            if record_id is None:
                raise BridgeError(
                    "Embedding ID missing: "
                    f"{path}:{line_number}"
                )

            if record_id in identifiers:
                raise BridgeError(
                    "Duplicate embedding ID: "
                    f"{record_id}"
                )

            vector = extract_vector(value)

            if vector is None:
                raise BridgeError(
                    "Embedding vector missing for "
                    f"{record_id}"
                )

            if len(vector) != expected_dimension:
                raise BridgeError(
                    "Unexpected vector dimension for "
                    f"{record_id}: {len(vector)}; "
                    f"expected {expected_dimension}"
                )

            norm = math.sqrt(
                sum(
                    component * component
                    for component in vector
                )
            )

            minimum_norm = (
                norm
                if minimum_norm is None
                else min(minimum_norm, norm)
            )

            maximum_norm = (
                norm
                if maximum_norm is None
                else max(maximum_norm, norm)
            )

            total_norm += norm
            line_count += 1
            identifiers.add(record_id)

    return {
        "line_count": line_count,
        "ids": identifiers,
        "minimum_vector_norm": (
            minimum_norm
            if minimum_norm is not None
            else 0.0
        ),
        "maximum_vector_norm": (
            maximum_norm
            if maximum_norm is not None
            else 0.0
        ),
        "average_vector_norm": (
            total_norm / line_count
            if line_count
            else 0.0
        ),
    }


def reconstruct_bridge(
    *,
    records_path: Path,
    embeddings_path: Path,
    embedding_report_path: Path,
    titan_manifest_path: Path,
    book_id: str,
    version: str,
    expected_count: int,
    expected_dimension: int,
    model_id: str,
    provenance_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not records_path.is_file():
        raise BridgeError(
            f"Embedding records missing: "
            f"{records_path}"
        )

    if not embeddings_path.is_file():
        raise BridgeError(
            f"Embeddings missing: "
            f"{embeddings_path}"
        )

    records = inspect_records(
        records_path
    )

    embeddings = inspect_embeddings(
        embeddings_path,
        expected_dimension=expected_dimension,
    )

    if records["line_count"] != expected_count:
        raise BridgeError(
            "Unexpected embedding-record count: "
            f"{records['line_count']}; "
            f"expected {expected_count}"
        )

    if embeddings["line_count"] != expected_count:
        raise BridgeError(
            "Unexpected embedding count: "
            f"{embeddings['line_count']}; "
            f"expected {expected_count}"
        )

    if records["ids"] != embeddings["ids"]:
        missing_embeddings = (
            records["ids"]
            - embeddings["ids"]
        )

        extra_embeddings = (
            embeddings["ids"]
            - records["ids"]
        )

        raise BridgeError(
            "Record and embedding ID sets differ. "
            f"Missing embeddings: "
            f"{len(missing_embeddings)}; "
            f"extra embeddings: "
            f"{len(extra_embeddings)}"
        )

    if records["missing_local_assets"]:
        raise BridgeError(
            "Missing referenced local assets: "
            f"{len(records['missing_local_assets'])}"
        )

    if records["citation_count"] != expected_count:
        raise BridgeError(
            "Citation-bearing record count is "
            f"{records['citation_count']}; "
            f"expected {expected_count}"
        )

    provenance: dict[str, Any] = {
        "method": (
            "reconstructed-from-validated-"
            "promoted-artifacts"
        )
    }

    if provenance_path is not None:
        if not provenance_path.is_file():
            raise BridgeError(
                "Provenance receipt missing: "
                f"{provenance_path}"
            )

        provenance.update(
            {
                "receipt": str(
                    provenance_path
                ),
                "receipt_sha256": (
                    sha256_file(
                        provenance_path
                    )
                ),
            }
        )

    records_sha256 = sha256_file(
        records_path
    )

    embeddings_sha256 = sha256_file(
        embeddings_path
    )

    modality_counts = records[
        "modalities"
    ]

    embedding_report = {
        "schema_version": "1.0",
        "status": "PREPARED",
        "book_id": book_id,
        "book_version": version,
        "normalized_dir": str(
            records_path.parent.parent
        ),
        "input_content_units": (
            expected_count
        ),
        "input_tables": (
            modality_counts.get("table", 0)
        ),
        "input_figures": (
            modality_counts.get("figure", 0)
            + modality_counts.get("image", 0)
        ),
        "embedding_record_count": (
            expected_count
        ),
        "skipped_unit_count": 0,
        "skipped_by_reason": {},
        "records_by_modality": (
            modality_counts
        ),
        "minimum_characters": records[
            "minimum_characters"
        ],
        "maximum_characters": records[
            "maximum_characters"
        ],
        "average_characters": records[
            "average_characters"
        ],
        "chunking": {
            "mode": (
                "preserved-authoritative-records"
            )
        },
        "policy": {
            "rerun_bda": False,
            "rerun_titan": False,
            "expected_dimension": (
                expected_dimension
            ),
        },
        "input_records_jsonl": str(
            records_path
        ),
        "input_records_jsonl_sha256": (
            records_sha256
        ),
        "validation": {
            "record_count": expected_count,
            "unique_record_count": (
                len(records["ids"])
            ),
            "citation_count": records[
                "citation_count"
            ],
            "unique_local_asset_count": (
                records[
                    "unique_local_asset_count"
                ]
            ),
            "missing_local_asset_count": 0,
            "record_embedding_ids_equal": True,
        },
        "provenance": provenance,
    }

    titan_manifest = {
        "schema_version": "1.0",
        "status": "COMPLETED",
        "book_id": book_id,
        "book_version": version,
        "input_record_count": expected_count,
        "completed_record_count": (
            expected_count
        ),
        "new_embedding_count": 0,
        "reused_checkpoint_count": (
            expected_count
        ),
        "seeded_smoke_test_count": 0,
        "records_with_token_count": 0,
        "total_input_tokens": 0,
        "minimum_vector_norm": embeddings[
            "minimum_vector_norm"
        ],
        "maximum_vector_norm": embeddings[
            "maximum_vector_norm"
        ],
        "average_vector_norm": embeddings[
            "average_vector_norm"
        ],
        "configuration": {
            "model_id": model_id,
            "dimensions": (
                expected_dimension
            ),
            "normalize": True,
        },
        "embedding_sources": {
            "promoted_authoritative": (
                expected_count
            )
        },
        "input_records_jsonl": str(
            records_path
        ),
        "input_records_jsonl_sha256": (
            records_sha256
        ),
        "embeddings_jsonl": str(
            embeddings_path
        ),
        "embeddings_jsonl_sha256": (
            embeddings_sha256
        ),
        "provenance": provenance,
        "validation": {
            "unique_input_ids": (
                len(records["ids"])
            ),
            "unique_embedding_ids": (
                len(embeddings["ids"])
            ),
            "record_embedding_ids_equal": True,
            "embedding_dimensions": (
                expected_dimension
            ),
        },
    }

    report_status = atomic_write_json(
        embedding_report_path,
        embedding_report,
        overwrite=overwrite,
    )

    manifest_status = atomic_write_json(
        titan_manifest_path,
        titan_manifest,
        overwrite=overwrite,
    )

    return {
        "status": "COMPLETED",
        "book_id": book_id,
        "version": version,
        "record_count": expected_count,
        "embedding_count": expected_count,
        "embedding_report": str(
            embedding_report_path
        ),
        "embedding_report_status": (
            report_status
        ),
        "titan_manifest": str(
            titan_manifest_path
        ),
        "titan_manifest_status": (
            manifest_status
        ),
        "records_sha256": (
            records_sha256
        ),
        "embeddings_sha256": (
            embeddings_sha256
        ),
    }


def load_runner():
    runner_path = Path(
        "workers/multimodal-ingestion/scripts/"
        "run_all_textbooks.py"
    )

    spec = importlib.util.spec_from_file_location(
        "run_all_textbooks",
        runner_path,
    )

    if spec is None or spec.loader is None:
        raise BridgeError(
            "Unable to load run_all_textbooks.py"
        )

    runner = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(runner)

    return runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct truthful embedding "
            "preparation and Titan manifest "
            "contracts from validated existing "
            "records and embeddings."
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
        "--model-id",
        default=(
            "amazon.titan-embed-text-v2:0"
        ),
    )

    parser.add_argument(
        "--provenance",
        type=Path,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = load_runner()

    paths = runner.paths_for_book(
        args.book_id,
        args.version,
    )

    downstream = (
        runner.downstream_artifact_paths(
            paths
        )
    )

    titan = runner.titan_artifact_paths(
        paths
    )

    result = reconstruct_bridge(
        records_path=downstream[
            "embedding_records"
        ],
        embeddings_path=titan[
            "embeddings"
        ],
        embedding_report_path=downstream[
            "embedding_report"
        ],
        titan_manifest_path=titan[
            "manifest"
        ],
        book_id=args.book_id,
        version=args.version,
        expected_count=args.expected_count,
        expected_dimension=(
            args.expected_dimension
        ),
        model_id=args.model_id,
        provenance_path=args.provenance,
        overwrite=args.overwrite,
    )

    print("Bridge status:", result["status"])
    print(
        "Embedding records:",
        result["record_count"],
    )
    print(
        "Embeddings:",
        result["embedding_count"],
    )
    print(
        "Embedding report:",
        result["embedding_report"],
    )
    print(
        "Embedding report write status:",
        result["embedding_report_status"],
    )
    print(
        "Titan manifest:",
        result["titan_manifest"],
    )
    print(
        "Titan manifest write status:",
        result["titan_manifest_status"],
    )
    print(
        "Records SHA256:",
        result["records_sha256"],
    )
    print(
        "Embeddings SHA256:",
        result["embeddings_sha256"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
