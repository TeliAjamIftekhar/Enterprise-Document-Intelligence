from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
)


DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_DIMENSIONS = 1024
DEFAULT_NORMALIZE = True

RETRYABLE_ERROR_CODES = {
    "InternalServerException",
    "ModelErrorException",
    "ModelNotReadyException",
    "ModelTimeoutException",
    "ServiceUnavailableException",
    "ThrottlingException",
    "TooManyRequestsException",
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
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
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, "
                    f"line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path}, "
                    f"line {line_number}."
                )

            records.append(record)

    return records


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
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        path,
    )


def atomic_write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    os.replace(
        temporary_path,
        path,
    )


def get_item_path(
    items_dir: Path,
    record_id: str,
) -> Path:
    filename = (
        sha256_text(record_id)
        + ".json"
    )

    return items_dir / filename


def extract_embedding(
    response_body: dict[str, Any],
) -> list[float]:
    embedding = response_body.get(
        "embedding"
    )

    if not isinstance(embedding, list):
        embeddings_by_type = (
            response_body.get(
                "embeddingsByType",
                {},
            )
        )

        if isinstance(
            embeddings_by_type,
            dict,
        ):
            embedding = embeddings_by_type.get(
                "float"
            )

    if not isinstance(embedding, list):
        raise RuntimeError(
            "Titan response contains no float embedding."
        )

    vector: list[float] = []

    for index, value in enumerate(embedding):
        if not isinstance(value, (int, float)):
            raise RuntimeError(
                "Embedding contains a non-numeric value "
                f"at index {index}."
            )

        float_value = float(value)

        if not math.isfinite(float_value):
            raise RuntimeError(
                "Embedding contains a non-finite value "
                f"at index {index}."
            )

        vector.append(float_value)

    return vector


def calculate_vector_norm(
    vector: list[float],
) -> float:
    return math.sqrt(
        sum(
            value * value
            for value in vector
        )
    )


def validate_vector(
    vector: list[float],
    dimensions: int,
    normalize: bool,
) -> float:
    if len(vector) != dimensions:
        raise RuntimeError(
            "Unexpected vector length. "
            f"Expected={dimensions}, "
            f"actual={len(vector)}"
        )

    norm = calculate_vector_norm(vector)

    if not math.isfinite(norm):
        raise RuntimeError(
            "Embedding vector norm is not finite."
        )

    if normalize and not (
        0.98 <= norm <= 1.02
    ):
        raise RuntimeError(
            "Normalized embedding has an unexpected "
            f"L2 norm: {norm}"
        )

    return norm


def validate_input_records(
    records: list[dict[str, Any]],
) -> None:
    if not records:
        raise RuntimeError(
            "No embedding records were found."
        )

    seen_ids: set[str] = set()

    for index, record in enumerate(records):
        record_id = record.get("record_id")
        embedding_text = record.get(
            "embedding_text"
        )

        if not isinstance(
            record_id,
            str,
        ) or not record_id:
            raise ValueError(
                f"Record {index} has no valid record_id."
            )

        if record_id in seen_ids:
            raise ValueError(
                f"Duplicate record_id: {record_id}"
            )

        seen_ids.add(record_id)

        if not isinstance(
            embedding_text,
            str,
        ) or not embedding_text.strip():
            raise ValueError(
                f"Record {record_id} has empty "
                "embedding_text."
            )

        if len(embedding_text) > 50_000:
            raise ValueError(
                f"Record {record_id} exceeds "
                "50,000 characters."
            )


def build_run_configuration(
    input_path: Path,
    input_sha256: str,
    region: str,
    model_id: str,
    dimensions: int,
    normalize: bool,
) -> dict[str, Any]:
    return {
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "region": region,
        "model_id": model_id,
        "dimensions": dimensions,
        "normalize": normalize,
    }


def validate_existing_manifest(
    manifest: dict[str, Any],
    run_configuration: dict[str, Any],
) -> None:
    existing_configuration = manifest.get(
        "configuration"
    )

    if not isinstance(
        existing_configuration,
        dict,
    ):
        raise RuntimeError(
            "Existing manifest has no configuration."
        )

    fields = (
        "input_sha256",
        "region",
        "model_id",
        "dimensions",
        "normalize",
    )

    mismatches = []

    for field in fields:
        if (
            existing_configuration.get(field)
            != run_configuration.get(field)
        ):
            mismatches.append(
                {
                    "field": field,
                    "existing": (
                        existing_configuration.get(field)
                    ),
                    "requested": (
                        run_configuration.get(field)
                    ),
                }
            )

    if mismatches:
        raise RuntimeError(
            "Existing output directory belongs to an "
            "incompatible embedding run:\n"
            + json.dumps(
                mismatches,
                indent=2,
            )
        )


def build_item(
    record: dict[str, Any],
    vector: list[float],
    input_token_count: int | None,
    model_id: str,
    dimensions: int,
    normalize: bool,
    source: str,
) -> dict[str, Any]:
    embedding_text = str(
        record["embedding_text"]
    )

    norm = validate_vector(
        vector=vector,
        dimensions=dimensions,
        normalize=normalize,
    )

    return {
        "generated_at": utc_now(),
        "source": source,
        "record_id": record["record_id"],
        "source_unit_id": record.get(
            "source_unit_id"
        ),
        "book_id": record.get("book_id"),
        "book_version": record.get(
            "book_version"
        ),
        "element_type": record.get(
            "element_type"
        ),
        "element_sub_type": record.get(
            "element_sub_type"
        ),
        "modality": record.get("modality"),
        "source_page_numbers": record.get(
            "source_page_numbers",
            [],
        ),
        "citation_label": record.get(
            "citation_label"
        ),
        "input_character_count": len(
            embedding_text
        ),
        "input_text_sha256": sha256_text(
            embedding_text
        ),
        "input_token_count": (
            input_token_count
        ),
        "model_id": model_id,
        "dimensions": dimensions,
        "normalize": normalize,
        "vector_length": len(vector),
        "vector_l2_norm": norm,
        "embedding": vector,
    }


def validate_saved_item(
    item: dict[str, Any],
    record: dict[str, Any],
    model_id: str,
    dimensions: int,
    normalize: bool,
) -> None:
    record_id = str(
        record["record_id"]
    )

    embedding_text = str(
        record["embedding_text"]
    )

    expected_text_sha256 = sha256_text(
        embedding_text
    )

    expected_values = {
        "record_id": record_id,
        "input_text_sha256": (
            expected_text_sha256
        ),
        "model_id": model_id,
        "dimensions": dimensions,
        "normalize": normalize,
    }

    for field, expected in expected_values.items():
        actual = item.get(field)

        if actual != expected:
            raise RuntimeError(
                f"Saved embedding mismatch for "
                f"{record_id}: field={field}, "
                f"expected={expected!r}, "
                f"actual={actual!r}"
            )

    vector = item.get("embedding")

    if not isinstance(vector, list):
        raise RuntimeError(
            f"Saved item has no embedding: {record_id}"
        )

    numeric_vector = [
        float(value)
        for value in vector
    ]

    validate_vector(
        vector=numeric_vector,
        dimensions=dimensions,
        normalize=normalize,
    )


def seed_from_smoke_test(
    smoke_test_path: Path,
    records_by_id: dict[str, dict[str, Any]],
    items_dir: Path,
    model_id: str,
    dimensions: int,
    normalize: bool,
) -> int:
    if not smoke_test_path.exists():
        print(
            "Smoke-test vector: not found; "
            "nothing seeded."
        )
        return 0

    smoke = load_json(
        smoke_test_path
    )

    record_id = smoke.get("record_id")

    if not isinstance(
        record_id,
        str,
    ) or record_id not in records_by_id:
        print(
            "Smoke-test vector: record is not "
            "present in the current input; skipped."
        )
        return 0

    record = records_by_id[record_id]

    expected_text_sha256 = sha256_text(
        str(record["embedding_text"])
    )

    checks = {
        "model_id": model_id,
        "dimensions": dimensions,
        "normalize": normalize,
        "input_text_sha256": (
            expected_text_sha256
        ),
    }

    for field, expected in checks.items():
        if smoke.get(field) != expected:
            print(
                "Smoke-test vector: incompatible "
                f"{field}; skipped."
            )
            return 0

    vector = smoke.get("embedding")

    if not isinstance(vector, list):
        print(
            "Smoke-test vector: embedding missing; "
            "skipped."
        )
        return 0

    numeric_vector = [
        float(value)
        for value in vector
    ]

    item = build_item(
        record=record,
        vector=numeric_vector,
        input_token_count=smoke.get(
            "input_token_count"
        ),
        model_id=model_id,
        dimensions=dimensions,
        normalize=normalize,
        source="smoke_test",
    )

    item_path = get_item_path(
        items_dir=items_dir,
        record_id=record_id,
    )

    if item_path.exists():
        existing = load_json(item_path)

        validate_saved_item(
            item=existing,
            record=record,
            model_id=model_id,
            dimensions=dimensions,
            normalize=normalize,
        )

        print(
            "Smoke-test vector: matching checkpoint "
            "already exists."
        )

        return 0

    atomic_write_json(
        item_path,
        item,
    )

    print(
        "Smoke-test vector: seeded one matching "
        "embedding checkpoint."
    )

    return 1


def is_retryable_client_error(
    error: ClientError,
) -> bool:
    code = str(
        error.response.get(
            "Error",
            {},
        ).get(
            "Code",
            "",
        )
    )

    return code in RETRYABLE_ERROR_CODES


def invoke_with_retries(
    client: Any,
    embedding_text: str,
    model_id: str,
    dimensions: int,
    normalize: bool,
    max_attempts: int,
    base_delay_seconds: float,
) -> tuple[
    list[float],
    int | None,
]:
    request_body = {
        "inputText": embedding_text,
        "dimensions": dimensions,
        "normalize": normalize,
    }

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        try:
            response = client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body),
            )

            response_body = json.loads(
                response["body"].read()
            )

            if not isinstance(
                response_body,
                dict,
            ):
                raise RuntimeError(
                    "Titan response is not a JSON object."
                )

            vector = extract_embedding(
                response_body
            )

            validate_vector(
                vector=vector,
                dimensions=dimensions,
                normalize=normalize,
            )

            token_count = response_body.get(
                "inputTextTokenCount"
            )

            if not isinstance(
                token_count,
                int,
            ):
                token_count = None

            return vector, token_count

        except ClientError as exc:
            retryable = is_retryable_client_error(
                exc
            )

            if (
                not retryable
                or attempt >= max_attempts
            ):
                raise

            delay = min(
                base_delay_seconds
                * (2 ** (attempt - 1)),
                20.0,
            )

            code = exc.response.get(
                "Error",
                {},
            ).get(
                "Code",
                "Unknown",
            )

            print(
                f"    Retryable AWS error {code}; "
                f"retrying after {delay:.1f}s."
            )

            time.sleep(delay)

        except BotoCoreError:
            if attempt >= max_attempts:
                raise

            delay = min(
                base_delay_seconds
                * (2 ** (attempt - 1)),
                20.0,
            )

            print(
                "    Temporary SDK/network error; "
                f"retrying after {delay:.1f}s."
            )

            time.sleep(delay)

    raise RuntimeError(
        "Embedding invocation exhausted retries."
    )


def build_consolidated_record(
    source_record: dict[str, Any],
    item: dict[str, Any],
) -> dict[str, Any]:
    return {
        **source_record,
        "embedding_model_id": item[
            "model_id"
        ],
        "embedding_dimensions": item[
            "dimensions"
        ],
        "embedding_normalized": item[
            "normalize"
        ],
        "input_text_sha256": item[
            "input_text_sha256"
        ],
        "input_token_count": item.get(
            "input_token_count"
        ),
        "vector_length": item[
            "vector_length"
        ],
        "vector_l2_norm": item[
            "vector_l2_norm"
        ],
        "embedding": item["embedding"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate resumable Titan Text "
            "Embeddings V2 vectors."
        )
    )

    parser.add_argument(
        "input_jsonl",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--smoke-test",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        choices=(256, 512, 1024),
        default=DEFAULT_DIMENSIONS,
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--base-retry-delay",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.1,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    normalize = not args.no_normalize

    if args.max_attempts < 1:
        raise ValueError(
            "max-attempts must be at least 1."
        )

    if args.request_delay < 0:
        raise ValueError(
            "request-delay cannot be negative."
        )

    records = load_jsonl(
        args.input_jsonl
    )

    validate_input_records(records)

    input_sha256 = sha256_file(
        args.input_jsonl
    )

    output_dir: Path = args.output_dir

    items_dir = (
        output_dir / "items"
    )

    items_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_dir
        / "embedding-manifest.json"
    )

    embeddings_path = (
        output_dir
        / "embeddings.jsonl"
    )

    failures_path = (
        output_dir
        / "failed-records.jsonl"
    )

    run_configuration = (
        build_run_configuration(
            input_path=args.input_jsonl,
            input_sha256=input_sha256,
            region=args.region,
            model_id=args.model_id,
            dimensions=args.dimensions,
            normalize=normalize,
        )
    )

    if manifest_path.exists():
        manifest = load_json(
            manifest_path
        )

        validate_existing_manifest(
            manifest=manifest,
            run_configuration=(
                run_configuration
            ),
        )

    else:
        manifest = {
            "schema_version": "1.0",
            "status": "IN_PROGRESS",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "configuration": (
                run_configuration
            ),
            "input_record_count": len(records),
            "completed_record_count": 0,
        }

        atomic_write_json(
            manifest_path,
            manifest,
        )

    print("============================================")
    print("TITAN TEXT V2 BATCH EMBEDDING")
    print("============================================")
    print(f"Input:       {args.input_jsonl}")
    print(f"Input SHA:   {input_sha256}")
    print(f"Records:     {len(records)}")
    print(f"Region:      {args.region}")
    print(f"Model:       {args.model_id}")
    print(f"Dimensions:  {args.dimensions}")
    print(f"Normalize:   {normalize}")
    print(f"Output:      {output_dir}")
    print()

    records_by_id = {
        str(record["record_id"]): record
        for record in records
    }

    seeded_count = 0

    if args.smoke_test is not None:
        seeded_count = seed_from_smoke_test(
            smoke_test_path=args.smoke_test,
            records_by_id=records_by_id,
            items_dir=items_dir,
            model_id=args.model_id,
            dimensions=args.dimensions,
            normalize=normalize,
        )

        print()

    client = boto3.client(
        "bedrock-runtime",
        region_name=args.region,
        config=Config(
            retries={
                "mode": "standard",
                "max_attempts": 5,
            },
            connect_timeout=30,
            read_timeout=120,
        ),
    )

    new_embedding_count = 0
    reused_count = 0

    failures: list[dict[str, Any]] = []

    for index, record in enumerate(
        records,
        start=1,
    ):
        record_id = str(
            record["record_id"]
        )

        item_path = get_item_path(
            items_dir=items_dir,
            record_id=record_id,
        )

        if item_path.exists():
            item = load_json(item_path)

            validate_saved_item(
                item=item,
                record=record,
                model_id=args.model_id,
                dimensions=args.dimensions,
                normalize=normalize,
            )

            reused_count += 1

            print(
                f"[{index:02d}/{len(records):02d}] "
                f"REUSE {record_id}"
            )

            continue

        print(
            f"[{index:02d}/{len(records):02d}] "
            f"EMBED {record_id}"
        )

        try:
            vector, token_count = (
                invoke_with_retries(
                    client=client,
                    embedding_text=str(
                        record["embedding_text"]
                    ),
                    model_id=args.model_id,
                    dimensions=args.dimensions,
                    normalize=normalize,
                    max_attempts=args.max_attempts,
                    base_delay_seconds=(
                        args.base_retry_delay
                    ),
                )
            )

            item = build_item(
                record=record,
                vector=vector,
                input_token_count=token_count,
                model_id=args.model_id,
                dimensions=args.dimensions,
                normalize=normalize,
                source="bedrock_invoke_model",
            )

            atomic_write_json(
                item_path,
                item,
            )

            new_embedding_count += 1

            print(
                "    Saved | "
                f"tokens={token_count} | "
                f"norm={item['vector_l2_norm']:.8f}"
            )

            if args.request_delay:
                time.sleep(
                    args.request_delay
                )

        except Exception as exc:
            failure = {
                "record_id": record_id,
                "record_index": index,
                "error_type": (
                    type(exc).__name__
                ),
                "error_message": str(exc),
                "failed_at": utc_now(),
            }

            failures.append(failure)

            atomic_write_jsonl(
                failures_path,
                failures,
            )

            manifest.update(
                {
                    "status": "FAILED",
                    "updated_at": utc_now(),
                    "failed_record": failure,
                }
            )

            atomic_write_json(
                manifest_path,
                manifest,
            )

            raise

    consolidated_records: list[
        dict[str, Any]
    ] = []

    norms: list[float] = []
    token_counts: list[int] = []

    source_counts: dict[str, int] = {}

    for record in records:
        record_id = str(
            record["record_id"]
        )

        item_path = get_item_path(
            items_dir=items_dir,
            record_id=record_id,
        )

        item = load_json(item_path)

        validate_saved_item(
            item=item,
            record=record,
            model_id=args.model_id,
            dimensions=args.dimensions,
            normalize=normalize,
        )

        consolidated_records.append(
            build_consolidated_record(
                source_record=record,
                item=item,
            )
        )

        norms.append(
            float(item["vector_l2_norm"])
        )

        token_count = item.get(
            "input_token_count"
        )

        if isinstance(token_count, int):
            token_counts.append(
                token_count
            )

        source = str(
            item.get(
                "source",
                "unknown",
            )
        )

        source_counts[source] = (
            source_counts.get(source, 0)
            + 1
        )

    atomic_write_jsonl(
        embeddings_path,
        consolidated_records,
    )

    if failures_path.exists():
        failures_path.unlink()

    manifest.update(
        {
            "status": "COMPLETED",
            "updated_at": utc_now(),
            "completed_at": utc_now(),
            "input_record_count": len(records),
            "completed_record_count": len(
                consolidated_records
            ),
            "new_embedding_count": (
                new_embedding_count
            ),
            "reused_checkpoint_count": (
                reused_count
            ),
            "seeded_smoke_test_count": (
                seeded_count
            ),
            "embedding_sources": (
                source_counts
            ),
            "total_input_tokens": sum(
                token_counts
            ),
            "records_with_token_count": len(
                token_counts
            ),
            "minimum_vector_norm": min(
                norms
            ),
            "maximum_vector_norm": max(
                norms
            ),
            "average_vector_norm": (
                sum(norms) / len(norms)
            ),
            "embeddings_jsonl": str(
                embeddings_path
            ),
            "embeddings_jsonl_sha256": (
                sha256_file(embeddings_path)
            ),
        }
    )

    atomic_write_json(
        manifest_path,
        manifest,
    )

    print()
    print("============================================")
    print("TITAN BATCH EMBEDDING COMPLETED")
    print("============================================")
    print(
        f"Records completed:  "
        f"{len(consolidated_records)}"
    )
    print(
        f"New model calls:    "
        f"{new_embedding_count}"
    )
    print(
        f"Reused checkpoints: "
        f"{reused_count}"
    )
    print(
        f"Smoke-test seeded:  "
        f"{seeded_count}"
    )
    print(
        f"Total input tokens: "
        f"{sum(token_counts)}"
    )
    print(
        f"Vector dimensions:  "
        f"{args.dimensions}"
    )
    print(
        f"Norm range:         "
        f"{min(norms):.8f} - "
        f"{max(norms):.8f}"
    )
    print(f"Embeddings:         {embeddings_path}")
    print(f"Manifest:           {manifest_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (ClientError, BotoCoreError) as exc:
        print(
            f"AWS embedding error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"Batch embedding failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
