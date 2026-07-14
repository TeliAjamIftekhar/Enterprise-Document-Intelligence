from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError


REGION = "us-east-1"
BUCKET = "edi-documents-ajam-2026"

BOOK_ID = "grade-9-english-kaveri"
BOOK_VERSION = "v1"
GRADE = "grade-9"

SAMPLE_RANGE = "pages-0089-0093"

INVOCATION_ID = (
    "b4320c20-e407-4d7b-905f-7f440e843941"
)

LOCAL_ROOT = Path(
    "data/multimodal-output"
    f"/{BOOK_ID}/{BOOK_VERSION}"
    "/opensearch-serverless"
)

S3_PREFIX = (
    f"derived-artifacts/{GRADE}/"
    f"{BOOK_ID}/{BOOK_VERSION}/"
    "retrieval/samples/"
    f"{SAMPLE_RANGE}/"
    f"{INVOCATION_ID}/"
    "opensearch-validation"
)

LOCAL_MANIFEST_PATH = (
    LOCAL_ROOT
    / "opensearch-validation-publication-manifest.json"
)


ARTIFACTS: dict[str, Path] = {
    "preflight-report.json": (
        LOCAL_ROOT
        / "preflight-report.json"
    ),
    "nextgen-capability-report.json": (
        LOCAL_ROOT
        / "nextgen-capability-report.json"
    ),
    "provisioning-report.json": (
        LOCAL_ROOT
        / "provisioning-report.json"
    ),
    "index-provisioning-report.json": (
        LOCAL_ROOT
        / "index-provisioning-report.json"
    ),
    "bulk/bulk-preparation-report.json": (
        LOCAL_ROOT
        / "bulk"
        / SAMPLE_RANGE
        / "bulk-preparation-report.json"
    ),
    "bulk/bulk-upload-report.json": (
        LOCAL_ROOT
        / "bulk"
        / SAMPLE_RANGE
        / "upload"
        / "bulk-upload-report.json"
    ),
    "evaluation/vector-retrieval-evaluation-report.json": (
        LOCAL_ROOT
        / "vector-retrieval-evaluation-report.json"
    ),
    "evaluation/hybrid-retrieval-evaluation-report.json": (
        LOCAL_ROOT
        / "hybrid-retrieval-evaluation-report.json"
    ),
    "evaluation/rag-evaluation-report.json": (
        LOCAL_ROOT
        / "rag-evaluation-report.json"
    ),
    "examples/mirror-work-rag-aligned.json": (
        LOCAL_ROOT
        / "mirror-work-rag-aligned.json"
    ),
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(
            1024 * 1024
        ):
            digest.update(chunk)

    return digest.hexdigest()


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
            default=str,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        path,
    )


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return value


def validate_required_reports() -> dict[str, Any]:
    missing = [
        str(path)
        for path in ARTIFACTS.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Required validation artifacts "
            "are missing:\n- "
            + "\n- ".join(missing)
        )

    vector_report = load_json_object(
        ARTIFACTS[
            "evaluation/"
            "vector-retrieval-evaluation-report.json"
        ]
    )

    hybrid_report = load_json_object(
        ARTIFACTS[
            "evaluation/"
            "hybrid-retrieval-evaluation-report.json"
        ]
    )

    rag_report = load_json_object(
        ARTIFACTS[
            "evaluation/"
            "rag-evaluation-report.json"
        ]
    )

    bulk_report = load_json_object(
        ARTIFACTS[
            "bulk/bulk-upload-report.json"
        ]
    )

    checks = {
        "bulk_upload_completed": (
            bulk_report.get("status")
            == "COMPLETED"
        ),
        "bulk_final_count_40": (
            bulk_report.get("final_count")
            == 40
        ),
        "bulk_failures_zero": (
            bulk_report.get(
                "bulk_result",
                {},
            ).get("failure_count")
            == 0
        ),
        "vector_all_tests_passed": (
            vector_report.get(
                "all_tests_passed"
            )
            is True
        ),
        "vector_top_1_accuracy_1": (
            vector_report.get(
                "top_1_accuracy"
            )
            == 1.0
        ),
        "hybrid_all_tests_passed": (
            hybrid_report.get(
                "all_tests_passed"
            )
            is True
        ),
        "hybrid_top_1_accuracy_1": (
            hybrid_report.get(
                "top_1_accuracy"
            )
            == 1.0
        ),
        "rag_all_tests_passed": (
            rag_report.get(
                "all_tests_passed"
            )
            is True
        ),
        "rag_pass_rate_1": (
            rag_report.get("pass_rate")
            == 1.0
        ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed_checks:
        raise RuntimeError(
            "Validation reports are not ready "
            "for publication:\n- "
            + "\n- ".join(failed_checks)
        )

    return {
        "checks": checks,
        "bulk_document_count": (
            bulk_report.get("final_count")
        ),
        "vector_test_count": (
            vector_report.get("test_count")
        ),
        "vector_top_1_accuracy": (
            vector_report.get(
                "top_1_accuracy"
            )
        ),
        "hybrid_test_count": (
            hybrid_report.get("test_count")
        ),
        "hybrid_top_1_accuracy": (
            hybrid_report.get(
                "top_1_accuracy"
            )
        ),
        "rag_test_count": (
            rag_report.get("test_count")
        ),
        "rag_pass_rate": (
            rag_report.get("pass_rate")
        ),
    }


def upload_artifact(
    s3_client: Any,
    relative_key: str,
    local_path: Path,
) -> dict[str, Any]:
    sha256 = sha256_file(
        local_path
    )

    size_bytes = local_path.stat().st_size

    s3_key = (
        f"{S3_PREFIX}/"
        f"{relative_key}"
    )

    s3_client.upload_file(
        Filename=str(local_path),
        Bucket=BUCKET,
        Key=s3_key,
        ExtraArgs={
            "ContentType": (
                "application/json"
            ),
            "ServerSideEncryption": "AES256",
            "Metadata": {
                "sha256": sha256,
                "book-id": BOOK_ID,
                "book-version": BOOK_VERSION,
                "sample-range": SAMPLE_RANGE,
            },
        },
    )

    head = s3_client.head_object(
        Bucket=BUCKET,
        Key=s3_key,
    )

    remote_size = int(
        head["ContentLength"]
    )

    remote_sha256 = (
        head.get("Metadata", {})
        .get("sha256")
    )

    if remote_size != size_bytes:
        raise RuntimeError(
            f"Uploaded size mismatch for "
            f"{relative_key}: "
            f"local={size_bytes}, "
            f"remote={remote_size}"
        )

    if remote_sha256 != sha256:
        raise RuntimeError(
            f"Uploaded checksum metadata "
            f"mismatch for {relative_key}."
        )

    return {
        "relative_key": relative_key,
        "local_path": str(local_path),
        "s3_uri": (
            f"s3://{BUCKET}/{s3_key}"
        ),
        "size_bytes": size_bytes,
        "sha256": sha256,
        "etag": str(
            head.get("ETag", "")
        ).strip('"'),
        "version_id": head.get(
            "VersionId"
        ),
        "server_side_encryption": (
            head.get(
                "ServerSideEncryption"
            )
        ),
        "verified": True,
    }


def main() -> int:
    validation_summary = (
        validate_required_reports()
    )

    s3_client = boto3.client(
        "s3",
        region_name=REGION,
    )

    print(
        "============================================"
    )
    print(
        "PUBLISH OPENSEARCH VALIDATION ARTIFACTS"
    )
    print(
        "============================================"
    )
    print(f"Region:       {REGION}")
    print(f"Bucket:       {BUCKET}")
    print(f"Book:         {BOOK_ID}")
    print(f"Version:      {BOOK_VERSION}")
    print(f"Sample:       {SAMPLE_RANGE}")
    print(
        f"Invocation:   {INVOCATION_ID}"
    )
    print(f"S3 prefix:    {S3_PREFIX}")
    print(
        f"Artifacts:    {len(ARTIFACTS)}"
    )
    print()

    uploaded_artifacts: list[
        dict[str, Any]
    ] = []

    for index, (
        relative_key,
        local_path,
    ) in enumerate(
        ARTIFACTS.items(),
        start=1,
    ):
        print(
            f"[{index}/{len(ARTIFACTS)}] "
            f"{relative_key}"
        )

        artifact = upload_artifact(
            s3_client=s3_client,
            relative_key=relative_key,
            local_path=local_path,
        )

        uploaded_artifacts.append(
            artifact
        )

        print(
            f"    VERIFIED | "
            f"{artifact['size_bytes']:,} bytes"
        )

    manifest = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": "PUBLISHED",
        "region": REGION,
        "bucket": BUCKET,
        "book_id": BOOK_ID,
        "book_version": BOOK_VERSION,
        "grade": GRADE,
        "sample_range": SAMPLE_RANGE,
        "bda_invocation_id": (
            INVOCATION_ID
        ),
        "s3_prefix": (
            f"s3://{BUCKET}/{S3_PREFIX}/"
        ),
        "validation_summary": (
            validation_summary
        ),
        "artifact_count": len(
            uploaded_artifacts
        ),
        "artifacts": uploaded_artifacts,
    }

    atomic_write_json(
        LOCAL_MANIFEST_PATH,
        manifest,
    )

    manifest_sha256 = sha256_file(
        LOCAL_MANIFEST_PATH
    )

    manifest_key = (
        f"{S3_PREFIX}/"
        "opensearch-validation-"
        "publication-manifest.json"
    )

    s3_client.upload_file(
        Filename=str(
            LOCAL_MANIFEST_PATH
        ),
        Bucket=BUCKET,
        Key=manifest_key,
        ExtraArgs={
            "ContentType": (
                "application/json"
            ),
            "ServerSideEncryption": "AES256",
            "Metadata": {
                "sha256": (
                    manifest_sha256
                ),
                "book-id": BOOK_ID,
                "book-version": (
                    BOOK_VERSION
                ),
                "publication-status": (
                    "published"
                ),
            },
        },
    )

    manifest_head = (
        s3_client.head_object(
            Bucket=BUCKET,
            Key=manifest_key,
        )
    )

    if (
        manifest_head.get(
            "Metadata",
            {},
        ).get("sha256")
        != manifest_sha256
    ):
        raise RuntimeError(
            "Publication manifest checksum "
            "verification failed."
        )

    print()
    print(
        "============================================"
    )
    print(
        "PUBLICATION RESULT"
    )
    print(
        "============================================"
    )
    print("Status:          PUBLISHED")
    print(
        f"Artifacts:       "
        f"{len(uploaded_artifacts)}"
    )
    print(
        "Verified:        "
        f"{sum(1 for item in uploaded_artifacts if item['verified'])}"
    )
    print(
        "Bulk documents:  "
        f"{validation_summary['bulk_document_count']}"
    )
    print(
        "Vector tests:    "
        f"{validation_summary['vector_test_count']}/"
        f"{validation_summary['vector_test_count']}"
    )
    print(
        "Hybrid tests:    "
        f"{validation_summary['hybrid_test_count']}/"
        f"{validation_summary['hybrid_test_count']}"
    )
    print(
        "RAG tests:       "
        f"{validation_summary['rag_test_count']}/"
        f"{validation_summary['rag_test_count']}"
    )
    print(
        f"Manifest SHA256: "
        f"{manifest_sha256}"
    )
    print(
        f"Local manifest:  "
        f"{LOCAL_MANIFEST_PATH}"
    )
    print(
        f"S3 manifest:     "
        f"s3://{BUCKET}/{manifest_key}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (
        ClientError,
        BotoCoreError,
    ) as exc:
        print(
            f"AWS publication error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"Publication failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
