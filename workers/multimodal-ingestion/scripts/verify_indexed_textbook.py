from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.final_verification import (
    evaluate_final_verification,
)


MARKER_NAME = "PIPELINE_VERIFIED"
REPORT_NAME = "final-verification-report.json"


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(
            f"Unable to read JSON report "
            f"{path}: {error}"
        ) from error

    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected a JSON object: {path}"
        )

    return value


def write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate generic per-book "
            "OpenSearch and RAG evidence."
        )
    )

    parser.add_argument(
        "--book-id",
        required=True,
    )
    parser.add_argument(
        "--book-version",
        required=True,
    )
    parser.add_argument(
        "--bulk-upload-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--vector-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--hybrid-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--rag-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    report = evaluate_final_verification(
        book_id=args.book_id,
        book_version=args.book_version,
        bulk_upload_report=(
            load_json_object(
                args.bulk_upload_report
            )
        ),
        vector_report=load_json_object(
            args.vector_report
        ),
        hybrid_report=load_json_object(
            args.hybrid_report
        ),
        rag_report=load_json_object(
            args.rag_report
        ),
    )

    report["generated_at"] = utc_now()

    report["source_reports"] = {
        "bulk_upload": str(
            args.bulk_upload_report
        ),
        "vector": str(
            args.vector_report
        ),
        "hybrid": str(
            args.hybrid_report
        ),
        "rag": str(
            args.rag_report
        ),
    }

    report_path = (
        args.output_dir / REPORT_NAME
    )
    marker_path = (
        args.output_dir / MARKER_NAME
    )

    write_json(
        report_path,
        report,
    )

    if report["status"] != "VERIFIED":
        marker_path.unlink(
            missing_ok=True
        )

        print(
            "Final verification failed:",
            ", ".join(
                report["failed_checks"]
            ),
        )
        print("Report:", report_path)
        return 1

    marker_path.write_text(
        "VERIFIED\n",
        encoding="utf-8",
    )

    print("Final verification: VERIFIED")
    print("Report:", report_path)
    print("Marker:", marker_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
