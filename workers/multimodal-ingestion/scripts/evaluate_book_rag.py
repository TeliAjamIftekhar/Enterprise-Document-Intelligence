from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.generic_rag_evaluation import (
    build_answer_command,
    evaluate_answer_output,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
ANSWER_SCRIPT = (
    SCRIPT_ROOT / "answer_opensearch_rag.py"
)


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
            f"Unable to read JSON file "
            f"{path}: {error}"
        ) from error

    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected JSON object: {path}"
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
            "Run generic config-driven "
            "textbook RAG evaluation."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Book configuration JSON.",
    )

    parser.add_argument(
        "--test-cases",
        type=Path,
        required=True,
        help=(
            "Per-book RAG test-case JSON."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "RAG evaluation report path."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    test_document = load_json_object(
        args.test_cases
    )

    tests = test_document.get("tests")

    if (
        not isinstance(tests, list)
        or not tests
    ):
        raise RuntimeError(
            "RAG test-case document must "
            "contain a non-empty tests list."
        )

    book_id = str(
        test_document.get(
            "book_id",
            "",
        )
    ).strip()

    book_version = str(
        test_document.get(
            "book_version",
            "",
        )
    ).strip()

    if not book_id or not book_version:
        raise RuntimeError(
            "RAG test-case document requires "
            "book_id and book_version."
        )

    answer_root = (
        args.output.parent
        / "rag-answer-results"
    )

    answer_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for index, test_case in enumerate(
        tests,
        start=1,
    ):
        if not isinstance(
            test_case,
            dict,
        ):
            raise RuntimeError(
                "Each RAG test case must "
                "be a JSON object."
            )

        test_id = str(
            test_case.get(
                "test_id",
                f"test-{index:04d}",
            )
        ).strip()

        safe_test_id = "".join(
            character
            if (
                character.isalnum()
                or character in "-_"
            )
            else "-"
            for character in test_id
        ).strip("-")

        if not safe_test_id:
            safe_test_id = (
                f"test-{index:04d}"
            )

        question = str(
            test_case.get(
                "question",
                "",
            )
        ).strip()

        top_k = int(
            test_case.get(
                "top_k",
                5,
            )
        )

        modality = test_case.get(
            "modality"
        )

        if modality is not None:
            modality = str(
                modality
            ).strip() or None

        answer_path = (
            answer_root
            / f"{safe_test_id}.json"
        )

        command = build_answer_command(
            answer_script=ANSWER_SCRIPT,
            config_path=args.config,
            question=question,
            top_k=top_k,
            modality=modality,
            output_path=answer_path,
        )

        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )

        if answer_path.is_file():
            try:
                answer_output = (
                    load_json_object(
                        answer_path
                    )
                )
            except RuntimeError:
                answer_output = {}
        else:
            answer_output = {}

        result = evaluate_answer_output(
            test_case=test_case,
            answer_output=answer_output,
            command_return_code=(
                completed.returncode
            ),
        )

        result["answer_output_path"] = str(
            answer_path
        )
        result["command_stdout"] = (
            completed.stdout
        )
        result["command_stderr"] = (
            completed.stderr
        )

        results.append(result)

        print(
            f"[{index}/{len(tests)}] "
            f"{test_id}: "
            f"{result['status']}"
        )

    passed_count = sum(
        1
        for result in results
        if result["passed"]
    )

    failed_count = (
        len(results) - passed_count
    )

    all_tests_passed = (
        failed_count == 0
    )

    report = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "runtime_mode": "book_config",
        "config_path": str(
            args.config
        ),
        "test_cases_path": str(
            args.test_cases
        ),
        "status": (
            "PASSED"
            if all_tests_passed
            else "FAILED"
        ),
        "book_id": book_id,
        "book_version": book_version,
        "test_count": len(results),
        "passed_test_count": (
            passed_count
        ),
        "failed_test_count": (
            failed_count
        ),
        "pass_rate": (
            passed_count / len(results)
        ),
        "all_tests_passed": (
            all_tests_passed
        ),
        "tests": results,
    }

    write_json(
        args.output,
        report,
    )

    print(
        "RAG evaluation status:",
        report["status"],
    )
    print("Report:", args.output)

    return (
        0
        if all_tests_passed
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
