from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any


PAGE_PATTERN = re.compile(
    r"\[\s*Pages?\s+([0-9,\-\s]+)\]",
    flags=re.IGNORECASE,
)

INSUFFICIENT_PHRASES = (
    "do not contain enough information",
    "does not contain enough information",
    "not enough information",
    "insufficient information",
    "cannot be answered from the provided",
    "cannot answer from the provided",
)


def normalize_text(
    value: Any,
) -> str:
    return " ".join(
        str(value).casefold().split()
    )


def _collect_integer_pages(
    value: Any,
) -> set[int]:
    pages: set[int] = set()

    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    ):
        pages.add(value)

    elif isinstance(value, str):
        for token in re.findall(
            r"\d+",
            value,
        ):
            number = int(token)

            if number > 0:
                pages.add(number)

    elif isinstance(value, list):
        for item in value:
            pages.update(
                _collect_integer_pages(item)
            )

    return pages


def extract_citation_pages(
    answer_output: dict[str, Any],
) -> list[int]:
    pages: set[int] = set()

    answer = str(
        answer_output.get(
            "answer",
            "",
        )
    )

    for match in PAGE_PATTERN.finditer(
        answer
    ):
        value = match.group(1)

        for token in value.split(","):
            token = token.strip()

            if "-" in token:
                start_text, end_text = (
                    token.split("-", 1)
                )

                if (
                    start_text.strip().isdigit()
                    and end_text.strip().isdigit()
                ):
                    start = int(
                        start_text.strip()
                    )
                    end = int(
                        end_text.strip()
                    )

                    if (
                        start > 0
                        and end >= start
                        and end - start <= 50
                    ):
                        pages.update(
                            range(start, end + 1)
                        )

            elif token.isdigit():
                pages.add(int(token))

    exact_page_keys = {
        "citation_pages",
        "source_page_numbers",
        "page_numbers",
        "allowed_pages",
        "allowed_citation_pages",
    }

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized_key = (
                    str(key).casefold()
                )

                if (
                    normalized_key
                    in exact_page_keys
                    or (
                        "citation" in normalized_key
                        and "page" in normalized_key
                    )
                ):
                    pages.update(
                        _collect_integer_pages(
                            item
                        )
                    )

                walk(item)

        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(answer_output)

    return sorted(
        page
        for page in pages
        if page > 0
    )


def detect_insufficient_information(
    answer_output: dict[str, Any],
) -> bool:
    for key in (
        "insufficient_detected",
        "insufficient_information",
    ):
        if answer_output.get(key) is True:
            return True

    reason = normalize_text(
        answer_output.get(
            "reason",
            "",
        )
    )

    if reason == "insufficient_information":
        return True

    answer = normalize_text(
        answer_output.get(
            "answer",
            "",
        )
    )

    return any(
        phrase in answer
        for phrase in INSUFFICIENT_PHRASES
    )


def normalize_required_term_groups(
    value: Any,
) -> list[list[str]]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(
            "required_term_groups must be a list."
        )

    groups: list[list[str]] = []

    for raw_group in value:
        if isinstance(raw_group, str):
            raw_group = [raw_group]

        if not isinstance(raw_group, list):
            raise ValueError(
                "Each required term group must "
                "be a string or list of strings."
            )

        group = [
            str(term).strip()
            for term in raw_group
            if str(term).strip()
        ]

        if not group:
            raise ValueError(
                "Required term groups cannot "
                "be empty."
            )

        groups.append(group)

    return groups


def evaluate_answer_output(
    *,
    test_case: dict[str, Any],
    answer_output: dict[str, Any],
    command_return_code: int = 0,
) -> dict[str, Any]:
    test_id = str(
        test_case.get(
            "test_id",
            "",
        )
    ).strip()

    question = str(
        test_case.get(
            "question",
            "",
        )
    ).strip()

    if not test_id:
        raise ValueError(
            "RAG test case requires test_id."
        )

    if not question:
        raise ValueError(
            "RAG test case requires question."
        )

    expected_pages = sorted(
        {
            int(page)
            for page in test_case.get(
                "expected_citation_pages",
                [],
            )
            if (
                isinstance(page, int)
                and not isinstance(page, bool)
                and page > 0
            )
        }
    )

    required_groups = (
        normalize_required_term_groups(
            test_case.get(
                "required_term_groups",
                [],
            )
        )
    )

    expect_insufficient = bool(
        test_case.get(
            "expect_insufficient",
            False,
        )
    )

    answer = str(
        answer_output.get(
            "answer",
            "",
        )
    ).strip()

    normalized_answer = normalize_text(
        answer
    )

    actual_pages = (
        extract_citation_pages(
            answer_output
        )
    )

    insufficient_detected = (
        detect_insufficient_information(
            answer_output
        )
    )

    group_results = []

    for group in required_groups:
        matched_terms = [
            term
            for term in group
            if normalize_text(term)
            in normalized_answer
        ]

        group_results.append(
            {
                "alternatives": group,
                "passed": bool(
                    matched_terms
                ),
                "matched_terms": (
                    matched_terms
                ),
            }
        )

    required_terms_passed = all(
        result["passed"]
        for result in group_results
    )

    citation_pages_passed = (
        all(
            page in actual_pages
            for page in expected_pages
        )
        if not expect_insufficient
        else True
    )

    insufficient_behavior_passed = (
        insufficient_detected
        == expect_insufficient
    )

    answer_present = bool(answer)

    command_passed = (
        command_return_code == 0
    )

    checks = {
        "command_passed": command_passed,
        "answer_present": answer_present,
        "required_terms_passed": (
            required_terms_passed
        ),
        "citation_pages_passed": (
            citation_pages_passed
        ),
        "insufficient_behavior_passed": (
            insufficient_behavior_passed
        ),
    }

    passed = all(checks.values())

    return {
        "test_id": test_id,
        "question": question,
        "status": (
            "PASSED"
            if passed
            else "FAILED"
        ),
        "passed": passed,
        "checks": checks,
        "answer": answer,
        "expected_citation_pages": (
            expected_pages
        ),
        "actual_citation_pages": (
            actual_pages
        ),
        "required_term_groups": (
            group_results
        ),
        "expect_insufficient": (
            expect_insufficient
        ),
        "insufficient_detected": (
            insufficient_detected
        ),
        "command_return_code": (
            command_return_code
        ),
    }


def build_answer_command(
    *,
    answer_script: Path,
    config_path: Path,
    question: str,
    top_k: int,
    output_path: Path,
    modality: str | None = None,
) -> list[str]:
    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    command = [
        sys.executable,
        str(answer_script),
        "--config",
        str(config_path),
        "--query",
        question,
        "--top-k",
        str(top_k),
        "--output",
        str(output_path),
    ]

    if modality:
        command.extend(
            [
                "--modality",
                modality,
            ]
        )

    return command
