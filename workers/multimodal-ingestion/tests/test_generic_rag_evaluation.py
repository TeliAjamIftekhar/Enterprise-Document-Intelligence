from pathlib import Path

from src.generic_rag_evaluation import (
    build_answer_command,
    detect_insufficient_information,
    evaluate_answer_output,
    extract_citation_pages,
)


def test_extracts_answer_and_source_pages() -> None:
    output = {
        "answer": (
            "The answer is shown on "
            "[Page 7] and [Pages 9-10]."
        ),
        "sources": [
            {
                "source_page_numbers": [
                    12,
                ]
            }
        ],
    }

    assert extract_citation_pages(
        output
    ) == [
        7,
        9,
        10,
        12,
    ]


def test_normal_answer_passes() -> None:
    result = evaluate_answer_output(
        test_case={
            "test_id": "lesson-purpose",
            "question": (
                "What is the lesson about?"
            ),
            "expected_citation_pages": [
                7,
            ],
            "required_term_groups": [
                ["education"],
                [
                    "learning",
                    "literacy",
                ],
            ],
            "expect_insufficient": False,
        },
        answer_output={
            "answer": (
                "The lesson explains education "
                "and learning. [Page 7]"
            ),
        },
    )

    assert result["status"] == "PASSED"
    assert result["passed"] is True


def test_insufficient_guardrail_passes() -> None:
    output = {
        "answer": (
            "The provided textbook sources "
            "do not contain enough information "
            "to answer this."
        ),
        "reason": (
            "insufficient_information"
        ),
    }

    assert (
        detect_insufficient_information(
            output
        )
        is True
    )

    result = evaluate_answer_output(
        test_case={
            "test_id": "outside-book",
            "question": (
                "What is outside this book?"
            ),
            "expected_citation_pages": [],
            "required_term_groups": [],
            "expect_insufficient": True,
        },
        answer_output=output,
    )

    assert result["status"] == "PASSED"


def test_missing_required_term_fails() -> None:
    result = evaluate_answer_output(
        test_case={
            "test_id": "missing-term",
            "question": "Question",
            "expected_citation_pages": [
                4,
            ],
            "required_term_groups": [
                ["required phrase"],
            ],
            "expect_insufficient": False,
        },
        answer_output={
            "answer": (
                "Different answer. [Page 4]"
            ),
        },
    )

    assert result["status"] == "FAILED"
    assert (
        result["checks"][
            "required_terms_passed"
        ]
        is False
    )


def test_build_answer_command() -> None:
    command = build_answer_command(
        answer_script=Path(
            "answer_opensearch_rag.py"
        ),
        config_path=Path(
            "book-config.json"
        ),
        question="Test question",
        top_k=8,
        modality="figure",
        output_path=Path(
            "answer.json"
        ),
    )

    assert "--config" in command
    assert "book-config.json" in command
    assert "--query" in command
    assert "Test question" in command
    assert "--top-k" in command
    assert "8" in command
    assert "--modality" in command
    assert "figure" in command
    assert "--output" in command
    assert "answer.json" in command
