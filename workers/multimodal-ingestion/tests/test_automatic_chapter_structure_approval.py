from pathlib import Path

from run_all_textbooks import (
    append_chapter_structure_argument,
    chapter_structure_path_for_book,
    paths_for_book,
)


def test_tracked_chapter_structure_path():
    expected = Path(
        "workers/multimodal-ingestion/config/"
        "chapter-structures/"
        "grade-6-mathematics-"
        "ganita-prakash-v1.json"
    )

    assert chapter_structure_path_for_book(
        "grade-6-mathematics-ganita-prakash",
        "v1",
    ) == expected

    assert paths_for_book(
        "grade-6-mathematics-ganita-prakash",
        "v1",
    )["chapter_structure"] == expected


def test_existing_approval_is_added(
    tmp_path,
):
    approval = tmp_path / "approval.json"
    approval.write_text(
        "{}",
        encoding="utf-8",
    )

    command = ["python", "generator.py"]

    result = append_chapter_structure_argument(
        command,
        approval,
    )

    assert result == [
        "python",
        "generator.py",
        "--chapter-structure",
        str(approval),
    ]


def test_missing_approval_is_not_added(
    tmp_path,
):
    command = ["python", "generator.py"]

    result = append_chapter_structure_argument(
        command,
        tmp_path / "missing.json",
    )

    assert result == command
