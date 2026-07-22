from resolve_chapter_structure import (
    merge_split_number_rows,
    reject_toc_title,
)


def test_merge_number_split_across_midpoint() -> None:
    rows = [
        {
            "column": "left",
            "y": 407.52,
            "x": 286.84,
            "text": "17.",
            "spans": [{"text": "17."}],
        },
        {
            "column": "right",
            "y": 407.52,
            "x": 322.83,
            "text": "हवा 94",
            "spans": [
                {"text": "हवा"},
                {"text": "94"},
            ],
        },
    ]

    merged = merge_split_number_rows(
        rows
    )

    assert len(merged) == 1
    assert merged[0]["text"] == (
        "17. हवा 94"
    )


def test_reject_combined_front_matter() -> None:
    assert reject_toc_title(
        "आमुख पाठ्यपुस्तक के बारे में"
    )


def test_do_not_reject_lesson_title() -> None:
    assert not reject_toc_title(
        "कितनी प्यारी है ये दुनिया"
    )
