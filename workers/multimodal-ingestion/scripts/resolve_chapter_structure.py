from __future__ import annotations

import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


NUMBERED_ENTRY_PATTERN = re.compile(
    r"^\s*([0-9०-९]{1,2})\.\s*(.*)$"
)

NUMBER_ONLY_ROW_PATTERN = re.compile(
    r"^\s*([0-9०-९]{1,2})\.\s*$"
)

TRAILING_PAGE_PATTERN = re.compile(
    r"^(.*?)"
    r"(?:\s+)"
    r"([0-9०-९]{1,4})"
    r"\s*$"
)

TOC_DECORATIVE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:\*|★|•|●|▪|◦)+\s*"
)


DEVANAGARI_DIGITS = str.maketrans(
    "०१२३४५६७८९",
    "0123456789",
)

REJECT_TOC_TITLES = {
    "आमुख",
    "पाठ्यपुस्तक के बारे में",
    "पाठ्यपुस्‍तक के बारे में",
    "विषय सूची",
    "विषय-सूची",
    "अनुक्रमणिका",
    "contents",
    "foreword",
    "preface",
}

ZERO_WIDTH_CHARACTERS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
    "\u00ad",
}


def normalize_spaces(
    value: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def normalize_display_text(
    value: str,
) -> str:
    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    for character in (
        ZERO_WIDTH_CHARACTERS
    ):
        value = value.replace(
            character,
            "",
        )

    value = value.replace(
        " ू ",
        "ू",
    )

    value = value.replace(
        " -",
        "-",
    )

    value = value.replace(
        "- ",
        "-",
    )

    return normalize_spaces(value)


def strip_toc_decorative_prefix(
    value: str,
) -> str:
    """
    Remove TOC-only bullets or star markers
    without modifying punctuation inside titles.

    Examples:
        * चंदा मामा दूर के
        • Supplementary Reading
        ★ اضافی سبق
    """

    value = normalize_display_text(value)

    value = (
        TOC_DECORATIVE_PREFIX_PATTERN.sub(
            "",
            value,
            count=1,
        )
    )

    return normalize_display_text(value)


def normalize_match_key(
    value: str,
) -> str:
    value = normalize_display_text(
        value
    ).casefold()

    return "".join(
        character
        for character in value
        if character.isalnum()
    )


def title_similarity(
    first: str,
    second: str,
) -> float:
    first_key = normalize_match_key(
        first
    )

    second_key = normalize_match_key(
        second
    )

    if not first_key or not second_key:
        return 0.0

    return SequenceMatcher(
        None,
        first_key,
        second_key,
    ).ratio()


def parse_integer(
    value: str,
) -> int:
    return int(
        value.translate(
            DEVANAGARI_DIGITS
        )
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


def atomic_write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def contains_devanagari(
    value: str,
) -> bool:
    return any(
        "\u0900" <= character <= "\u097f"
        for character in value
    )


def reject_toc_title(
    value: str,
) -> bool:
    title = normalize_display_text(
        value
    )

    if not title:
        return True

    title_casefolded = title.casefold()

    if title_casefolded in {
        item.casefold()
        for item in REJECT_TOC_TITLES
    }:
        return True

    front_matter_markers = (
        "आमुख",
        "पाठ्यपुस्तक के बारे में",
        "पाठ्यपुस्‍तक के बारे में",
        "अनुक्रमणिका",
        "विषय सूची",
        "विषय-सूची",
    )

    if any(
        marker.casefold()
        in title_casefolded
        for marker in front_matter_markers
    ):
        return True

    if title.startswith(
        (
            "इकाई ",
            "अध्याय ",
            "पाठ ",
        )
    ):
        return True

    if "तारांकित पाठ" in title:
        return True

    if title.casefold().startswith(
        "reprint"
    ):
        return True

    if re.fullmatch(
        r"[0-9०-९ivxlcdmIVXLCDM.\-–— ]+",
        title,
    ):
        return True

    if not contains_devanagari(
        title
    ):
        return True

    if len(title) > 150:
        return True

    return False


def group_page_rows(
    page: dict[str, Any],
) -> list[dict[str, Any]]:
    width = float(
        page["width"]
    )

    midpoint = width / 2

    source_spans = [
        span
        for span in page["spans"]
        if bool(span.get("bold"))
        and float(
            span.get("font_size", 0)
        )
        >= 20
        and float(
            span["bbox"][1]
        )
        < 700
    ]

    all_rows: list[
        dict[str, Any]
    ] = []

    for column_name, (
        column_left,
        column_right,
    ) in {
        "left": (0.0, midpoint),
        "right": (midpoint, width),
    }.items():
        spans = [
            dict(span)
            for span in source_spans
            if (
                column_left
                <= float(
                    span["bbox"][0]
                )
                < column_right
            )
        ]

        spans.sort(
            key=lambda item: (
                float(item["bbox"][1]),
                float(item["bbox"][0]),
            )
        )

        rows: list[
            dict[str, Any]
        ] = []

        for span in spans:
            y_value = float(
                span["bbox"][1]
            )

            if (
                rows
                and abs(
                    y_value
                    - rows[-1]["y"]
                )
                <= 6
            ):
                rows[-1][
                    "spans"
                ].append(span)

                rows[-1]["y"] = min(
                    rows[-1]["y"],
                    y_value,
                )

            else:
                rows.append({
                    "column": column_name,
                    "y": y_value,
                    "spans": [span],
                })

        for row in rows:
            row["spans"].sort(
                key=lambda item: float(
                    item["bbox"][0]
                )
            )

            row["x"] = min(
                float(
                    span["bbox"][0]
                )
                for span in row["spans"]
            )

            row["text"] = (
                normalize_display_text(
                    " ".join(
                        str(span["text"])
                        for span
                        in row["spans"]
                    )
                )
            )

            all_rows.append(row)

    all_rows.sort(
        key=lambda row: (
            int(page["page_number"]),
            0
            if row["column"] == "left"
            else 1,
            row["y"],
        )
    )

    return all_rows


def merge_split_number_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Join a lesson number and title when a
    fixed page-midpoint split places them in
    different columns.

    Example:
        left row:  17.
        right row: हवा 94
    """

    used_indexes: set[int] = set()

    merged_rows: list[
        dict[str, Any]
    ] = []

    for index, row in enumerate(rows):
        if index in used_indexes:
            continue

        number_match = (
            NUMBER_ONLY_ROW_PATTERN.match(
                normalize_display_text(
                    row["text"]
                )
            )
        )

        if not number_match:
            merged_rows.append(
                dict(row)
            )
            used_indexes.add(index)
            continue

        number_x = float(
            row["x"]
        )

        number_y = float(
            row["y"]
        )

        possible_matches: list[
            tuple[float, int, dict[str, Any]]
        ] = []

        for candidate_index, candidate in enumerate(
            rows
        ):
            if candidate_index == index:
                continue

            if candidate_index in used_indexes:
                continue

            candidate_text = (
                normalize_display_text(
                    candidate["text"]
                )
            )

            if NUMBER_ONLY_ROW_PATTERN.match(
                candidate_text
            ):
                continue

            candidate_x = float(
                candidate["x"]
            )

            candidate_y = float(
                candidate["y"]
            )

            horizontal_gap = (
                candidate_x - number_x
            )

            vertical_gap = abs(
                candidate_y - number_y
            )

            if vertical_gap > 6:
                continue

            # A split number/title pair should
            # be immediately to the right.
            if not 20 <= horizontal_gap <= 120:
                continue

            possible_matches.append(
                (
                    horizontal_gap,
                    candidate_index,
                    candidate,
                )
            )

        if not possible_matches:
            merged_rows.append(
                dict(row)
            )
            used_indexes.add(index)
            continue

        (
            _,
            candidate_index,
            candidate,
        ) = min(
            possible_matches,
            key=lambda item: item[0],
        )

        merged_rows.append({
            "column": candidate[
                "column"
            ],
            "y": min(
                number_y,
                float(candidate["y"]),
            ),
            "x": number_x,
            "text": normalize_display_text(
                row["text"]
                + " "
                + candidate["text"]
            ),
            "spans": (
                list(row["spans"])
                + list(candidate["spans"])
            ),
        })

        used_indexes.add(index)
        used_indexes.add(
            candidate_index
        )

    merged_rows.sort(
        key=lambda row: (
            0
            if row["column"] == "left"
            else 1,
            float(row["y"]),
            float(row["x"]),
        )
    )

    return merged_rows


def combine_continuation_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    combined: list[
        dict[str, Any]
    ] = []

    index = 0

    while index < len(rows):
        current = dict(rows[index])

        current["source_rows"] = [
            current["text"]
        ]

        while index + 1 < len(rows):
            next_row = rows[
                index + 1
            ]

            if (
                next_row["column"]
                != current["column"]
            ):
                break

            vertical_gap = (
                float(next_row["y"])
                - float(current["y"])
            )

            if not (
                8
                <= vertical_gap
                <= 33
            ):
                break

            current_number_match = (
                NUMBERED_ENTRY_PATTERN.match(
                    current["text"]
                )
            )

            next_number_match = (
                NUMBERED_ENTRY_PATTERN.match(
                    next_row["text"]
                )
            )

            if next_number_match:
                break

            current_page_match = (
                TRAILING_PAGE_PATTERN.match(
                    current["text"]
                )
            )

            if current_page_match:
                break

            expected_title_x = (
                float(current["x"])
            )

            if current_number_match:
                numbered_spans = (
                    current["spans"]
                )

                if len(numbered_spans) >= 2:
                    expected_title_x = float(
                        numbered_spans[1][
                            "bbox"
                        ][0]
                    )

            if abs(
                float(next_row["x"])
                - expected_title_x
            ) > 30:
                break

            current["text"] = (
                normalize_display_text(
                    current["text"]
                    + " "
                    + next_row["text"]
                )
            )

            current[
                "source_rows"
            ].append(
                next_row["text"]
            )

            current["spans"] = (
                current["spans"]
                + next_row["spans"]
            )

            index += 1

        combined.append(current)
        index += 1

    return combined


def parse_toc_entries(
    layout: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[
        dict[str, Any]
    ] = []

    for page in layout[
        "contents_pages"
    ]:
        rows = group_page_rows(page)

        rows = merge_split_number_rows(
            rows
        )

        rows = (
            combine_continuation_rows(
                rows
            )
        )

        for row in rows:
            text = normalize_display_text(
                row["text"]
            )

            number = None

            number_match = (
                NUMBERED_ENTRY_PATTERN.match(
                    text
                )
            )

            if number_match:
                number = parse_integer(
                    number_match.group(1)
                )

                text = normalize_display_text(
                    number_match.group(2)
                )

            printed_page = None

            page_match = (
                TRAILING_PAGE_PATTERN.match(
                    text
                )
            )

            if page_match:
                possible_title = (
                    normalize_display_text(
                        page_match.group(1)
                    )
                )

                possible_page = (
                    parse_integer(
                        page_match.group(2)
                    )
                )

                if 1 <= possible_page <= 9999:
                    text = possible_title
                    printed_page = (
                        possible_page
                    )

            text = (
                strip_toc_decorative_prefix(
                    text
                )
            )

            if reject_toc_title(text):
                continue

            entries.append({
                "lesson_number": number,
                "toc_title": text,
                "printed_page": (
                    printed_page
                ),
                "contents_page": int(
                    page["page_number"]
                ),
                "column": row["column"],
                "y": round(
                    float(row["y"]),
                    2,
                ),
                "source_rows": row.get(
                    "source_rows",
                    [row["text"]],
                ),
            })

    # Remove accidental duplicates while
    # preserving TOC order.
    unique: list[
        dict[str, Any]
    ] = []

    seen: set[
        tuple[int | None, str]
    ] = set()

    for entry in entries:
        key = (
            entry["lesson_number"],
            normalize_match_key(
                entry["toc_title"]
            ),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(entry)

    return unique


def flatten_internal_candidates(
    internal: dict[str, Any],
) -> list[dict[str, Any]]:
    values: list[
        dict[str, Any]
    ] = []

    for document in internal[
        "documents"
    ]:
        if (
            document["document_id"]
            == "front-matter"
        ):
            continue

        for candidate in document[
            "title_candidates"
        ]:
            values.append({
                "document_id": (
                    document[
                        "document_id"
                    ]
                ),
                "source_filename": (
                    document[
                        "source_filename"
                    ]
                ),
                "title": normalize_display_text(
                    candidate["title"]
                ),
                "canonical_page": int(
                    candidate[
                        "canonical_page"
                    ]
                ),
                "source_page": int(
                    candidate[
                        "source_page"
                    ]
                ),
                "font_size": float(
                    candidate[
                        "font_size"
                    ]
                ),
            })

    return values


def find_best_internal_match(
    title: str,
    candidates: list[
        dict[str, Any]
    ],
) -> tuple[
    dict[str, Any] | None,
    float,
]:
    best_candidate = None
    best_score = 0.0

    for candidate in candidates:
        score = title_similarity(
            title,
            candidate["title"],
        )

        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate, best_score


def find_document_for_page(
    documents: list[dict[str, Any]],
    canonical_page: int,
) -> dict[str, Any] | None:
    for document in documents:
        start_page = int(
            document[
                "canonical_start_page"
            ]
        )

        end_page = int(
            document[
                "canonical_end_page"
            ]
        )

        if (
            start_page
            <= canonical_page
            <= end_page
        ):
            return document

    return None


def assess_toc_scope(
    *,
    toc_entries: list[dict[str, Any]],
    resolved: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Determine whether detected contents pages
    represent the complete textbook or only an
    index inside one lesson/document.
    """

    lesson_documents = [
        document
        for document in documents
        if document["document_type"]
        != "front_matter"
    ]

    lesson_document_ids = {
        str(document["document_id"])
        for document in lesson_documents
    }

    resolved_document_ids = {
        str(entry["document_id"])
        for entry in resolved
        if entry.get("document_id")
        in lesson_document_ids
    }

    toc_count = len(toc_entries)
    resolved_count = len(resolved)
    lesson_document_count = len(
        lesson_documents
    )

    resolution_ratio = (
        resolved_count / toc_count
        if toc_count
        else 0.0
    )

    document_coverage_ratio = (
        len(resolved_document_ids)
        / lesson_document_count
        if lesson_document_count
        else 0.0
    )

    low_resolution = (
        toc_count >= 3
        and resolution_ratio < 0.50
    )

    narrow_document_coverage = (
        lesson_document_count >= 3
        and document_coverage_ratio <= 0.25
    )

    concentrated_in_one_document = (
        lesson_document_count >= 3
        and len(resolved_document_ids) <= 1
        and resolved_count > 0
    )

    rejected = (
        low_resolution
        and narrow_document_coverage
        and concentrated_in_one_document
    )

    reason = None

    if rejected:
        reason = (
            "Detected contents entries resolve "
            "mostly inside one lesson document "
            "and do not represent the complete "
            "textbook."
        )

    return {
        "status": (
            "REJECTED"
            if rejected
            else "ACCEPTED"
        ),
        "reason": reason,
        "toc_entry_count": toc_count,
        "resolved_entry_count": (
            resolved_count
        ),
        "unresolved_entry_count": (
            len(unresolved)
        ),
        "lesson_document_count": (
            lesson_document_count
        ),
        "resolved_document_count": (
            len(resolved_document_ids)
        ),
        "resolution_ratio": round(
            resolution_ratio,
            4,
        ),
        "document_coverage_ratio": round(
            document_coverage_ratio,
            4,
        ),
    }


def choose_document_boundary_title(
    document: dict[str, Any],
    internal_candidates: list[
        dict[str, Any]
    ],
    *,
    chapter_number: int,
) -> tuple[
    str,
    dict[str, Any] | None,
]:
    """
    Choose the strongest available internal title
    for a chapter-wise source PDF.

    Page-one candidates are preferred, followed by
    the earliest and largest visual heading.
    """

    candidates = [
        candidate
        for candidate in internal_candidates
        if (
            candidate["document_id"]
            == document["document_id"]
            or candidate["source_filename"]
            == document["source_filename"]
        )
    ]

    candidates.sort(
        key=lambda candidate: (
            0
            if int(
                candidate["source_page"]
            ) == 1
            else 1,
            int(candidate["source_page"]),
            -float(candidate["font_size"]),
        )
    )

    if candidates:
        selected = candidates[0]

        return (
            normalize_display_text(
                selected["title"]
            ),
            selected,
        )

    return (
        f"Chapter {chapter_number}",
        None,
    )


def build_document_boundary_fallback(
    *,
    documents: list[dict[str, Any]],
    internal_candidates: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    """
    Build one chapter per non-front-matter PDF.

    This is used only when a detected TOC is proven
    to be lesson-internal rather than book-level.
    """

    lesson_documents = sorted(
        (
            document
            for document in documents
            if document["document_type"]
            != "front_matter"
        ),
        key=lambda document:
        int(document["order"]),
    )

    fallback: list[
        dict[str, Any]
    ] = []

    for chapter_number, document in enumerate(
        lesson_documents,
        start=1,
    ):
        (
            title,
            matched_internal,
        ) = choose_document_boundary_title(
            document,
            internal_candidates,
            chapter_number=chapter_number,
        )

        canonical_start = int(
            document["canonical_start_page"]
        )

        fallback.append({
            "lesson_number": chapter_number,
            "toc_title": title,
            "printed_page": None,
            "contents_page": None,
            "column": "document",
            "y": 0.0,
            "source_rows": [title],
            "canonical_start_page": (
                canonical_start
            ),
            "document_id": str(
                document["document_id"]
            ),
            "source_filename": str(
                document["source_filename"]
            ),
            "source_start_page": 1,
            "resolution_evidence": (
                "document-boundary-fallback"
            ),
            "matched_internal_title": (
                matched_internal["title"]
                if matched_internal
                else None
            ),
            "title_similarity": (
                1.0
                if matched_internal
                else None
            ),
        })

    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve textbook chapter/readings "
            "using TOC entries, printed pages "
            "and internal layout evidence."
        )
    )

    parser.add_argument(
        "--extraction-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--toc-layout",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--internal-titles",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    extraction = load_json_object(
        args.extraction_report
    )

    layout = load_json_object(
        args.toc_layout
    )

    internal = load_json_object(
        args.internal_titles
    )

    documents = extraction[
        "documents"
    ]

    front_matter_pages = sum(
        int(document["page_count"])
        for document in documents
        if document["document_type"]
        == "front_matter"
    )

    toc_entries = parse_toc_entries(
        layout
    )

    internal_candidates = (
        flatten_internal_candidates(
            internal
        )
    )

    resolved: list[
        dict[str, Any]
    ] = []

    unresolved: list[
        dict[str, Any]
    ] = []

    for entry in toc_entries:
        canonical_start = None

        evidence = None
        matched_internal = None
        similarity = None

        if entry["printed_page"] is not None:
            canonical_start = (
                front_matter_pages
                + int(
                    entry[
                        "printed_page"
                    ]
                )
            )

            evidence = (
                "toc-printed-page"
            )

            same_page_candidates = [
                candidate
                for candidate
                in internal_candidates
                if (
                    candidate[
                        "canonical_page"
                    ]
                    == canonical_start
                )
            ]

            if same_page_candidates:
                matched_internal = max(
                    same_page_candidates,
                    key=lambda candidate:
                    title_similarity(
                        entry["toc_title"],
                        candidate["title"],
                    ),
                )

                similarity = (
                    title_similarity(
                        entry["toc_title"],
                        matched_internal[
                            "title"
                        ],
                    )
                )

        else:
            (
                best_candidate,
                best_score,
            ) = find_best_internal_match(
                entry["toc_title"],
                internal_candidates,
            )

            if (
                best_candidate is not None
                and best_score >= 0.72
            ):
                canonical_start = int(
                    best_candidate[
                        "canonical_page"
                    ]
                )

                matched_internal = (
                    best_candidate
                )

                similarity = best_score

                evidence = (
                    "toc-title-layout-match"
                )

        if canonical_start is None:
            unresolved.append(entry)
            continue

        source_document = (
            find_document_for_page(
                documents,
                canonical_start,
            )
        )

        if source_document is None:
            unresolved.append({
                **entry,
                "resolution_error": (
                    "canonical page outside "
                    "source documents"
                ),
                "canonical_start_page": (
                    canonical_start
                ),
            })

            continue

        resolved.append({
            **entry,
            "canonical_start_page": (
                canonical_start
            ),
            "document_id": (
                source_document[
                    "document_id"
                ]
            ),
            "source_filename": (
                source_document[
                    "source_filename"
                ]
            ),
            "source_start_page": (
                canonical_start
                - int(
                    source_document[
                        "canonical_start_page"
                    ]
                )
                + 1
            ),
            "resolution_evidence": (
                evidence
            ),
            "matched_internal_title": (
                matched_internal[
                    "title"
                ]
                if matched_internal
                else None
            ),
            "title_similarity": (
                round(similarity, 4)
                if similarity is not None
                else None
            ),
        })

    toc_scope = assess_toc_scope(
        toc_entries=toc_entries,
        resolved=resolved,
        unresolved=unresolved,
        documents=documents,
    )

    rejected_toc_entries: list[
        dict[str, Any]
    ] = []

    resolution_mode = "book-level-toc"

    if toc_scope["status"] == "REJECTED":
        rejected_toc_entries = [
            dict(entry)
            for entry in toc_entries
        ]

        resolved = (
            build_document_boundary_fallback(
                documents=documents,
                internal_candidates=(
                    internal_candidates
                ),
            )
        )

        unresolved = []
        resolution_mode = (
            "document-boundary-fallback"
        )

    resolved.sort(
        key=lambda entry: (
            int(
                entry[
                    "canonical_start_page"
                ]
            ),
            (
                entry["lesson_number"]
                if entry[
                    "lesson_number"
                ]
                is not None
                else 999
            ),
        )
    )

    # Duplicate chapter starts indicate a
    # parser or TOC-quality problem.
    duplicate_starts = []

    starts: dict[
        int,
        list[str]
    ] = {}

    for entry in resolved:
        start = int(
            entry[
                "canonical_start_page"
            ]
        )

        starts.setdefault(
            start,
            [],
        ).append(
            entry["toc_title"]
        )

    for start, titles in starts.items():
        if len(titles) > 1:
            duplicate_starts.append({
                "canonical_page": start,
                "titles": titles,
            })

    supplementary_counter = 0

    for index, entry in enumerate(
        resolved
    ):
        source_document = (
            find_document_for_page(
                documents,
                int(
                    entry[
                        "canonical_start_page"
                    ]
                ),
            )
        )

        if source_document is None:
            raise RuntimeError(
                "Resolved source document "
                "unexpectedly missing."
            )

        document_end = int(
            source_document[
                "canonical_end_page"
            ]
        )

        if index + 1 < len(resolved):
            next_start = int(
                resolved[
                    index + 1
                ][
                    "canonical_start_page"
                ]
            )

            canonical_end = min(
                document_end,
                next_start - 1,
            )

        else:
            canonical_end = (
                document_end
            )

        source_end = (
            canonical_end
            - int(
                source_document[
                    "canonical_start_page"
                ]
            )
            + 1
        )

        entry[
            "canonical_end_page"
        ] = canonical_end

        entry[
            "source_end_page"
        ] = source_end

        if entry["lesson_number"] is not None:
            entry["chapter_id"] = (
                f"chapter-"
                f"{int(entry['lesson_number']):02d}"
            )

            entry["chapter_type"] = (
                "numbered"
            )

        else:
            supplementary_counter += 1

            entry["chapter_id"] = (
                "supplementary-"
                f"{supplementary_counter:02d}"
            )

            entry["chapter_type"] = (
                "supplementary"
            )

        entry["chapter_title"] = (
            entry["toc_title"]
        )

    numbered_entries = [
        entry
        for entry in resolved
        if entry["lesson_number"]
        is not None
    ]

    supplementary_entries = [
        entry
        for entry in resolved
        if entry["lesson_number"]
        is None
    ]

    numbered_values = sorted(
        int(entry["lesson_number"])
        for entry in numbered_entries
    )

    expected_numbered_values = (
        list(
            range(
                1,
                max(numbered_values) + 1,
            )
        )
        if numbered_values
        else []
    )

    numbered_sequence_valid = (
        numbered_values
        == expected_numbered_values
    )

    status = "READY"

    if (
        unresolved
        or duplicate_starts
        or not numbered_sequence_valid
    ):
        status = (
            "NEEDS_REVIEW"
        )

    payload = {
        "schema_version": "1.0",
        "status": status,
        "book_id": extraction[
            "book_id"
        ],
        "book_version": extraction[
            "book_version"
        ],
        "front_matter_page_count": (
            front_matter_pages
        ),
        "resolution_mode": (
            resolution_mode
        ),
        "toc_scope": toc_scope,
        "rejected_toc_entries": (
            rejected_toc_entries
        ),
        "toc_entry_count": len(
            toc_entries
        ),
        "resolved_entry_count": len(
            resolved
        ),
        "unresolved_entry_count": len(
            unresolved
        ),
        "numbered_chapter_count": len(
            numbered_entries
        ),
        "supplementary_chapter_count": (
            len(supplementary_entries)
        ),
        "numbered_sequence_valid": (
            numbered_sequence_valid
        ),
        "duplicate_starts": (
            duplicate_starts
        ),
        "unresolved_entries": (
            unresolved
        ),
        "chapters": resolved,
        "safety": {
            "aws_calls": 0,
            "s3_writes": 0,
            "bedrock_calls": 0,
            "opensearch_calls": 0,
        },
    }

    atomic_write_json(
        args.output,
        payload,
    )

    print("=" * 100)
    print("RESOLVED CHAPTER STRUCTURE")
    print("=" * 100)
    print(
        "Book ID:               ",
        extraction["book_id"],
    )
    print(
        "Front-matter pages:     ",
        front_matter_pages,
    )
    print(
        "Resolution mode:        ",
        resolution_mode,
    )
    print(
        "TOC scope:              ",
        toc_scope["status"],
    )
    print(
        "TOC entries:            ",
        len(toc_entries),
    )
    print(
        "Resolved entries:       ",
        len(resolved),
    )
    print(
        "Unresolved entries:     ",
        len(unresolved),
    )
    print(
        "Numbered chapters:      ",
        len(numbered_entries),
    )
    print(
        "Supplementary readings: ",
        len(supplementary_entries),
    )
    print(
        "Number sequence valid:  ",
        numbered_sequence_valid,
    )
    print(
        "Duplicate starts:       ",
        len(duplicate_starts),
    )
    print("Status:                 ", status)

    print()
    print("CHAPTER/READING RANGES")
    print("-" * 100)

    for entry in resolved:
        number_text = (
            str(entry["lesson_number"])
            if entry["lesson_number"]
            is not None
            else "*"
        )

        print(
            f"{number_text:>2} | "
            f"{entry['chapter_title']} | "
            f"{entry['source_filename']} | "
            f"source="
            f"{entry['source_start_page']}-"
            f"{entry['source_end_page']} | "
            f"canonical="
            f"{entry['canonical_start_page']}-"
            f"{entry['canonical_end_page']} | "
            f"{entry['resolution_evidence']}"
        )

    if unresolved:
        print()
        print("UNRESOLVED ENTRIES")
        print("-" * 100)

        for entry in unresolved:
            print(
                entry.get(
                    "lesson_number",
                    "*",
                ),
                "|",
                entry.get(
                    "toc_title"
                ),
            )

    print()
    print("Report:", args.output)
    print("AWS calls: 0")

    return 0 if status == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())