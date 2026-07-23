

from __future__ import annotations

import argparse
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import fitz

from src.book_config import BookConfig
from src.chapter_manifest import ChapterManifest


CONFIG_ROOT = Path(
    "workers/multimodal-ingestion/config/books"
)

MANIFEST_ROOT = (
    CONFIG_ROOT / "manifests"
)

REPORT_ROOT = Path(
    "data/textbook-automation/"
    "config-generation"
)

BDA_PROJECT_ARN = (
    "arn:aws:bedrock:us-east-1:"
    "334590195171:"
    "data-automation-project/"
    "1894a0411c15"
)

BDA_PROFILE_ARN = (
    "arn:aws:bedrock:us-east-1:"
    "334590195171:"
    "data-automation-profile/"
    "us.data-automation-v1"
)

OPENSEARCH_ENDPOINT = (
    "https://"
    "kqjqddn0b5gmcfvgsd2e."
    "aoss.us-east-1.on.aws"
)


HEADING_PATTERN = re.compile(
    r"^(?:"
    r"chapter|unit|lesson|"
    r"अध्याय|पाठ|इकाई|धडा|एकक|"
    r"अध्यायः|पाठः|"
    r"باب|سبق"
    r")"
    r"\s*"
    r"[0-9०-९IVXivx]*"
    r"\s*"
    r"[:.\-–—]?"
    r"\s*"
    r"(.*)$",
    re.IGNORECASE,
)

BOILERPLATE_PATTERNS = [
    re.compile(
        pattern,
        re.IGNORECASE,
    )
    for pattern in [
        r"^isbn\b",
        r"^reprint\b",
        r"^first edition\b",
        r"^published\b",
        r"^publication\b",
        r"^copyright\b",
        r"^all rights reserved\b",
        r"^national council\b",
        r"^ncert\b",
        r"^printed at\b",
        r"^foreword$",
        r"^preface$",
        r"^contents?$",
        r"^acknowledgements?$",
        r"^textbook development",
        r"^constitution of india",
        r"^भारत का संविधान",
    ]
]


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_json_object(
    path: Path,
) -> dict[str, Any]:
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


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
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
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def find_registry_book(
    registry: dict[str, Any],
    book_id: str,
) -> dict[str, Any]:
    books = registry.get("books")

    if not isinstance(books, list):
        raise ValueError(
            "Registry field 'books' must "
            "be a list."
        )

    matches = [
        book
        for book in books
        if isinstance(book, dict)
        and book.get("book_id") == book_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one registry "
            f"record for {book_id}; "
            f"found {len(matches)}."
        )

    return dict(matches[0])


def normalize_line(
    value: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def is_boilerplate(
    value: str,
) -> bool:
    return any(
        pattern.search(value)
        for pattern in BOILERPLATE_PATTERNS
    )


def is_title_candidate(
    value: str,
) -> bool:
    if len(value) < 3:
        return False

    if len(value) > 160:
        return False

    if value.isdigit():
        return False

    if re.fullmatch(
        r"[0-9०-९IVXivx .\-–—]+",
        value,
    ):
        return False

    if "http://" in value.casefold():
        return False

    if "https://" in value.casefold():
        return False

    if "www." in value.casefold():
        return False

    if "@" in value:
        return False

    if is_boilerplate(value):
        return False

    alphanumeric = sum(
        character.isalnum()
        for character in value
    )

    if alphanumeric < 3:
        return False

    digit_count = sum(
        character.isdigit()
        for character in value
    )

    if (
        alphanumeric > 0
        and digit_count / alphanumeric > 0.6
    ):
        return False

    return True


def extract_sample_lines(
    archive: ZipFile,
    archive_member: str,
    *,
    maximum_pages: int = 4,
) -> list[str]:
    with tempfile.NamedTemporaryFile(
        suffix=".pdf"
    ) as temporary:
        with archive.open(
            archive_member,
            "r",
        ) as source:
            while True:
                chunk = source.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                temporary.write(chunk)

        temporary.flush()

        with fitz.open(
            temporary.name
        ) as document:
            line_values: list[str] = []

            for page_number in range(
                min(
                    maximum_pages,
                    document.page_count,
                )
            ):
                text = document[
                    page_number
                ].get_text("text")

                for raw_line in (
                    text.splitlines()
                ):
                    line = normalize_line(
                        raw_line
                    )

                    if line:
                        line_values.append(
                            line
                        )

            return line_values


LAYOUT_SECTION_LABELS = {
    "चित्र और बातचीत",
    "चित्र और बातचचीत",
    "सुनें कहानी",
    "मिलकर पढ़िए",
    "मिलकर पढ़िए",
    "आनंदमयी कविता",
    "आनंददमयी कविता",
    "खेल गीत",
    "बातचीत के लिए",
    "बातचचीत के लिए",
    "शब्दों का खेल",
    "शबदों का खेल",
    "झटपट कहिए",
    "चित्रकारी",
    "खोजें-जानें",
    "रंग भरिए",
    "शिक्षण-संकेत",
    "शिक्षण संकेत",
    "read together",
    "listen to the story",
    "picture and conversation",
    "joyful poem",
    "word game",
}


def layout_span_is_bold(
    span: dict[str, Any],
) -> bool:
    font = str(
        span.get("font", "")
    ).casefold()

    return any(
        marker in font
        for marker in (
            "bold",
            "black",
            "heavy",
            "semibold",
            "demi",
        )
    )


def is_layout_section_label(
    value: str,
) -> bool:
    normalized = normalize_line(
        value
    )

    casefolded = (
        normalized.casefold()
    )

    if normalized in LAYOUT_SECTION_LABELS:
        return True

    if casefolded in {
        item.casefold()
        for item in LAYOUT_SECTION_LABELS
    }:
        return True

    if re.match(
        r"^(?:इकाई|अध्याय|पाठ)"
        r"\s*[0-9०-९IVXivx]*"
        r"\s*[:.\-–—]?",
        normalized,
    ):
        return True

    if re.match(
        r"^(?:unit|chapter|lesson)"
        r"\s*[0-9IVXivx]*"
        r"\s*[:.\-–—]?",
        casefolded,
    ):
        return True

    return False


def valid_layout_title_piece(
    *,
    text: str,
    font_size: float,
    bold: bool,
    y0: float,
    page_height: float,
) -> bool:
    if not bold:
        return False

    if font_size < 27:
        return False

    if y0 < 0:
        return False

    if y0 > page_height * 0.58:
        return False

    if len(text) < 2 or len(text) > 120:
        return False

    if re.fullmatch(
        r"[0-9०-९IVXivx .)\-–—]+",
        text,
    ):
        return False

    if is_layout_section_label(text):
        return False

    if text.casefold().startswith(
        "reprint"
    ):
        return False

    if text.startswith(
        (
            "शिक्षण-संकेत",
            "शिक्षण संकेत",
        )
    ):
        return False

    return True


def extract_layout_title(
    archive: ZipFile,
    archive_member: str,
) -> dict[str, Any] | None:
    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
    ) as temporary:
        with archive.open(
            archive_member,
            "r",
        ) as source:
            while True:
                chunk = source.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                temporary.write(chunk)

        temporary.flush()

        with fitz.open(
            temporary.name
        ) as document:
            if document.page_count < 1:
                return None

            page = document[0]

            payload = page.get_text(
                "dict",
                sort=True,
            )

            candidates: list[
                dict[str, Any]
            ] = []

            for block in payload.get(
                "blocks",
                [],
            ):
                if block.get("type") != 0:
                    continue

                for line in block.get(
                    "lines",
                    [],
                ):
                    for span in line.get(
                        "spans",
                        [],
                    ):
                        value = normalize_line(
                            str(
                                span.get(
                                    "text",
                                    "",
                                )
                            )
                        )

                        if not value:
                            continue

                        bbox = [
                            float(item)
                            for item in span.get(
                                "bbox",
                                [0, 0, 0, 0],
                            )
                        ]

                        font_size = float(
                            span.get(
                                "size",
                                0,
                            )
                        )

                        bold = (
                            layout_span_is_bold(
                                span
                            )
                        )

                        if not valid_layout_title_piece(
                            text=value,
                            font_size=font_size,
                            bold=bold,
                            y0=bbox[1],
                            page_height=(
                                page.rect.height
                            ),
                        ):
                            continue

                        candidates.append({
                            "text": value,
                            "font_size": (
                                font_size
                            ),
                            "bbox": bbox,
                        })

            if not candidates:
                return None

            maximum_size = max(
                item["font_size"]
                for item in candidates
            )

            # Keep only the largest title-sized
            # pieces. This removes section labels.
            largest = [
                item
                for item in candidates
                if item["font_size"]
                >= maximum_size - 1.0
            ]

            largest.sort(
                key=lambda item: (
                    item["bbox"][1],
                    item["bbox"][0],
                )
            )

            first = largest[0]
            selected = [first]

            previous_y = float(
                first["bbox"][1]
            )

            first_center = (
                float(first["bbox"][0])
                + float(first["bbox"][2])
            ) / 2

            for item in largest[1:]:
                current_y = float(
                    item["bbox"][1]
                )

                current_center = (
                    float(item["bbox"][0])
                    + float(item["bbox"][2])
                ) / 2

                maximum_vertical_gap = max(
                    65.0,
                    maximum_size * 1.9,
                )

                if (
                    current_y - previous_y
                    > maximum_vertical_gap
                ):
                    break

                if abs(
                    current_center
                    - first_center
                ) > page.rect.width * 0.32:
                    break

                selected.append(item)
                previous_y = current_y

            title = normalize_line(
                " ".join(
                    str(item["text"])
                    for item in selected
                )
            )

            if not title:
                return None

            return {
                "title": title,
                "confidence": "high",
                "source": (
                    "first-page-layout-title"
                ),
                "candidates": [
                    str(item["text"])
                    for item in largest[:10]
                ],
            }


TOC_NUMBER_PATTERN = re.compile(
    r"^\s*([0-9०-९]{1,2})\.\s*(.*?)\s*$"
)

DEVANAGARI_DIGITS = str.maketrans(
    "०१२३४५६७८९",
    "0123456789",
)


def parse_toc_number(
    value: str,
) -> int:
    return int(
        value.translate(
            DEVANAGARI_DIGITS
        )
    )


def is_toc_page_number(
    value: str,
) -> bool:
    normalized = normalize_line(
        value
    )

    if re.fullmatch(
        r"[0-9०-९]{1,4}",
        normalized,
    ):
        return True

    if re.fullmatch(
        r"[ivxlcdmIVXLCDM]{1,8}",
        normalized,
    ):
        return True

    return False


TOC_TRAILING_PAGE_PATTERN = re.compile(
    r"\s+"
    r"(?:"
    r"[0-9०-९]{1,4}"
    r"|"
    r"[ivxlcdmIVXLCDM]{1,8}"
    r")"
    r"\s*$"
)


def normalize_toc_title(
    parts: list[str],
) -> str:
    """
    Join TOC title spans and remove a printed
    page number that PyMuPDF may merge into the
    same span as the lesson title.
    """

    value = normalize_line(
        " ".join(parts)
    )

    previous = None

    while value and value != previous:
        previous = value

        value = normalize_line(
            TOC_TRAILING_PAGE_PATTERN.sub(
                "",
                value,
            )
        )

    return value


def extract_numbered_toc_entries(
    archive: ZipFile,
    archive_member: str,
) -> dict[int, str]:
    """
    Extract numbered textbook contents entries.

    The parser uses layout positions instead of
    ordinary text order, so it supports:
    - two-column contents pages;
    - titles split across multiple spans;
    - titles continued on the next line;
    - number and title inside one span.
    """

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
    ) as temporary:
        with archive.open(
            archive_member,
            "r",
        ) as source:
            while True:
                chunk = source.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                temporary.write(chunk)

        temporary.flush()

        entries: dict[int, str] = {}

        with fitz.open(
            temporary.name
        ) as document:
            # Contents pages are normally near
            # the end of the front matter.
            first_page = max(
                0,
                document.page_count - 8,
            )

            for page_number in range(
                first_page,
                document.page_count,
            ):
                page = document[
                    page_number
                ]

                payload = page.get_text(
                    "dict",
                    sort=True,
                )

                spans: list[
                    dict[str, Any]
                ] = []

                for block in payload.get(
                    "blocks",
                    [],
                ):
                    if block.get("type") != 0:
                        continue

                    for line in block.get(
                        "lines",
                        [],
                    ):
                        for span in line.get(
                            "spans",
                            [],
                        ):
                            text = normalize_line(
                                str(
                                    span.get(
                                        "text",
                                        "",
                                    )
                                )
                            )

                            if not text:
                                continue

                            bbox = [
                                float(value)
                                for value
                                in span.get(
                                    "bbox",
                                    [0, 0, 0, 0],
                                )
                            ]

                            spans.append({
                                "text": text,
                                "bbox": bbox,
                                "font_size": float(
                                    span.get(
                                        "size",
                                        0,
                                    )
                                ),
                                "bold": (
                                    layout_span_is_bold(
                                        span
                                    )
                                ),
                            })

                spans.sort(
                    key=lambda item: (
                        item["bbox"][1],
                        item["bbox"][0],
                    )
                )

                numbered_spans = []

                for item in spans:
                    match = (
                        TOC_NUMBER_PATTERN.match(
                            item["text"]
                        )
                    )

                    if not match:
                        continue

                    if not item["bold"]:
                        continue

                    if item["font_size"] < 20:
                        continue

                    numbered_spans.append(
                        (item, match)
                    )

                for item, match in numbered_spans:
                    number = parse_toc_number(
                        match.group(1)
                    )

                    if not 1 <= number <= 99:
                        continue

                    start_y = float(
                        item["bbox"][1]
                    )

                    start_x = float(
                        item["bbox"][0]
                    )

                    page_midpoint = (
                        page.rect.width / 2
                    )

                    if start_x < page_midpoint:
                        column_left = 0.0
                        column_right = (
                            page_midpoint
                        )
                    else:
                        column_left = (
                            page_midpoint
                        )
                        column_right = (
                            page.rect.width
                        )

                    title_parts: list[str] = []

                    inline_title = (
                        normalize_line(
                            match.group(2)
                        )
                    )

                    if inline_title:
                        title_parts.append(
                            inline_title
                        )

                    # Same-row title spans.
                    same_row = [
                        candidate
                        for candidate in spans
                        if (
                            column_left
                            <= candidate["bbox"][0]
                            < column_right
                        )
                        and abs(
                            candidate["bbox"][1]
                            - start_y
                        )
                        <= 5
                        and candidate is not item
                    ]

                    same_row.sort(
                        key=lambda candidate:
                        candidate["bbox"][0]
                    )

                    for candidate in same_row:
                        candidate_text = (
                            candidate["text"]
                        )

                        if (
                            TOC_NUMBER_PATTERN.match(
                                candidate_text
                            )
                        ):
                            continue

                        if is_toc_page_number(
                            candidate_text
                        ):
                            continue

                        if candidate_text.casefold().startswith(
                            "reprint"
                        ):
                            continue

                        title_parts.append(
                            candidate_text
                        )

                    # Some titles continue on the
                    # immediately following row.
                    possible_continuation_y = [
                        candidate["bbox"][1]
                        for candidate in spans
                        if (
                            column_left
                            <= candidate["bbox"][0]
                            < column_right
                        )
                        and (
                            start_y + 7
                            < candidate["bbox"][1]
                            <= start_y + 33
                        )
                        and candidate["bold"]
                        and candidate["font_size"]
                        >= 20
                    ]

                    if possible_continuation_y:
                        continuation_y = min(
                            possible_continuation_y
                        )

                        continuation = [
                            candidate
                            for candidate in spans
                            if (
                                column_left
                                <= candidate["bbox"][0]
                                < column_right
                            )
                            and abs(
                                candidate["bbox"][1]
                                - continuation_y
                            )
                            <= 5
                        ]

                        continuation.sort(
                            key=lambda candidate:
                            candidate["bbox"][0]
                        )

                        for candidate in continuation:
                            candidate_text = (
                                candidate["text"]
                            )

                            if (
                                TOC_NUMBER_PATTERN.match(
                                    candidate_text
                                )
                            ):
                                continue

                            if is_toc_page_number(
                                candidate_text
                            ):
                                continue

                            if candidate_text.casefold().startswith(
                                "reprint"
                            ):
                                continue

                            title_parts.append(
                                candidate_text
                            )

                    title = normalize_toc_title(
                        title_parts
                    )

                    if (
                        len(title) >= 2
                        and len(title) <= 140
                    ):
                        # Later contents pages
                        # override accidental earlier
                        # numbered-list matches.
                        entries[number] = title

        return dict(
            sorted(entries.items())
        )


def infer_document_title(
    lines: list[str],
    *,
    fallback: str,
) -> dict[str, Any]:
    candidates = [
        line
        for line in lines
        if is_title_candidate(line)
    ]

    for index, line in enumerate(lines):
        match = HEADING_PATTERN.match(
            line
        )

        if not match:
            continue

        remainder = normalize_line(
            match.group(1)
        )

        if (
            remainder
            and is_title_candidate(
                remainder
            )
        ):
            return {
                "title": remainder,
                "confidence": "high",
                "source": (
                    "heading-line"
                ),
                "candidates": (
                    candidates[:10]
                ),
            }

        for next_line in lines[
            index + 1:index + 6
        ]:
            next_line = normalize_line(
                next_line
            )

            if is_title_candidate(
                next_line
            ):
                return {
                    "title": next_line,
                    "confidence": "high",
                    "source": (
                        "line-after-heading"
                    ),
                    "candidates": (
                        candidates[:10]
                    ),
                }

    if candidates:
        return {
            "title": candidates[0],
            "confidence": "medium",
            "source": "first-title-candidate",
            "candidates": candidates[:10],
        }

    return {
        "title": fallback,
        "confidence": "low",
        "source": "generated-fallback",
        "candidates": [],
    }


def apply_resolved_chapter_title(
    inferred: dict[str, Any],
    chapters: list[dict[str, Any]],
) -> dict[str, Any]:
    """Use a reviewed title for one-chapter PDFs.

    Multiple approved chapters do not automatically
    become a single document title. In that case the
    inferred title and its review status are retained.
    """
    approved_titles: list[str] = []

    for chapter in chapters:
        title = normalize_line(
            str(
                chapter.get(
                    "chapter_title",
                    "",
                )
            )
        )

        if (
            title
            and title not in approved_titles
        ):
            approved_titles.append(title)

    if len(approved_titles) != 1:
        return inferred

    approved_title = approved_titles[0]

    return {
        "title": approved_title,
        "confidence": "high",
        "source": (
            "resolved-chapter-structure"
        ),
        "candidates": [approved_title],
    }


def is_front_matter(
    source_filename: str,
    order: int,
) -> bool:
    stem = Path(
        source_filename
    ).stem.casefold()

    return (
        stem.endswith("ps")
        or "front" in stem
        or "prelim" in stem
        or (
            order == 1
            and stem.endswith("00")
        )
    )


def subject_display(
    subject: str,
) -> str:
    return subject.replace(
        "-",
        " ",
    ).title()


def build_book_title(
    book: dict[str, Any],
) -> str:
    title = str(
        book["title"]
    ).strip()

    subject = subject_display(
        str(book["subject"])
    )

    grade = int(
        book["grade"]
    )

    if subject.casefold() in (
        title.casefold()
    ):
        return (
            f"{title} Textbook for Grade "
            f"{grade}"
        )

    return (
        f"{title} {subject} Textbook "
        f"for Grade {grade}"
    )


def load_resolved_chapter_structure(
    path: Path,
    *,
    book_id: str,
    version: str,
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
]:
    payload = load_json_object(path)

    if payload.get("status") != "READY":
        raise ValueError(
            "Chapter structure must have "
            "status READY."
        )

    if payload.get("book_id") != book_id:
        raise ValueError(
            "Chapter structure book ID mismatch: "
            f"{payload.get('book_id')!r} != "
            f"{book_id!r}"
        )

    if payload.get("book_version") != version:
        raise ValueError(
            "Chapter structure version mismatch: "
            f"{payload.get('book_version')!r} != "
            f"{version!r}"
        )

    raw_chapters = payload.get("chapters")

    if (
        not isinstance(raw_chapters, list)
        or not raw_chapters
    ):
        raise ValueError(
            "Chapter structure contains no chapters."
        )

    grouped: dict[
        str,
        list[dict[str, Any]]
    ] = {}

    seen_chapter_ids: set[str] = set()

    required_fields = {
        "chapter_id",
        "chapter_title",
        "source_filename",
        "source_start_page",
        "source_end_page",
        "canonical_start_page",
        "canonical_end_page",
    }

    for raw_chapter in raw_chapters:
        if not isinstance(raw_chapter, dict):
            raise ValueError(
                "Chapter structure entry must "
                "be a JSON object."
            )

        missing = sorted(
            required_fields
            - set(raw_chapter)
        )

        if missing:
            raise ValueError(
                "Chapter structure entry is "
                "missing fields: "
                + ", ".join(missing)
            )

        chapter_id = str(
            raw_chapter["chapter_id"]
        ).strip()

        if chapter_id in seen_chapter_ids:
            raise ValueError(
                f"Duplicate chapter ID: {chapter_id}"
            )

        seen_chapter_ids.add(chapter_id)

        source_filename = str(
            raw_chapter["source_filename"]
        )

        chapter = {
            "chapter_id": chapter_id,
            "chapter_title": str(
                raw_chapter["chapter_title"]
            ).strip(),
            "source_start_page": int(
                raw_chapter["source_start_page"]
            ),
            "source_end_page": int(
                raw_chapter["source_end_page"]
            ),
            "canonical_start_page": int(
                raw_chapter[
                    "canonical_start_page"
                ]
            ),
            "canonical_end_page": int(
                raw_chapter[
                    "canonical_end_page"
                ]
            ),
        }

        grouped.setdefault(
            source_filename,
            [],
        ).append(chapter)

    for chapters in grouped.values():
        chapters.sort(
            key=lambda chapter: (
                chapter["source_start_page"],
                chapter["canonical_start_page"],
            )
        )

    return payload, grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a validated chapter "
            "manifest and BookConfig from an "
            "archive inspection report."
        )
    )

    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--inspection",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--book-id",
        required=True,
    )

    parser.add_argument(
        "--version",
        default="v1",
    )

    parser.add_argument(
        "--chapter-structure",
        type=Path,
        help=(
            "Optional READY chapter structure "
            "report containing one or more "
            "chapters per source PDF."
        ),
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Write generated config, manifest "
            "and generation report. Without "
            "this option, perform a dry-run."
        ),
    )

    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Replace existing generated files."
        ),
    )

    return parser.parse_args()


def inspection_allows_config_generation(
    inspection: dict[str, Any],
) -> tuple[bool, str]:
    """
    Accept fully passed inspections.

    A PASSED_WITH_WARNING report is accepted only
    when every PDF is valid and the sole warning
    indicates an unreadable/scanned text layer that
    has been routed to visual BDA extraction.
    """

    status = inspection.get(
        "inspection_status"
    )

    if status == "PASSED":
        return True, "passed"

    if status != "PASSED_WITH_WARNING":
        return (
            False,
            f"unsupported inspection status: "
            f"{status!r}",
        )

    book = inspection.get("book")
    documents = inspection.get("documents")
    evidence = inspection.get(
        "script_evidence"
    )
    warnings = inspection.get("warnings")

    if not isinstance(book, dict):
        return False, "missing book metadata"

    if (
        not isinstance(documents, list)
        or not documents
    ):
        return False, "missing inspected documents"

    if not isinstance(evidence, dict):
        return False, "missing script evidence"

    if (
        not isinstance(warnings, list)
        or not warnings
    ):
        return False, "warning status without warnings"

    if any(
        not isinstance(document, dict)
        or document.get("pdf_status")
        != "valid"
        for document in documents
    ):
        return False, "one or more PDFs are invalid"

    if (
        book.get("detected_script")
        != "unknown"
        or book.get("script_verification")
        != "unverified"
        or book.get("extraction_route")
        != "visual-bda"
    ):
        return (
            False,
            "warning is not an approved "
            "visual-BDA routing case",
        )

    if (
        evidence.get("text_evidence_status")
        != "insufficient"
        or evidence.get("extraction_route")
        != "visual-bda"
    ):
        return (
            False,
            "script evidence does not support "
            "visual-BDA routing",
        )

    allowed_warning_prefix = (
        "PDF text layer does not provide enough "
        "meaningful script evidence;"
    )

    if any(
        not str(warning).startswith(
            allowed_warning_prefix
        )
        for warning in warnings
    ):
        return (
            False,
            "inspection contains additional "
            "unapproved warnings",
        )

    return True, "visual-bda-scanned-textbook"


def main() -> int:
    args = parse_args()

    registry = load_json_object(
        args.registry
    )

    inspection = load_json_object(
        args.inspection
    )

    book = find_registry_book(
        registry,
        args.book_id,
    )

    inspected_book_id = (
        inspection.get(
            "book",
            {},
        ).get("book_id")
    )

    if inspected_book_id != args.book_id:
        raise ValueError(
            "Inspection book ID mismatch: "
            f"{inspected_book_id!r} != "
            f"{args.book_id!r}"
        )

    (
        inspection_accepted,
        inspection_acceptance_mode,
    ) = inspection_allows_config_generation(
        inspection
    )

    if not inspection_accepted:
        raise ValueError(
            "Inspection report cannot be used "
            "for config generation: "
            f"{inspection_acceptance_mode}"
        )

    inspected_documents = (
        inspection.get("documents")
    )

    if (
        not isinstance(
            inspected_documents,
            list,
        )
        or not inspected_documents
    ):
        raise ValueError(
            "Inspection contains no documents."
        )

    invalid_documents = [
        document
        for document in inspected_documents
        if document.get("pdf_status")
        != "valid"
    ]

    if invalid_documents:
        raise ValueError(
            "Inspection contains invalid PDFs."
        )

    archive_path = Path(
        inspection["source_archive"][
            "local_path"
        ]
    )

    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Downloaded archive missing: "
            f"{archive_path}"
        )

    version = args.version

    chapter_structure = None

    resolved_chapters_by_filename: dict[
        str,
        list[dict[str, Any]]
    ] = {}

    if args.chapter_structure is not None:
        (
            chapter_structure,
            resolved_chapters_by_filename,
        ) = load_resolved_chapter_structure(
            args.chapter_structure,
            book_id=args.book_id,
            version=version,
        )

    manifest_path = (
        MANIFEST_ROOT
        / (
            f"{args.book_id}-"
            f"{version}-chapters.json"
        )
    )

    config_path = (
        CONFIG_ROOT
        / (
            f"{args.book_id}-"
            f"{version}.json"
        )
    )

    generation_report_path = (
        REPORT_ROOT
        / (
            f"{args.book_id}-"
            f"{version}.json"
        )
    )

    if args.write and not args.replace:
        existing = [
            path
            for path in (
                manifest_path,
                config_path,
                generation_report_path,
            )
            if path.exists()
        ]

        if existing:
            raise FileExistsError(
                "Generated output already exists: "
                + ", ".join(
                    str(path)
                    for path in existing
                )
                + ". Use --replace only after "
                "reviewing existing files."
            )

    full_book_title = build_book_title(
        book
    )

    documents: list[
        dict[str, Any]
    ] = []

    title_metadata: list[
        dict[str, Any]
    ] = []

    canonical_page = 1
    unit_number = 0

    with ZipFile(
        archive_path,
        "r",
    ) as archive:
        ordered_documents = sorted(
            inspected_documents,
            key=lambda document: int(
                document[
                    "candidate_order"
                ]
            ),
        )

        toc_entries: dict[int, str] = {}

        for candidate in ordered_documents:
            candidate_order = int(
                candidate[
                    "candidate_order"
                ]
            )

            candidate_filename = str(
                candidate[
                    "source_filename"
                ]
            )

            if not is_front_matter(
                candidate_filename,
                candidate_order,
            ):
                continue

            candidate_toc = (
                extract_numbered_toc_entries(
                    archive,
                    str(
                        candidate[
                            "archive_path"
                        ]
                    ),
                )
            )

            if len(candidate_toc) > len(
                toc_entries
            ):
                toc_entries = candidate_toc

        for document in ordered_documents:
            order = int(
                document[
                    "candidate_order"
                ]
            )

            source_filename = str(
                document[
                    "source_filename"
                ]
            )

            source_page_count = int(
                document[
                    "source_page_count"
                ]
            )

            archive_member = str(
                document["archive_path"]
            )

            canonical_start = (
                canonical_page
            )

            canonical_end = (
                canonical_start
                + source_page_count
                - 1
            )

            front_matter = is_front_matter(
                source_filename,
                order,
            )

            if front_matter:
                document_id = (
                    "front-matter"
                    if not any(
                        item["document_id"]
                        == "front-matter"
                        for item in documents
                    )
                    else (
                        f"front-matter-{order}"
                    )
                )

                document_type = (
                    "front_matter"
                )

                current_unit_number = None

                inferred = {
                    "title": (
                        f"{book['title']} "
                        "Front Matter"
                    ),
                    "confidence": "high",
                    "source": (
                        "front-matter-filename"
                    ),
                    "candidates": [],
                }

                chapters: list[
                    dict[str, Any]
                ] = []

            else:
                unit_number += 1

                document_id = (
                    f"unit-{unit_number}"
                )

                document_type = "unit"

                current_unit_number = (
                    unit_number
                )

                inferred = (
                    extract_layout_title(
                        archive,
                        archive_member,
                    )
                )

                toc_title = toc_entries.get(
                    unit_number
                )

                if (
                    inferred is None
                    or inferred.get(
                        "confidence"
                    )
                    != "high"
                ):
                    if toc_title:
                        inferred = {
                            "title": toc_title,
                            "confidence": "high",
                            "source": (
                                "front-matter-"
                                "numbered-toc"
                            ),
                            "candidates": [
                                toc_title
                            ],
                        }
                    else:
                        sample_lines = (
                            extract_sample_lines(
                                archive,
                                archive_member,
                            )
                        )

                        inferred = (
                            infer_document_title(
                                sample_lines,
                                fallback=(
                                    f"{book['title']} "
                                    f"Unit {unit_number}"
                                ),
                            )
                        )

                chapter_title = str(
                    inferred["title"]
                )

                if chapter_structure is not None:
                    chapters = [
                        dict(chapter)
                        for chapter
                        in resolved_chapters_by_filename.get(
                            source_filename,
                            [],
                        )
                    ]

                    for chapter in chapters:
                        if not (
                            1
                            <= chapter[
                                "source_start_page"
                            ]
                            <= chapter[
                                "source_end_page"
                            ]
                            <= source_page_count
                        ):
                            raise ValueError(
                                "Resolved chapter source "
                                "range is outside PDF: "
                                f"{source_filename} / "
                                f"{chapter}"
                            )

                        if not (
                            canonical_start
                            <= chapter[
                                "canonical_start_page"
                            ]
                            <= chapter[
                                "canonical_end_page"
                            ]
                            <= canonical_end
                        ):
                            raise ValueError(
                                "Resolved chapter canonical "
                                "range is outside document: "
                                f"{source_filename} / "
                                f"{chapter}"
                            )

                    inferred = (
                        apply_resolved_chapter_title(
                            inferred,
                            chapters,
                        )
                    )

                else:
                    chapters = [{
                        "chapter_id": (
                            f"unit-{unit_number}"
                        ),
                        "chapter_title": (
                            chapter_title
                        ),
                        "source_start_page": 1,
                        "source_end_page": (
                            source_page_count
                        ),
                        "canonical_start_page": (
                            canonical_start
                        ),
                        "canonical_end_page": (
                            canonical_end
                        ),
                    }]

            documents.append({
                "order": order,
                "document_id": document_id,
                "document_type": (
                    document_type
                ),
                "unit_number": (
                    current_unit_number
                ),
                "source_filename": (
                    source_filename
                ),
                "source_page_count": (
                    source_page_count
                ),
                "canonical_start_page": (
                    canonical_start
                ),
                "canonical_end_page": (
                    canonical_end
                ),
                "title": inferred["title"],
                "chapters": chapters,
            })

            title_metadata.append({
                "order": order,
                "source_filename": (
                    source_filename
                ),
                "document_id": document_id,
                "selected_title": (
                    inferred["title"]
                ),
                "title_confidence": (
                    inferred["confidence"]
                ),
                "title_source": (
                    inferred["source"]
                ),
                "title_candidates": (
                    inferred["candidates"]
                ),
            })

            canonical_page = (
                canonical_end + 1
            )

    if chapter_structure is not None:
        manifest_source_filenames = {
            document["source_filename"]
            for document in documents
        }

        unknown_structure_files = sorted(
            set(
                resolved_chapters_by_filename
            )
            - manifest_source_filenames
        )

        if unknown_structure_files:
            raise ValueError(
                "Chapter structure references "
                "unknown source PDFs: "
                + ", ".join(
                    unknown_structure_files
                )
            )

    total_pages = canonical_page - 1

    expected_pages = int(
        inspection[
            "canonical_candidate"
        ][
            "canonical_page_count"
        ]
    )

    if total_pages != expected_pages:
        raise ValueError(
            "Generated page total differs from "
            "inspection: "
            f"{total_pages} != {expected_pages}"
        )

    manifest_payload = {
        "schema_version": "1.0",
        "book_id": args.book_id,
        "book_version": version,
        "title": full_book_title,
        "ordering_strategy": "manifest",
        "source_archive": {
            "bucket": book[
                "source_bucket"
            ],
            "key": book[
                "source_zip_key"
            ],
            "archive_root": inspection[
                "source_archive"
            ].get("archive_root"),
            "supplementary_assets": (
                inspection.get(
                    "supplementary_assets",
                    [],
                )
            ),
        },
        "canonical_layout": {
            "leading_blank_pages": 0,
            "source_document_pages": (
                total_pages
            ),
            "trailing_blank_pages": 0,
            "canonical_page_count": (
                total_pages
            ),
            "source_to_canonical_page_offset": 0,
        },
        "documents": documents,
    }

    manifest_model = (
        ChapterManifest.model_validate(
            manifest_payload
        )
    )

    chapter_directory = (
        Path("data/source-archives")
        / args.book_id
        / version
        / "extracted"
    )

    local_root = (
        Path("data/multimodal-output")
        / args.book_id
        / version
    )

    index_name = (
        f"{args.book_id}-{version}"
    )

    grade = int(book["grade"])

    config_payload = {
        "schema_version": "1.0",
        "book": {
            "book_id": args.book_id,
            "title": full_book_title,
            "grade": grade,
            "subject": book["subject"],
            "language": book["language"],
            "board": book.get(
                "board",
                "ncert",
            ),
            "version": version,
            "page_count": total_pages,
            "academic_year": None,
            "status": "draft",
        },
        "aws": {
            "region": "us-east-1",
            "bucket": book[
                "source_bucket"
            ],
        },
        "bda": {
            "project_arn": (
                BDA_PROJECT_ARN
            ),
            "profile_arn": (
                BDA_PROFILE_ARN
            ),
            "stage": "DEVELOPMENT",
        },
        "opensearch": {
            "collection_endpoint": (
                OPENSEARCH_ENDPOINT
            ),
            "index_name": index_name,
            "vector_field": "embedding",
        },
        "source": {
            "mode": "chapter_folder",
            "chapter_directory": (
                chapter_directory.as_posix()
            ),
            "chapter_order": "manifest",
            "chapter_manifest": (
                manifest_path.as_posix()
            ),
            "merged_pdf_name": (
                "textbook.pdf"
            ),
        },
        "storage": {
            "source_s3_key": (
                f"source-documents/"
                f"grade-{grade}/"
                f"{args.book_id}/"
                f"versions/{version}/"
                f"textbook.pdf"
            ),
            "derived_prefix": (
                f"derived-artifacts/"
                f"grade-{grade}/"
                f"{args.book_id}/"
                f"{version}"
            ),
            "bda_input_prefix": (
                f"bda-input/"
                f"grade-{grade}/"
                f"{args.book_id}/"
                f"{version}"
            ),
            "local_root": (
                local_root.as_posix()
            ),
        },
        "models": {
            "embedding": {
                "model_id": (
                    "amazon.titan-"
                    "embed-text-v2:0"
                ),
                "dimensions": 1024,
                "normalize": True,
            },
            "generation": {
                "model_id": (
                    "amazon.nova-lite-v1:0"
                ),
                "maximum_output_tokens": 700,
                "temperature": 0,
            },
        },
        "processing": {
            "page_batch_size": 20,
            "minimum_text_similarity": (
                0.999
            ),
            "embedding_checkpoint_interval": 10,
        },
        "retrieval": {
            "vector_candidate_limit": 20,
            "bm25_candidate_limit": 20,
            "result_limit": 5,
            "rrf_constant": 60,
            "vector_weight": 6.0,
            "bm25_weight": 1.0,
        },
    }

    config_model = (
        BookConfig.model_validate(
            config_payload
        )
    )

    title_review_items = [
        item
        for item in title_metadata
        if item["title_confidence"]
        != "high"
    ]

    low_confidence = [
        item
        for item in title_metadata
        if item["title_confidence"]
        == "low"
    ]

    generation_report = {
        "schema_version": "1.0",
        "status": (
            "READY"
            if not title_review_items
            else "NEEDS_TITLE_REVIEW"
        ),
        "generated_at": utc_now(),
        "write_enabled": bool(
            args.write
        ),
        "book_id": args.book_id,
        "version": version,
        "inspection_path": str(
            args.inspection
        ),
        "chapter_structure_path": (
            str(args.chapter_structure)
            if args.chapter_structure
            is not None
            else None
        ),
        "chapter_structure_status": (
            chapter_structure.get("status")
            if chapter_structure
            is not None
            else None
        ),
        "resolved_structure_used": (
            chapter_structure is not None
        ),
        "config_path": str(
            config_path
        ),
        "manifest_path": str(
            manifest_path
        ),
        "document_count": len(
            documents
        ),
        "chapter_count": (
            manifest_model.chapter_count
        ),
        "canonical_page_count": (
            total_pages
        ),
        "archive_root": (
            manifest_model
            .source_archive
            .archive_root
        ),
        "numbered_toc_entries": {
            str(number): title
            for number, title
            in toc_entries.items()
        },
        "numbered_toc_entry_count": (
            len(toc_entries)
        ),
        "title_metadata": (
            title_metadata
        ),
        "title_review_count": (
            len(title_review_items)
        ),
        "title_review_documents": [
            {
                "source_filename": (
                    item["source_filename"]
                ),
                "selected_title": (
                    item["selected_title"]
                ),
                "confidence": (
                    item["title_confidence"]
                ),
                "source": (
                    item["title_source"]
                ),
            }
            for item in title_review_items
        ],
        "low_confidence_title_count": (
            len(low_confidence)
        ),
        "validation": {
            "manifest": "PASSED",
            "book_config": "PASSED",
        },
        "safety": {
            "aws_calls": 0,
            "s3_writes": 0,
            "bedrock_calls": 0,
            "opensearch_calls": 0,
        },
    }

    if args.write:
        atomic_write_json(
            manifest_path,
            manifest_model.model_dump(
                mode="json"
            ),
        )

        atomic_write_json(
            config_path,
            config_model.model_dump(
                mode="json"
            ),
        )

        atomic_write_json(
            generation_report_path,
            generation_report,
        )

    print("=" * 80)
    print("BOOK CONFIG AND MANIFEST GENERATION")
    print("=" * 80)
    print("Mode:             ",
          "WRITE" if args.write else "DRY RUN")
    print("Book ID:          ", args.book_id)
    print("Version:          ", version)
    print("Book title:       ", full_book_title)
    print("Documents:        ", len(documents))
    print("Chapters:         ",
          manifest_model.chapter_count)
    print("Canonical pages:  ", total_pages)
    print(
        "Resolved structure:",
        "YES"
        if chapter_structure is not None
        else "NO",
    )
    print("Archive root:     ",
          manifest_model.source_archive.archive_root)
    print("TOC entries:      ",
          len(toc_entries))
    print("Manifest valid:   PASS")
    print("BookConfig valid: PASS")
    print(
        "Titles needing review:",
        len(title_review_items),
    )
    print(
        "Low-confidence titles:",
        len(low_confidence),
    )

    print()
    print("GENERATED DOCUMENT METADATA")
    print("-" * 80)

    for item in title_metadata:
        print(
            f"{item['order']:02}. "
            f"{item['source_filename']} | "
            f"{item['document_id']} | "
            f"{item['selected_title']} | "
            f"confidence="
            f"{item['title_confidence']} | "
            f"source="
            f"{item['title_source']}"
        )

    print()
    print("OUTPUT PATHS")
    print("-" * 80)
    print("Config:   ", config_path)
    print("Manifest: ", manifest_path)
    print(
        "Report:   ",
        generation_report_path,
    )

    print()
    print("AWS calls:        0")
    print("S3 writes:        0")
    print("Bedrock calls:    0")
    print("OpenSearch calls: 0")

    if not args.write:
        print()
        print(
            "Dry-run only. Use --write after "
            "reviewing generated titles."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())