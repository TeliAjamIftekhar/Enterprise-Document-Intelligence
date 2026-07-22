from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import fitz


BOLD_FONT_MARKERS = (
    "bold",
    "black",
    "heavy",
    "semibold",
    "demi",
)

ZERO_WIDTH_CHARACTERS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
    "\u00ad",
}

DIGIT_CHARACTERS = (
    "0-9"
    "०-९"
    "٠-٩"
    "۰-۹"
)

NUMBER_TOKEN_PATTERN = re.compile(
    rf"^\s*[\(\[]?"
    rf"[{DIGIT_CHARACTERS}]{{1,4}}"
    rf"[\.\):：\-–—]?"
    rf"[\)\]]?\s*$"
)

NUMBERED_PREFIX_PATTERN = re.compile(
    rf"^\s*[\(\[]?"
    rf"[{DIGIT_CHARACTERS}]{{1,3}}"
    rf"[\.\):：\-–—]"
    rf"\s*\S+"
)

ROMAN_PAGE_PATTERN = re.compile(
    r"^\s*[ivxlcdmIVXLCDM]{1,8}\s*$"
)

TOC_MARKERS = (
    # English
    "contents",
    "table of contents",
    "lessons",
    "chapters",

    # Hindi / Marathi / Sanskrit
    "अनुक्रमणिका",
    "विषय सूची",
    "विषय-सूची",
    "पाठ सूची",
    "पाठ-सूची",
    "अध्याय सूची",
    "अध्याय-सूची",

    # Urdu
    "فہرست",
    "فہرست مضامین",
    "مضامین",
    "اسباق",
)

LANGUAGE_HINTS = (
    "english",
    "hindi",
    "marathi",
    "sanskrit",
    "urdu",
)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    for character in ZERO_WIDTH_CHARACTERS:
        value = value.replace(
            character,
            "",
        )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return payload


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
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def character_script(
    character: str,
) -> str | None:
    codepoint = ord(character)

    if (
        0x0900 <= codepoint <= 0x097F
        or 0xA8E0 <= codepoint <= 0xA8FF
    ):
        return "devanagari"

    if (
        0x0600 <= codepoint <= 0x06FF
        or 0x0750 <= codepoint <= 0x077F
        or 0x08A0 <= codepoint <= 0x08FF
        or 0xFB50 <= codepoint <= 0xFDFF
        or 0xFE70 <= codepoint <= 0xFEFF
    ):
        return "arabic"

    name = unicodedata.name(
        character,
        "",
    )

    if "LATIN" in name:
        return "latin"

    if character.isalpha():
        return "other"

    return None


def count_scripts(
    value: str,
) -> dict[str, int]:
    counts = {
        "latin": 0,
        "devanagari": 0,
        "arabic": 0,
        "other": 0,
    }

    for character in value:
        script = character_script(
            character
        )

        if script is not None:
            counts[script] += 1

    return counts


def merge_script_counts(
    target: dict[str, int],
    source: dict[str, int],
) -> None:
    for script, count in source.items():
        target[script] = (
            target.get(script, 0)
            + int(count)
        )


def language_hint_from_book_id(
    book_id: str,
) -> str | None:
    lowered = book_id.casefold()

    for language in LANGUAGE_HINTS:
        if language in lowered:
            return language

    return None


def detect_script_profile(
    counts: dict[str, int],
    *,
    book_id: str,
) -> dict[str, Any]:
    ranked = sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    dominant_script = (
        ranked[0][0]
        if ranked and ranked[0][1] > 0
        else "unknown"
    )

    total_letters = sum(counts.values())

    dominant_count = counts.get(
        dominant_script,
        0,
    )

    confidence = (
        dominant_count / total_letters
        if total_letters
        else 0.0
    )

    active_scripts = [
        script
        for script, count in counts.items()
        if count > 0
    ]

    if (
        len(active_scripts) > 1
        and confidence < 0.75
    ):
        profile = "mixed-script"
    else:
        profile = dominant_script

    direction = (
        "rtl"
        if dominant_script == "arabic"
        else "ltr"
    )

    return {
        "profile": profile,
        "dominant_script": dominant_script,
        "direction": direction,
        "language_hint": (
            language_hint_from_book_id(
                book_id
            )
        ),
        "confidence": round(
            confidence,
            4,
        ),
        "character_counts": counts,
    }


def span_is_bold(
    span: dict[str, Any],
) -> bool:
    font = str(
        span.get("font", "")
    ).casefold()

    if any(
        marker in font
        for marker in BOLD_FONT_MARKERS
    ):
        return True

    flags = int(span.get("flags", 0))

    # PyMuPDF font flag bit commonly used for
    # bold text. Font-name detection remains
    # the primary signal.
    return bool(flags & 16)


def page_payload(
    page: fitz.Page,
    *,
    canonical_page: int,
) -> dict[str, Any]:
    raw = page.get_text(
        "dict",
        sort=True,
    )

    spans: list[dict[str, Any]] = []
    page_text_parts: list[str] = []

    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            for span in line.get(
                "spans",
                [],
            ):
                text = normalize_text(
                    str(span.get("text", ""))
                )

                if not text:
                    continue

                bbox = [
                    round(float(value), 2)
                    for value in span.get(
                        "bbox",
                        [0, 0, 0, 0],
                    )
                ]

                font_size = round(
                    float(span.get("size", 0)),
                    2,
                )

                bold = span_is_bold(span)

                spans.append({
                    "text": text,
                    "font_size": font_size,
                    "font": str(
                        span.get("font", "")
                    ),
                    "bold": bold,
                    "bbox": bbox,
                    "score": round(
                        font_size * 10
                        + (40 if bold else 0),
                        2,
                    ),
                })

                page_text_parts.append(text)

    page_text = normalize_text(
        " ".join(page_text_parts)
    )

    marker_hits = sum(
        1
        for marker in TOC_MARKERS
        if marker.casefold()
        in page_text.casefold()
    )

    numeric_tokens = sum(
        1
        for span in spans
        if (
            NUMBER_TOKEN_PATTERN.fullmatch(
                span["text"]
            )
            or ROMAN_PAGE_PATTERN.fullmatch(
                span["text"]
            )
        )
    )

    numbered_prefixes = sum(
        1
        for span in spans
        if NUMBERED_PREFIX_PATTERN.match(
            span["text"]
        )
    )

    large_bold_spans = sum(
        1
        for span in spans
        if (
            bool(span["bold"])
            and float(
                span["font_size"]
            ) >= 18
        )
    )

    toc_score = (
        marker_hits * 20
        + numbered_prefixes * 5
        + numeric_tokens * 2
        + min(large_bold_spans, 20)
    )

    is_toc_candidate = (
        marker_hits > 0
        or numbered_prefixes >= 2
        or (
            numeric_tokens >= 2
            and large_bold_spans >= 4
        )
    )

    return {
        "page_number": canonical_page,
        "width": float(page.rect.width),
        "height": float(page.rect.height),
        "spans": spans,
        "toc_score": toc_score,
        "toc_signals": {
            "marker_hits": marker_hits,
            "numbered_prefixes": (
                numbered_prefixes
            ),
            "numeric_tokens": numeric_tokens,
            "large_bold_spans": (
                large_bold_spans
            ),
        },
        "is_toc_candidate": (
            is_toc_candidate
        ),
        "_page_text": page_text,
    }


def contiguous_groups(
    pages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: list[
        list[dict[str, Any]]
    ] = []

    for page in sorted(
        pages,
        key=lambda item:
        int(item["page_number"]),
    ):
        if (
            not groups
            or int(page["page_number"])
            != int(
                groups[-1][-1][
                    "page_number"
                ]
            )
            + 1
        ):
            groups.append([page])
        else:
            groups[-1].append(page)

    return groups


def choose_contents_pages(
    front_pages: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    candidates = [
        page
        for page in front_pages
        if page["is_toc_candidate"]
    ]

    if candidates:
        groups = contiguous_groups(
            candidates
        )

        selected = max(
            groups,
            key=lambda group: (
                sum(
                    int(page["toc_score"])
                    for page in group
                ),
                len(group),
                int(
                    group[-1]["page_number"]
                ),
            ),
        )

        return selected

    # Conservative fallback: inspect the last
    # eight front-matter pages. Resolver quality
    # gates will prevent unsafe final manifests.
    return front_pages[-8:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect extracted textbook layout, "
            "detect script/direction and produce "
            "resolver-compatible TOC page data."
        )
    )

    parser.add_argument(
        "--extraction-report",
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

    if extraction.get("status") != "VALID":
        raise ValueError(
            "Extraction report must have "
            "status VALID."
        )

    book_id = str(
        extraction["book_id"]
    )

    version = str(
        extraction["book_version"]
    )

    target_directory = Path(
        extraction["target_directory"]
    )

    front_pages: list[
        dict[str, Any]
    ] = []

    lesson_documents: list[
        dict[str, Any]
    ] = []

    script_counts = {
        "latin": 0,
        "devanagari": 0,
        "arabic": 0,
        "other": 0,
    }

    for document in extraction["documents"]:
        source_filename = str(
            document["source_filename"]
        )

        pdf_path = (
            target_directory
            / source_filename
        )

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"Missing extracted PDF: "
                f"{pdf_path}"
            )

        canonical_start = int(
            document[
                "canonical_start_page"
            ]
        )

        with fitz.open(pdf_path) as pdf:
            if (
                pdf.page_count
                != int(document["page_count"])
            ):
                raise ValueError(
                    "PDF page count mismatch: "
                    f"{source_filename}"
                )

            document_script_counts = {
                "latin": 0,
                "devanagari": 0,
                "arabic": 0,
                "other": 0,
            }

            for local_index, page in enumerate(
                pdf
            ):
                canonical_page = (
                    canonical_start
                    + local_index
                )

                payload = page_payload(
                    page,
                    canonical_page=(
                        canonical_page
                    ),
                )

                page_counts = count_scripts(
                    payload.pop("_page_text")
                )

                merge_script_counts(
                    document_script_counts,
                    page_counts,
                )

                merge_script_counts(
                    script_counts,
                    page_counts,
                )

                if (
                    document[
                        "document_type"
                    ]
                    == "front_matter"
                ):
                    front_pages.append(
                        payload
                    )

            if (
                document["document_type"]
                != "front_matter"
            ):
                lesson_documents.append({
                    "order": int(
                        document["order"]
                    ),
                    "document_id": str(
                        document["document_id"]
                    ),
                    "source_filename": (
                        source_filename
                    ),
                    "canonical_start_page": (
                        canonical_start
                    ),
                    "canonical_end_page": int(
                        document[
                            "canonical_end_page"
                        ]
                    ),
                    "page_count": int(
                        document["page_count"]
                    ),
                    "script_counts": (
                        document_script_counts
                    ),
                })

    if not front_pages:
        raise ValueError(
            "No front-matter pages were found."
        )

    contents_pages = (
        choose_contents_pages(
            front_pages
        )
    )

    for page in contents_pages:
        page.pop(
            "is_toc_candidate",
            None,
        )

    profile = detect_script_profile(
        script_counts,
        book_id=book_id,
    )

    status = (
        "READY"
        if contents_pages
        and profile[
            "dominant_script"
        ] != "unknown"
        else "NEEDS_REVIEW"
    )

    output = {
        "schema_version": "1.0",
        "book_id": book_id,
        "version": version,
        "status": status,
        "script_profile": profile,
        "front_matter_page_count": (
            len(front_pages)
        ),
        "contents_pages": (
            contents_pages
        ),
        "lesson_documents": (
            lesson_documents
        ),
        "aws_calls": 0,
    }

    atomic_write_json(
        args.output,
        output,
    )

    print("=" * 100)
    print("TEXTBOOK LAYOUT INSPECTION")
    print("=" * 100)
    print("Book ID:          ", book_id)
    print("Version:          ", version)
    print(
        "Dominant script:  ",
        profile["dominant_script"],
    )
    print(
        "Direction:        ",
        profile["direction"],
    )
    print(
        "Language hint:    ",
        profile["language_hint"],
    )
    print(
        "Script confidence:",
        profile["confidence"],
    )
    print(
        "Front pages:      ",
        len(front_pages),
    )
    print(
        "Contents pages:   ",
        len(contents_pages),
    )
    print(
        "Contents page IDs:",
        [
            page["page_number"]
            for page in contents_pages
        ],
    )
    print("Status:           ", status)
    print("Output:           ", args.output)
    print("AWS calls:         0")

    return 0 if status == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())