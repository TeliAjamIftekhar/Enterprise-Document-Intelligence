from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz


DEFAULT_SOURCE_PDF = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/"
    "source/textbook.pdf"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/multimodal-output/"
    "grade-9-english-kaveri/v1/"
    "page-scan"
)

EDUCATIONAL_KEYWORDS = (
    "activity",
    "answer",
    "chapter",
    "classroom",
    "complete",
    "discuss",
    "exercise",
    "explain",
    "lesson",
    "listen",
    "poem",
    "project",
    "question",
    "read",
    "story",
    "think",
    "unit",
    "write",
)

FRONT_MATTER_KEYWORDS = (
    "all rights reserved",
    "first edition",
    "isbn",
    "publication division",
    "publication team",
    "printed at",
    "printed on",
)


def clean_preview(
    text: str,
    limit: int = 240,
) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:limit]


def rectangle_area(
    value: Any,
    page_rect: fitz.Rect,
) -> float:
    try:
        rectangle = fitz.Rect(value)
        intersection = rectangle & page_rect

        if intersection.is_empty:
            return 0.0

        return max(
            0.0,
            intersection.width * intersection.height,
        )

    except Exception:
        return 0.0


def get_image_digest(
    image: dict[str, Any],
) -> str | None:
    digest = image.get("digest")

    if isinstance(digest, bytes):
        return digest.hex()

    if isinstance(digest, str) and digest:
        return digest

    return None


def count_keyword_hits(
    text: str,
    keywords: tuple[str, ...],
) -> list[str]:
    lower_text = text.lower()

    return [
        keyword
        for keyword in keywords
        if re.search(
            rf"\b{re.escape(keyword)}\b",
            lower_text,
        )
    ]


def scan_page(
    page: fitz.Page,
) -> dict[str, Any]:
    page_number = page.number + 1
    page_rect = page.rect

    page_area = max(
        1.0,
        page_rect.width * page_rect.height,
    )

    errors: list[str] = []

    try:
        text = page.get_text(
            "text",
            sort=True,
        )
    except Exception as exc:
        text = ""
        errors.append(
            f"text_extraction: {type(exc).__name__}: {exc}"
        )

    try:
        words = page.get_text("words")
        word_count = len(words)
    except Exception as exc:
        word_count = 0
        errors.append(
            f"word_extraction: {type(exc).__name__}: {exc}"
        )

    try:
        images = page.get_image_info(
            hashes=True,
            xrefs=False,
        )
    except Exception as exc:
        images = []
        errors.append(
            f"image_inspection: {type(exc).__name__}: {exc}"
        )

    image_area = 0.0
    image_digests: list[str] = []

    for image in images:
        if not isinstance(image, dict):
            continue

        image_area += rectangle_area(
            image.get("bbox"),
            page_rect,
        )

        digest = get_image_digest(image)

        if digest:
            image_digests.append(digest)

    image_area_ratio = min(
        1.0,
        image_area / page_area,
    )

    try:
        drawings = page.get_drawings()
    except Exception as exc:
        drawings = []
        errors.append(
            f"drawing_inspection: {type(exc).__name__}: {exc}"
        )

    try:
        drawing_clusters = page.cluster_drawings(
            drawings=drawings,
        )
    except Exception as exc:
        drawing_clusters = []
        errors.append(
            f"drawing_clustering: {type(exc).__name__}: {exc}"
        )

    try:
        table_finder = page.find_tables(
            strategy="lines_strict",
            paths=drawings,
        )

        strict_table_count = len(
            table_finder.tables
        )

    except Exception as exc:
        strict_table_count = 0
        errors.append(
            f"table_detection: {type(exc).__name__}: {exc}"
        )

    educational_hits = count_keyword_hits(
        text,
        EDUCATIONAL_KEYWORDS,
    )

    front_matter_hits = count_keyword_hits(
        text,
        FRONT_MATTER_KEYWORDS,
    )

    non_whitespace_chars = len(
        re.sub(r"\s+", "", text)
    )

    return {
        "page_number": page_number,
        "page_index": page.number,
        "page_width_points": page_rect.width,
        "page_height_points": page_rect.height,
        "text_char_count": len(text.strip()),
        "non_whitespace_char_count": (
            non_whitespace_chars
        ),
        "word_count": word_count,
        "displayed_image_count": len(images),
        "image_area_ratio": round(
            image_area_ratio,
            6,
        ),
        "image_digests": image_digests,
        "drawing_path_count": len(drawings),
        "drawing_cluster_count": len(
            drawing_clusters
        ),
        "strict_table_count": (
            strict_table_count
        ),
        "educational_keyword_hits": (
            educational_hits
        ),
        "educational_keyword_count": len(
            educational_hits
        ),
        "front_matter_keyword_hits": (
            front_matter_hits
        ),
        "front_matter_keyword_count": len(
            front_matter_hits
        ),
        "text_preview": clean_preview(text),
        "scan_errors": errors,
    }


def calculate_page_score(
    page: dict[str, Any],
) -> float:
    score = 0.0

    unique_images = int(
        page["unique_image_count"]
    )

    recurring_images = int(
        page["recurring_image_count"]
    )

    score += min(unique_images, 6) * 8.0
    score += min(
        float(page["image_area_ratio"]),
        0.85,
    ) * 22.0

    score += min(
        int(page["drawing_cluster_count"]),
        12,
    ) * 1.5

    score += min(
        int(page["strict_table_count"]),
        3,
    ) * 12.0

    score += min(
        int(page["educational_keyword_count"]),
        8,
    ) * 2.5

    text_chars = int(page["text_char_count"])

    if 300 <= text_chars <= 7000:
        score += 6.0

    elif text_chars < 80:
        score -= 6.0

    score -= min(
        recurring_images,
        5,
    ) * 1.5

    score -= min(
        int(page["front_matter_keyword_count"]),
        5,
    ) * 8.0

    if page["scan_errors"]:
        score -= 5.0

    return round(score, 3)


def build_candidate_windows(
    pages: list[dict[str, Any]],
    window_size: int,
    exclude_front_pages: int,
    exclude_end_pages: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    last_allowed_page = (
        len(pages) - exclude_end_pages
    )

    for start_index in range(
        exclude_front_pages,
        last_allowed_page - window_size + 1,
    ):
        window = pages[
            start_index:start_index + window_size
        ]

        if len(window) != window_size:
            continue

        page_numbers = [
            page["page_number"]
            for page in window
        ]

        visual_page_count = sum(
            1
            for page in window
            if (
                page["unique_image_count"] > 0
                or page["drawing_cluster_count"] > 0
            )
        )

        educational_page_count = sum(
            1
            for page in window
            if page[
                "educational_keyword_count"
            ] > 0
        )

        table_page_count = sum(
            1
            for page in window
            if page["strict_table_count"] > 0
        )

        total_unique_images = sum(
            page["unique_image_count"]
            for page in window
        )

        total_drawings = sum(
            page["drawing_cluster_count"]
            for page in window
        )

        total_words = sum(
            page["word_count"]
            for page in window
        )

        base_score = sum(
            page["page_score"]
            for page in window
        )

        diversity_bonus = 0.0

        if visual_page_count >= 2:
            diversity_bonus += 8.0

        if educational_page_count >= 2:
            diversity_bonus += 8.0

        if table_page_count:
            diversity_bonus += 8.0

        if total_words >= 800:
            diversity_bonus += 5.0

        score = round(
            base_score + diversity_bonus,
            3,
        )

        candidates.append(
            {
                "start_page": page_numbers[0],
                "end_page": page_numbers[-1],
                "page_numbers": page_numbers,
                "score": score,
                "base_page_score": round(
                    base_score,
                    3,
                ),
                "diversity_bonus": (
                    diversity_bonus
                ),
                "visual_page_count": (
                    visual_page_count
                ),
                "educational_page_count": (
                    educational_page_count
                ),
                "table_page_count": (
                    table_page_count
                ),
                "unique_image_count": (
                    total_unique_images
                ),
                "drawing_cluster_count": (
                    total_drawings
                ),
                "word_count": total_words,
                "page_previews": [
                    {
                        "page_number": (
                            page["page_number"]
                        ),
                        "preview": (
                            page["text_preview"]
                        ),
                    }
                    for page in window
                ],
            }
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate["score"],
            candidate["unique_image_count"],
            candidate["educational_page_count"],
        ),
        reverse=True,
    )


def write_csv(
    path: Path,
    pages: list[dict[str, Any]],
) -> None:
    fields = [
        "page_number",
        "text_char_count",
        "word_count",
        "displayed_image_count",
        "unique_image_count",
        "recurring_image_count",
        "image_area_ratio",
        "drawing_path_count",
        "drawing_cluster_count",
        "strict_table_count",
        "educational_keyword_count",
        "front_matter_keyword_count",
        "page_score",
        "text_preview",
        "scan_errors",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for page in pages:
            row = {
                field: page.get(field)
                for field in fields
            }

            row["scan_errors"] = json.dumps(
                row["scan_errors"],
                ensure_ascii=False,
            )

            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a textbook PDF and rank multimodal "
            "five-page BDA sample candidates."
        )
    )

    parser.add_argument(
        "--source-pdf",
        type=Path,
        default=DEFAULT_SOURCE_PDF,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--exclude-front-pages",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--exclude-end-pages",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--top-results",
        type=int,
        default=15,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.source_pdf.exists():
        raise FileNotFoundError(
            f"Source PDF not found: {args.source_pdf}"
        )

    if args.window_size < 1:
        raise ValueError(
            "window-size must be at least 1."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("============================================")
    print("SCANNING TEXTBOOK PAGES")
    print("============================================")
    print(f"Source PDF: {args.source_pdf}")
    print(f"Output:     {args.output_dir}")
    print(
        f"Window:     {args.window_size} pages"
    )
    print()

    pages: list[dict[str, Any]] = []

    with fitz.open(args.source_pdf) as document:
        if document.needs_pass:
            raise RuntimeError(
                "Source PDF is password protected."
            )

        page_count = document.page_count

        print(f"Total pages: {page_count}")

        for page_index in range(page_count):
            page = document.load_page(page_index)
            pages.append(scan_page(page))

            completed = page_index + 1

            if (
                completed == 1
                or completed % 25 == 0
                or completed == page_count
            ):
                print(
                    f"Scanned {completed}/{page_count} pages"
                )

    digest_frequencies = Counter(
        digest
        for page in pages
        for digest in page["image_digests"]
    )

    for page in pages:
        image_digests = page["image_digests"]

        page["unique_image_count"] = sum(
            1
            for digest in image_digests
            if digest_frequencies[digest] <= 2
        )

        page["recurring_image_count"] = sum(
            1
            for digest in image_digests
            if digest_frequencies[digest] > 2
        )

        page["page_score"] = (
            calculate_page_score(page)
        )

    candidate_pages = sorted(
        pages,
        key=lambda page: (
            page["page_score"],
            page["unique_image_count"],
            page["educational_keyword_count"],
        ),
        reverse=True,
    )

    candidate_windows = build_candidate_windows(
        pages=pages,
        window_size=args.window_size,
        exclude_front_pages=(
            args.exclude_front_pages
        ),
        exclude_end_pages=(
            args.exclude_end_pages
        ),
    )

    inventory_path = (
        args.output_dir / "page-inventory.json"
    )

    inventory_csv_path = (
        args.output_dir / "page-inventory.csv"
    )

    candidate_pages_path = (
        args.output_dir / "candidate-pages.json"
    )

    candidate_windows_path = (
        args.output_dir / "candidate-windows.json"
    )

    report_path = (
        args.output_dir / "scan-report.json"
    )

    inventory_path.write_text(
        json.dumps(
            pages,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_csv(
        inventory_csv_path,
        pages,
    )

    candidate_pages_path.write_text(
        json.dumps(
            candidate_pages,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    candidate_windows_path.write_text(
        json.dumps(
            candidate_windows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = {
        "generated_at": (
            datetime.now(timezone.utc).isoformat()
        ),
        "source_pdf": str(args.source_pdf),
        "page_count": len(pages),
        "window_size": args.window_size,
        "exclude_front_pages": (
            args.exclude_front_pages
        ),
        "exclude_end_pages": (
            args.exclude_end_pages
        ),
        "pages_with_images": sum(
            1
            for page in pages
            if page["displayed_image_count"] > 0
        ),
        "pages_with_unique_images": sum(
            1
            for page in pages
            if page["unique_image_count"] > 0
        ),
        "pages_with_drawing_clusters": sum(
            1
            for page in pages
            if page["drawing_cluster_count"] > 0
        ),
        "pages_with_strict_tables": sum(
            1
            for page in pages
            if page["strict_table_count"] > 0
        ),
        "pages_with_scan_errors": sum(
            1
            for page in pages
            if page["scan_errors"]
        ),
        "candidate_window_count": len(
            candidate_windows
        ),
        "top_window": (
            candidate_windows[0]
            if candidate_windows
            else None
        ),
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("============================================")
    print("PAGE SCAN COMPLETED")
    print("============================================")
    print(
        "Pages with displayed images: "
        f"{report['pages_with_images']}"
    )
    print(
        "Pages with unique images:    "
        f"{report['pages_with_unique_images']}"
    )
    print(
        "Pages with drawing clusters: "
        f"{report['pages_with_drawing_clusters']}"
    )
    print(
        "Pages with strict tables:    "
        f"{report['pages_with_strict_tables']}"
    )
    print(
        "Pages with scan errors:      "
        f"{report['pages_with_scan_errors']}"
    )

    print()
    print(
        f"Top {min(args.top_results, len(candidate_windows))} "
        "five-page candidates:"
    )
    print("-" * 78)

    for rank, candidate in enumerate(
        candidate_windows[:args.top_results],
        start=1,
    ):
        print(
            f"{rank:02d}. Pages "
            f"{candidate['start_page']}-"
            f"{candidate['end_page']} | "
            f"score={candidate['score']:.3f} | "
            f"unique_images="
            f"{candidate['unique_image_count']} | "
            f"visual_pages="
            f"{candidate['visual_page_count']} | "
            f"educational_pages="
            f"{candidate['educational_page_count']} | "
            f"tables="
            f"{candidate['table_page_count']} | "
            f"words={candidate['word_count']}"
        )

    print()
    print(f"Inventory JSON:   {inventory_path}")
    print(f"Inventory CSV:    {inventory_csv_path}")
    print(f"Candidate pages:  {candidate_pages_path}")
    print(f"Candidate windows:{candidate_windows_path}")
    print(f"Report:           {report_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as exc:
        print(
            f"Textbook page scan failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
