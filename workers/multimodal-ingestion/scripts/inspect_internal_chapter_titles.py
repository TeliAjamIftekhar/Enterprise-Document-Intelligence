#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import fitz


SECTION_LABELS = {
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
}


def normalize(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def is_bold(span: dict[str, Any]) -> bool:
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


def reject_title(value: str) -> bool:
    text = normalize(value)

    if not text:
        return True

    if text in SECTION_LABELS:
        return True

    if text.casefold() in {
        item.casefold()
        for item in SECTION_LABELS
    }:
        return True

    if re.fullmatch(
        r"[0-9०-९IVXivx .)\-–—]+",
        text,
    ):
        return True

    if re.match(
        r"^(?:इकाई|अध्याय|पाठ)"
        r"\s*[0-9०-९IVXivx]*"
        r"\s*[:.\-–—]?",
        text,
    ):
        return True

    if text.casefold().startswith(
        "reprint"
    ):
        return True

    if len(text) > 120:
        return True

    return False


def page_title_candidates(
    page: fitz.Page,
) -> list[dict[str, Any]]:
    payload = page.get_text(
        "dict",
        sort=True,
    )

    spans: list[dict[str, Any]] = []

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
                text = normalize(
                    str(
                        span.get(
                            "text",
                            "",
                        )
                    )
                )

                size = float(
                    span.get(
                        "size",
                        0,
                    )
                )

                bbox = [
                    float(value)
                    for value in span.get(
                        "bbox",
                        [0, 0, 0, 0],
                    )
                ]

                if not is_bold(span):
                    continue

                if size < 27:
                    continue

                if bbox[1] < 0:
                    continue

                if bbox[1] > (
                    page.rect.height * 0.65
                ):
                    continue

                if reject_title(text):
                    continue

                spans.append({
                    "text": text,
                    "font_size": round(
                        size,
                        2,
                    ),
                    "bbox": [
                        round(value, 2)
                        for value in bbox
                    ],
                })

    if not spans:
        return []

    spans.sort(
        key=lambda item: (
            item["bbox"][1],
            item["bbox"][0],
        )
    )

    groups: list[
        list[dict[str, Any]]
    ] = []

    for span in spans:
        if not groups:
            groups.append([span])
            continue

        previous = groups[-1][-1]

        vertical_gap = (
            span["bbox"][1]
            - previous["bbox"][1]
        )

        same_font = abs(
            span["font_size"]
            - previous["font_size"]
        ) <= 1.5

        if (
            same_font
            and vertical_gap <= max(
                65,
                span["font_size"] * 1.9,
            )
        ):
            groups[-1].append(span)
        else:
            groups.append([span])

    candidates: list[
        dict[str, Any]
    ] = []

    for group in groups:
        title = normalize(
            " ".join(
                item["text"]
                for item in group
            )
        )

        if reject_title(title):
            continue

        candidates.append({
            "title": title,
            "font_size": max(
                item["font_size"]
                for item in group
            ),
            "y": min(
                item["bbox"][1]
                for item in group
            ),
            "pieces": [
                item["text"]
                for item in group
            ],
        })

    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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

    report = json.loads(
        args.extraction_report.read_text(
            encoding="utf-8"
        )
    )

    target_directory = Path(
        report["target_directory"]
    )

    results: list[
        dict[str, Any]
    ] = []

    for document in report["documents"]:
        source_filename = document[
            "source_filename"
        ]

        pdf_path = (
            target_directory
            / source_filename
        )

        canonical_start = int(
            document[
                "canonical_start_page"
            ]
        )

        document_candidates = []

        with fitz.open(
            pdf_path
        ) as pdf:
            for page_index in range(
                pdf.page_count
            ):
                page = pdf[page_index]

                candidates = (
                    page_title_candidates(page)
                )

                for candidate in candidates:
                    candidate.update({
                        "source_page": (
                            page_index + 1
                        ),
                        "canonical_page": (
                            canonical_start
                            + page_index
                        ),
                    })

                    document_candidates.append(
                        candidate
                    )

        results.append({
            "order": document["order"],
            "document_id": (
                document["document_id"]
            ),
            "source_filename": (
                source_filename
            ),
            "source_page_count": (
                document["page_count"]
            ),
            "canonical_start_page": (
                document[
                    "canonical_start_page"
                ]
            ),
            "canonical_end_page": (
                document[
                    "canonical_end_page"
                ]
            ),
            "title_candidates": (
                document_candidates
            ),
        })

    payload = {
        "schema_version": "1.0",
        "book_id": report["book_id"],
        "book_version": (
            report["book_version"]
        ),
        "documents": results,
        "aws_calls": 0,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 90)
    print("INTERNAL CHAPTER TITLE CANDIDATES")
    print("=" * 90)

    total_candidates = 0

    for document in results:
        candidates = document[
            "title_candidates"
        ]

        total_candidates += len(
            candidates
        )

        print()
        print(
            f"{document['order']:02}. "
            f"{document['source_filename']} | "
            f"pages="
            f"{document['source_page_count']} | "
            f"candidates={len(candidates)}"
        )

        for candidate in candidates:
            print(
                "    "
                f"source_page="
                f"{candidate['source_page']:>2} | "
                f"canonical_page="
                f"{candidate['canonical_page']:>3} | "
                f"size="
                f"{candidate['font_size']:>5} | "
                f"{candidate['title']}"
            )

    print()
    print("-" * 90)
    print("Documents:", len(results))
    print(
        "Internal title candidates:",
        total_candidates,
    )
    print("Report:", args.output)
    print("AWS calls: 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
