from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TABLE_KEYWORDS = {
    "table",
    "tables",
    "cell",
    "cells",
    "row",
    "rows",
    "column",
    "columns",
    "csv",
}


def contains_markdown_table(markdown: str) -> bool:
    lines = markdown.splitlines()

    for index in range(len(lines) - 1):
        current = lines[index]
        following = lines[index + 1]

        if "|" not in current:
            continue

        if re.search(
            r"\|\s*:?-{3,}:?\s*(\||$)",
            following,
        ):
            return True

    return False


def recursive_search(
    value: Any,
    path: str,
    matches: list[dict[str, Any]],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            normalized_key = key.lower()

            if any(
                keyword in normalized_key
                for keyword in TABLE_KEYWORDS
            ):
                matches.append(
                    {
                        "path": child_path,
                        "reason": "key",
                        "value_preview": repr(child)[:500],
                    }
                )

            recursive_search(
                value=child,
                path=child_path,
                matches=matches,
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            recursive_search(
                value=child,
                path=f"{path}[{index}]",
                matches=matches,
            )

    elif isinstance(value, str):
        lower_value = value.lower()

        indicators = [
            "<table",
            "</table>",
            ".csv",
            '"table"',
        ]

        found = [
            indicator
            for indicator in indicators
            if indicator in lower_value
        ]

        if contains_markdown_table(value):
            found.append("markdown-table")

        if found:
            matches.append(
                {
                    "path": path,
                    "reason": ",".join(found),
                    "value_preview": value[:500],
                }
            )


def get_element_pages(
    element: dict[str, Any],
) -> list[int]:
    page_indices = element.get("page_indices", [])

    if isinstance(page_indices, list):
        return [
            page_index + 1
            for page_index in page_indices
            if isinstance(page_index, int)
        ]

    return []


def preview_text(
    value: Any,
    limit: int = 500,
) -> str:
    if not isinstance(value, str):
        return ""

    return " ".join(value.split())[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose table and figure representation "
            "inside a BDA document result."
        )
    )

    parser.add_argument(
        "result_json",
        type=Path,
    )

    args = parser.parse_args()

    data = json.loads(
        args.result_json.read_text(encoding="utf-8")
    )

    pages = data.get("pages", [])
    elements = data.get("elements", [])
    document = data.get("document", {})

    print("============================================")
    print("BDA SCHEMA DIAGNOSTIC")
    print("============================================")

    print("\nDocument statistics:")

    statistics = (
        document.get("statistics", {})
        if isinstance(document, dict)
        else {}
    )

    print(json.dumps(statistics, indent=2))

    print("\nPer-page statistics:")
    print("-" * 70)

    pages_reporting_tables: list[int] = []

    for page in pages:
        if not isinstance(page, dict):
            continue

        physical_page = int(page["page_index"]) + 1
        page_statistics = page.get("statistics", {})

        print(
            f"Physical page {physical_page}: "
            f"{json.dumps(page_statistics, sort_keys=True)}"
        )

        if (
            isinstance(page_statistics, dict)
            and page_statistics.get("table_count", 0) > 0
        ):
            pages_reporting_tables.append(
                physical_page
            )

    print(
        "\nPages reporting tables:",
        pages_reporting_tables,
    )

    print("\nElement type/subtype and raw keys:")
    print("-" * 70)

    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            continue

        print(
            f"{index:02d} | "
            f"{element.get('type')} / "
            f"{element.get('sub_type')} | "
            f"pages={get_element_pages(element)} | "
            f"keys={sorted(element.keys())}"
        )

    matches: list[dict[str, Any]] = []

    recursive_search(
        value=data,
        path="$",
        matches=matches,
    )

    print("\nTable-related keys and representations:")
    print("-" * 70)

    if matches:
        for match in matches:
            print(
                f"Path: {match['path']}\n"
                f"Reason: {match['reason']}\n"
                f"Preview: {match['value_preview']}\n"
            )
    else:
        print(
            "No table-specific keys, HTML tables, CSV references, "
            "or Markdown table syntax were found."
        )

    print("\nElements on pages reporting tables:")
    print("-" * 70)

    table_page_elements: list[dict[str, Any]] = []

    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            continue

        element_pages = get_element_pages(element)

        if not set(element_pages).intersection(
            pages_reporting_tables
        ):
            continue

        representation = element.get(
            "representation",
            {},
        )

        if not isinstance(representation, dict):
            representation = {}

        record = {
            "index": index,
            "id": element.get("id"),
            "type": element.get("type"),
            "sub_type": element.get("sub_type"),
            "pages": element_pages,
            "reading_order": element.get(
                "reading_order"
            ),
            "raw_keys": sorted(element.keys()),
            "text": representation.get("text"),
            "markdown": representation.get(
                "markdown"
            ),
            "crop_images": element.get(
                "crop_images",
                [],
            ),
        }

        table_page_elements.append(record)

        print(
            f"\nIndex: {index}\n"
            f"Type: {record['type']} / "
            f"{record['sub_type']}\n"
            f"Pages: {element_pages}\n"
            f"Reading order: {record['reading_order']}\n"
            f"Keys: {record['raw_keys']}\n"
            f"Text: {preview_text(record['text'])}\n"
            f"Markdown: "
            f"{preview_text(record['markdown'])}\n"
            f"Crop images: {record['crop_images']}"
        )

    figure_records = []

    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            continue

        if element.get("type") != "FIGURE":
            continue

        figure_records.append(
            {
                "index": index,
                "sub_type": element.get(
                    "sub_type"
                ),
                "pages": get_element_pages(
                    element
                ),
                "has_summary": bool(
                    element.get("summary")
                ),
                "crop_count": len(
                    element.get(
                        "crop_images",
                        [],
                    )
                ),
            }
        )

    report = {
        "document_statistics": statistics,
        "pages_reporting_tables": (
            pages_reporting_tables
        ),
        "table_related_matches": matches,
        "table_page_elements": (
            table_page_elements
        ),
        "figure_records": figure_records,
    }

    output_path = (
        args.result_json.parent
        / "schema-diagnostic.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nFigure records:")
    print("-" * 70)

    for figure in figure_records:
        print(
            f"Index {figure['index']}: "
            f"{figure['sub_type']}, "
            f"pages={figure['pages']}, "
            f"summary={figure['has_summary']}, "
            f"crops={figure['crop_count']}"
        )

    print(f"\nDiagnostic saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
