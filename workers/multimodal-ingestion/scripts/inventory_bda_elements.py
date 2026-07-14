from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def get_representation(
    element: dict[str, Any],
) -> tuple[str, str]:
    representation = element.get("representation", {})

    if not isinstance(representation, dict):
        return "", ""

    text = representation.get("text", "")
    markdown = representation.get("markdown", "")

    return (
        text if isinstance(text, str) else "",
        markdown if isinstance(markdown, str) else "",
    )


def get_page_indices(
    element: dict[str, Any],
) -> list[int]:
    page_indices = element.get("page_indices")

    if isinstance(page_indices, list):
        return [
            int(value)
            for value in page_indices
            if isinstance(value, int)
        ]

    locations = element.get("locations", [])

    indices: list[int] = []

    if isinstance(locations, list):
        for location in locations:
            if not isinstance(location, dict):
                continue

            page_index = location.get("page_index")

            if isinstance(page_index, int):
                indices.append(page_index)

    return sorted(set(indices))


def get_crop_images(
    element: dict[str, Any],
) -> list[str]:
    crop_images = ensure_list(
        element.get("crop_images")
    )

    results: list[str] = []

    for item in crop_images:
        if isinstance(item, str):
            results.append(item)

        elif isinstance(item, dict):
            for key in (
                "s3_uri",
                "s3Uri",
                "uri",
                "path",
            ):
                value = item.get(key)

                if isinstance(value, str):
                    results.append(value)
                    break

    return results


def get_asset_names(
    crop_images: list[str],
) -> list[str]:
    names: list[str] = []

    for uri in crop_images:
        parsed = urlparse(uri)

        if parsed.scheme == "s3":
            name = Path(parsed.path).name
        else:
            name = Path(uri).name

        if name:
            names.append(name)

    return names


def get_locations(
    element: dict[str, Any],
) -> list[dict[str, Any]]:
    locations = element.get("locations", [])

    if not isinstance(locations, list):
        return []

    return [
        location
        for location in locations
        if isinstance(location, dict)
    ]


def build_inventory(
    data: dict[str, Any],
) -> dict[str, Any]:
    elements = data.get("elements", [])
    pages = data.get("pages", [])
    document = data.get("document", {})

    if not isinstance(elements, list):
        raise ValueError(
            "result.json contains no valid elements list."
        )

    if not isinstance(pages, list):
        raise ValueError(
            "result.json contains no valid pages list."
        )

    records: list[dict[str, Any]] = []

    type_counter: Counter[str] = Counter()
    subtype_counter: Counter[str] = Counter()
    page_counter: Counter[int] = Counter()
    type_subtype_counter: Counter[str] = Counter()

    for index, raw_element in enumerate(elements):
        if not isinstance(raw_element, dict):
            continue

        element_type = str(
            raw_element.get("type", "UNKNOWN")
        )

        subtype = str(
            raw_element.get("sub_type", "UNKNOWN")
        )

        text, markdown = get_representation(
            raw_element
        )

        zero_based_pages = get_page_indices(
            raw_element
        )

        physical_pages = [
            page_index + 1
            for page_index in zero_based_pages
        ]

        crop_images = get_crop_images(
            raw_element
        )

        locations = get_locations(
            raw_element
        )

        title = raw_element.get("title")
        summary = raw_element.get("summary")

        record = {
            "element_index": index,
            "id": raw_element.get("id"),
            "type": element_type,
            "sub_type": subtype,
            "reading_order": raw_element.get(
                "reading_order"
            ),
            "page_indices": zero_based_pages,
            "physical_page_numbers": physical_pages,
            "title": (
                title if isinstance(title, str) else None
            ),
            "summary": (
                summary
                if isinstance(summary, str)
                else None
            ),
            "text": text,
            "markdown": markdown,
            "text_length": len(text),
            "markdown_length": len(markdown),
            "crop_images": crop_images,
            "crop_asset_names": get_asset_names(
                crop_images
            ),
            "crop_image_count": len(crop_images),
            "locations": locations,
            "location_count": len(locations),
            "raw_keys": sorted(raw_element.keys()),
        }

        records.append(record)

        type_counter[element_type] += 1
        subtype_counter[subtype] += 1
        type_subtype_counter[
            f"{element_type}/{subtype}"
        ] += 1

        for page_number in physical_pages:
            page_counter[page_number] += 1

    document_statistics = {}

    if isinstance(document, dict):
        statistics = document.get("statistics")

        if isinstance(statistics, dict):
            document_statistics = statistics

    return {
        "source_statistics": document_statistics,
        "page_count": len(pages),
        "element_count": len(records),
        "type_counts": dict(
            sorted(type_counter.items())
        ),
        "subtype_counts": dict(
            sorted(subtype_counter.items())
        ),
        "type_subtype_counts": dict(
            sorted(type_subtype_counter.items())
        ),
        "elements_per_physical_page": {
            str(key): value
            for key, value in sorted(
                page_counter.items()
            )
        },
        "elements": records,
    }


def save_csv(
    inventory: dict[str, Any],
    output_path: Path,
) -> None:
    fields = [
        "element_index",
        "id",
        "type",
        "sub_type",
        "physical_page_numbers",
        "reading_order",
        "title",
        "summary",
        "text_length",
        "markdown_length",
        "crop_image_count",
        "crop_asset_names",
        "location_count",
        "raw_keys",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for element in inventory["elements"]:
            row = {
                key: element.get(key)
                for key in fields
            }

            for key in (
                "physical_page_numbers",
                "crop_asset_names",
                "raw_keys",
            ):
                row[key] = json.dumps(
                    row[key],
                    ensure_ascii=False,
                )

            writer.writerow(row)


def print_summary(
    inventory: dict[str, Any],
) -> None:
    print("============================================")
    print("BDA ELEMENT INVENTORY")
    print("============================================")
    print(
        f"Pages:        {inventory['page_count']}"
    )
    print(
        f"Elements:     {inventory['element_count']}"
    )
    print()

    print("Source document statistics:")

    for key, value in (
        inventory["source_statistics"].items()
    ):
        print(f"- {key}: {value}")

    print()
    print("Element types:")

    for key, value in (
        inventory["type_counts"].items()
    ):
        print(f"- {key}: {value}")

    print()
    print("Element subtypes:")

    for key, value in (
        inventory["subtype_counts"].items()
    ):
        print(f"- {key}: {value}")

    print()
    print("Type/subtype combinations:")

    for key, value in (
        inventory[
            "type_subtype_counts"
        ].items()
    ):
        print(f"- {key}: {value}")

    print()
    print("Non-text and cropped elements:")
    print("-" * 60)

    interesting = [
        element
        for element in inventory["elements"]
        if (
            element["type"] != "TEXT"
            or element["crop_image_count"] > 0
            or element["sub_type"]
            not in {"PARAGRAPH", "UNKNOWN"}
        )
    ]

    if not interesting:
        print("No non-text elements found.")
        return

    for element in interesting:
        print(
            f"Index: {element['element_index']} | "
            f"Type: {element['type']} | "
            f"Subtype: {element['sub_type']} | "
            f"Pages: {element['physical_page_numbers']}"
        )

        if element["title"]:
            print(f"Title: {element['title']}")

        if element["summary"]:
            preview = (
                element["summary"]
                .replace("\n", " ")[:240]
            )
            print(f"Summary: {preview}")

        if element["crop_asset_names"]:
            print(
                "Assets: "
                + ", ".join(
                    element["crop_asset_names"]
                )
            )

        if element["text"]:
            preview = (
                element["text"]
                .replace("\n", " ")[:180]
            )
            print(f"Text: {preview}")

        print("-" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an inventory of BDA document elements."
        )
    )

    parser.add_argument(
        "result_json",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.result_json.exists():
        raise FileNotFoundError(
            f"BDA result not found: {args.result_json}"
        )

    data = json.loads(
        args.result_json.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, dict):
        raise ValueError(
            "BDA result root must be a JSON object."
        )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.result_json.parent
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory = build_inventory(data)

    json_path = (
        output_dir
        / "element-inventory.json"
    )

    csv_path = (
        output_dir
        / "element-inventory.csv"
    )

    json_path.write_text(
        json.dumps(
            inventory,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    save_csv(
        inventory=inventory,
        output_path=csv_path,
    )

    print_summary(inventory)

    print()
    print(f"JSON inventory: {json_path}")
    print(f"CSV inventory:  {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
