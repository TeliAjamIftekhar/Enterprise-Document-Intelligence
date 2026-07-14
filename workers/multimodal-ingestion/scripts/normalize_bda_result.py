from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SCHEMA_VERSION = "1.1"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return value


def ensure_string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def unique_nonempty(values: list[str]) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = " ".join(value.split())

        if not cleaned or cleaned in seen:
            continue

        seen.add(cleaned)
        results.append(cleaned)

    return results


def get_representation(
    element: dict[str, Any],
) -> tuple[str, str]:
    representation = element.get("representation", {})

    if not isinstance(representation, dict):
        return "", ""

    return (
        ensure_string(representation.get("text")),
        ensure_string(representation.get("markdown")),
    )


def get_locations(
    element: dict[str, Any],
    source_start_page: int,
) -> list[dict[str, Any]]:
    raw_locations = element.get("locations", [])

    if not isinstance(raw_locations, list):
        return []

    locations: list[dict[str, Any]] = []

    for raw_location in raw_locations:
        if not isinstance(raw_location, dict):
            continue

        page_index = raw_location.get("page_index")

        if not isinstance(page_index, int):
            continue

        bounding_box = raw_location.get("bounding_box")
        normalized_box = None

        if isinstance(bounding_box, dict):
            normalized_box = {
                "left": bounding_box.get("left"),
                "top": bounding_box.get("top"),
                "width": bounding_box.get("width"),
                "height": bounding_box.get("height"),
            }

        locations.append(
            {
                "sample_page_index": page_index,
                "sample_page_number": page_index + 1,
                "source_page_number": (
                    source_start_page + page_index
                ),
                "bounding_box": normalized_box,
            }
        )

    return locations


def get_page_indices(
    element: dict[str, Any],
) -> tuple[list[int], bool]:
    """
    Return zero-based page indices and whether they were recovered
    from locations because page_indices was absent or empty.
    """
    raw_page_indices = element.get("page_indices")

    if isinstance(raw_page_indices, list):
        explicit = sorted(
            {
                value
                for value in raw_page_indices
                if isinstance(value, int)
            }
        )

        if explicit:
            return explicit, False

    raw_locations = element.get("locations", [])

    recovered: set[int] = set()

    if isinstance(raw_locations, list):
        for location in raw_locations:
            if not isinstance(location, dict):
                continue

            page_index = location.get("page_index")

            if isinstance(page_index, int):
                recovered.add(page_index)

    return sorted(recovered), bool(recovered)


def get_crop_uris(
    element: dict[str, Any],
) -> list[str]:
    raw_crops = element.get("crop_images", [])

    if not isinstance(raw_crops, list):
        return []

    crop_uris: list[str] = []

    for crop in raw_crops:
        if isinstance(crop, str):
            crop_uris.append(crop)
            continue

        if not isinstance(crop, dict):
            continue

        for key in ("s3_uri", "s3Uri", "uri", "path"):
            value = crop.get(key)

            if isinstance(value, str) and value:
                crop_uris.append(value)
                break

    return crop_uris



def normalize_string_list(
    value: Any,
) -> list[str]:
    if not isinstance(value, list):
        return []

    return [
        item
        for item in value
        if isinstance(item, str)
    ]


def get_table_metadata(
    element: dict[str, Any],
) -> dict[str, Any]:
    representation = element.get(
        "representation",
        {},
    )

    if not isinstance(representation, dict):
        representation = {}

    return {
        "csv_text": ensure_string(
            representation.get("csv")
        ),
        "csv_s3_uri": ensure_string(
            element.get("csv_s3_uri")
        ),
        "headers": normalize_string_list(
            element.get("headers")
        ),
        "footers": normalize_string_list(
            element.get("footers")
        ),
    }


def get_asset_filename(uri: str) -> str:
    parsed = urlparse(uri)

    if parsed.scheme == "s3":
        return Path(parsed.path).name

    return Path(uri).name


def determine_modality(
    element_type: str,
    subtype: str,
) -> str:
    normalized_type = element_type.upper()
    normalized_subtype = subtype.upper()

    if normalized_type == "TABLE":
        return "table"

    if normalized_type == "FIGURE":
        mapping = {
            "QRCODE": "qr_code",
            "LOGO": "logo",
            "ICON": "icon",
            "IMAGE": "figure",
            "DIAGRAM": "diagram",
            "FLOWCHART": "flowchart",
            "GRAPH": "graph",
            "ILLUSTRATION": "illustration",
            "INFOGRAPHIC": "infographic",
        }

        return mapping.get(normalized_subtype, "figure")

    if normalized_subtype in {
        "TITLE",
        "SECTION_HEADER",
        "HEADER",
    }:
        return "heading"

    if normalized_subtype == "LIST":
        return "list"

    return "paragraph"


def determine_retrieval_priority(
    element_type: str,
    subtype: str,
) -> str:
    normalized_type = element_type.upper()
    normalized_subtype = subtype.upper()

    if normalized_subtype in {"TITLE", "SECTION_HEADER"}:
        return "high"

    if normalized_subtype in {
        "HEADER",
        "FOOTER",
        "LOGO",
        "ICON",
    }:
        return "low"

    if normalized_type in {"TABLE", "FIGURE"}:
        return "normal"

    return "normal"


def build_search_text(
    element_type: str,
    raw_text: str,
    markdown: str,
    title: str,
    summary: str,
) -> str:
    if element_type.upper() == "FIGURE":
        parts = [
            title,
            summary,
            raw_text,
        ]

    elif element_type.upper() == "TABLE":
        parts = [
            title,
            markdown,
            raw_text,
        ]

    else:
        parts = [raw_text]

    return "\n\n".join(
        unique_nonempty(parts)
    )


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")


def normalize(
    result: dict[str, Any],
    sample_metadata: dict[str, Any],
    result_json_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    elements = result.get("elements", [])

    if not isinstance(elements, list):
        raise ValueError(
            "BDA result contains no elements list."
        )

    result_metadata = result.get("metadata", {})

    if not isinstance(result_metadata, dict):
        result_metadata = {}

    document = result.get("document", {})

    if not isinstance(document, dict):
        document = {}

    source_statistics = document.get(
        "statistics",
        {},
    )

    if not isinstance(source_statistics, dict):
        source_statistics = {}

    book_id = ensure_string(
        sample_metadata.get("book_id")
    )

    book_version = ensure_string(
        sample_metadata.get("book_version")
    )

    source_start_page = int(
        sample_metadata.get("source_start_page", 1)
    )

    source_pdf = ensure_string(
        sample_metadata.get("source_pdf")
    )

    source_sample_s3_uri = ensure_string(
        sample_metadata.get("sample_s3_uri")
    )

    asset_directory = (
        result_json_path.parent / "assets"
    )

    content_units: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []

    recovered_page_reference_count = 0
    missing_page_reference_count = 0
    missing_asset_count = 0
    missing_assets: list[str] = []

    for element_index, raw_element in enumerate(elements):
        if not isinstance(raw_element, dict):
            continue

        bda_element_id = ensure_string(
            raw_element.get("id")
        )

        if not bda_element_id:
            bda_element_id = f"element-{element_index:06d}"

        element_type = ensure_string(
            raw_element.get("type")
        ).upper() or "UNKNOWN"

        subtype = ensure_string(
            raw_element.get("sub_type")
        ).upper() or "UNKNOWN"

        raw_text, markdown = get_representation(
            raw_element
        )

        title = ensure_string(
            raw_element.get("title")
        )

        summary = ensure_string(
            raw_element.get("summary")
        )

        sample_page_indices, recovered = (
            get_page_indices(raw_element)
        )

        if recovered:
            recovered_page_reference_count += 1

        if not sample_page_indices:
            missing_page_reference_count += 1

        source_page_numbers = [
            source_start_page + page_index
            for page_index in sample_page_indices
        ]

        locations = get_locations(
            element=raw_element,
            source_start_page=source_start_page,
        )

        crop_uris = get_crop_uris(raw_element)

        crop_asset_names = [
            get_asset_filename(uri)
            for uri in crop_uris
            if get_asset_filename(uri)
        ]

        local_asset_paths = [
            str(asset_directory / filename)
            for filename in crop_asset_names
        ]

        table_metadata = get_table_metadata(
            raw_element
        )

        csv_s3_uri = table_metadata[
            "csv_s3_uri"
        ]

        csv_asset_name = (
            get_asset_filename(csv_s3_uri)
            if csv_s3_uri
            else ""
        )

        csv_local_path = (
            str(asset_directory / csv_asset_name)
            if csv_asset_name
            else ""
        )

        all_asset_s3_uris = list(crop_uris)

        if csv_s3_uri:
            all_asset_s3_uris.append(
                csv_s3_uri
            )

        all_asset_local_paths = list(
            local_asset_paths
        )

        if csv_local_path:
            all_asset_local_paths.append(
                csv_local_path
            )

        for local_asset_path in all_asset_local_paths:
            if not Path(local_asset_path).exists():
                missing_asset_count += 1
                missing_assets.append(local_asset_path)

        modality = determine_modality(
            element_type=element_type,
            subtype=subtype,
        )

        retrieval_priority = (
            determine_retrieval_priority(
                element_type=element_type,
                subtype=subtype,
            )
        )

        quality_flags: list[str] = []

        if recovered:
            quality_flags.append(
                "page_reference_recovered_from_locations"
            )

        if not sample_page_indices:
            quality_flags.append(
                "missing_page_reference"
            )

        if (
            element_type == "FIGURE"
            and not summary
        ):
            quality_flags.append(
                "missing_generated_summary"
            )

        if crop_uris and not local_asset_paths:
            quality_flags.append(
                "missing_local_asset_mapping"
            )

        if element_type == "TABLE":
            if not table_metadata["csv_text"]:
                quality_flags.append(
                    "missing_inline_csv"
                )

            if not csv_s3_uri:
                quality_flags.append(
                    "missing_csv_s3_uri"
                )

        if subtype == "QRCODE":
            subtype_evidence = " ".join(
                [
                    title,
                    summary,
                    raw_text,
                ]
            ).lower()

            if "qr" not in subtype_evidence:
                quality_flags.append(
                    "possible_visual_subtype_misclassification"
                )

        unit_id = (
            f"{book_id}:{book_version}:"
            f"bda:{bda_element_id}"
        )

        content_unit = {
            "schema_version": SCHEMA_VERSION,
            "unit_id": unit_id,
            "book_id": book_id,
            "book_version": book_version,
            "source_kind": "bda_standard_output",
            "source_pdf": source_pdf,
            "source_sample_s3_uri": source_sample_s3_uri,
            "bda_element_id": bda_element_id,
            "element_index": element_index,
            "element_type": element_type,
            "element_sub_type": subtype,
            "modality": modality,
            "reading_order": raw_element.get(
                "reading_order"
            ),
            "sample_page_indices": (
                sample_page_indices
            ),
            "source_page_numbers": (
                source_page_numbers
            ),
            "locations": locations,
            "raw_text": raw_text,
            "markdown": markdown,
            "generated_title": title,
            "generated_summary": summary,
            "search_text": build_search_text(
                element_type=element_type,
                raw_text=raw_text,
                markdown=markdown,
                title=title,
                summary=summary,
            ),
            "asset_s3_uris": all_asset_s3_uris,
            "asset_local_paths": all_asset_local_paths,
            "retrieval_priority": retrieval_priority,
            "quality_flags": quality_flags,
        }

        content_units.append(content_unit)

        if element_type == "FIGURE":
            figures.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "figure_id": unit_id,
                    "book_id": book_id,
                    "book_version": book_version,
                    "bda_element_id": bda_element_id,
                    "element_index": element_index,
                    "figure_sub_type": subtype,
                    "modality": modality,
                    "reading_order": raw_element.get(
                        "reading_order"
                    ),
                    "source_page_numbers": (
                        source_page_numbers
                    ),
                    "locations": locations,
                    "generated_title": title,
                    "generated_summary": summary,
                    "ocr_text": raw_text,
                    "markdown": markdown,
                    "search_text": (
                        content_unit["search_text"]
                    ),
                    "crop_s3_uris": crop_uris,
                    "crop_local_paths": (
                        local_asset_paths
                    ),
                    "retrieval_priority": (
                        retrieval_priority
                    ),
                    "quality_flags": quality_flags,
                }
            )

        if element_type == "TABLE":
            tables.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "table_id": unit_id,
                    "book_id": book_id,
                    "book_version": book_version,
                    "bda_element_id": bda_element_id,
                    "element_index": element_index,
                    "reading_order": raw_element.get(
                        "reading_order"
                    ),
                    "source_page_numbers": (
                        source_page_numbers
                    ),
                    "locations": locations,
                    "generated_title": title,
                    "generated_summary": summary,
                    "plain_text": raw_text,
                    "markdown": markdown,
                    "csv_text": table_metadata[
                        "csv_text"
                    ],
                    "csv_s3_uri": csv_s3_uri,
                    "csv_local_path": csv_local_path,
                    "headers": table_metadata[
                        "headers"
                    ],
                    "footers": table_metadata[
                        "footers"
                    ],
                    "crop_s3_uris": crop_uris,
                    "crop_local_paths": (
                        local_asset_paths
                    ),
                    "quality_flags": quality_flags,
                }
            )

    reported_table_count = int(
        source_statistics.get("table_count", 0)
    )

    reported_figure_count = int(
        source_statistics.get("figure_count", 0)
    )

    warnings: list[str] = []

    if reported_table_count != len(tables):
        warnings.append(
            "BDA reported "
            f"{reported_table_count} table(s), but emitted "
            f"{len(tables)} explicit TABLE element(s). "
            "No inferred table records were created."
        )

    if reported_figure_count != len(figures):
        warnings.append(
            "BDA reported "
            f"{reported_figure_count} figure(s), but emitted "
            f"{len(figures)} FIGURE element(s)."
        )

    if missing_page_reference_count:
        warnings.append(
            f"{missing_page_reference_count} element(s) "
            "have no recoverable page reference."
        )

    if missing_asset_count:
        warnings.append(
            f"{missing_asset_count} referenced local asset(s) "
            "were not found."
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": (
            datetime.now(timezone.utc).isoformat()
        ),
        "result_json": str(result_json_path),
        "book_id": book_id,
        "book_version": book_version,
        "source_start_page": source_start_page,
        "source_page_count": result_metadata.get(
            "number_of_pages"
        ),
        "source_statistics": source_statistics,
        "input_element_count": len(elements),
        "normalized_content_unit_count": len(
            content_units
        ),
        "normalized_figure_count": len(figures),
        "normalized_table_count": len(tables),
        "recovered_page_reference_count": (
            recovered_page_reference_count
        ),
        "missing_page_reference_count": (
            missing_page_reference_count
        ),
        "missing_asset_count": missing_asset_count,
        "missing_assets": sorted(
            set(missing_assets)
        ),
        "warnings": warnings,
        "normalization_policy": {
            "statistics_are_advisory": True,
            "tables_require_explicit_table_elements": True,
            "table_csv_is_preserved": True,
            "figure_ocr_and_summary_are_separate": True,
            "page_indices_fallback_to_locations": True,
            "source_pages_are_one_based": True,
        },
    }

    return content_units, figures, tables, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize Amazon BDA document output "
            "into JSONL retrieval records."
        )
    )

    parser.add_argument(
        "result_json",
        type=Path,
    )

    parser.add_argument(
        "--sample-metadata",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    result = load_json(args.result_json)
    sample_metadata = load_json(
        args.sample_metadata
    )

    (
        content_units,
        figures,
        tables,
        report,
    ) = normalize(
        result=result,
        sample_metadata=sample_metadata,
        result_json_path=args.result_json,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    content_units_path = (
        args.output_dir / "content-units.jsonl"
    )

    figures_path = (
        args.output_dir / "figures.jsonl"
    )

    tables_path = (
        args.output_dir / "tables.jsonl"
    )

    report_path = (
        args.output_dir
        / "normalization-report.json"
    )

    write_jsonl(
        content_units_path,
        content_units,
    )

    write_jsonl(
        figures_path,
        figures,
    )

    write_jsonl(
        tables_path,
        tables,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("============================================")
    print("BDA NORMALIZATION COMPLETED")
    print("============================================")
    print(
        "Content units:       "
        f"{len(content_units)}"
    )
    print(
        "Figures:             "
        f"{len(figures)}"
    )
    print(
        "Tables:              "
        f"{len(tables)}"
    )
    print(
        "Recovered page refs: "
        f"{report['recovered_page_reference_count']}"
    )
    print(
        "Missing page refs:   "
        f"{report['missing_page_reference_count']}"
    )
    print(
        "Missing assets:      "
        f"{report['missing_asset_count']}"
    )

    if report["warnings"]:
        print()
        print("Warnings:")

        for warning in report["warnings"]:
            print(f"- {warning}")

    print()
    print(f"Content units: {content_units_path}")
    print(f"Figures:       {figures_path}")
    print(f"Tables:        {tables_path}")
    print(f"Report:        {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
