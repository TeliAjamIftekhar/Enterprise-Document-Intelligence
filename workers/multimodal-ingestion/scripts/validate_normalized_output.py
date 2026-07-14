from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_CONTENT_UNIT_FIELDS = {
    "schema_version",
    "unit_id",
    "book_id",
    "book_version",
    "bda_element_id",
    "element_index",
    "element_type",
    "element_sub_type",
    "modality",
    "source_page_numbers",
    "locations",
    "raw_text",
    "markdown",
    "generated_title",
    "generated_summary",
    "search_text",
    "asset_s3_uris",
    "asset_local_paths",
    "retrieval_priority",
    "quality_flags",
}

REQUIRED_FIGURE_FIELDS = {
    "schema_version",
    "figure_id",
    "book_id",
    "book_version",
    "bda_element_id",
    "element_index",
    "figure_sub_type",
    "modality",
    "source_page_numbers",
    "locations",
    "generated_title",
    "generated_summary",
    "ocr_text",
    "markdown",
    "search_text",
    "crop_s3_uris",
    "crop_local_paths",
    "retrieval_priority",
    "quality_flags",
}

REQUIRED_TABLE_FIELDS = {
    "schema_version",
    "table_id",
    "book_id",
    "book_version",
    "bda_element_id",
    "element_index",
    "source_page_numbers",
    "locations",
    "plain_text",
    "markdown",
    "csv_text",
    "csv_s3_uri",
    "csv_local_path",
    "headers",
    "footers",
    "crop_s3_uris",
    "crop_local_paths",
    "quality_flags",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path} "
                    f"at line {line_number}."
                )

            records.append(record)

    return records


def validate_required_fields(
    record: dict[str, Any],
    required_fields: set[str],
    record_name: str,
    errors: list[str],
) -> None:
    missing = sorted(required_fields - set(record))

    if missing:
        errors.append(
            f"{record_name} is missing fields: {missing}"
        )


def validate_unique_values(
    records: list[dict[str, Any]],
    field: str,
    record_type: str,
    errors: list[str],
) -> None:
    seen: set[str] = set()

    for index, record in enumerate(records):
        value = record.get(field)

        if not isinstance(value, str) or not value:
            errors.append(
                f"{record_type} {index} has invalid {field}: "
                f"{value!r}"
            )
            continue

        if value in seen:
            errors.append(
                f"Duplicate {field} in {record_type}: {value}"
            )

        seen.add(value)


def validate_page_numbers(
    record: dict[str, Any],
    source_start_page: int,
    source_end_page: int,
    record_name: str,
    errors: list[str],
) -> None:
    pages = record.get("source_page_numbers")

    if not isinstance(pages, list) or not pages:
        errors.append(
            f"{record_name} has no source page numbers."
        )
        return

    for page in pages:
        if not isinstance(page, int):
            errors.append(
                f"{record_name} has non-integer page: {page!r}"
            )
            continue

        if not source_start_page <= page <= source_end_page:
            errors.append(
                f"{record_name} page {page} is outside "
                f"{source_start_page}-{source_end_page}."
            )


def validate_local_assets(
    record: dict[str, Any],
    field: str,
    record_name: str,
    errors: list[str],
) -> None:
    paths = record.get(field, [])

    if not isinstance(paths, list):
        errors.append(
            f"{record_name}.{field} must be a list."
        )
        return

    for value in paths:
        if not isinstance(value, str) or not value:
            errors.append(
                f"{record_name} has invalid asset path: "
                f"{value!r}"
            )
            continue

        path = Path(value)

        if not path.exists():
            errors.append(
                f"{record_name} references missing asset: "
                f"{path}"
            )

        elif not path.is_file():
            errors.append(
                f"{record_name} asset is not a file: {path}"
            )



def validate_local_file(
    value: Any,
    record_name: str,
    field_name: str,
    errors: list[str],
) -> None:
    if not isinstance(value, str) or not value:
        errors.append(
            f"{record_name}.{field_name} "
            "must be a non-empty string."
        )
        return

    path = Path(value)

    if not path.exists():
        errors.append(
            f"{record_name}.{field_name} "
            f"does not exist: {path}"
        )

    elif not path.is_file():
        errors.append(
            f"{record_name}.{field_name} "
            f"is not a file: {path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate normalized multimodal BDA records."
        )
    )

    parser.add_argument(
        "normalized_dir",
        type=Path,
    )

    args = parser.parse_args()
    normalized_dir = args.normalized_dir

    content_units_path = (
        normalized_dir / "content-units.jsonl"
    )
    figures_path = normalized_dir / "figures.jsonl"
    tables_path = normalized_dir / "tables.jsonl"
    report_path = (
        normalized_dir / "normalization-report.json"
    )

    content_units = load_jsonl(content_units_path)
    figures = load_jsonl(figures_path)
    tables = load_jsonl(tables_path)
    report = load_json(report_path)

    errors: list[str] = []
    warnings: list[str] = []

    source_start_page = int(
        report.get("source_start_page", 1)
    )

    source_page_count = int(
        report.get("source_page_count", 0)
    )

    source_end_page = (
        source_start_page + source_page_count - 1
    )

    if len(content_units) != report.get(
        "normalized_content_unit_count"
    ):
        errors.append(
            "Content-unit JSONL count does not match "
            "normalization report."
        )

    if len(figures) != report.get(
        "normalized_figure_count"
    ):
        errors.append(
            "Figure JSONL count does not match "
            "normalization report."
        )

    if len(tables) != report.get(
        "normalized_table_count"
    ):
        errors.append(
            "Table JSONL count does not match "
            "normalization report."
        )

    if len(content_units) != report.get(
        "input_element_count"
    ):
        errors.append(
            "Not all BDA elements were preserved as "
            "content units."
        )

    validate_unique_values(
        content_units,
        "unit_id",
        "content unit",
        errors,
    )

    validate_unique_values(
        figures,
        "figure_id",
        "figure",
        errors,
    )

    validate_unique_values(
        tables,
        "table_id",
        "table",
        errors,
    )

    content_unit_ids = {
        record.get("unit_id")
        for record in content_units
    }

    for index, record in enumerate(content_units):
        record_name = f"content unit {index}"

        validate_required_fields(
            record,
            REQUIRED_CONTENT_UNIT_FIELDS,
            record_name,
            errors,
        )

        validate_page_numbers(
            record,
            source_start_page,
            source_end_page,
            record_name,
            errors,
        )

        validate_local_assets(
            record,
            "asset_local_paths",
            record_name,
            errors,
        )

        search_text = record.get("search_text")

        if not isinstance(search_text, str):
            errors.append(
                f"{record_name}.search_text must be a string."
            )

        elif not search_text.strip():
            warnings.append(
                f"{record_name} has empty search text."
            )

        element_type = record.get("element_type")

        if element_type == "TABLE":
            table_id = record.get("unit_id")

            if not any(
                table.get("table_id") == table_id
                for table in tables
            ):
                errors.append(
                    f"{record_name} is TABLE but has no "
                    "corresponding table record."
                )

        if element_type == "FIGURE":
            figure_id = record.get("unit_id")

            if not any(
                figure.get("figure_id") == figure_id
                for figure in figures
            ):
                errors.append(
                    f"{record_name} is FIGURE but has no "
                    "corresponding figure record."
                )

    for index, record in enumerate(figures):
        record_name = f"figure {index}"

        validate_required_fields(
            record,
            REQUIRED_FIGURE_FIELDS,
            record_name,
            errors,
        )

        validate_page_numbers(
            record,
            source_start_page,
            source_end_page,
            record_name,
            errors,
        )

        validate_local_assets(
            record,
            "crop_local_paths",
            record_name,
            errors,
        )

        if record.get("figure_id") not in content_unit_ids:
            errors.append(
                f"{record_name} has no matching content unit."
            )

        crop_uris = record.get("crop_s3_uris", [])
        crop_paths = record.get("crop_local_paths", [])

        if len(crop_uris) != len(crop_paths):
            errors.append(
                f"{record_name} crop URI/path counts differ."
            )

        if not crop_paths:
            warnings.append(
                f"{record_name} has no crop image."
            )

    for index, record in enumerate(tables):
        record_name = f"table {index}"

        validate_required_fields(
            record,
            REQUIRED_TABLE_FIELDS,
            record_name,
            errors,
        )

        validate_page_numbers(
            record,
            source_start_page,
            source_end_page,
            record_name,
            errors,
        )

        validate_local_assets(
            record,
            "crop_local_paths",
            record_name,
            errors,
        )

        csv_text = record.get("csv_text")

        if not isinstance(csv_text, str):
            errors.append(
                f"{record_name}.csv_text "
                "must be a string."
            )

        elif not csv_text.strip():
            errors.append(
                f"{record_name}.csv_text is empty."
            )

        csv_s3_uri = record.get("csv_s3_uri")

        if (
            not isinstance(csv_s3_uri, str)
            or not csv_s3_uri.startswith("s3://")
        ):
            errors.append(
                f"{record_name}.csv_s3_uri "
                "must be a valid S3 URI."
            )

        validate_local_file(
            record.get("csv_local_path"),
            record_name,
            "csv_local_path",
            errors,
        )

        headers = record.get("headers")
        footers = record.get("footers")

        if not isinstance(headers, list):
            errors.append(
                f"{record_name}.headers must be a list."
            )

        if not isinstance(footers, list):
            errors.append(
                f"{record_name}.footers must be a list."
            )

        if record.get("table_id") not in content_unit_ids:
            errors.append(
                f"{record_name} has no matching content unit."
            )

    reported_table_count = int(
        report.get(
            "source_statistics",
            {},
        ).get("table_count", 0)
    )

    if reported_table_count and not tables:
        warnings.append(
            "BDA statistics reported tables, but no explicit "
            "TABLE records were emitted. This is documented "
            "and no tables were inferred."
        )

    print("============================================")
    print("NORMALIZED OUTPUT VALIDATION")
    print("============================================")
    print(f"Content units: {len(content_units)}")
    print(f"Figures:       {len(figures)}")
    print(f"Tables:        {len(tables)}")
    print(
        f"Page range:    "
        f"{source_start_page}-{source_end_page}"
    )
    print(f"Errors:        {len(errors)}")
    print(f"Warnings:      {len(warnings)}")

    if warnings:
        print()
        print("Warnings:")

        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print()
        print("Errors:")

        for error in errors:
            print(f"- {error}")

        print()
        print("VALIDATION RESULT: FAILED")
        return 1

    print()
    print("VALIDATION RESULT: PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"Validation failed unexpectedly: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
