from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def describe_value(
    value: Any,
    path: str,
    depth: int,
    max_depth: int,
    lines: list[str],
) -> None:
    indent = "  " * depth

    if isinstance(value, dict):
        lines.append(
            f"{indent}{path}: object ({len(value)} keys)"
        )

        if depth >= max_depth:
            return

        for key, child in value.items():
            describe_value(
                value=child,
                path=str(key),
                depth=depth + 1,
                max_depth=max_depth,
                lines=lines,
            )

    elif isinstance(value, list):
        lines.append(
            f"{indent}{path}: array ({len(value)} items)"
        )

        if value and depth < max_depth:
            describe_value(
                value=value[0],
                path="[0]",
                depth=depth + 1,
                max_depth=max_depth,
                lines=lines,
            )

    elif isinstance(value, str):
        preview = value.replace("\n", " ")[:100]
        lines.append(
            f"{indent}{path}: string "
            f"({len(value)} chars) {preview!r}"
        )

    else:
        lines.append(
            f"{indent}{path}: "
            f"{type(value).__name__} {value!r}"
        )


def walk_for_keys(
    value: Any,
    counter: Counter[str],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            counter[key] += 1
            walk_for_keys(child, counter)

    elif isinstance(value, list):
        for child in value:
            walk_for_keys(child, counter)


def collect_candidate_elements(
    value: Any,
    elements: list[dict[str, Any]],
) -> None:
    if isinstance(value, dict):
        keys = set(value)

        candidate_markers = {
            "type",
            "element_type",
            "representation",
            "bounding_box",
            "boundingBox",
            "page_index",
            "page_number",
            "pageNumber",
        }

        if keys & candidate_markers:
            elements.append(value)

        for child in value.values():
            collect_candidate_elements(child, elements)

    elif isinstance(value, list):
        for child in value:
            collect_candidate_elements(child, elements)


def find_type_value(
    element: dict[str, Any],
) -> str:
    for key in (
        "type",
        "element_type",
        "elementType",
        "category",
    ):
        value = element.get(key)

        if isinstance(value, str):
            return value

    return "UNKNOWN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an Amazon BDA result.json file."
    )

    parser.add_argument("result_json", type=Path)

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.result_json.exists():
        raise FileNotFoundError(
            f"Result file not found: {args.result_json}"
        )

    data = json.loads(
        args.result_json.read_text(encoding="utf-8")
    )

    lines: list[str] = []

    lines.append("BDA RESULT INSPECTION")
    lines.append("=" * 60)
    lines.append(f"File: {args.result_json}")
    lines.append(
        f"Root type: {type(data).__name__}"
    )
    lines.append("")

    lines.append("STRUCTURE")
    lines.append("-" * 60)

    describe_value(
        value=data,
        path="$",
        depth=0,
        max_depth=args.max_depth,
        lines=lines,
    )

    key_counter: Counter[str] = Counter()
    walk_for_keys(data, key_counter)

    lines.append("")
    lines.append("MOST COMMON JSON KEYS")
    lines.append("-" * 60)

    for key, count in key_counter.most_common(40):
        lines.append(f"{key}: {count}")

    candidate_elements: list[dict[str, Any]] = []
    collect_candidate_elements(
        data,
        candidate_elements,
    )

    type_counter = Counter(
        find_type_value(element)
        for element in candidate_elements
    )

    lines.append("")
    lines.append("CANDIDATE ELEMENT TYPES")
    lines.append("-" * 60)
    lines.append(
        f"Candidate objects found: "
        f"{len(candidate_elements)}"
    )

    for element_type, count in type_counter.most_common():
        lines.append(f"{element_type}: {count}")

    report = "\n".join(lines)

    print(report)

    output_path = args.output

    if output_path is None:
        output_path = (
            args.result_json.parent
            / "inspection-report.txt"
        )

    output_path.write_text(
        report + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Inspection report saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
