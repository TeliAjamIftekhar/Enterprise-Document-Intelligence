#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import fitz


ARABIC_PATTERN = re.compile(
    r"[\u0600-\u06FF"
    r"\u0750-\u077F"
    r"\u08A0-\u08FF"
    r"\uFB50-\uFDFF"
    r"\uFE70-\uFEFF]"
)

LATIN_PATTERN = re.compile(r"[A-Za-z]")


OCR_PROMPT = """
Perform strict OCR transcription of the supplied
Urdu school textbook page image.

Return only text that is visibly printed in the image.

Rules:
- Preserve natural right-to-left logical reading order.
- Preserve visible headings, paragraphs, labels, lists and numbers.
- Do not translate, romanize, summarize or describe illustrations.
- Do not output JSON, Markdown or explanations.
- Do not add a page number, lesson number, heading or sentence unless it is visibly printed.
- Do not use metadata or information outside the image as textbook content.
- Do not repeat text unless it is visibly repeated.
- Write [غیر واضح] only for genuinely unreadable fragments.
""".strip()


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_json(path: Path) -> dict[str, Any]:
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
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def render_page(
    *,
    pdf_path: Path,
    page_number: int,
    dpi: int,
) -> bytes:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Canonical PDF not found: {pdf_path}"
        )

    document = fitz.open(pdf_path)

    try:
        if not 1 <= page_number <= len(document):
            raise ValueError(
                f"Page {page_number} outside "
                f"PDF range 1-{len(document)}."
            )

        page = document.load_page(
            page_number - 1
        )

        scale = dpi / 72.0

        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(
                scale,
                scale,
            ),
            alpha=False,
        )

        return pixmap.tobytes("png")

    finally:
        document.close()


def extract_model_text(
    response: dict[str, Any],
) -> str:
    content = (
        response.get("output", {})
        .get("message", {})
        .get("content", [])
    )

    parts: list[str] = []

    if isinstance(content, list):
        for item in content:
            if (
                isinstance(item, dict)
                and isinstance(
                    item.get("text"),
                    str,
                )
            ):
                parts.append(item["text"])

    return "\n".join(parts).strip()


def strip_code_fence(value: str) -> str:
    stripped = value.strip()

    if stripped.startswith("```"):
        stripped = re.sub(
            r"^```(?:json)?\s*",
            "",
            stripped,
            count=1,
            flags=re.IGNORECASE,
        )

        stripped = re.sub(
            r"\s*```$",
            "",
            stripped,
            count=1,
        )

    return stripped.strip()


def parse_model_json(
    value: str,
) -> dict[str, Any] | None:
    candidate = strip_code_fence(value)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    return (
        parsed
        if isinstance(parsed, dict)
        else None
    )


def assess_ocr_quality(
    value: str,
    *,
    stop_reason: str | None,
) -> dict[str, Any]:
    """
    Validate script quality, response completion and
    repetition before OCR is allowed downstream.
    """

    cleaned = strip_code_fence(value)

    lines = [
        " ".join(line.split())
        for line in cleaned.splitlines()
        if line.strip()
    ]

    frequencies: dict[str, int] = {}

    for line in lines:
        frequencies[line] = (
            frequencies.get(line, 0) + 1
        )

    arabic_count = len(
        ARABIC_PATTERN.findall(cleaned)
    )

    latin_count = len(
        LATIN_PATTERN.findall(cleaned)
    )

    script_total = (
        arabic_count + latin_count
    )

    arabic_ratio = (
        arabic_count / script_total
        if script_total
        else 0.0
    )

    max_line_repeat = max(
        frequencies.values(),
        default=0,
    )

    unique_line_count = len(
        frequencies
    )

    duplicate_line_ratio = (
        1.0
        - (
            unique_line_count
            / len(lines)
        )
        if lines
        else 1.0
    )

    response_completed = (
        stop_reason == "end_turn"
    )

    checks = {
        "response_completed": (
            response_completed
        ),
        "arabic_text_present": (
            arabic_count >= 50
        ),
        "arabic_dominant": (
            arabic_ratio >= 0.80
        ),
        "repetition_within_limit": (
            max_line_repeat <= 3
        ),
        "duplicate_ratio_within_limit": (
            duplicate_line_ratio <= 0.35
        ),
    }

    status = (
        "OCR_VALID"
        if all(checks.values())
        else "NEEDS_REVIEW"
    )

    return {
        "status": status,
        "stop_reason": stop_reason,
        "line_count": len(lines),
        "unique_line_count": (
            unique_line_count
        ),
        "max_line_repeat": (
            max_line_repeat
        ),
        "duplicate_line_ratio": round(
            duplicate_line_ratio,
            6,
        ),
        "arabic_characters": (
            arabic_count
        ),
        "latin_characters": (
            latin_count
        ),
        "arabic_ratio": round(
            arabic_ratio,
            6,
        ),
        "checks": checks,
        "cleaned_text": cleaned,
        "lines": lines,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render and optionally OCR one "
            "textbook page with Amazon Nova."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--page",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--model-id",
        default="amazon.nova-lite-v1:0",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Invoke Amazon Bedrock. Without this "
            "flag only render and plan locally."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    book = config.get("book", {})
    aws = config.get("aws", {})
    storage = config.get("storage", {})

    book_id = str(book.get("book_id"))
    version = str(book.get("version"))
    region = str(
        aws.get("region", "us-east-1")
    )

    local_root = Path(
        str(storage["local_root"])
    )

    pdf_path = (
        local_root
        / "source"
        / "textbook.pdf"
    )

    output_dir = (
        args.output_dir
        or (
            local_root
            / "ocr-fallback"
            / "nova"
            / f"page-{args.page:04d}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_bytes = render_page(
        pdf_path=pdf_path,
        page_number=args.page,
        dpi=args.dpi,
    )

    image_path = (
        output_dir
        / f"page-{args.page:04d}.png"
    )

    image_path.write_bytes(image_bytes)

    prompt = OCR_PROMPT.format(
        page_number=args.page
    )

    plan = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "book_id": book_id,
        "book_version": version,
        "page_number": args.page,
        "source_pdf": str(pdf_path),
        "rendered_image": str(image_path),
        "rendered_image_bytes": len(
            image_bytes
        ),
        "dpi": args.dpi,
        "region": region,
        "model_id": args.model_id,
        "language": "urdu",
        "expected_script": "arabic",
        "reading_direction": "rtl",
        "execute_requested": args.execute,
        "prompt": prompt,
    }

    atomic_write_json(
        output_dir / "request-plan.json",
        plan,
    )

    print("=" * 90)
    print("NOVA OCR PAGE PILOT")
    print("=" * 90)
    print("Book ID:       ", book_id)
    print("Page:          ", args.page)
    print("Source PDF:    ", pdf_path)
    print("Rendered image:", image_path)
    print("Image bytes:   ", len(image_bytes))
    print("DPI:           ", args.dpi)
    print("Model ID:      ", args.model_id)
    print("Execute:       ", args.execute)

    if not args.execute:
        print()
        print("Status:         DRY_RUN_READY")
        print("AWS calls:      0")
        print(
            "Plan:          ",
            output_dir / "request-plan.json",
        )
        return 0

    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
    )

    response = client.converse(
        modelId=args.model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "png",
                            "source": {
                                "bytes": image_bytes,
                            },
                        }
                    },
                    {
                        "text": prompt,
                    },
                ],
            }
        ],
        inferenceConfig={
            "maxTokens": 4096,
        },
    )

    atomic_write_json(
        output_dir / "raw-response.json",
        response,
    )

    model_text = extract_model_text(
        response
    )

    stop_reason = response.get(
        "stopReason"
    )

    quality = assess_ocr_quality(
        model_text,
        stop_reason=(
            str(stop_reason)
            if stop_reason is not None
            else None
        ),
    )

    cleaned_text = quality[
        "cleaned_text"
    ]

    (
        output_dir / "model-output.txt"
    ).write_text(
        cleaned_text + "\n",
        encoding="utf-8",
    )

    local_result = {
        "schema_version": "1.0",
        "page_number": args.page,
        "language": "urdu",
        "script": "arabic",
        "reading_direction": "rtl",
        "text_blocks": [
            {
                "order": index,
                "text": line,
            }
            for index, line in enumerate(
                quality["lines"],
                start=1,
            )
        ],
        "full_text": cleaned_text,
    }

    atomic_write_json(
        output_dir / "ocr-result.json",
        local_result,
    )

    status = str(
        quality["status"]
    )

    report = {
        **plan,
        "status": status,
        "execute_requested": True,
        "response_received": True,
        "local_result_created": True,
        "stop_reason": quality[
            "stop_reason"
        ],
        "line_count": quality[
            "line_count"
        ],
        "unique_line_count": quality[
            "unique_line_count"
        ],
        "max_line_repeat": quality[
            "max_line_repeat"
        ],
        "duplicate_line_ratio": quality[
            "duplicate_line_ratio"
        ],
        "arabic_characters": quality[
            "arabic_characters"
        ],
        "latin_characters": quality[
            "latin_characters"
        ],
        "arabic_ratio": quality[
            "arabic_ratio"
        ],
        "quality_checks": quality[
            "checks"
        ],
        "usage": response.get("usage"),
        "metrics": response.get("metrics"),
        "model_output": str(
            output_dir / "model-output.txt"
        ),
        "parsed_output": str(
            output_dir / "ocr-result.json"
        ),
    }

    atomic_write_json(
        output_dir / "ocr-report.json",
        report,
    )

    print()
    print("Status:        ", status)
    print(
        "Stop reason:   ",
        quality["stop_reason"],
    )
    print(
        "Lines:         ",
        quality["line_count"],
    )
    print(
        "Unique lines:  ",
        quality["unique_line_count"],
    )
    print(
        "Max repetition:",
        quality["max_line_repeat"],
    )
    print(
        "Duplicate ratio:",
        f"{quality['duplicate_line_ratio']:.2%}",
    )
    print(
        "Arabic chars:  ",
        quality["arabic_characters"],
    )
    print(
        "Latin chars:   ",
        quality["latin_characters"],
    )
    print(
        "Arabic ratio:  ",
        f"{quality['arabic_ratio']:.2%}",
    )
    print(
        "Output:        ",
        output_dir / "model-output.txt",
    )
    print(
        "Report:        ",
        output_dir / "ocr-report.json",
    )
    print("Bedrock calls:  1")

    return (
        0
        if status == "OCR_VALID"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
