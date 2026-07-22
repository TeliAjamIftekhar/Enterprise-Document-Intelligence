#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ARABIC_PATTERN = re.compile(
    r"[\u0600-\u06FF"
    r"\u0750-\u077F"
    r"\u08A0-\u08FF"
    r"\uFB50-\uFDFF"
    r"\uFE70-\uFEFF]"
)

LATIN_PATTERN = re.compile(r"[A-Za-z]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Tesseract Urdu OCR using multiple "
            "preprocessing and page-segmentation modes."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--language",
        default="urd",
    )

    return parser.parse_args()


def otsu_threshold(image: Image.Image) -> int:
    histogram = image.histogram()
    total = sum(histogram)

    weighted_sum = sum(
        index * count
        for index, count in enumerate(histogram)
    )

    background_weight = 0
    background_sum = 0.0
    maximum_variance = -1.0
    threshold = 127

    for index, count in enumerate(histogram):
        background_weight += count

        if background_weight == 0:
            continue

        foreground_weight = total - background_weight

        if foreground_weight == 0:
            break

        background_sum += index * count

        background_mean = (
            background_sum / background_weight
        )

        foreground_mean = (
            weighted_sum - background_sum
        ) / foreground_weight

        variance = (
            background_weight
            * foreground_weight
            * (
                background_mean
                - foreground_mean
            )
            ** 2
        )

        if variance > maximum_variance:
            maximum_variance = variance
            threshold = index

    return threshold


def prepare_variants(
    source: Path,
    output_dir: Path,
) -> dict[str, Path]:
    image = Image.open(source).convert("RGB")

    variants_dir = output_dir / "images"
    variants_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    variants: dict[str, Path] = {
        "original": source,
    }

    grayscale = ImageOps.grayscale(image)

    autocontrast = ImageOps.autocontrast(
        grayscale,
        cutoff=1,
    )

    autocontrast_path = (
        variants_dir / "autocontrast.png"
    )

    autocontrast.save(
        autocontrast_path,
        dpi=(300, 300),
    )

    variants["autocontrast"] = (
        autocontrast_path
    )

    sharpened = (
        ImageEnhance.Contrast(
            autocontrast
        )
        .enhance(1.5)
        .filter(ImageFilter.SHARPEN)
    )

    sharpened_path = (
        variants_dir / "contrast-sharpen.png"
    )

    sharpened.save(
        sharpened_path,
        dpi=(300, 300),
    )

    variants["contrast_sharpen"] = (
        sharpened_path
    )

    threshold = otsu_threshold(
        autocontrast
    )

    binary = autocontrast.point(
        lambda pixel: (
            255
            if pixel > threshold
            else 0
        ),
        mode="1",
    )

    binary_path = (
        variants_dir / "binary-otsu.png"
    )

    binary.save(
        binary_path,
        dpi=(300, 300),
    )

    variants["binary_otsu"] = binary_path

    return variants


def ngram_stats(
    words: list[str],
    size: int,
) -> dict[str, Any]:
    if len(words) < size:
        return {
            "size": size,
            "maximum_repetition": 1,
            "unique_ratio": 1.0,
        }

    counts = Counter(
        tuple(words[index:index + size])
        for index in range(
            len(words) - size + 1
        )
    )

    total = sum(counts.values())

    return {
        "size": size,
        "maximum_repetition": max(
            counts.values(),
            default=1,
        ),
        "unique_ratio": round(
            len(counts) / total,
            6,
        ),
    }


def analyse_output(
    text_path: Path,
    tsv_path: Path,
) -> dict[str, Any]:
    text = text_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    words = text.split()

    arabic_count = len(
        ARABIC_PATTERN.findall(text)
    )

    latin_count = len(
        LATIN_PATTERN.findall(text)
    )

    script_total = (
        arabic_count + latin_count
    )

    arabic_ratio = (
        arabic_count / script_total
        if script_total
        else 0.0
    )

    confidences: list[float] = []

    with tsv_path.open(
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        for row in csv.DictReader(
            handle,
            delimiter="\t",
        ):
            value = (
                row.get("text") or ""
            ).strip()

            try:
                confidence = float(
                    row.get("conf", "-1")
                )
            except ValueError:
                confidence = -1.0

            if value and confidence >= 0:
                confidences.append(
                    confidence
                )

    mean_confidence = (
        statistics.mean(confidences)
        if confidences
        else 0.0
    )

    median_confidence = (
        statistics.median(confidences)
        if confidences
        else 0.0
    )

    high_confidence_ratio = (
        sum(
            value >= 70
            for value in confidences
        )
        / len(confidences)
        if confidences
        else 0.0
    )

    low_confidence_ratio = (
        sum(
            value < 30
            for value in confidences
        )
        / len(confidences)
        if confidences
        else 1.0
    )

    five_gram = ngram_stats(
        words,
        5,
    )

    eight_gram = ngram_stats(
        words,
        8,
    )

    score = (
        mean_confidence * 0.40
        + median_confidence * 0.25
        + high_confidence_ratio * 20
        + arabic_ratio * 15
        - low_confidence_ratio * 15
        - max(
            0,
            eight_gram[
                "maximum_repetition"
            ]
            - 2,
        )
        * 10
    )

    return {
        "characters": len(text),
        "words": len(words),
        "arabic_characters": arabic_count,
        "latin_characters": latin_count,
        "arabic_ratio": round(
            arabic_ratio,
            6,
        ),
        "recognized_words": len(
            confidences
        ),
        "mean_confidence": round(
            mean_confidence,
            2,
        ),
        "median_confidence": round(
            median_confidence,
            2,
        ),
        "high_confidence_ratio": round(
            high_confidence_ratio,
            6,
        ),
        "low_confidence_ratio": round(
            low_confidence_ratio,
            6,
        ),
        "five_word_ngram": five_gram,
        "eight_word_ngram": eight_gram,
        "score": round(score, 2),
        "text_path": str(text_path),
        "tsv_path": str(tsv_path),
    }


def main() -> int:
    args = parse_args()

    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    tesseract = shutil.which(
        "tesseract"
    )

    if not tesseract:
        raise RuntimeError(
            "Tesseract binary not found."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    variants = prepare_variants(
        image_path,
        output_dir,
    )

    page_modes = [3, 4, 6, 11]

    results: list[dict[str, Any]] = []

    for variant_name, variant_path in (
        variants.items()
    ):
        for page_mode in page_modes:
            run_dir = (
                output_dir
                / variant_name
                / f"psm-{page_mode}"
            )

            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_base = (
                run_dir / "ocr"
            )

            command = [
                tesseract,
                str(variant_path),
                str(output_base),
                "-l",
                args.language,
                "--oem",
                "1",
                "--psm",
                str(page_mode),
                "txt",
                "tsv",
            ]

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

            log_path = (
                run_dir / "tesseract.log"
            )

            log_path.write_text(
                process.stdout
                + process.stderr,
                encoding="utf-8",
            )

            result: dict[str, Any] = {
                "variant": variant_name,
                "psm": page_mode,
                "return_code": (
                    process.returncode
                ),
                "log_path": str(log_path),
            }

            text_path = output_base.with_suffix(
                ".txt"
            )

            tsv_path = output_base.with_suffix(
                ".tsv"
            )

            if (
                process.returncode == 0
                and text_path.is_file()
                and tsv_path.is_file()
            ):
                result.update(
                    analyse_output(
                        text_path,
                        tsv_path,
                    )
                )

                result["status"] = (
                    "COMPLETED"
                )
            else:
                result["status"] = "FAILED"
                result["score"] = -999.0

            results.append(result)

    ranked = sorted(
        results,
        key=lambda item: float(
            item.get("score", -999.0)
        ),
        reverse=True,
    )

    json_path = (
        output_dir
        / "benchmark-results.json"
    )

    json_path.write_text(
        json.dumps(
            {
                "image": str(image_path),
                "language": args.language,
                "aws_calls": 0,
                "results": ranked,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = (
        output_dir
        / "benchmark-results.csv"
    )

    fields = [
        "variant",
        "psm",
        "status",
        "score",
        "words",
        "mean_confidence",
        "median_confidence",
        "high_confidence_ratio",
        "low_confidence_ratio",
        "arabic_ratio",
        "text_path",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for result in ranked:
            writer.writerow({
                field: result.get(field)
                for field in fields
            })

    print("=" * 110)
    print("TESSERACT URDU BENCHMARK")
    print("=" * 110)

    for position, result in enumerate(
        ranked[:8],
        start=1,
    ):
        print(
            f"{position:>2}. "
            f"variant={result['variant']:<18} "
            f"psm={result['psm']:<2} "
            f"score={result.get('score', 0):>6} "
            f"mean={result.get('mean_confidence', 0):>6} "
            f"median={result.get('median_confidence', 0):>6} "
            f"high={result.get('high_confidence_ratio', 0):>6.1%} "
            f"low={result.get('low_confidence_ratio', 0):>6.1%} "
            f"words={result.get('words', 0)}"
        )

    print()
    print("Best text:")
    print(ranked[0].get("text_path"))

    print()
    print("JSON:", json_path)
    print("CSV: ", csv_path)
    print("AWS calls: 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
