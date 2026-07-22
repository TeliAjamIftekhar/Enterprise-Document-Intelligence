#!/usr/bin/env bash

set -u
set -o pipefail

PROJECT_ROOT="/home/ec2-user/SageMaker/Enterprise-Document-Intelligence"
BOOK_ID="grade-1-urdu-shahnai"
VERSION="v1"

MAIN_PYTHON="${PROJECT_ROOT}/workers/multimodal-ingestion/.venv/bin/python"
SURYA_VENV="${PROJECT_ROOT}/workers/multimodal-ingestion/.venv-surya"

BOOK_ROOT="${PROJECT_ROOT}/data/multimodal-output/${BOOK_ID}/${VERSION}"
PILOT_ROOT="${BOOK_ROOT}/ocr-fallback/surya/multipage-v1"
INPUT_DIR="${PILOT_ROOT}/input"
RESULT_DIR="${PILOT_ROOT}/results"

STATUS_FILE="${PILOT_ROOT}/status.txt"
PID_FILE="${PILOT_ROOT}/runner.pid"
MANIFEST="${PILOT_ROOT}/pilot-manifest.json"

mkdir -p \
  "$INPUT_DIR" \
  "$RESULT_DIR" \
  "${PROJECT_ROOT}/workers/multimodal-ingestion/.cache/huggingface"

echo "PREPARING" > "$STATUS_FILE"
echo "$$" > "$PID_FILE"

if [ ! -x "$MAIN_PYTHON" ]; then
  echo "FAILED:MAIN_PYTHON_NOT_FOUND" > "$STATUS_FILE"
  exit 1
fi

if [ ! -f "${SURYA_VENV}/bin/activate" ]; then
  echo "FAILED:SURYA_ENV_NOT_FOUND" > "$STATUS_FILE"
  exit 1
fi

echo "================================================================================"
echo "SURYA SHAHNAI REPRESENTATIVE MULTI-PAGE PILOT"
echo "================================================================================"
echo "Started: $(date --iso-8601=seconds)"
echo

"$MAIN_PYTHON" - \
  "$BOOK_ROOT" \
  "$INPUT_DIR" \
  "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

import fitz

book_root = Path(sys.argv[1])
input_dir = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])

candidates = []

for pdf_path in book_root.rglob("*.pdf"):
    try:
        document = fitz.open(pdf_path)
        page_count = document.page_count
        document.close()

        candidates.append(
            (
                page_count,
                pdf_path.stat().st_size,
                pdf_path,
            )
        )
    except Exception:
        continue

if not candidates:
    raise SystemExit(
        f"NO_PDF_FOUND_UNDER: {book_root}"
    )

# Prefer the PDF with the highest page count, then largest size.
page_count, _, canonical_pdf = max(
    candidates,
    key=lambda item: (item[0], item[1]),
)

if page_count < 20:
    raise SystemExit(
        f"INVALID_CANONICAL_PDF_PAGE_COUNT: {page_count}"
    )

# Page 17 is retained as the known control page.
requested_pages = [5, 17, 35, 60, 85, 110]

selected_pages = []

for requested in requested_pages:
    selected = min(max(requested, 1), page_count)

    if selected not in selected_pages:
        selected_pages.append(selected)

document = fitz.open(canonical_pdf)

rendered = []

for page_number in selected_pages:
    page = document.load_page(page_number - 1)

    # Same high-resolution strategy used for successful page 17.
    matrix = fitz.Matrix(300 / 72, 300 / 72)
    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    output_path = (
        input_dir /
        f"page-{page_number:04d}.png"
    )

    pixmap.save(output_path)

    rendered.append(
        {
            "canonical_page": page_number,
            "image": str(output_path),
            "width": pixmap.width,
            "height": pixmap.height,
            "bytes": output_path.stat().st_size,
        }
    )

document.close()

manifest = {
    "book_id": "grade-1-urdu-shahnai",
    "version": "v1",
    "canonical_pdf": str(canonical_pdf),
    "canonical_page_count": page_count,
    "selected_pages": selected_pages,
    "render_dpi": 300,
    "rendered_images": rendered,
    "purpose": (
        "Representative Surya OCR quality pilot. "
        "Full-book processing remains blocked."
    ),
}

manifest_path.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print(f"Canonical PDF: {canonical_pdf}")
print(f"Page count:    {page_count}")
print(
    "Selected:      "
    + ", ".join(map(str, selected_pages))
)

for item in rendered:
    print(
        f"Rendered page {item['canonical_page']:>3}: "
        f"{item['width']}x{item['height']} | "
        f"{item['bytes']:,} bytes"
    )
PY

PREPARE_STATUS=$?

if [ "$PREPARE_STATUS" -ne 0 ]; then
  echo "FAILED:PAGE_PREPARATION:${PREPARE_STATUS}" \
    > "$STATUS_FILE"
  exit "$PREPARE_STATUS"
fi

source "${SURYA_VENV}/bin/activate"

export HF_HOME="${PROJECT_ROOT}/workers/multimodal-ingestion/.cache/huggingface"
export DOCKER_HF_CACHE_PATH="$HF_HOME"

export SURYA_INFERENCE_BACKEND="vllm"
export CUDA_VISIBLE_DEVICES="0"
export VLLM_GPUS="0"

export VLLM_GPU_TYPE="t4"
export VLLM_DTYPE="float16"

export SURYA_INFERENCE_PARALLEL="1"
export VLLM_GPU_MEMORY_UTILIZATION="0.80"
export VLLM_MAX_MODEL_LEN="8192"
export SURYA_INFERENCE_CTX_PER_SLOT="8192"
export SURYA_INFERENCE_CTX_SIZE="8192"
export SURYA_MAX_TOKENS_FULL_PAGE="4096"
export VLLM_ENABLE_MTP="false"

export SURYA_INFERENCE_STARTUP_TIMEOUT="1800"
export SURYA_INFERENCE_TIMEOUT_SECONDS="1200"
export SURYA_INFERENCE_KEEP_ALIVE="false"

echo "RUNNING" > "$STATUS_FILE"

echo
echo "Starting Surya OCR for representative pages..."
echo "Input:  $INPUT_DIR"
echo "Output: $RESULT_DIR"
echo

surya_ocr \
  "$INPUT_DIR" \
  --output_dir "$RESULT_DIR" \
  --images

EXIT_CODE=$?

echo
echo "Surya exit code: $EXIT_CODE"
echo "Finished:        $(date --iso-8601=seconds)"

if [ "$EXIT_CODE" -eq 0 ]; then
  echo "COMPLETED" > "$STATUS_FILE"
else
  echo "FAILED:${EXIT_CODE}" > "$STATUS_FILE"
fi

exit "$EXIT_CODE"
