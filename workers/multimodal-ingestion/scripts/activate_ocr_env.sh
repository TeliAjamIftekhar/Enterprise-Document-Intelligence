#!/usr/bin/env sh

OCR_PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "ERROR: Run this command from inside the project repository." >&2
    return 1 2>/dev/null || exit 1
}

OCR_PREFIX="${OCR_PROJECT_ROOT}/workers/multimodal-ingestion/.local/ocr"

export OCR_PREFIX
export PATH="${OCR_PREFIX}/bin${PATH:+:${PATH}}"
export PKG_CONFIG_PATH="${OCR_PREFIX}/lib/pkgconfig:${OCR_PREFIX}/lib64/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export LD_LIBRARY_PATH="${OCR_PREFIX}/lib:${OCR_PREFIX}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TESSDATA_PREFIX="${OCR_PREFIX}/share/tessdata"

echo "OCR environment activated"
echo "OCR prefix: ${OCR_PREFIX}"
