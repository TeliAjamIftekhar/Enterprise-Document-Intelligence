#!/usr/bin/env bash

set -u
set -o pipefail

PROJECT_ROOT="/home/ec2-user/SageMaker/Enterprise-Document-Intelligence"
VENV="${PROJECT_ROOT}/workers/multimodal-ingestion/.venv-surya"

BOOK_ID="grade-1-urdu-shahnai"

IMAGE="${PROJECT_ROOT}/data/multimodal-output/${BOOK_ID}/v1/ocr-fallback/nova/page-0017-pro-v1/page-0017.png"

OUTPUT_DIR="${PROJECT_ROOT}/data/multimodal-output/${BOOK_ID}/v1/ocr-fallback/surya/page-0017-v4"

STATUS_FILE="${OUTPUT_DIR}/status.txt"
PID_FILE="${OUTPUT_DIR}/runner.pid"

mkdir -p \
  "$OUTPUT_DIR" \
  "${PROJECT_ROOT}/workers/multimodal-ingestion/.cache/huggingface" \
  "${PROJECT_ROOT}/workers/multimodal-ingestion/.cache/surya-models"

echo "STARTING" > "$STATUS_FILE"

if [ ! -f "$IMAGE" ]; then
  echo "FAILED:IMAGE_NOT_FOUND" > "$STATUS_FILE"
  exit 1
fi

if [ ! -f "${VENV}/bin/activate" ]; then
  echo "FAILED:SURYA_ENV_NOT_FOUND" > "$STATUS_FILE"
  exit 1
fi

source "${VENV}/bin/activate"

export HF_HOME="${PROJECT_ROOT}/workers/multimodal-ingestion/.cache/huggingface"
export DOCKER_HF_CACHE_PATH="$HF_HOME"
export MODEL_CACHE_DIR="${PROJECT_ROOT}/workers/multimodal-ingestion/.cache/surya-models"

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

echo "$$" > "$PID_FILE"
echo "RUNNING" > "$STATUS_FILE"

echo "================================================================================"
echo "SURYA PAGE 17 DETACHED PILOT"
echo "================================================================================"
echo "Started:      $(date --iso-8601=seconds)"
echo "PID:          $$"
echo "Image:        $IMAGE"
echo "Output:       $OUTPUT_DIR"
echo "GPU type:     $VLLM_GPU_TYPE"
echo "Dtype:        $VLLM_DTYPE"
echo "Model length: $VLLM_MAX_MODEL_LEN"
echo

surya_ocr \
  "$IMAGE" \
  --output_dir "$OUTPUT_DIR" \
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
