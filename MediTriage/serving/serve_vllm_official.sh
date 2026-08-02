#!/bin/bash
# serve_vllm_official.sh - Serve official MBZUAI/MediX-R1-8B (VLM) via vLLM
#
# Inference base for the Agent system.
# Vision-enabled (Qwen3-VL backbone), OpenAI-compatible API on port 8000.
#
# Usage (inside medix-fix container):
#   bash MediTriage/serving/serve_vllm_official.sh   (从仓库根运行)
#
# GPU: uses GPU 1 by default; override with GPU=<id> env.
#   Override with: GPU=2 bash serve_vllm_official.sh

set -x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # serving/ 上一级即 MediTriage/，models/ 与 data/ 都在其下
MODEL_PATH="${MEDITRIAGE_MODELS:-$ROOT/models}/MediX-R1-8B"
GPU=${GPU:-1}
IMAGE_DIR="${MEDITRIAGE_DATA:-$ROOT/data}/med_image_samples"

export CUDA_VISIBLE_DEVICES=${GPU}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
unset HF_ENDPOINT

mkdir -p "${IMAGE_DIR}"

vllm serve "${MODEL_PATH}" \
    --served-model-name medix-r1-8b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image": 2}' \
    --allowed-local-media-path "${IMAGE_DIR}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
