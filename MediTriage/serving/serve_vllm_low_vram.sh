#!/bin/bash
# serve_vllm_low_vram.sh - 低显存 vLLM 启动方案（无需 72GB 大卡）
#
# 与 serve_vllm_official.sh 提供同一模型、同一 OpenAI 兼容契约（:8000），
# 仅通过三招显著降低显存占用：
#   1) 缩小上下文窗口 MAX_MODEL_LEN（默认 32768；256K 的 KV cache 才是显存大头）
#   2) 可选量化（--quantization awq / fp8，需先准备对应量化权重）
#   3) 可选 CPU offload（--cpu-offload-gb N：权重放内存，GPU 只留计算/KV）
#
# 适用：16~24GB 消费级显卡。显存仍不足时继续调小 MAX_MODEL_LEN 或增大 CPU_OFFLOAD_GB。
# 注意：上下文缩小后，超长文档/长对话由上层"上下文超限退避"机制兜底，功能不受影响。
#
# 用法（从仓库根运行）：
#   bash MediTriage/serving/serve_vllm_low_vram.sh
#   GPU=0 MAX_MODEL_LEN=8192 bash MediTriage/serving/serve_vllm_low_vram.sh
#   MODEL_PATH=models/MediX-R1-8B-AWQ QUANTIZATION=awq bash MediTriage/serving/serve_vllm_low_vram.sh
#   CPU_OFFLOAD_GB=8 MAX_MODEL_LEN=16384 bash MediTriage/serving/serve_vllm_low_vram.sh
#
# 切回全量 72GB 方案：serve_vllm_official.sh（保持原功能不变）。
set -x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MEDITRIAGE_MODELS:-$ROOT/models}/${MEDITRIAGE_MODEL_NAME:-MediX-R1-8B}"
GPU=${GPU:-0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
QUANTIZATION=${QUANTIZATION:-}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
IMAGE_DIR="${MEDITRIAGE_DATA:-$ROOT/data}/med_image_samples"

export CUDA_VISIBLE_DEVICES=${GPU}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
unset HF_ENDPOINT

mkdir -p "${IMAGE_DIR}"

EXTRA_ARGS=()
if [ -n "$QUANTIZATION" ]; then
  EXTRA_ARGS+=(--quantization "$QUANTIZATION")
fi
if [ "$CPU_OFFLOAD_GB" -gt 0 ] 2>/dev/null; then
  EXTRA_ARGS+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
fi

vllm serve "${MODEL_PATH}" \
    --served-model-name medix-r1-8b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image": 2}' \
    --allowed-local-media-path "${IMAGE_DIR}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    "${EXTRA_ARGS[@]}"