#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-${REPO_ROOT}/.cloud-cache}"
MODEL_PATH="${MODEL_PATH:-${PERSIST_ROOT}/models/Qwen2.5-Math-1.5B}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/experiments/vanilla_smoke_${RUN_TAG}}"

export HF_HOME="${HF_HOME:-${PERSIST_ROOT}/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PERSIST_ROOT}/uv}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"
uv run --no-sync python -c \
    "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

uv run --no-sync python scripts/train_grpo.py \
    --model-name-or-path "${MODEL_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --run-name "vanilla-smoke-${RUN_TAG}" \
    --sampling-strategy random \
    --reward-mode question_only \
    --format-reward-weight 0.1 \
    --answer-reward-weight 1.0 \
    --loss-type reinforce_with_baseline \
    --n-grpo-steps 1 \
    --train-samples 32 \
    --rollout-batch-size 8 \
    --group-size 4 \
    --train-batch-size 8 \
    --gradient-accumulation-steps 8 \
    --sampling-max-tokens 128 \
    --eval-max-tokens 128 \
    --eval-samples 8 \
    --final-eval-samples 8 \
    --eval-steps 1 \
    --checkpoint-steps 1 \
    --no-eval-before-training \
    --train-device cuda:0 \
    --vllm-device cuda:0 \
    --vllm-gpu-memory-utilization 0.35 \
    --attn-implementation sdpa \
    --wandb-mode disabled

echo "Smoke test output: ${OUTPUT_PATH}"
