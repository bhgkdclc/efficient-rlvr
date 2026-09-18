#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-${REPO_ROOT}/.cloud-cache}"
MODEL_PATH="${MODEL_PATH:-${PERSIST_ROOT}/models/Qwen2.5-Math-1.5B}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/experiments/difficulty_dynamic_smoke_${RUN_TAG}}"

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
    --run-name "difficulty-dynamic-smoke-${RUN_TAG}" \
    --seed 42 \
    --sampling-strategy difficulty_dynamic \
    --difficulty-ema-beta 0.5 \
    --sampling-uniform-epsilon 0.1 \
    --difficulty-warmup-groups 2 \
    --reward-mode question_only \
    --format-reward-weight 0.1 \
    --answer-reward-weight 1.0 \
    --zero-variance-epsilon 1e-8 \
    --loss-type reinforce_with_baseline \
    --learning-rate 2e-5 \
    --n-grpo-steps 2 \
    --train-samples 64 \
    --rollout-batch-size 16 \
    --group-size 8 \
    --train-batch-size 16 \
    --gradient-accumulation-steps 16 \
    --sampling-temperature 1.0 \
    --sampling-max-tokens 512 \
    --max-rollout-tokens 200000 \
    --eval-max-tokens 512 \
    --eval-samples 8 \
    --final-eval-samples 8 \
    --eval-steps 2 \
    --checkpoint-steps 2 \
    --no-eval-before-training \
    --train-device cuda:0 \
    --vllm-device cuda:0 \
    --vllm-gpu-memory-utilization 0.35 \
    --attn-implementation sdpa \
    --wandb-mode disabled

echo "Difficulty-Dynamic smoke output: ${OUTPUT_PATH}"
