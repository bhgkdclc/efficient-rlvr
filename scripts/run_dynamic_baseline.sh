#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-${REPO_ROOT}/.cloud-cache}"
MODEL_PATH="${MODEL_PATH:-${PERSIST_ROOT}/models/Qwen2.5-Math-1.5B}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/experiments/dynamic_${RUN_TAG}}"
WANDB_MODE="${WANDB_MODE:-offline}"

export HF_HOME="${HF_HOME:-${PERSIST_ROOT}/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PERSIST_ROOT}/uv}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"
uv run --no-sync python scripts/train_grpo.py \
    --model-name-or-path "${MODEL_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --run-name "dynamic-${RUN_TAG}" \
    --seed 42 \
    --sampling-strategy dynamic \
    --reward-mode question_only \
    --format-reward-weight 0.1 \
    --answer-reward-weight 1.0 \
    --zero-variance-epsilon 1e-8 \
    --loss-type reinforce_with_baseline \
    --learning-rate 2e-5 \
    --n-grpo-steps 200 \
    --rollout-batch-size 64 \
    --group-size 8 \
    --train-batch-size 64 \
    --gradient-accumulation-steps 64 \
    --sampling-temperature 1.0 \
    --sampling-max-tokens 512 \
    --max-rollout-tokens 4000000 \
    --eval-steps 10 \
    --eval-samples 256 \
    --final-eval-samples 0 \
    --eval-temperature 0.0 \
    --eval-max-tokens 512 \
    --train-device cuda:0 \
    --vllm-device cuda:0 \
    --vllm-gpu-memory-utilization 0.4 \
    --attn-implementation sdpa \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project efficient-rlvr

echo "Dynamic baseline output: ${OUTPUT_PATH}"
