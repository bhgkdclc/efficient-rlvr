#!/usr/bin/env bash
set -euo pipefail

# Full GSM8K control evaluation of the untouched base model. With zero GRPO
# steps and identical eval/final sample counts, this saves no checkpoint.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-${REPO_ROOT}/.cloud-cache}"
MODEL_PATH="${MODEL_PATH:-${PERSIST_ROOT}/models/Qwen2.5-Math-1.5B}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/experiments/base_full_eval_${RUN_TAG}}"

export HF_HOME="${HF_HOME:-${PERSIST_ROOT}/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PERSIST_ROOT}/uv}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"
uv run --no-sync python scripts/train_grpo.py \
    --model-name-or-path "${MODEL_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --run-name "base-full-eval-${RUN_TAG}" \
    --seed 42 \
    --n-grpo-steps 0 \
    --eval-samples 0 \
    --final-eval-samples 0 \
    --eval-temperature 0.0 \
    --eval-max-tokens 512 \
    --reward-mode question_only \
    --format-reward-weight 0.1 \
    --answer-reward-weight 1.0 \
    --checkpoint-steps 0 \
    --train-device cuda:0 \
    --vllm-device cuda:0 \
    --vllm-gpu-memory-utilization 0.4 \
    --attn-implementation sdpa \
    --wandb-mode disabled

echo "Base full evaluation output: ${OUTPUT_PATH}"

