#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-${REPO_ROOT}/.cloud-cache}"
MODEL_PATH="${MODEL_PATH:-${PERSIST_ROOT}/models/Qwen2.5-Math-1.5B}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/experiments/vanilla_random_${RUN_TAG}}"
WANDB_MODE="${WANDB_MODE:-offline}"
SEED="${SEED:-42}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-50}"
MAX_ROLLOUT_TOKENS="${MAX_ROLLOUT_TOKENS:-4000000}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-true}"
MATCHED_RANDOM_WARMUP="${MATCHED_RANDOM_WARMUP:-false}"
FULL_EVAL_ROLLOUT_TOKEN_MILESTONES="${FULL_EVAL_ROLLOUT_TOKEN_MILESTONES:-}"

MILESTONE_EVAL_ARGS=()
if [[ -n "${FULL_EVAL_ROLLOUT_TOKEN_MILESTONES}" ]]; then
    read -r -a milestone_values <<< "${FULL_EVAL_ROLLOUT_TOKEN_MILESTONES}"
    MILESTONE_EVAL_ARGS=(
        --full-eval-rollout-token-milestones "${milestone_values[@]}"
    )
fi

case "${SAVE_FINAL_CHECKPOINT}" in
    true) FINAL_CHECKPOINT_FLAG="--save-final-checkpoint" ;;
    false) FINAL_CHECKPOINT_FLAG="--no-save-final-checkpoint" ;;
    *)
        echo "SAVE_FINAL_CHECKPOINT must be true or false" >&2
        exit 2
        ;;
esac

case "${MATCHED_RANDOM_WARMUP}" in
    true) MATCHED_WARMUP_FLAG="--matched-random-warmup" ;;
    false) MATCHED_WARMUP_FLAG="--no-matched-random-warmup" ;;
    *)
        echo "MATCHED_RANDOM_WARMUP must be true or false" >&2
        exit 2
        ;;
esac

export HF_HOME="${HF_HOME:-${PERSIST_ROOT}/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PERSIST_ROOT}/uv}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"
uv run --no-sync python scripts/train_grpo.py \
    --model-name-or-path "${MODEL_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --run-name "vanilla-random-${RUN_TAG}" \
    --seed "${SEED}" \
    --sampling-strategy random \
    "${MATCHED_WARMUP_FLAG}" \
    --difficulty-ema-beta 0.5 \
    --sampling-uniform-epsilon 0.1 \
    --difficulty-warmup-groups 128 \
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
    --max-rollout-tokens "${MAX_ROLLOUT_TOKENS}" \
    "${MILESTONE_EVAL_ARGS[@]}" \
    --eval-steps 10 \
    --checkpoint-steps "${CHECKPOINT_STEPS}" \
    --eval-samples 256 \
    --final-eval-samples 0 \
    --eval-temperature 0.0 \
    --eval-max-tokens 1024 \
    "${FINAL_CHECKPOINT_FLAG}" \
    --train-device cuda:0 \
    --vllm-device cuda:0 \
    --vllm-gpu-memory-utilization 0.4 \
    --attn-implementation sdpa \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-project efficient-rlvr

echo "Vanilla baseline (seed=${SEED}) output: ${OUTPUT_PATH}"
