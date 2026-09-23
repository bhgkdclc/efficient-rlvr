#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-${REPO_ROOT}/.cloud-cache}"
MODEL_PATH="${MODEL_PATH:-${PERSIST_ROOT}/models/Qwen2.5-Math-1.5B}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-difficulty}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-difficulty}"
DIFFICULTY_EMA_BETA="${DIFFICULTY_EMA_BETA:-0.9}"
DIFFICULTY_COVERAGE_WEIGHT="${DIFFICULTY_COVERAGE_WEIGHT:-0.0}"
SAMPLING_UNIFORM_EPSILON="${SAMPLING_UNIFORM_EPSILON:-0.1}"
DIFFICULTY_WARMUP_GROUPS="${DIFFICULTY_WARMUP_GROUPS:-128}"
DIFFICULTY_SWITCH_ROLLOUT_TOKENS="${DIFFICULTY_SWITCH_ROLLOUT_TOKENS:-0}"
DIFFICULTY_REFRESH_CYCLE_TOKENS="${DIFFICULTY_REFRESH_CYCLE_TOKENS:-0}"
DIFFICULTY_REFRESH_RANDOM_TOKENS="${DIFFICULTY_REFRESH_RANDOM_TOKENS:-0}"
PROMPT_IMPORTANCE_CORRECTION="${PROMPT_IMPORTANCE_CORRECTION:-false}"
IMPORTANCE_WEIGHT_CLIP_MIN="${IMPORTANCE_WEIGHT_CLIP_MIN:-0.25}"
IMPORTANCE_WEIGHT_CLIP_MAX="${IMPORTANCE_WEIGHT_CLIP_MAX:-4.0}"
SEED="${SEED:-42}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-50}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-true}"
MAX_ROLLOUT_TOKENS="${MAX_ROLLOUT_TOKENS:-4000000}"
FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.1}"
ANSWER_REWARD_WEIGHT="${ANSWER_REWARD_WEIGHT:-1.0}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-64}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/experiments/${EXPERIMENT_NAME}_${RUN_TAG}}"
WANDB_MODE="${WANDB_MODE:-offline}"
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

case "${PROMPT_IMPORTANCE_CORRECTION}" in
    true) IMPORTANCE_CORRECTION_FLAG="--prompt-importance-correction" ;;
    false) IMPORTANCE_CORRECTION_FLAG="--no-prompt-importance-correction" ;;
    *)
        echo "PROMPT_IMPORTANCE_CORRECTION must be true or false" >&2
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
    --run-name "${EXPERIMENT_NAME}-${RUN_TAG}" \
    --seed "${SEED}" \
    --sampling-strategy "${SAMPLING_STRATEGY}" \
    --difficulty-ema-beta "${DIFFICULTY_EMA_BETA}" \
    --difficulty-coverage-weight "${DIFFICULTY_COVERAGE_WEIGHT}" \
    --sampling-uniform-epsilon "${SAMPLING_UNIFORM_EPSILON}" \
    --difficulty-warmup-groups "${DIFFICULTY_WARMUP_GROUPS}" \
    --difficulty-switch-rollout-tokens "${DIFFICULTY_SWITCH_ROLLOUT_TOKENS}" \
    --difficulty-refresh-cycle-tokens "${DIFFICULTY_REFRESH_CYCLE_TOKENS}" \
    --difficulty-refresh-random-tokens "${DIFFICULTY_REFRESH_RANDOM_TOKENS}" \
    "${IMPORTANCE_CORRECTION_FLAG}" \
    --importance-weight-clip-min "${IMPORTANCE_WEIGHT_CLIP_MIN}" \
    --importance-weight-clip-max "${IMPORTANCE_WEIGHT_CLIP_MAX}" \
    --reward-mode question_only \
    --format-reward-weight "${FORMAT_REWARD_WEIGHT}" \
    --answer-reward-weight "${ANSWER_REWARD_WEIGHT}" \
    --zero-variance-epsilon 1e-8 \
    --loss-type reinforce_with_baseline \
    --learning-rate 2e-5 \
    --n-grpo-steps 200 \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --group-size 8 \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
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

echo "Difficulty-aware baseline (strategy=${SAMPLING_STRATEGY}, EMA beta=${DIFFICULTY_EMA_BETA}, coverage weight=${DIFFICULTY_COVERAGE_WEIGHT}, seed=${SEED}) output: ${OUTPUT_PATH}"
