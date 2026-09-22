#!/usr/bin/env bash
set -euo pipefail

# Causal control for the 2M-token early-budget comparison.  Both arms consume
# the exact same 128 warmup prompt groups; only the post-warmup sampler differs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEED="${SEED:-42}"
MAX_ROLLOUT_TOKENS="${MAX_ROLLOUT_TOKENS:-2000000}"
PAIR_TAG="${PAIR_TAG:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-disabled}"

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "${MAX_ROLLOUT_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_ROLLOUT_TOKENS must be a positive integer" >&2
    exit 2
fi

export CHECKPOINT_STEPS=0
export SAVE_FINAL_CHECKPOINT=false
export MAX_ROLLOUT_TOKENS
export WANDB_MODE

cd "${REPO_ROOT}"

vanilla_output="${REPO_ROOT}/experiments/vanilla_matched_2m_seed${SEED}_${PAIR_TAG}"
echo "Starting matched-warmup 2M Vanilla: seed=${SEED}"
SEED="${SEED}" \
RUN_TAG="matched_2m_seed${SEED}_${PAIR_TAG}" \
OUTPUT_PATH="${vanilla_output}" \
MATCHED_RANDOM_WARMUP=true \
    bash scripts/run_vanilla_baseline.sh \
    2>&1 | tee "matched-warmup-vanilla-seed${SEED}-console.log"

difficulty_output="${REPO_ROOT}/experiments/difficulty_matched_2m_seed${SEED}_${PAIR_TAG}"
echo "Starting matched-warmup 2M Fast-EMA: seed=${SEED}"
SEED="${SEED}" \
RUN_TAG="matched_2m_seed${SEED}_${PAIR_TAG}" \
OUTPUT_PATH="${difficulty_output}" \
EXPERIMENT_NAME="difficulty_matched_2m" \
PROMPT_IMPORTANCE_CORRECTION=false \
    bash scripts/run_difficulty_fast_ema.sh \
    2>&1 | tee "matched-warmup-fastema-seed${SEED}-console.log"

echo "Completed matched-warmup 2M pair: seed=${SEED}"
echo "Vanilla output: ${vanilla_output}"
echo "Fast-EMA output: ${difficulty_output}"
