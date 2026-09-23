#!/usr/bin/env bash
set -euo pipefail

# One held-out, prompt-matched 4M-token pair. Both arms receive the same first
# 128 prompt groups and run full GSM8K evaluations at 1M/2M/3M/4M rollout
# tokens. Evaluation tokens are logged but do not count toward the training
# rollout-token budget.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEED="${SEED:-45}"
PAIR_TAG="${PAIR_TAG:-$(date +%Y%m%d_%H%M%S)}"
MAX_ROLLOUT_TOKENS=4000000
FULL_EVAL_ROLLOUT_TOKEN_MILESTONES="1000000 2000000 3000000 4000000"
WANDB_MODE="${WANDB_MODE:-disabled}"

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer" >&2
    exit 2
fi

export CHECKPOINT_STEPS=0
export SAVE_FINAL_CHECKPOINT=false
export MAX_ROLLOUT_TOKENS
export FULL_EVAL_ROLLOUT_TOKEN_MILESTONES
export WANDB_MODE

cd "${REPO_ROOT}"

vanilla_output="${REPO_ROOT}/experiments/vanilla_matched_4m_curve_seed${SEED}_${PAIR_TAG}"
echo "Starting matched 4M budget curve Vanilla: seed=${SEED}"
SEED="${SEED}" \
RUN_TAG="matched_4m_curve_seed${SEED}_${PAIR_TAG}" \
OUTPUT_PATH="${vanilla_output}" \
MATCHED_RANDOM_WARMUP=true \
    bash scripts/run_vanilla_baseline.sh \
    2>&1 | tee "matched-4m-curve-vanilla-seed${SEED}-console.log"

difficulty_output="${REPO_ROOT}/experiments/difficulty_matched_4m_curve_seed${SEED}_${PAIR_TAG}"
echo "Starting matched 4M budget curve Fast-EMA: seed=${SEED}"
SEED="${SEED}" \
RUN_TAG="matched_4m_curve_seed${SEED}_${PAIR_TAG}" \
OUTPUT_PATH="${difficulty_output}" \
EXPERIMENT_NAME="difficulty_matched_4m_curve" \
PROMPT_IMPORTANCE_CORRECTION=false \
    bash scripts/run_difficulty_fast_ema.sh \
    2>&1 | tee "matched-4m-curve-fastema-seed${SEED}-console.log"

result_dir="${REPO_ROOT}/results/matched_4m_curve_seed${SEED}_${PAIR_TAG}"
mkdir -p "${result_dir}"
uv run --no-sync python scripts/analyze_budget_curve_pair.py \
    "${vanilla_output}" \
    "${difficulty_output}" \
    --output "${result_dir}/summary.json" \
    | tee "matched-4m-curve-seed${SEED}-analysis-console.log"

echo "Completed matched 4M budget curve pair: seed=${SEED}"
echo "Vanilla output: ${vanilla_output}"
echo "Fast-EMA output: ${difficulty_output}"
echo "Analysis output: ${result_dir}/summary.json"
