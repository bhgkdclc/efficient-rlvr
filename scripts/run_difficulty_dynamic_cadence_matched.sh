#!/usr/bin/env bash
set -euo pipefail

# Follow-up to the three-seed Difficulty-Dynamic audit. Fast-EMA produced
# about five effective groups per optimizer step, while naive filtering packed
# eight into each step and reduced the update count from ~141 to ~89. Targeting
# five accepted groups (40 responses) restores the original update cadence
# without changing the sampler, reward, group size, or rollout-token budget.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-difficulty_dynamic_g5}"
export ROLLOUT_BATCH_SIZE=40
export TRAIN_BATCH_SIZE=40
export GRADIENT_ACCUMULATION_STEPS=40
export CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-0}"
export SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-false}"

exec bash "${SCRIPT_DIR}/run_difficulty_dynamic_baseline.sh"
