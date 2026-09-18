#!/usr/bin/env bash
set -euo pipefail

# Pre-generation Fast-EMA prompt selection plus post-generation filtering.
# Other than the strategy, this inherits the established Difficulty baseline
# hyperparameters and the same four-million rollout-token budget.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-difficulty_dynamic_beta05}"
export SAMPLING_STRATEGY="difficulty_dynamic"
export DIFFICULTY_EMA_BETA="${DIFFICULTY_EMA_BETA:-0.5}"
export DIFFICULTY_COVERAGE_WEIGHT="${DIFFICULTY_COVERAGE_WEIGHT:-0.0}"
export SAMPLING_UNIFORM_EPSILON="${SAMPLING_UNIFORM_EPSILON:-0.1}"
export DIFFICULTY_WARMUP_GROUPS="${DIFFICULTY_WARMUP_GROUPS:-128}"
export CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-0}"

exec bash "${SCRIPT_DIR}/run_difficulty_baseline.sh"
