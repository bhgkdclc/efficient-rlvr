#!/usr/bin/env bash
set -euo pipefail

# Fast-EMA difficulty sampling with clipped prompt-level importance correction.
# Sampling remains identical to the registered fast-EMA baseline; only the
# policy-gradient contribution is reweighted toward the uniform prompt target.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-difficulty_importance_beta05}"
export SAMPLING_STRATEGY="difficulty"
export DIFFICULTY_EMA_BETA="${DIFFICULTY_EMA_BETA:-0.5}"
export DIFFICULTY_COVERAGE_WEIGHT="${DIFFICULTY_COVERAGE_WEIGHT:-0.0}"
export SAMPLING_UNIFORM_EPSILON="${SAMPLING_UNIFORM_EPSILON:-0.1}"
export DIFFICULTY_WARMUP_GROUPS="${DIFFICULTY_WARMUP_GROUPS:-128}"
export PROMPT_IMPORTANCE_CORRECTION="true"
export IMPORTANCE_WEIGHT_CLIP_MIN="${IMPORTANCE_WEIGHT_CLIP_MIN:-0.25}"
export IMPORTANCE_WEIGHT_CLIP_MAX="${IMPORTANCE_WEIGHT_CLIP_MAX:-4.0}"

export CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-0}"
export SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-false}"

exec bash "${SCRIPT_DIR}/run_difficulty_baseline.sh"
