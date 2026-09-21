#!/usr/bin/env bash
set -euo pipefail

# Two-stage schedule: use fast-EMA difficulty sampling for the first half of
# the rollout-token budget, then return to uniform random sampling to restore
# prompt coverage.  The trainer performs a full evaluation at the switch.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-annealed_difficulty_beta05}"
export SAMPLING_STRATEGY="difficulty_then_random"
export DIFFICULTY_EMA_BETA="${DIFFICULTY_EMA_BETA:-0.5}"
export DIFFICULTY_COVERAGE_WEIGHT="${DIFFICULTY_COVERAGE_WEIGHT:-0.0}"
export SAMPLING_UNIFORM_EPSILON="${SAMPLING_UNIFORM_EPSILON:-0.1}"
export DIFFICULTY_WARMUP_GROUPS="${DIFFICULTY_WARMUP_GROUPS:-128}"
export DIFFICULTY_SWITCH_ROLLOUT_TOKENS="${DIFFICULTY_SWITCH_ROLLOUT_TOKENS:-2000000}"

# Avoid multi-gigabyte intermediate checkpoints.  The switch evaluation is
# metrics-only and the shared runner still saves the final checkpoint.
export CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-0}"

exec bash "${SCRIPT_DIR}/run_difficulty_baseline.sh"
