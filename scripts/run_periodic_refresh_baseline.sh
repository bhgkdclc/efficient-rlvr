#!/usr/bin/env bash
set -euo pipefail

# Periodic exploration/exploitation schedule.  Each one-million-token cycle
# begins with a 200k-token uniform-random refresh window, then returns to the
# fast-EMA difficulty sampler.  EMA state is retained and updated throughout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-difficulty_periodic_refresh_beta05}"
export SAMPLING_STRATEGY="difficulty_periodic_random"
export DIFFICULTY_EMA_BETA="${DIFFICULTY_EMA_BETA:-0.5}"
export DIFFICULTY_COVERAGE_WEIGHT="${DIFFICULTY_COVERAGE_WEIGHT:-0.0}"
export SAMPLING_UNIFORM_EPSILON="${SAMPLING_UNIFORM_EPSILON:-0.1}"
export DIFFICULTY_WARMUP_GROUPS="${DIFFICULTY_WARMUP_GROUPS:-128}"
export DIFFICULTY_REFRESH_CYCLE_TOKENS="${DIFFICULTY_REFRESH_CYCLE_TOKENS:-1000000}"
export DIFFICULTY_REFRESH_RANDOM_TOKENS="${DIFFICULTY_REFRESH_RANDOM_TOKENS:-200000}"

# Metrics and the final full evaluation are sufficient for this ablation.
# Avoid another multi-gigabyte checkpoint unless explicitly requested.
export CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-0}"
export SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-false}"

exec bash "${SCRIPT_DIR}/run_difficulty_baseline.sh"
