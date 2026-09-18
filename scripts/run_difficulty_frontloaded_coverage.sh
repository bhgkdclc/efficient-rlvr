#!/usr/bin/env bash
set -euo pipefail

# Controlled follow-up to the fixed coverage-weight ablation. Build a broader
# prompt pool during a 512-group unique warm-up, then switch to the unchanged
# beta=0.5 boundary sampler. This changes the exploration schedule only.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-difficulty_frontloaded_coverage}"
export DIFFICULTY_EMA_BETA="${DIFFICULTY_EMA_BETA:-0.5}"
export DIFFICULTY_COVERAGE_WEIGHT="${DIFFICULTY_COVERAGE_WEIGHT:-0.0}"
export SAMPLING_UNIFORM_EPSILON="${SAMPLING_UNIFORM_EPSILON:-0.1}"
export DIFFICULTY_WARMUP_GROUPS="${DIFFICULTY_WARMUP_GROUPS:-512}"
export CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-0}"

exec bash "${SCRIPT_DIR}/run_difficulty_baseline.sh"
