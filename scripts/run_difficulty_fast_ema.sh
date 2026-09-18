#!/usr/bin/env bash
set -euo pipefail

# Targeted ablation: change only the prompt-accuracy EMA from beta=0.9 to 0.5.
# All other training, rollout, evaluation, reward, and seed settings are inherited
# unchanged from run_difficulty_baseline.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-difficulty_beta05}"
export DIFFICULTY_EMA_BETA="${DIFFICULTY_EMA_BETA:-0.5}"

exec bash "${SCRIPT_DIR}/run_difficulty_baseline.sh"
