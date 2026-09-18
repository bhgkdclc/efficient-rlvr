#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-${REPO_ROOT}/.cloud-cache}"

if (( $# > 0 )); then
    SEEDS=("$@")
else
    SEEDS=(43 44)
fi

export PERSIST_ROOT
export CHECKPOINT_STEPS=0

cd "${REPO_ROOT}"
for seed in "${SEEDS[@]}"; do
    if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed: ${seed}" >&2
        exit 2
    fi

    vanilla_tag="seed${seed}_$(date +%Y%m%d_%H%M%S)"
    echo "Starting Vanilla replication: seed=${seed}, tag=${vanilla_tag}"
    SEED="${seed}" RUN_TAG="${vanilla_tag}" \
        bash scripts/run_vanilla_baseline.sh \
        2>&1 | tee "vanilla-seed${seed}-console.log"

    difficulty_tag="seed${seed}_$(date +%Y%m%d_%H%M%S)"
    echo "Starting Fast-EMA replication: seed=${seed}, tag=${difficulty_tag}"
    SEED="${seed}" RUN_TAG="${difficulty_tag}" \
        bash scripts/run_difficulty_fast_ema.sh \
        2>&1 | tee "difficulty-beta05-seed${seed}-console.log"
done

echo "Completed replication seeds: ${SEEDS[*]}"
