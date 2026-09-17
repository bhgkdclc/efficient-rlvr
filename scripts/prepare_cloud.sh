#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-${REPO_ROOT}/.cloud-cache}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-Math-1.5B}"
MODEL_DIR="${MODEL_DIR:-${PERSIST_ROOT}/models/Qwen2.5-Math-1.5B}"

mkdir -p "${PERSIST_ROOT}/uv" "${PERSIST_ROOT}/huggingface" "${MODEL_DIR}"
export UV_CACHE_DIR="${PERSIST_ROOT}/uv"
export HF_HOME="${PERSIST_ROOT}/huggingface"
export UV_LINK_MODE=copy

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh -o /tmp/install-uv.sh
    sh /tmp/install-uv.sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

cd "${REPO_ROOT}"

# --frozen avoids refreshing the unrelated alpaca-eval Git dependency.
# flash-attn is optional because the training scripts default to PyTorch SDPA.
uv sync --frozen \
    --no-install-package alpaca-eval \
    --no-install-package flash-attn

uv run --no-sync huggingface-cli download \
    "${MODEL_ID}" \
    --local-dir "${MODEL_DIR}"

echo "Cloud preparation complete."
echo "MODEL_DIR=${MODEL_DIR}"
echo "HF_HOME=${HF_HOME}"
