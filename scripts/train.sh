#!/usr/bin/env bash
# Train one configuration.
#
# Usage: bash scripts/train.sh <config.yaml> [extra --flags ...]
# Example:
#   bash scripts/train.sh configs/cppo_p75.yaml
#   bash scripts/train.sh configs/grpo.yaml --max-samples 512
#
# Set NUM_GPUS>1 to launch under `accelerate` with DeepSpeed ZeRO-2.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:?usage: scripts/train.sh <config.yaml> [extra flags]}"
shift || true

NUM_GPUS="${NUM_GPUS:-1}"
export TOKENIZERS_PARALLELISM=false

if [[ "$NUM_GPUS" -gt 1 ]]; then
  echo ">> Launching on $NUM_GPUS GPUs"
  accelerate launch \
    --config_file configs/accelerate_zero2.yaml \
    --num_processes "$NUM_GPUS" \
    -m cppo.train --config "$CONFIG" "$@"
else
  python -m cppo.train --config "$CONFIG" "$@"
fi
