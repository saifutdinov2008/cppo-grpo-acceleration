#!/usr/bin/env bash
# Measure how the update stage scales with the number of retained completions.
#
# Usage: bash scripts/benchmark_update_stage.sh [device] [output.json]
# Example:
#   bash scripts/benchmark_update_stage.sh cuda results/update_stage_a100.json
#   bash scripts/benchmark_update_stage.sh mps  results/update_stage_m1.json
#
# This isolates CPPO's effect from the rollout: no dataset, no generation,
# no reward model -- just forward, backward and optimiser step over a batch
# whose size is set by the pruning rate.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${1:-auto}"
OUTPUT="${2:-results/update_stage_benchmark.json}"

python benchmarks/update_stage_benchmark.py \
  --model "${MODEL:-Qwen/Qwen3-0.6B}" \
  --device "$DEVICE" \
  --num-generations "${NUM_GENERATIONS:-8}" \
  --questions "${QUESTIONS:-4}" \
  --completion-length "${COMPLETION_LENGTH:-512}" \
  --steps "${STEPS:-5}" \
  --warmup "${WARMUP:-2}" \
  --output "$OUTPUT"
