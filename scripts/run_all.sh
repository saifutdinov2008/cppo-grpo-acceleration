#!/usr/bin/env bash
# Reproduce the full study: baseline + three CPPO pruning rates + the
# no-allocation ablation, each trained, evaluated and then tabulated.
#
# Usage: bash scripts/run_all.sh
# Reference hardware: 1x A100/H100 80GB. Budget roughly 12-18 GPU-hours.
#
# Override the sweep with e.g. CONFIGS="configs/grpo.yaml configs/cppo_p75.yaml".
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export TOKENIZERS_PARALLELISM=false

CONFIGS="${CONFIGS:-configs/grpo.yaml configs/cppo_p50.yaml configs/cppo_p75.yaml configs/cppo_p875.yaml configs/cppo_p75_no_allocation.yaml}"
BACKEND="${BACKEND:-vllm}"

mkdir -p results

echo "=============================================================="
echo "Step 0: evaluate the untrained Qwen3-0.6B reference point"
echo "=============================================================="
BACKEND="$BACKEND" bash scripts/evaluate.sh Qwen/Qwen3-0.6B results/base/eval.json

PROFILES=()
EVALS=()
for CONFIG in $CONFIGS; do
  NAME="$(basename "$CONFIG" .yaml)"
  echo "=============================================================="
  echo "Training $NAME"
  echo "=============================================================="
  bash scripts/train.sh "$CONFIG" --profile-path "results/$NAME/profile.json"

  OUT_DIR="$(python - "$CONFIG" <<'PY'
import sys
from cppo.config import load_settings
print(load_settings(["--config", sys.argv[1]]).output_dir)
PY
)"

  echo "== Evaluating $NAME ($OUT_DIR) =="
  BACKEND="$BACKEND" bash scripts/evaluate.sh "$OUT_DIR" "results/$NAME/eval.json"

  PROFILES+=("results/$NAME/profile.json")
  EVALS+=("results/$NAME/eval.json")
done

echo "=============================================================="
echo "Step N: update-stage micro-benchmark"
echo "=============================================================="
bash scripts/benchmark_update_stage.sh auto results/update_stage_benchmark.json

echo "=============================================================="
echo "Rendering result tables"
echo "=============================================================="
python -m cppo.report \
  --profiles "${PROFILES[@]}" --evals "${EVALS[@]}" \
  --benchmark results/update_stage_benchmark.json \
  --format markdown --output results/tables.md
python -m cppo.report \
  --profiles "${PROFILES[@]}" --evals "${EVALS[@]}" \
  --benchmark results/update_stage_benchmark.json \
  --format latex --output report/generated_tables.tex

echo "Done. See results/tables.md"
