#!/usr/bin/env bash
# Reproduce the full study: the GRPO baseline, three CPPO pruning rates and
# the no-allocation ablation, each trained for one epoch over the same 8,192
# DAPO-Math problems, then evaluated on gsm8k, minerva_math and aime24.
#
# Usage:
#   bash scripts/preflight.sh      # ALWAYS run this first (about 5 minutes)
#   bash scripts/run_all.sh
#
# Reference hardware: 1x A100/H100 80GB. Budget roughly 6-8 GPU-hours.
# See docs/RUNBOOK.md for the rented-GPU walkthrough.
#
# The script is resumable: a stage whose artefact already exists is skipped,
# so a failure in run 4 does not cost you runs 1-3. Delete the artefact to
# force a re-run. Every stage also tees to results/logs/, so a dropped SSH
# session does not lose the output.
#
# Override the sweep with e.g.
#   CONFIGS="configs/grpo.yaml configs/cppo_p75.yaml" bash scripts/run_all.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export TOKENIZERS_PARALLELISM=false

CONFIGS="${CONFIGS:-configs/grpo.yaml configs/cppo_p50.yaml configs/cppo_p75.yaml configs/cppo_p875.yaml configs/cppo_p75_no_allocation.yaml}"
BACKEND="${BACKEND:-vllm}"
EVAL_LIMIT="${EVAL_LIMIT:-}"

mkdir -p results/logs
START_ALL=$SECONDS

# Run "$@", teeing to a log, unless the marker file already exists.
run_stage() {
  local marker="$1" label="$2"
  shift 2
  if [[ -f "$marker" ]]; then
    echo ">> SKIP  $label (found $marker)"
    return 0
  fi
  local log="results/logs/${label//\//_}.log"
  echo "=============================================================="
  echo ">> RUN   $label   (log: $log)"
  echo "=============================================================="
  local start=$SECONDS
  "$@" 2>&1 | tee "$log"
  echo ">> DONE  $label in $(( SECONDS - start ))s"
}

evaluate_checkpoint() {
  local model="$1" out="$2"
  local extra=()
  [[ -n "$EVAL_LIMIT" ]] && extra+=(--limit "$EVAL_LIMIT")
  BACKEND="$BACKEND" bash scripts/evaluate.sh "$model" "$out" "${extra[@]}"
}

echo "Sweep     : $CONFIGS"
echo "Eval      : backend=$BACKEND limit=${EVAL_LIMIT:-full}"
echo

# The untrained model is the reference point every trained row is read against.
run_stage results/base/eval.json "eval-base" \
  evaluate_checkpoint Qwen/Qwen3-0.6B results/base/eval.json

PROFILES=()
EVALS=()
for CONFIG in $CONFIGS; do
  NAME="$(basename "$CONFIG" .yaml)"
  OUT_DIR="$(python - "$CONFIG" <<'PY'
import sys
from cppo.config import load_settings
print(load_settings(["--config", sys.argv[1]]).output_dir)
PY
)"

  run_stage "results/$NAME/profile.json" "train-$NAME" \
    bash scripts/train.sh "$CONFIG" --profile-path "results/$NAME/profile.json"

  run_stage "results/$NAME/eval.json" "eval-$NAME" \
    evaluate_checkpoint "$OUT_DIR" "results/$NAME/eval.json"

  PROFILES+=("results/$NAME/profile.json")
  EVALS+=("results/$NAME/eval.json")
done

run_stage results/update_stage_benchmark.json "benchmark" \
  bash scripts/benchmark_update_stage.sh auto results/update_stage_benchmark.json

echo "=============================================================="
echo "Rendering figures and result tables"
echo "=============================================================="
python benchmarks/plot_results.py \
  --benchmark results/update_stage_benchmark.json --outdir report/figures

python -m cppo.report \
  --profiles "${PROFILES[@]}" --evals "${EVALS[@]}" \
  --baseline-eval results/base/eval.json \
  --benchmark results/update_stage_benchmark.json \
  --format markdown --output results/tables.md
python -m cppo.report \
  --profiles "${PROFILES[@]}" --evals "${EVALS[@]}" \
  --baseline-eval results/base/eval.json \
  --benchmark results/update_stage_benchmark.json \
  --format latex --output report/generated_tables.tex

echo
echo "=============================================================="
echo "Done in $(( (SECONDS - START_ALL) / 60 )) minutes."
echo "  tables  : results/tables.md"
echo "  figures : report/figures/"
echo "  logs    : results/logs/"
echo "  report  : cd report && make"
echo "=============================================================="
cat results/tables.md
