#!/usr/bin/env bash
# Evaluate a checkpoint on gsm8k, minerva_math and aime24 with lm_eval.
#
# Usage: bash scripts/evaluate.sh <model-path-or-hub-id> <output.json> [extra flags]
# Example:
#   bash scripts/evaluate.sh Qwen/Qwen3-0.6B results/base/eval.json
#   bash scripts/evaluate.sh outputs/cppo-p75 results/cppo-p75/eval.json --backend vllm
#
# The system prompt matches the one used during training so that the train and
# test prompt formats agree; see src/cppo/data.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${1:?usage: scripts/evaluate.sh <model> <output.json> [extra flags]}"
OUTPUT="${2:?usage: scripts/evaluate.sh <model> <output.json> [extra flags]}"
shift 2 || true

BACKEND="${BACKEND:-hf}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-2048}"
# Must exceed the longest few-shot prompt plus MAX_GEN_TOKS; minerva_math is
# 4-shot and overflows a 4096 window, which would left-truncate the prompt.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=0

python -m cppo.evaluate \
  --model-path "$MODEL" \
  --tasks gsm8k minerva_math aime24 \
  --backend "$BACKEND" \
  --batch-size "$BATCH_SIZE" \
  --max-gen-toks "$MAX_GEN_TOKS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --prompt-style boxed \
  --output-path "$OUTPUT" \
  "$@"
