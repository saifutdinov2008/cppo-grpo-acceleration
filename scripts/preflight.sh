#!/usr/bin/env bash
# Five-minute validation of the exact GPU code path before committing hours.
#
# Usage: bash scripts/preflight.sh
#
# Runs, on real CUDA hardware with the real backends:
#   1. environment and VRAM report
#   2. GRPO for 2 steps with the vLLM rollout
#   3. CPPO for 2 steps with the vLLM rollout
#   4. every lm_eval task the sweep uses, on a couple of documents each
#   5. table rendering
#
# If this passes, `bash scripts/run_all.sh` will not fail on plumbing. If it
# fails, it fails in minutes instead of three hours into an eight-hour sweep.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export TOKENIZERS_PARALLELISM=false

echo "=============================================================="
echo "0/5  Environment"
echo "=============================================================="
python - <<'PY'
import torch
print(f"torch          {torch.__version__}")
print(f"cuda available {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("preflight requires a CUDA device")
for index in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(index)
    total = torch.cuda.get_device_properties(index).total_memory / 1024**3
    major = torch.cuda.get_device_properties(index).major
    print(f"gpu {index}         {name}  {total:.0f} GiB  sm_{major}x")
    if total < 38:
        print("  NOTE: under 40 GiB. Lower --vllm-gpu-memory-utilization and")
        print("        --per-device-train-batch-size, or expect an OOM.")
    if major < 8:
        print("  NOTE: pre-Ampere. bfloat16 is unsupported; use --torch-dtype float16.")
PY
python -c "import vllm; print(f'vllm           {vllm.__version__}')" \
  || { echo "vLLM missing: bash scripts/setup.sh --with-vllm"; exit 1; }
python -c "import lm_eval; print(f'lm_eval        {lm_eval.__version__}')"

# Import every task module the sweep will load. `minerva_math` asserts a
# specific antlr4 runtime at import time, and that assertion would otherwise
# fire only once evaluation starts -- after the training runs have completed.
python - <<'PY'
from importlib.metadata import version
import importlib

print(f"antlr4         {version('antlr4-python3-runtime')}")
for module in ("lm_eval.tasks.minerva_math.utils", "lm_eval.tasks.aime.utils"):
    importlib.import_module(module)
    print(f"task module    {module} OK")
from math_verify import parse, verify
assert verify(parse("$1/2$"), parse("0.5"))
print("math_verify    OK")
PY

PRE=outputs/preflight
rm -rf "$PRE"

echo "=============================================================="
echo "1/5  GRPO, 2 steps, vLLM rollout"
echo "=============================================================="
python -m cppo.train --config configs/grpo.yaml \
  --run-name preflight-grpo --output-dir "$PRE/grpo" \
  --max-samples 64 --max-steps 2 --max-completion-length 128 \
  --no-save-final-model --profile-path "$PRE/grpo/profile.json"

echo "=============================================================="
echo "2/5  CPPO P=0.75, 2 steps, vLLM rollout"
echo "=============================================================="
python -m cppo.train --config configs/cppo_p75.yaml \
  --run-name preflight-cppo --output-dir "$PRE/cppo" \
  --max-samples 256 --max-steps 2 --max-completion-length 128 \
  --no-save-final-model --profile-path "$PRE/cppo/profile.json"

echo "=============================================================="
echo "2b/5  Completions must terminate, not all hit the length cap"
echo "=============================================================="
# A policy that never emits EOS has every completion truncated, and with
# mask_truncated_completions that zeroes the whole gradient: the sweep runs for
# hours and the model never moves. Check it at the real completion length.
python -m cppo.train --config configs/grpo.yaml \
  --run-name preflight-length --output-dir "$PRE/length" \
  --max-samples 64 --max-steps 1 \
  --no-save-final-model --profile-path "$PRE/length/profile.json" \
  2>&1 | tee "$PRE/length.log"

python -m cppo.preflight_checks "$PRE/length.log"

echo "=============================================================="
echo "3/5  Checkpoint save/load round trip"
echo "=============================================================="
python -m cppo.train --config configs/cppo_p75.yaml \
  --run-name preflight-save --output-dir "$PRE/ckpt" \
  --max-samples 64 --max-steps 1 --max-completion-length 64 \
  --profile-path "$PRE/ckpt/profile.json"
test -f "$PRE/ckpt/config.json" || { echo "checkpoint was not written"; exit 1; }

echo "=============================================================="
echo "4/5  lm_eval through the vLLM backend, all three tasks"
echo "=============================================================="
# All three tasks, not just gsm8k: each one loads its own task module and
# datasets, and a failure in any of them would otherwise only appear after
# the training sweep has already run.
python -m cppo.evaluate --model-path "$PRE/ckpt" \
  --tasks gsm8k minerva_math aime24 \
  --backend vllm --limit 2 --max-gen-toks 256 --prompt-style boxed \
  --gpu-memory-utilization 0.5 --output-path "$PRE/eval.json"

echo "=============================================================="
echo "5/5  Table rendering"
echo "=============================================================="
python -m cppo.report \
  --profiles "$PRE/grpo/profile.json" "$PRE/cppo/profile.json" \
  --evals none "$PRE/eval.json"

echo
echo "PREFLIGHT PASSED — scripts/run_all.sh is safe to start."
