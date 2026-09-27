#!/usr/bin/env bash
# Five-minute validation of the exact GPU code path before committing hours.
#
# Usage: bash scripts/preflight.sh
#
# Runs, on real CUDA hardware with the real backends:
#   1. environment and VRAM report
#   2. GRPO for 2 steps with the vLLM rollout
#   3. CPPO for 2 steps with the vLLM rollout
#   4. lm_eval on 8 GSM8K documents through the vLLM backend
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
echo "3/5  Checkpoint save/load round trip"
echo "=============================================================="
python -m cppo.train --config configs/cppo_p75.yaml \
  --run-name preflight-save --output-dir "$PRE/ckpt" \
  --max-samples 64 --max-steps 1 --max-completion-length 64 \
  --profile-path "$PRE/ckpt/profile.json"
test -f "$PRE/ckpt/config.json" || { echo "checkpoint was not written"; exit 1; }

echo "=============================================================="
echo "4/5  lm_eval through the vLLM backend, 8 documents"
echo "=============================================================="
python -m cppo.evaluate --model-path "$PRE/ckpt" --tasks gsm8k \
  --backend vllm --limit 8 --max-gen-toks 256 --prompt-style boxed \
  --gpu-memory-utilization 0.5 --output-path "$PRE/eval.json"

echo "=============================================================="
echo "5/5  Table rendering"
echo "=============================================================="
python -m cppo.report \
  --profiles "$PRE/grpo/profile.json" "$PRE/cppo/profile.json" \
  --evals none "$PRE/eval.json"

echo
echo "PREFLIGHT PASSED — scripts/run_all.sh is safe to start."
