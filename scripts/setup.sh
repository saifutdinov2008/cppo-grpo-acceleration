#!/usr/bin/env bash
# Create the project virtual environment and install every dependency.
#
# Usage: bash scripts/setup.sh [--with-vllm]
#
# The base install is enough to train on CPU/CUDA with the Transformers
# generation backend and to evaluate with lm-evaluation-harness. Pass
# --with-vllm on a CUDA machine to also install vLLM, which the reference
# configs use for the rollout stage.
#
# On a cloud image that already ships a CUDA build of torch (RunPod, Lambda,
# Colab), set CPPO_NO_VENV=1 to install into the existing environment instead
# of building a virtualenv, which would re-download several GB of torch:
#
#   CPPO_NO_VENV=1 bash scripts/setup.sh --with-vllm
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

if [[ "${CPPO_NO_VENV:-0}" == "1" ]]; then
  echo ">> Installing into the current environment (CPPO_NO_VENV=1)"
else
  if [[ ! -d "$VENV" ]]; then
    echo ">> Creating virtual environment in $VENV"
    "$PYTHON" -m venv "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

echo ">> Upgrading build tooling"
python -m pip install --upgrade pip setuptools wheel

echo ">> Installing the project with its eval and dev extras"
python -m pip install -e ".[eval,dev]"

if [[ "${1:-}" == "--with-vllm" ]]; then
  echo ">> Installing vLLM (CUDA only)"
  python -m pip install -e ".[vllm]"
fi

echo ">> Environment summary"
python - <<'PY'
import torch, transformers, trl
print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")
print(f"trl          {trl.__version__}")
print(f"cuda         {torch.cuda.is_available()} ({torch.cuda.device_count()} device(s))")
PY

if [[ "${CPPO_NO_VENV:-0}" == "1" ]]; then
  echo ">> Done. Next: bash scripts/preflight.sh"
else
  echo ">> Done. Activate with: source $VENV/bin/activate"
fi
