#!/usr/bin/env bash
# End-to-end validation with no GPU: lint, types, unit tests, then two real
# training steps of GRPO and of CPPO on Qwen3-0.6B and DAPO-Math-17k.
#
# Usage: bash scripts/smoke_test.sh
#
# Expect a few minutes on a laptop. This is the fastest way to confirm that a
# fresh checkout is wired up correctly before committing GPU hours.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export TOKENIZERS_PARALLELISM=false

echo "== pylint =="
python -m pylint src/cppo tests benchmarks

echo "== mypy =="
python -m mypy

echo "== unit tests (including the tiny end-to-end trainer tests) =="
python -m pytest tests -q

echo "== GRPO: 2 real training steps on CPU =="
python -m cppo.train --config configs/smoke_cpu.yaml \
  --run-name smoke-grpo --output-dir outputs/smoke-grpo --pruning-rate 0.0

echo "== CPPO: 2 real training steps on CPU (P = 0.5) =="
python -m cppo.train --config configs/smoke_cpu.yaml \
  --run-name smoke-cppo --output-dir outputs/smoke-cppo --pruning-rate 0.5

echo "== summary =="
python -m cppo.report \
  --profiles outputs/smoke-grpo/profile.json outputs/smoke-cppo/profile.json

echo "OK"
