#!/usr/bin/env bash
# Run every static check the repository is required to pass.
#
# Usage: bash scripts/lint.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== pylint =="
python -m pylint src/cppo tests benchmarks

echo "== mypy (strict) =="
python -m mypy

echo "== pytest =="
python -m pytest tests -q

echo "All checks passed."
