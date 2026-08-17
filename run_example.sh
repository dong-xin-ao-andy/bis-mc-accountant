#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python3}"

echo "Starting a small two-sided BIS Monte Carlo calibration."
echo "No fixed seed is supplied; EVR uses fresh OS entropy."

"$python_bin" bis_calibration.py \
  --eps 8.0 \
  --delta 1.25e-5 \
  --T 391 \
  --k 5 \
  --initial-sigma 0.60 \
  --sigma-step 0.05 \
  --min-sigma 0.50 \
  --samples-per-sigma 2000000 \
  --num-workers 4 \
  --directions both
