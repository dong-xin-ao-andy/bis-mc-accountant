#!/bin/bash
# run_example.sh
# A quick, small-scale demonstration of the BIS Monte Carlo calibrator.

echo "Starting BIS Monte Carlo Calibration Example..."
echo "This will run a fast sweep with the --run-optimistic flag enabled."
echo "------------------------------------------------------------------"

python bis_calibration.py \
    --eps 8.0 \
    --delta 1.25e-5 \
    --T 391 \
    --k 5 \
    --initial-sigma 0.60 \
    --sigma-step 0.05 \
    --min-sigma 0.50 \
    --samples-per-sigma 2000000 \
    --num-workers 4 \
    --run-optimistic