# What is the Optimal Subsampling Scheme for DP-SGD? Theory and Near-Exact Accounting

This repository contains the official code for the paper **[Less Random, More Private: What is the Optimal Subsampling Scheme for DP-SGD?]**, authored by Andy Dong and Ayfer Özgür (Stanford University, 2026).

## Overview

For a decade, Poisson subsampling has been the default mechanism for differentially private machine learning (DP-SGD). In our work, we prove that under a fixed expected participation constraint, the optimal sampling scheme in high-utility, low-noise regimes must strictly eliminate participation variance. This establishes **Balanced Iteration Subsampling (BIS)** as the mathematically optimal independent-example mechanism.

To bypass the analytical slack of existing composition-based RDP and PLD accountants, this repository provides a practical, near-exact **Monte Carlo accounting framework**. It is powered by:
1. An highly efficient `O(Tk)` dynamic program for exact log-likelihood ratio evaluation.
2. An ultra-fast `O(T)` screening bound that acts as a computational filter.

This tool provides near-exact numerical evidence establishing BIS as a strictly superior alternative to Poisson sampling in practical DP-SGD regimes.

## Repository Structure

```text
├── README.md
├── requirements.txt
├── run_example.sh            # Quick start demonstration script
├── bis_calibration.py        # Main CLI tool for Monte Carlo calibration
├── symmetric_polynomial.py   # Core logic for DP evaluation and screening
└── delta_calculation.py      # Statistical EVR verification functions
```

## Installation

This codebase was tested on **Python 3.11.5**. To install the required dependencies (`numpy` and `scipy`), run:

```bash
pip install -r requirements.txt
```

## Quick Start

To verify that your environment is set up correctly and see the calibrator in action, run the included example shell script. This executes a fast, small-scale sweep that outputs both the rigorous certified bounds and the optimistic estimates:

```bash
bash run_example.sh
```

## Usage: Running the Calibrator

`bis_calibration.py` is a command-line tool that performs a sequential line search to find the minimum required noise multiplier (`sigma`) to satisfy a target `(ε, δ)`-DP guarantee. It utilizes multiprocessing to heavily parallelize the Monte Carlo sampling.

**Example Command:**
```bash
python bis_calibration.py \
    --eps 8.0 \
    --delta 1.25e-5 \
    --T 391 \
    --k 5 \
    --initial-sigma 0.60 \
    --sigma-step 0.05 \
    --min-sigma 0.50 \
    --samples-per-sigma 2000000 \
    --num-workers 4
```

### Key Arguments:
* `--eps`, `--delta`: Your target privacy budget.
* `--T`: Total number of training iterations.
* `--k`: Exact number of participations per example (for BIS).
* `--initial-sigma`, `--min-sigma`, `--sigma-step`: The parameters defining the line search for the noise multiplier. The search sweeps from `initial-sigma` downwards.
* `--samples-per-sigma`: Number of Monte Carlo samples to generate per candidate noise multiplier.
* `--num-workers`: Number of parallel CPU processes to use. *(Note: If running on a Slurm cluster, this will automatically default to `$SLURM_CPUS_PER_TASK` if not explicitly provided.)*
* `--run-optimistic`: (Optional flag) If included, the script will simultaneously compute and sweep based on unbiased optimistic Monte Carlo estimates, representing the fundamental privacy limit given unlimited compute.

## Acknowledgments and Licensing

* The exact sampling algorithms and DP evaluations are provided in `symmetric_polynomial.py` and `bis_calibration.py`.
* `delta_calculation.py` is included here as a standalone copy to ensure repository stability. It is originally derived from Google DeepMind's privacy libraries and remains subject to the Apache 2.0 license provided in its header.

## Citation

If you use this code or our theoretical results in your research, please cite our paper:

```bibtex
@article{dong2026less,
  title={Less Random, More Private: What is the Optimal Subsampling Scheme for DP-SGD?},
  author={Dong, Andy and {\"O}zg{\"u}r, Ayfer},
  journal={arXiv preprint},
  year={2026}
}
```