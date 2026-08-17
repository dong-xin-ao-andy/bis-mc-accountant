# BIS Monte Carlo Accountant

This repository contains the implementation accompanying **[Less Random, More
Private: A Design Theory for Independent-Example Subsampling in
DP-SGD](https://arxiv.org/abs/2605.07072)** by Andy Dong and Ayfer Özgür.

Balanced Iteration Subsampling (BIS), also called random allocation, lets each
example independently choose a uniformly random `k`-subset of `T` training
iterations. The paper characterizes why fixing participation counts can improve
privacy in low-noise regimes, proves exact finite-noise optimality of BIS among
fixed-count participation laws, and gives a complementary local high-noise
hierarchy. The claims are scoped to the independent-example family and the
zero-out Gaussian dominating pair; they are not universal claims over fixed-size
minibatches, reshuffling, or replacement adjacency.

The accountant evaluates the full BIS Gaussian-mixture likelihood. It combines:

1. an `O(Tk)` log-domain elementary-symmetric-polynomial dynamic program;
2. an `O(T)` arithmetic-mean screen for the forward hockey-stick divergence;
3. an `O(T)` geometric-mean screen for the reverse divergence; and
4. ordered Estimate--Verify--Release (EVR) calibration.

Both privacy directions are checked by default. Screening only avoids exact
evaluations whose hockey-stick contribution is provably zero; it does not alter
the estimator.

## Repository structure

```text
├── bis_calibration.py                     # Two-sided calibration CLI
├── symmetric_polynomial.py                # Exact likelihood and two screens
├── delta_calculation.py                   # EVR statistical calculations
├── ACCOUNTING.md                          # Guarantee/implementation details
├── benchmark_screening.py                 # Screening benchmark
├── benchmark_screening_results.json       # Recorded benchmark output
├── submit_bis_calibration.sbatch          # Example fixed configuration
├── submit_bis_calibration_template.sbatch # Environment-variable template
├── run_example.sh                         # Small local demonstration
├── requirements.txt
└── tests/test_bis_accounting.py
```

## Installation and tests

The revised code was tested with Python 3.12.13, NumPy 1.26.3, and SciPy 1.12.0.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Quick start

```bash
bash run_example.sh
```

Equivalently:

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
  --num-workers 4 \
  --directions both
```

Candidates must be ordered from more private to less private. The sweep freezes
at its first failure and returns the last passing grid point. This is a certified
selection from the supplied ordered grid, not a proof that no smaller noise
multiplier could work. If the first candidate fails, the certified result is a
data-independent no-training fallback.

The CLI also reports raw empirical, or "optimistic," outcomes for diagnosis.
Those values are not end-to-end DP guarantees.

### Important arguments

- `--directions both` is the default and checks `H(P||Q)` and `H(Q||P)` using
  independent samples. `--directions forward` is diagnostic/reproduction-only.
- Omitting `--seed` uses fresh OS entropy, as required by the operational
  randomized EVR guarantee. A fixed `--seed` is useful for reproducible research
  but does not itself instantiate that information-theoretic randomized pipeline.
- If `--num-workers` is omitted, the CLI uses `SLURM_CPUS_PER_TASK` when set and
  otherwise uses one worker.
- `--samples-per-sigma` is the number of verifier draws **per direction and per
  candidate**.

See [ACCOUNTING.md](ACCOUNTING.md) for the precise adjacency model, directional
estimands, stopping/fallback rule, randomness requirements, and numerical
limitations.

## Benchmark

```bash
python benchmark_screening.py \
  --T 2000 --k 655 --sigma 20.5 --epsilon 3 \
  --samples 512 --repeats 7 --audit-samples 1000000
```

The benchmark isolates likelihood evaluation; it is not an end-to-end
calibration runtime measurement. The checked-in JSON records one run of the
released log-domain implementation.

## Numerical scope

The mathematical recurrence and screens are exact identities over the reals.
The implementation uses NumPy pseudorandomness, binary64 arithmetic, and
floating-point Gaussian samples without directed rounding. EVR controls
statistical verification error in the ideal sampling model; it does not turn
ordinary floating-point evaluation into a bit-level formal certificate.

## Licensing

The repository is MIT licensed. `delta_calculation.py` is derived from Google
DeepMind privacy code and retains its Apache 2.0 header and terms.

## Citation

```bibtex
@article{dong2026less,
  title={Less Random, More Private: A Design Theory for Independent-Example
         Subsampling in DP-SGD},
  author={Dong, Andy and {\"O}zg{\"u}r, Ayfer},
  journal={arXiv preprint arXiv:2605.07072},
  year={2026}
}
```
