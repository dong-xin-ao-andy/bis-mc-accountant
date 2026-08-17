# Certified BIS Monte Carlo accounting

`bis_calibration.py` calibrates the Gaussian noise for balanced iteration
subsampling (BIS) under zero-out adjacency. Let

- `P` be the BIS Gaussian mixture obtained when the distinguished record has a
  uniformly random binary participation vector of Hamming weight `k`; and
- `Q = N(0, sigma^2 I_T)` be the output when that record is zeroed out.

The realized BIS participation schedule is internal mechanism randomness. The
accounting guarantee does not cover releasing per-record schedules or their
random seeds alongside the noisy output.

The default command verifies the full two-sided condition
`max(H_exp(eps)(P || Q), H_exp(eps)(Q || P)) <= delta`. It follows the
convention used by DeepMind's JAX Privacy Monte Carlo utilities:

- forward draws are sampled from `P` and store `log(P / Q)`;
- reverse draws are sampled from `Q` and store `log(Q / P)`; and
- a candidate passes only if both empirical hockey-stick divergences are at
  most the certified base threshold.

The upstream convention is implemented in
[`perform_calibration_from_samples`](https://github.com/google-deepmind/jax_privacy/blob/main/jax_privacy/experimental/monte_carlo/delta_calculation.py)
and its sample generator negates `log(P / Q)` for an excluded-record draw.

## Running the accountant

The default is two-sided:

```bash
python3 -u bis_calibration.py \
  --eps 8 \
  --delta 1e-5 \
  --T 10000 \
  --k 205 \
  --initial-sigma 1.42 \
  --samples-per-sigma 100000000 \
  --sigma-step 0.02 \
  --directions both
```

`--samples-per-sigma` is the number of independent draws **per requested
direction**. Thus, `--directions both` draws that many forward samples and that
many reverse samples for every candidate noise value. Independent seed streams
are used for the two directions.

By default, the CLI obtains a fresh calibration seed from OS entropy. Passing
`--seed` is supported only for reproducible scientific audits: an operational
information-theoretic EVR deployment must use fresh verifier coins rather than
a fixed public seed. The fresh seed is not printed; publishing verifier coins
together with a release is outside the stated end-to-end guarantee.

For a one-sided diagnostic or to reproduce a forward-only analysis, pass
`--directions forward`. This mode does not by itself certify the two-sided
zero-out DP definition unless forward dominance is proved separately.

The Slurm templates set `--directions both` explicitly and omit `--seed` by
default, so each submitted verifier uses fresh OS entropy. To reproduce an
audit run, submit with a nonempty `SEED` environment variable; the scripts then
pass that value and print a warning that the fixed seed is for reproducibility
only. Existing job files keep their format; their sample-count column now
denotes samples per direction.

## Exact likelihood and screening

For weights `w_i = exp((2 y_i - 1) / (2 sigma^2))`, the likelihood ratio is
`e_k(w) / binom(T, k)`. The exact elementary-symmetric-polynomial dynamic
program costs `O(T k)`. Maclaurin's inequalities give the certified bounds

```text
k * mean(log(w)) <= log(e_k(w) / binom(T,k)) <= k * log(mean(w)).
```

The upper bound screens forward draws whose `log(P / Q)` cannot exceed `eps`.
The lower bound screens reverse draws whose `log(Q / P)` cannot exceed `eps`.
Only draws that survive the appropriate `O(T)` screen use the exact dynamic
program. Exact evaluation is performed from shifted log-weights to avoid
overflow.

### Numerical model

The dynamic-program recurrence and Maclaurin screens are exact identities over
the real numbers. The EVR theorem assumes ideal independent verifier draws,
whereas this implementation uses a standard pseudorandom generator, ordinary
float64 arithmetic, and floating-point Gaussian samples. EVR controls Monte
Carlo verification error in the ideal model; it does not certify the random
source, directed rounding, or a finite-precision Gaussian mechanism.
Consequently, a bit-level operational DP claim requires a validated random
source, a separately validated finite-precision Gaussian mechanism, and
outward-rounded (or otherwise certified) likelihood arithmetic. The paper uses
“near-exact” for this reason.

## Calibration output and stopping rule

Every candidate reports `forward_delta_hat`, `reverse_delta_hat`,
`max_delta_hat`, and `dominating_direction`. It also distinguishes
`*_exact_count` (draws that survive the screen and invoke the dynamic program)
from `*_positive_count` (draws whose exact hockey-stick contribution is
nonzero). These counts need not agree because a bound may conservatively send a
zero-contribution draw to exact evaluation. The certified selection is frozen
at the first failure while sweeping candidates from higher to lower noise, as
required by the ordered EVR calibration procedure. Later Monte Carlo
fluctuations cannot reopen a failed sweep. `Best optimistic sigma` remains a
clearly labeled, uncertified diagnostic.

The CLI rejects nonfinite privacy and noise parameters. The streaming
calibrator also rejects nonfinite privacy-loss samples and requires compressed
counts to be finite, nonnegative integers with an exact positive total. If a
numerical anomaly nevertheless makes an empirical divergence nonfinite or
outside `[0,1]`, that estimate is replaced by infinity, so the candidate fails
closed rather than passing through a comparison with `NaN`.

If the first (most private) candidate fails, none of the Monte Carlo-tested
candidates may be deployed. The CLI explicitly designates a data-independent
output/no-training mechanism as its safe `(0,0)`-DP fallback. A caller may
replace this with another fallback only if that mechanism has been certified
independently at the required base privacy level.

EVR certifies the full randomized **calibrate-then-run** procedure. A selected
noise value printed by one calibration run is not, by itself, a reusable
deterministic certificate conditional on that realized selection; operational
use must include the calibrated selection procedure (and its fallback) in the
mechanism.

## Local tests

With NumPy and SciPy installed, run:

```bash
python3 -m unittest discover -s tests -v
```

The tests compare the dynamic program with brute-force likelihoods, verify both
Maclaurin bounds, check the exact forward and reverse sampling conventions,
exercise compression in both directions, reject malformed numerical inputs,
confirm that a nonfinite estimator fails closed, and verify that first-failure
fallback and freeze semantics cannot be reopened by a later candidate.

## Screening wall-clock benchmark

The file **benchmark_screening.py** pre-generates forward BIS log-weights and
times only the likelihood-evaluation stage, comparing exact evaluation of every
draw with the production upper-bound screen:

    python3 benchmark_screening.py \
      --T 2000 --k 655 --sigma 20.5 --epsilon 3 \
      --samples 512 --repeats 7 \
      --audit-samples 1000000

The script alternates timing order across repetitions and verifies that the
screened and exact hockey-stick sums agree. Because it excludes random-number
generation and multiprocessing, its reported speedup is a screening
microbenchmark rather than an end-to-end calibration speedup. The optional
large audit counts screen survivors without running the exact likelihood on
every draw, and reports both the trigger fraction and the corresponding
`T k` versus `T + f T k` operation-model speedup.

`benchmark_screening_results.json` records the output of this command for the
released log-domain implementation and the environment reported in that file.
