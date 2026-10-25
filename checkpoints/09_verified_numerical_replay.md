# Checkpoint 09: verified numerical LSPI deletion replay

## Breakthrough

This checkpoint establishes a bounded fixed-feature breakthrough: trajectory
deletions can be replayed with binary64 linear algebra while certifying the
**exact rational LSPI policy sequence**, including changed-policy branches. An
outward-rounded interval verifier becomes faster than the strongest exact
policy-only inverse-maintenance control at feature dimensions 48 and 64 after
charging construction, original learning, verification, and all deletion
requests.

The result is conditional and deliberately narrow. It is a policy-sequence
certificate for a declared finite LSPI schedule, not equality of weights, a
full exact trace, privacy unlearning, or a neural-RL result.

## Method and correctness argument

For each retained LSTD system `A x = b`, partial-pivoted binary64 elimination
produces an approximate inverse `R` and solution `x_hat`. With

```text
E = I - R A,       eta = ||E||_infinity < 1,
r = b - A x_hat,
```

the exact rational solution satisfies

```text
||x_hat - A^{-1}b||_infinity
    <= ||R r||_infinity / (1-eta).
```

The interval implementation encloses exact-rational input conversion and every
binary64 product, sum, subtraction, and division with one-ULP outward
`nextafter` expansion. It accepts a greedy action only when its lower-bounded
score gap exceeds the feature-difference norm times the certified weight-error
bound. Inconclusive evaluations invoke the existing exact cached replay, and
singular requests do not commit deletion state.

Induction over the shared data-independent initial policy proves equality of
the complete finite policy sequence. The full proof, platform assumptions, and
prior-art boundary are in
[the numerical audit](../audits/verified_float_lspi.md).

## Numerical failure caught

The regression suite contains a two-state, two-action example with feature
scales from `10^-12` to `10^9`, discount `0.999999`, and ridge `10^-12`.
Ordinary binary64 solves followed by ordinary binary64 action-score dot
products return the wrong policy at every one of three LSPI iterations. The
exact-rational verifier certifies the correct sequence without fallback. The
outward interval variant detects that the ordinary proposed action cannot be
certified and uses exact fallback. This is a concrete silent numerical policy
failure rather than a weight-error-only example.

## Reproduction and validation

The implementation has 126 passing tests. The clean experiment revisions and
commands are:

```powershell
# Dimensions 8 and 16; source 3302a1871a80a1119f10b7952b2ee28646ec6811
python scripts/run_lspi_verified_float.py --seeds 0 1 --repeats 1 --requests 10 --output results/lspi_verified_interval.json.gz --summary results/lspi_verified_interval_summary.md

# Dimensions 24 and 32; source 41a9dbaad24afb4e0c80b200f7e6da1eae86d492
python scripts/run_lspi_verified_float.py --crossover --seeds 0 1 --repeats 1 --requests 5 --output results/lspi_verified_crossover.json.gz --summary results/lspi_verified_crossover_summary.md

# Dimensions 48 and 64; source 2655d88c414eb54b9ffc285e556bba68a41e1052
python scripts/run_lspi_verified_float.py --highdim --seeds 0 1 --repeats 1 --requests 2 --output results/lspi_verified_highdim.json.gz --summary results/lspi_verified_highdim_summary.md
```

Together these contain 20 clean dataset runs using two seeds per setting, 128
sequential deletion requests, and 504 method executions with zero policy-sequence
mismatches. Every ordinary synthetic request is certified without fallback.
The adversarial regression separately exercises fallback and atomicity.

## Lifecycle results

At dimensions 8 through 32, interval verification remains slower than exact
inverse maintenance. The preliminary one-seed dimension-32 win disappeared in
the stronger two-seed, five-deletion run and is retained as a negative result.

The high-dimensional clean archive gives:

| Regime | States | Dimension | Scratch retrain ms | Exact inverse ms | Interval verified ms | Speedup over inverse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fragile | 128 | 48 | 114146.851 | 21396.579 | 15908.444 | 1.35x |
| fragile | 128 | 64 | 213101.641 | 43101.550 | 30012.822 | 1.44x |

The interval method wins on every high-dimensional seed: 1.48x and 1.21x at
dimension 48, and 1.43x and 1.45x at dimension 64. It certifies all eight
requests with no exact fallback. Maximum interval inverse defects are
`8.50e-13` and `1.54e-12`, both far below the required bound of one.

This is an empirical crossover, not an asymptotic theorem. Exact rational cost
depends on operand bit growth; interval verification still performs cubic
fixed-width work. The two-seed medians are descriptive, not significance
estimates.

## Prior-art and novelty boundary

Approximate-inverse residual verification is established numerical analysis,
and LSPI plus incremental inverse updates are established RL methods. Current
RL unlearning work also prevents any broad claim to exact RL unlearning. A
focused search did not locate the exact combination of sequential trajectory
deletion, finite LSPI policy-sequence replay, outward numerical certification,
and atomic retained-only fallback. That absence does not prove novelty.

The defensible project contribution is therefore:

1. a deletion-specific composition with a precise policy-sequence contract;
2. a concrete binary64 policy failure that the certificate catches;
3. exact retained-only validation through policy changes and sequential
   deletions; and
4. a measured high-dimensional crossover against exact inverse maintenance,
   with all lower-dimensional losing regimes preserved.

A publication claim still requires a broader systematic bibliography, more
independent timings, memory measurements, and external tasks with frozen
features.

## Artifacts

- [Standard summary](../output/benchmarks/lspi_verified_interval_summary.md)
  and [raw archive](../output/benchmarks/lspi_verified_interval.json.gz)
- [Crossover summary](../output/benchmarks/lspi_verified_crossover_summary.md)
  and [raw archive](../output/benchmarks/lspi_verified_crossover.json.gz)
- [High-dimensional summary](../output/benchmarks/lspi_verified_highdim_summary.md)
  and [raw archive](../output/benchmarks/lspi_verified_highdim.json.gz)
