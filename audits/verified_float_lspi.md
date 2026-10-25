# Verified binary64 LSPI deletion replay

Status: candidate numerical checkpoint, 24 October 2026. This note states the
proof contract and prior-art boundary. It does not claim a new verified-linear-
algebra theorem, a privacy guarantee, or publication novelty.

## Output contract

The declared learner is the repository's finite-iteration, fixed-feature LSPI
schedule with rational inputs, data-independent initial policy, and fixed action
order. The verified binary64 path returns only the policy sequence, final
policy, cycle/fixed-point status, and diagnostics. It does not claim equality of
binary64 weights with exact rational weights or return a full exact trace.

For every retained LSTD system `A x = b`, partial-pivoted binary64 elimination
produces an approximate inverse `R` and solution `x_hat`. The verifier interprets
their IEEE-754 values as exact dyadic rationals and computes

```text
E = I - R A,
eta = ||E||_infinity,
r = b - A x_hat.
```

If `eta < 1`, then `R A = I-E` is nonsingular, hence both square matrices `R`
and `A` are nonsingular. Moreover,

```text
A^{-1} = (I-E)^{-1} R
||x_hat - A^{-1}b||_infinity
  <= ||R r||_infinity / (1-eta).
```

All quantities on the right are evaluated with exact rational arithmetic. For
chosen action `c` and competitor `a`, Holder's inequality bounds the score error
by

```text
||(phi(s,c)-phi(s,a))||_1 * weight_error_bound.
```

The fast decision is accepted only when every nonidentical competitor clears
this bound strictly. Identical feature rows preserve the declared action order.
An inconclusive inverse or action comparison triggers complete exact cached
replay. A request commits retained statistics only after either the verified
schedule or exact fallback completes nonsingularly.

Induction over the fixed LSPI schedule proves equality of every output policy:
the initial policy is shared; a verified evaluation selects the exact rational
greedy policy; therefore the next evaluation matrix is the same; and the
argument repeats. Exact fallback inherits the existing retained-only replay
proof.

## Failure found in ordinary binary64 scoring

The regression suite contains a two-state, two-action, two-feature LSPI problem
with feature scales from `10^-12` to `10^9`, discount `0.999999`, and ridge
`10^-12`. Ordinary binary64 solves followed by ordinary binary64 dot products
choose `((0,0),(0,0),(0,0))`. Exact rational LSPI chooses
`((1,0),(1,0),(1,0))`. The rational verifier returns the exact sequence and
accepts all three evaluations. The interval verifier detects that the ordinary
binary64 proposal is not certifiable and returns the exact sequence through
fallback. This establishes a concrete silent policy error in the naive
numerical path and two safe outcomes; it does not establish frequency on
realistic data.

## Prior-art boundary

Verified linear-system solution using approximate inverses and residual
inclusions is established numerical analysis. Rump's survey gives a general
verified dense-system algorithm based on an approximate inverse, residual, and
iteration matrix: [Verification methods: Rigorous results using floating-point
arithmetic](https://www.tuhh.de/ti3/rump/intlab/ActaNumerica2010.pdf). Rump also
showed that approximate inverses of highly conditioned matrices can remain
useful as preconditioners: [Approximate inverses of almost singular matrices
still contain useful information](https://doi.org/10.15480/882.319).

LSPI and incremental LSTD inverse updates are established by Lagoudakis and
Parr: [Least-Squares Policy Iteration](https://jmlr.csail.mit.edu/papers/volume4/lagoudakis03a/lagoudakis03a.pdf).
Finite-sample LSPI error propagation is also established: [Lazaric,
Ghavamzadeh, and Munos](https://jmlr.org/papers/v13/lazaric12a.html).

The repository-specific result is the composition of a posteriori numerical
verification with sequential trajectory deletion and exact LSPI policy-sequence
replay. A focused search did not locate this exact composition, but that absence
is not evidence of novelty. A paper claim requires a broader bibliography,
measured advantage over strong exact and verified controls, and a result beyond
applying standard verification to the existing replay algebra.

The broader unlearning boundary has also moved. Nguyen-Tang and Arora's
[Exact Unlearning in Reinforcement Learning](https://proceedings.mlr.press/v306/nguyen-tang26a.html)
provides an exact-unlearning framework for online tabular RL using total-
variation stability; it is a different learner and guarantee, but prevents a
broad claim to “exact RL unlearning.” The recent
[TOUR benchmark](https://arxiv.org/abs/2607.21111) reinforces the need for
retraining references, retained-utility anchors, and multiple privacy attacks
before any neural offline-RL claim.

A second focused search on 25 October used combinations of “verified LSPI,”
“verified LSTD,” numerical policy certificates, trajectory deletion, and exact
policy sequences. It recovered LSPI numerical/convergence literature and
general machine-unlearning verification, but no primary work with this exact
deletion-plus-numerical-certificate contract. This is a search result, not proof
of priority.

## Cost and remaining gate

The exact-rational verifier performs rational `R A`, residual, and `R r`
products. The second implementation encloses every exact-input conversion and
every binary64 product, sum, subtraction, and final division by stepping one
representable value outward with `math.nextafter`. It requires the recorded
platform to expose radix-2, 53-bit IEEE-754 floats, rejects nonfinite
intermediates, and uses the upper interval norm in the same theorem. Random
tests compare its bound with the rational verifier and exact solutions. This is
still cubic in feature dimension, and wall time includes the failed fast path
when fallback occurs. The rational verifier remains the oracle for the interval
implementation.

The completed high-dimensional experiment finds the empirical crossover that
was absent at dimensions 8–32. Across two independently generated fragile
datasets and two sequential deletions per dataset, the interval verifier is
1.35x faster than exact policy-only inverse maintenance at dimension 48 and
1.44x faster at dimension 64. It wins on all four individual seeds/settings,
certifies every request without fallback, and matches direct retained-only
scratch LSPI. These are descriptive synthetic timings, not an asymptotic or
significance result.
