# Verified binary64 LSPI policy-sequence control

Source `41a9dbaad24afb4e0c80b200f7e6da1eae86d492`; clean at launch: True.

**8 runs, 40 requests, 160 executions; 0 mismatches.**

Lifecycle time includes construction, initial learning, verification, exact fallback, and every sequential request. Direct retained-only scratch LSPI is the reference.

## Lifecycle median milliseconds

| Regime | States | Dimension | scratch_retrain | inverse_policy | verified_rational | verified_interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fragile | 64 | 24 | 38898.887 | 2827.464 | 3271.702 | 3029.453 |
| fragile | 96 | 32 | 76573.089 | 7447.490 | 8420.578 | 7747.519 |
| stable | 64 | 24 | 37905.236 | 1754.405 | 2662.324 | 2352.397 |
| stable | 96 | 32 | 67359.381 | 3573.884 | 5549.174 | 5068.969 |

## Verification outcomes

| Regime | States | Dimension | Verifier | Certified requests | Exact fallbacks | Maximum inverse defect (decimal) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| fragile | 64 | 24 | verified_rational | 10 / 10 | 0 | 2.492e-14 |
| fragile | 64 | 24 | verified_interval | 10 / 10 | 0 | 1.377e-13 |
| fragile | 96 | 32 | verified_rational | 10 / 10 | 0 | 4.907e-14 |
| fragile | 96 | 32 | verified_interval | 10 / 10 | 0 | 2.883e-13 |
| stable | 64 | 24 | verified_rational | 10 / 10 | 0 | 2.374e-14 |
| stable | 64 | 24 | verified_interval | 10 / 10 | 0 | 1.312e-13 |
| stable | 96 | 32 | verified_rational | 10 / 10 | 0 | 5.594e-14 |
| stable | 96 | 32 | verified_interval | 10 / 10 | 0 | 2.905e-13 |

## Interpretation limits

- The certificate proves equality of the finite LSPI policy sequence, not equality of floating weights or a full matrix trace.
- The rational verifier treats computed binary64 numbers as exact dyadic rationals. The interval verifier instead encloses every conversion, product, and sum with directed `nextafter` expansion. Both apply the same inverse-defect theorem; inconclusive cases use exact replay.
- The rational verifier performs exact matrix products; the interval verifier performs cubic outward-rounded binary64 products. Any speed result is measured rather than inferred.
- These synthetic fixed-feature tasks do not establish neural-RL unlearning, privacy, historical-state erasure, universal speedup, or publication novelty.
