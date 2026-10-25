# Verified binary64 LSPI policy-sequence control

Source `2655d88c414eb54b9ffc285e556bba68a41e1052`; clean at launch: True.

**4 runs, 8 requests, 24 executions; 0 mismatches.**

Lifecycle time includes construction, initial learning, verification, exact fallback, and every sequential request. Direct retained-only scratch LSPI is the reference.

## Lifecycle median milliseconds

| Regime | States | Dimension | scratch_retrain | inverse_policy | verified_interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| fragile | 128 | 48 | 114146.851 | 21396.579 | 15908.444 |
| fragile | 128 | 64 | 213101.641 | 43101.550 | 30012.822 |

## Verification outcomes

| Regime | States | Dimension | Verifier | Certified requests | Exact fallbacks | Maximum inverse defect (decimal) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| fragile | 128 | 48 | verified_interval | 4 / 4 | 0 | 8.500e-13 |
| fragile | 128 | 64 | verified_interval | 4 / 4 | 0 | 1.543e-12 |

## Interpretation limits

- The certificate proves equality of the finite LSPI policy sequence, not equality of floating weights or a full matrix trace.
- The rational verifier treats computed binary64 numbers as exact dyadic rationals. The interval verifier instead encloses every conversion, product, and sum with directed `nextafter` expansion. Both apply the same inverse-defect theorem; inconclusive cases use exact replay.
- The rational verifier performs exact matrix products; the interval verifier performs cubic outward-rounded binary64 products. Any speed result is measured rather than inferred.
- These synthetic fixed-feature tasks do not establish neural-RL unlearning, privacy, historical-state erasure, universal speedup, or publication novelty.
