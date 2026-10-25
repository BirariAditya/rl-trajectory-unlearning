# Verified binary64 LSPI policy-sequence control

Source `3302a1871a80a1119f10b7952b2ee28646ec6811`; clean at launch: True.

**8 runs, 80 requests, 320 executions; 0 mismatches.**

Lifecycle time includes construction, initial learning, verification, exact fallback, and every sequential request. Direct retained-only scratch LSPI is the reference.

## Lifecycle median milliseconds

| Regime | States | Dimension | scratch_retrain | inverse_policy | verified_rational | verified_interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fragile | 32 | 8 | 7795.983 | 270.965 | 384.192 | 346.421 |
| fragile | 128 | 16 | 43921.400 | 3973.991 | 5894.402 | 5703.813 |
| stable | 32 | 8 | 26634.433 | 652.926 | 1207.802 | 935.857 |
| stable | 128 | 16 | 80616.040 | 4168.183 | 7741.247 | 7197.902 |

## Verification outcomes

| Regime | States | Dimension | Verifier | Certified requests | Exact fallbacks | Maximum inverse defect (decimal) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| fragile | 32 | 8 | verified_rational | 20 / 20 | 0 | 1.936e-15 |
| fragile | 32 | 8 | verified_interval | 20 / 20 | 0 | 8.619e-15 |
| fragile | 128 | 16 | verified_rational | 20 / 20 | 0 | 8.315e-15 |
| fragile | 128 | 16 | verified_interval | 20 / 20 | 0 | 3.669e-14 |
| stable | 32 | 8 | verified_rational | 20 / 20 | 0 | 1.936e-15 |
| stable | 32 | 8 | verified_interval | 20 / 20 | 0 | 8.583e-15 |
| stable | 128 | 16 | verified_rational | 20 / 20 | 0 | 9.225e-15 |
| stable | 128 | 16 | verified_interval | 20 / 20 | 0 | 3.628e-14 |

## Interpretation limits

- The certificate proves equality of the finite LSPI policy sequence, not equality of floating weights or a full matrix trace.
- The rational verifier treats computed binary64 numbers as exact dyadic rationals. The interval verifier instead encloses every conversion, product, and sum with directed `nextafter` expansion. Both apply the same inverse-defect theorem; inconclusive cases use exact replay.
- The rational verifier performs exact matrix products; the interval verifier performs cubic outward-rounded binary64 products. Any speed result is measured rather than inferred.
- These synthetic fixed-feature tasks do not establish neural-RL unlearning, privacy, historical-state erasure, universal speedup, or publication novelty.
