# Verified binary64 LSPI policy-sequence control

Source `7cf2a1989e4eaebde8522a16168de8c016d5a31a`; clean at launch: True.

**6 runs, 12 requests, 36 executions; 0 mismatches.**

Lifecycle time includes construction, initial learning, verification, exact fallback, and every sequential request. Direct retained-only scratch LSPI is the reference.

## Lifecycle median milliseconds

| Regime | States | Dimension | scratch_retrain | inverse_policy | verified_interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| fragile | 128 | 48 | 127487.110 | 23730.607 | 21497.633 |
| fragile | 128 | 64 | 256890.159 | 46097.419 | 34516.752 |

## Verification outcomes

| Regime | States | Dimension | Verifier | Certified requests | Exact fallbacks | Maximum inverse defect (decimal) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| fragile | 128 | 48 | verified_interval | 6 / 6 | 0 | 8.500e-13 |
| fragile | 128 | 64 | verified_interval | 6 / 6 | 0 | 1.543e-12 |

## Python allocation measurements

These are separate `tracemalloc` passes. Input problems, trajectory logs, and deletion batches are allocated before tracing and excluded. Peaks include method construction, initial replay, verification or fallback state, and all sequential requests. They are Python allocations, not process RSS or accelerator memory.

| Regime | States | Dimension | Method | Lifecycle peak MiB | End MiB | Maximum request increment MiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| fragile | 128 | 48 | inverse_policy | 5.618 | 2.870 | 2.750 |
| fragile | 128 | 48 | verified_interval | 1.466 | 0.678 | 0.789 |
| fragile | 128 | 64 | inverse_policy | 11.653 | 5.406 | 6.263 |
| fragile | 128 | 64 | verified_interval | 2.373 | 0.928 | 1.449 |

## Interpretation limits

- The certificate proves equality of the finite LSPI policy sequence, not equality of floating weights or a full matrix trace.
- The rational verifier treats computed binary64 numbers as exact dyadic rationals. The interval verifier instead encloses every conversion, product, and sum with directed `nextafter` expansion. Both apply the same inverse-defect theorem; inconclusive cases use exact replay.
- The rational verifier performs exact matrix products; the interval verifier performs cubic outward-rounded binary64 products. Any speed result is measured rather than inferred.
- Different seeds regenerate independent datasets. Repeats for one seed time the same generated dataset with independently shuffled method order; they are timing repeats, not additional dataset samples.
- These synthetic fixed-feature tasks do not establish neural-RL unlearning, privacy, historical-state erasure, universal speedup, or publication novelty.
