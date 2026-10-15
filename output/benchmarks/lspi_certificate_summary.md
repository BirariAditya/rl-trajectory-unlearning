# Policy-sequence certificate experiment

Source `97db5085ecdad055fd94206199d07cf0dedc5bd7`; clean at launch: True.

**45 runs, 900 requests, 3600 method executions, 0 mismatches.**

The certificate promises the same policy sequence/status only. Other methods additionally reproduce weights/matrices. This is an explicit narrower output contract, not full-state unlearning. The inverse control reuses both evaluations and greedy outputs for repeated policies.

## Request median milliseconds

| Regime | States | Features | cached_memo | replay_memo | inverse_reuse | policy_certificate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fragile | 16 | 4 | 2.858 | 2.152 | 1.911 | 0.432 |
| stable | 16 | 4 | 4.059 | 3.437 | 2.639 | 0.591 |
| stable | 128 | 4 | 28.122 | 23.085 | 10.464 | 0.612 |
| stable | 128 | 8 | 80.040 | 58.594 | 22.825 | 1.583 |
| switch | 2 | 4 | 0.518 | 0.479 | 0.571 | 0.198 |

## Lifecycle median milliseconds

| Regime | States | Features | cached_memo | replay_memo | inverse_reuse | policy_certificate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fragile | 16 | 4 | 87.281 | 70.793 | 65.925 | 83.808 |
| stable | 16 | 4 | 118.558 | 108.850 | 89.190 | 47.491 |
| stable | 128 | 4 | 637.327 | 553.150 | 269.630 | 78.241 |
| stable | 128 | 8 | 1874.282 | 1562.270 | 693.163 | 426.610 |
| switch | 2 | 4 | 18.125 | 18.179 | 23.378 | 17.704 |

## Certificate coverage and tail cost

| Regime | States | Features | Certified / requests | Fallbacks | Singular rejections | Median of run-max request ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fragile | 16 | 4 | 117 / 180 | 63 | 0 | 27.596 |
| stable | 16 | 4 | 180 / 180 | 0 | 0 | 0.889 |
| stable | 128 | 4 | 180 / 180 | 0 | 0 | 1.103 |
| stable | 128 | 8 | 171 / 180 | 9 | 0 | 184.620 |
| switch | 2 | 4 | 171 / 180 | 9 | 0 | 7.274 |

## Separate Python allocation pass

Seed 0 per setting, separate from timings. Inputs/features are allocated before tracing and excluded. Peaks include training, stored caches and requests; these are traced Python allocations, not process RSS or GPU memory.

| Regime | States | Features | Method | Lifecycle peak KiB | End KiB | Max request incremental KiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| stable | 16 | 4 | cached_memo | 41.3 | 33.5 | 12.8 |
| stable | 16 | 4 | replay_memo | 41.0 | 32.4 | 11.9 |
| stable | 16 | 4 | inverse_reuse | 59.6 | 43.3 | 20.1 |
| stable | 16 | 4 | policy_certificate | 43.7 | 32.7 | 5.1 |
| stable | 128 | 4 | cached_memo | 91.5 | 68.8 | 33.4 |
| stable | 128 | 4 | replay_memo | 92.6 | 68.8 | 32.6 |
| stable | 128 | 4 | inverse_reuse | 109.7 | 76.4 | 38.5 |
| stable | 128 | 4 | policy_certificate | 98.5 | 41.1 | 5.1 |
| stable | 128 | 8 | cached_memo | 143.3 | 108.1 | 55.1 |
| stable | 128 | 8 | replay_memo | 142.3 | 105.0 | 51.2 |
| stable | 128 | 8 | inverse_reuse | 204.8 | 134.1 | 79.0 |
| stable | 128 | 8 | policy_certificate | 210.5 | 77.0 | 143.6 |
| fragile | 16 | 4 | cached_memo | 41.5 | 33.7 | 12.7 |
| fragile | 16 | 4 | replay_memo | 41.2 | 32.6 | 11.7 |
| fragile | 16 | 4 | inverse_reuse | 59.4 | 43.4 | 20.1 |
| fragile | 16 | 4 | policy_certificate | 68.3 | 36.8 | 35.0 |
| switch | 2 | 4 | cached_memo | 25.6 | 21.3 | 10.5 |
| switch | 2 | 4 | replay_memo | 25.0 | 20.0 | 9.1 |
| switch | 2 | 4 | inverse_reuse | 41.2 | 31.8 | 14.8 |
| switch | 2 | 4 | policy_certificate | 36.7 | 26.3 | 24.5 |

## Interpretation limits

- Request medians are medians of within-run medians; lifecycle includes original learning, inverse/margin preparation, every certificate check and every fallback. Tail reports show why a low median can hide expensive rebuilding.
- Stable/fragile cases use designed shared positive feature vectors with opposite action signs. Stable rewards favor one action; fragile rewards are random. Switch cases repeat the earlier two-state fixture. These are deliberately controlled synthetic regimes.
- Seeds create different stable/fragile logs; switch seeds reorder the same batch. Timing repeats are not independent datasets. All methods run in randomized order; no statistical significance claim is made.
- Exact rational arithmetic avoids numerical errors but does not establish numerical stability. Factorizations, rational bit growth, larger feature dimensions, realistic tasks and external optimized libraries remain gates.
- Accepted certificates keep old internal anchors and return policy references. They do not erase internal influence or certify privacy. Serializing a whole policy table would cost at least its output size and is outside these in-process timings.
- Sherman–Morrison, perturbation bounds and unchanged-prediction certificates have prior art. This experiment cannot establish publication-level novelty.
