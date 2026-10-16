# LSPI feature-alias and selective-repair controls

Source `1fb3e300aaee97786be1b115c45b03606df042d2`; clean at launch: True.

**81 runs, 972 requests, 6804 method executions; 0 mismatches.**

Every method is checked against an independent per-transition retained-only LSPI reference. Full traces are additionally compared when exposed. Preparation includes feature indexing, initial training and certificate construction; request costs include fallbacks. Reference evaluation and report serialization are outside timings.

## Request median milliseconds

| Regime | Features | States | Dimension | inverse_reuse | inverse_deduplicated | inverse_deduplicated_policy_only | policy_certificate | selective_repair | selective_repair_policy_only | shared_anchor_repair |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fragile | aliased | 16 | 4 | 1.829 | 1.456 | 1.433 | 0.496 | 1.395 | 1.378 | 1.395 |
| fragile | aliased | 128 | 4 | 6.615 | 1.593 | 1.637 | 45.207 | 1.562 | 1.307 | 1.425 |
| fragile | unique | 16 | 4 | 1.702 | 1.735 | 1.725 | 0.401 | 1.484 | 1.492 | 1.466 |
| fragile | unique | 128 | 4 | 7.219 | 7.191 | 7.005 | 45.963 | 3.871 | 3.895 | 3.900 |
| stable | aliased | 16 | 4 | 1.577 | 1.289 | 1.267 | 0.347 | 1.017 | 0.998 | 1.039 |
| stable | aliased | 128 | 4 | 7.382 | 1.575 | 1.587 | 0.433 | 1.248 | 1.247 | 1.296 |
| stable | unique | 16 | 4 | 1.817 | 1.815 | 1.810 | 0.402 | 1.135 | 1.133 | 1.156 |
| stable | unique | 128 | 4 | 7.416 | 7.347 | 7.119 | 0.434 | 1.258 | 1.254 | 1.288 |
| switch | one_hot | 2 | 4 | 0.536 | 0.542 | 0.535 | 0.181 | 0.526 | 0.523 | 0.577 |

## Lifecycle median milliseconds

| Regime | Features | States | Dimension | inverse_reuse | inverse_deduplicated | inverse_deduplicated_policy_only | policy_certificate | selective_repair | selective_repair_policy_only | shared_anchor_repair |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fragile | aliased | 16 | 4 | 46.872 | 41.553 | 42.485 | 137.353 | 43.453 | 42.106 | 42.334 |
| fragile | aliased | 128 | 4 | 130.042 | 55.191 | 54.794 | 526.379 | 54.554 | 52.703 | 53.999 |
| fragile | unique | 16 | 4 | 45.429 | 45.641 | 44.182 | 134.509 | 41.235 | 41.598 | 40.939 |
| fragile | unique | 128 | 4 | 137.671 | 139.231 | 138.243 | 569.484 | 123.163 | 117.291 | 103.918 |
| stable | aliased | 16 | 4 | 41.992 | 37.509 | 36.731 | 27.189 | 33.312 | 33.925 | 34.560 |
| stable | aliased | 128 | 4 | 125.980 | 50.134 | 51.171 | 51.000 | 46.175 | 47.802 | 46.553 |
| stable | unique | 16 | 4 | 48.014 | 47.726 | 48.945 | 30.516 | 39.672 | 40.005 | 41.504 |
| stable | unique | 128 | 4 | 129.015 | 126.722 | 126.953 | 55.017 | 62.653 | 62.309 | 62.033 |
| switch | one_hot | 2 | 4 | 16.748 | 16.440 | 16.460 | 14.799 | 17.319 | 16.793 | 17.385 |

## Feature counts, changes and certificate coverage

| Regime | Features | States | Unique vectors / rows | Changed final policies / requests | Changed sequences | Certified | Fallbacks | Median run-max certificate ms |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| fragile | aliased | 16 | 8 / 8 | 3 / 108 | 3 | 69 | 39 | 30.015 |
| fragile | aliased | 128 | 8 / 8 | 12 / 108 | 21 | 36 | 72 | 49.464 |
| fragile | unique | 16 | 32 / 16 | 3 / 108 | 6 | 66 | 42 | 26.661 |
| fragile | unique | 128 | 256 / 128 | 15 / 108 | 18 | 36 | 72 | 53.355 |
| stable | aliased | 16 | 8 / 8 | 0 / 108 | 0 | 108 | 0 | 0.386 |
| stable | aliased | 128 | 8 / 8 | 0 / 108 | 0 | 108 | 0 | 0.598 |
| stable | unique | 16 | 32 / 16 | 0 / 108 | 0 | 108 | 0 | 0.551 |
| stable | unique | 128 | 256 / 128 | 0 / 108 | 0 | 108 | 0 | 0.536 |
| switch | one_hot | 2 | 4 / 2 | 9 / 108 | 9 | 99 | 9 | 6.892 |

## Request-only decision work

Counts below sum deletion requests across all seeds and timing repeats; preparation is excluded from this table and included in lifecycle timings above. Scores count individual action evaluations. Groups are ordered action-feature rows, and the same group may be evaluated for multiple distinct input policies in one request. A dash means the baseline does not expose that screening counter.

| Regime / features / states | Method | Action scores | New margin groups / request | Certified groups | Scored groups | Decisions changed from anchor |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fragile / aliased / 16 | inverse_deduplicated | 3504 | — | — | — | — |
| fragile / aliased / 16 | selective_repair | 1596 | 0.00 | 954 | 798 | 66 |
| fragile / aliased / 16 | shared_anchor_repair | 1596 | 0.00 | 954 | 798 | 12 |
| fragile / aliased / 128 | inverse_deduplicated | 4032 | — | — | — | — |
| fragile / aliased / 128 | selective_repair | 1644 | 1.11 | 1194 | 822 | 132 |
| fragile / aliased / 128 | shared_anchor_repair | 1464 | 0.00 | 1284 | 732 | 264 |
| fragile / unique / 16 | inverse_deduplicated | 7008 | — | — | — | — |
| fragile / unique / 16 | selective_repair | 2808 | 0.89 | 2100 | 1404 | 69 |
| fragile / unique / 16 | shared_anchor_repair | 3204 | 0.00 | 1902 | 1602 | 135 |
| fragile / unique / 128 | inverse_deduplicated | 61440 | — | — | — | — |
| fragile / unique / 128 | selective_repair | 25548 | 24.89 | 17946 | 12774 | 2103 |
| fragile / unique / 128 | shared_anchor_repair | 21966 | 0.00 | 19737 | 10983 | 4005 |
| stable / aliased / 16 | inverse_deduplicated | 3456 | — | — | — | — |
| stable / aliased / 16 | selective_repair | 0 | 0.00 | 1728 | 0 | 0 |
| stable / aliased / 16 | shared_anchor_repair | 0 | 0.00 | 1728 | 0 | 0 |
| stable / aliased / 128 | inverse_deduplicated | 3456 | — | — | — | — |
| stable / aliased / 128 | selective_repair | 0 | 0.00 | 1728 | 0 | 0 |
| stable / aliased / 128 | shared_anchor_repair | 0 | 0.00 | 1728 | 0 | 0 |
| stable / unique / 16 | inverse_deduplicated | 6912 | — | — | — | — |
| stable / unique / 16 | selective_repair | 0 | 0.00 | 3456 | 0 | 0 |
| stable / unique / 16 | shared_anchor_repair | 0 | 0.00 | 3456 | 0 | 0 |
| stable / unique / 128 | inverse_deduplicated | 55296 | — | — | — | — |
| stable / unique / 128 | selective_repair | 0 | 0.00 | 27648 | 0 | 0 |
| stable / unique / 128 | shared_anchor_repair | 0 | 0.00 | 27648 | 0 | 0 |
| switch / one_hot / 2 | inverse_deduplicated | 1296 | — | — | — | — |
| switch / one_hot / 2 | selective_repair | 504 | 0.33 | 396 | 252 | 216 |
| switch / one_hot / 2 | shared_anchor_repair | 1296 | 0.00 | 0 | 648 | 432 |

“Decisions changed from anchor” counts feature-group choices that differ from the selected historical margin anchor, summed across evaluations. It is **not** a count of new changes caused by the current deletion: the same persistent difference can be counted again on later requests, and a shared anchor can come from another input policy. The earlier coverage table compares each retained-only final policy/sequence with the previous successfully retained dataset. New-anchor choices are scored and indexed but are not counted as changes from an old anchor.

## Separate Python allocation pass

One seed-0 pass per case/method, outside the timing experiment, over the full request sequence. Input features/logs and deletion lists are allocated before tracing and excluded. Training, feature/index preparation, caches and requests are included. End allocations include the final returned result; earlier returned results are released. These are traced Python allocations, not process RSS or GPU memory.

| Regime / features / states | Method | Lifecycle peak KiB | End KiB | Max request incremental KiB |
| --- | --- | ---: | ---: | ---: |
| stable / aliased / 16 | inverse_reuse | 55.5 | 39.9 | 19.5 |
| stable / aliased / 16 | inverse_deduplicated | 56.4 | 40.7 | 19.4 |
| stable / aliased / 16 | inverse_deduplicated_policy_only | 56.4 | 39.5 | 19.4 |
| stable / aliased / 16 | policy_certificate | 43.1 | 31.9 | 5.0 |
| stable / aliased / 16 | selective_repair | 62.5 | 46.7 | 19.8 |
| stable / aliased / 16 | selective_repair_policy_only | 62.5 | 45.4 | 19.7 |
| stable / aliased / 16 | shared_anchor_repair | 62.5 | 46.7 | 19.8 |
| stable / unique / 16 | inverse_reuse | 57.9 | 41.3 | 20.3 |
| stable / unique / 16 | inverse_deduplicated | 59.4 | 42.8 | 20.3 |
| stable / unique / 16 | inverse_deduplicated_policy_only | 59.4 | 41.5 | 20.2 |
| stable / unique / 16 | policy_certificate | 44.7 | 32.6 | 5.1 |
| stable / unique / 16 | selective_repair | 68.3 | 51.6 | 20.5 |
| stable / unique / 16 | selective_repair_policy_only | 68.3 | 50.3 | 20.4 |
| stable / unique / 16 | shared_anchor_repair | 68.4 | 51.6 | 20.5 |
| stable / aliased / 128 | inverse_reuse | 102.0 | 70.0 | 37.0 |
| stable / aliased / 128 | inverse_deduplicated | 103.7 | 71.7 | 36.9 |
| stable / aliased / 128 | inverse_deduplicated_policy_only | 103.7 | 70.4 | 36.8 |
| stable / aliased / 128 | policy_certificate | 93.6 | 40.2 | 5.0 |
| stable / aliased / 128 | selective_repair | 109.8 | 77.8 | 36.2 |
| stable / aliased / 128 | selective_repair_policy_only | 109.8 | 76.5 | 36.1 |
| stable / aliased / 128 | shared_anchor_repair | 109.8 | 77.8 | 36.2 |
| stable / unique / 128 | inverse_reuse | 111.6 | 78.0 | 38.6 |
| stable / unique / 128 | inverse_deduplicated | 121.6 | 88.1 | 38.6 |
| stable / unique / 128 | inverse_deduplicated_policy_only | 121.6 | 86.8 | 38.5 |
| stable / unique / 128 | policy_certificate | 102.8 | 41.1 | 5.2 |
| stable / unique / 128 | selective_repair | 171.0 | 137.4 | 37.8 |
| stable / unique / 128 | selective_repair_policy_only | 171.0 | 136.1 | 37.7 |
| stable / unique / 128 | shared_anchor_repair | 171.0 | 137.4 | 37.8 |
| fragile / aliased / 16 | inverse_reuse | 55.4 | 39.8 | 19.5 |
| fragile / aliased / 16 | inverse_deduplicated | 56.2 | 40.6 | 19.4 |
| fragile / aliased / 16 | inverse_deduplicated_policy_only | 56.2 | 39.4 | 19.4 |
| fragile / aliased / 16 | policy_certificate | 67.5 | 36.6 | 34.3 |
| fragile / aliased / 16 | selective_repair | 62.9 | 47.1 | 19.8 |
| fragile / aliased / 16 | selective_repair_policy_only | 62.9 | 45.8 | 19.7 |
| fragile / aliased / 16 | shared_anchor_repair | 62.9 | 47.1 | 19.8 |
| fragile / unique / 16 | inverse_reuse | 57.6 | 41.1 | 20.2 |
| fragile / unique / 16 | inverse_deduplicated | 59.0 | 42.6 | 20.2 |
| fragile / unique / 16 | inverse_deduplicated_policy_only | 59.0 | 41.3 | 20.1 |
| fragile / unique / 16 | policy_certificate | 69.2 | 37.1 | 35.6 |
| fragile / unique / 16 | selective_repair | 68.5 | 52.0 | 20.4 |
| fragile / unique / 16 | selective_repair_policy_only | 68.5 | 50.7 | 20.3 |
| fragile / unique / 16 | shared_anchor_repair | 68.7 | 52.1 | 20.4 |
| fragile / aliased / 128 | inverse_reuse | 101.7 | 69.8 | 36.8 |
| fragile / aliased / 128 | inverse_deduplicated | 103.3 | 71.5 | 36.8 |
| fragile / aliased / 128 | inverse_deduplicated_policy_only | 103.3 | 70.2 | 36.7 |
| fragile / aliased / 128 | policy_certificate | 121.8 | 47.2 | 78.1 |
| fragile / aliased / 128 | selective_repair | 111.7 | 79.7 | 36.1 |
| fragile / aliased / 128 | selective_repair_policy_only | 111.7 | 78.4 | 35.9 |
| fragile / aliased / 128 | shared_anchor_repair | 111.7 | 79.7 | 36.1 |
| fragile / unique / 128 | inverse_reuse | 111.1 | 77.7 | 38.4 |
| fragile / unique / 128 | inverse_deduplicated | 121.1 | 87.8 | 38.4 |
| fragile / unique / 128 | inverse_deduplicated_policy_only | 121.1 | 86.5 | 38.3 |
| fragile / unique / 128 | policy_certificate | 130.9 | 47.9 | 86.3 |
| fragile / unique / 128 | selective_repair | 170.7 | 137.3 | 37.6 |
| fragile / unique / 128 | selective_repair_policy_only | 170.7 | 136.0 | 37.5 |
| fragile / unique / 128 | shared_anchor_repair | 170.8 | 137.4 | 37.6 |
| switch / one_hot / 2 | inverse_reuse | 38.2 | 29.2 | 14.9 |
| switch / one_hot / 2 | inverse_deduplicated | 38.4 | 29.5 | 15.3 |
| switch / one_hot / 2 | inverse_deduplicated_policy_only | 38.3 | 28.6 | 15.0 |
| switch / one_hot / 2 | policy_certificate | 36.7 | 25.7 | 24.5 |
| switch / one_hot / 2 | selective_repair | 42.9 | 33.8 | 16.2 |
| switch / one_hot / 2 | selective_repair_policy_only | 42.8 | 32.9 | 15.9 |
| switch / one_hot / 2 | shared_anchor_repair | 41.6 | 32.5 | 15.4 |


## Interpretation limits

- Exact rational arithmetic, synthetic fixed features, two actions, finite LSPI schedule and small feature dimension. Feature uniqueness does not imply a difficult or realistic representation.
- Aliased and unique cases share trajectory state/action draws and deletion IDs. Stable rewards use the corresponding feature-map teacher, so rewards differ slightly; fragile rewards are identical between the pair.
- Stable rewards target a mixed-sign feature map whose preferred action varies by state. Unique maps add deterministic small perturbations; their arithmetic bit lengths also differ from the aliased maps.
- The switch fixture repeats a deterministic two-state dataset; its seeds only reorder one deletion batch. Timing repeats are not independent scientific datasets.
- Certificate and prediction-only controls expose policy sequences/status. Other methods return exact full traces; both contracts are checked explicitly. These in-process measurements do not charge serialization of a whole policy table.
- No privacy, neural-RL, universal speedup, statistical significance or publication novelty claim follows from these controls. Optional allocation measurements exclude shared input objects and do not establish total process memory.
