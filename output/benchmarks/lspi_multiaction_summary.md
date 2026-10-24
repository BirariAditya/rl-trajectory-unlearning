# Multi-action fixed-feature LSPI deletion controls

Source `720e0a3b206e19583b6542f9223489c0d48ff432`; clean at launch: True.

**8 runs, 80 requests, 320 method executions; 0 full-trace mismatches.**

Every method is compared with direct retained-only LSPI. Lifecycle time includes initial learning, indexes, and the full sequential deletion workload. Reference comparison and serialization are outside competitor timings.

## Lifecycle median milliseconds

| Regime | States | Dimension | Actions | scratch_retrain | inverse_deduplicated | selective_repair | shared_anchor_repair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fragile | 32 | 8 | 3 | 8113.064 | 278.523 | 269.881 | 246.763 |
| fragile | 128 | 16 | 3 | 70241.789 | 7627.397 | 9508.872 | 7971.463 |
| stable | 32 | 8 | 3 | 27094.652 | 724.735 | 447.700 | 448.851 |
| stable | 128 | 16 | 3 | 106470.888 | 4347.826 | 3189.980 | 3178.098 |

## Changes and decision work

| Regime | States | Dimension | Changed final policies / requests | Changed full traces | Method | Action scores | Certified groups | Scored groups |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| fragile | 32 | 8 | 5 / 20 | 20 | inverse_deduplicated | 4608 | — | — |
| fragile | 32 | 8 | 5 / 20 | 20 | selective_repair | 2325 | 761 | 775 |
| fragile | 32 | 8 | 5 / 20 | 20 | shared_anchor_repair | 1896 | 904 | 632 |
| fragile | 128 | 16 | 20 / 20 | 20 | inverse_deduplicated | 23040 | — | — |
| fragile | 128 | 16 | 20 / 20 | 20 | selective_repair | 21369 | 557 | 7123 |
| fragile | 128 | 16 | 20 / 20 | 20 | shared_anchor_repair | 19557 | 1161 | 6519 |
| stable | 32 | 8 | 0 / 20 | 20 | inverse_deduplicated | 3840 | — | — |
| stable | 32 | 8 | 0 / 20 | 20 | selective_repair | 0 | 1280 | 0 |
| stable | 32 | 8 | 0 / 20 | 20 | shared_anchor_repair | 0 | 1280 | 0 |
| stable | 128 | 16 | 0 / 20 | 20 | inverse_deduplicated | 15360 | — | — |
| stable | 128 | 16 | 0 / 20 | 20 | selective_repair | 0 | 5120 | 0 |
| stable | 128 | 16 | 0 / 20 | 20 | shared_anchor_repair | 0 | 5120 | 0 |

## Interpretation limits

- These are synthetic fixed features with three actions, eight or sixteen dimensions, five scheduled LSPI iterations, and exact rational arithmetic.
- Feature vectors and ordered action rows are unique and no within-state action pair is an exact sign opposite. This removes the earlier alias and opposite-action shortcuts but does not make the task realistic.
- A full trace includes matrices, right-hand sides, weights, policies and status, so it can change after a deletion even when the final policy is unchanged.
- Timing repetitions, when requested, reuse a generated dataset. The bounded archive uses one timing execution for each of two independently generated seeds; medians are descriptive.
- Full-trace equality covers matrices, right-hand sides, weights, policies and status for the declared learner. It does not imply privacy or deletion of historical anchors.
- No universal speedup, neural-RL result, floating-point stability result, or publication novelty follows from this experiment.

