# Exact local repair checkpoint

Source: `70f13773b8e467e99cb429115cc2b437c008708e`; clean at execution: `True`.

**54 runs, 162 sequential requests, 810 method executions, 0 failures.** Rational methods were checked for exact policy and value equality against scratch rational PI; warm binary64 VI was checked for policy agreement.

## Request medians (milliseconds)

| Regime | States | Local exact | Pruned exact | Warm exact PI | Scratch exact PI | Warm float VI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 8 | 1.621 | 1.531 | 1.809 | 2.846 | 10.640 |
| dense | 16 | 6.782 | 6.837 | 6.807 | 10.790 | 63.461 |
| irrelevant_bridges | 8 | 1.657 | 0.436 | 1.652 | 1.878 | 1.936 |
| irrelevant_bridges | 16 | 10.205 | 0.448 | 9.847 | 10.687 | 4.417 |
| modular | 8 | 0.408 | 0.415 | 1.669 | 1.785 | 1.673 |
| modular | 16 | 0.422 | 0.434 | 9.876 | 10.436 | 3.866 |

## Full sequence including preparation (milliseconds)

Median of (preparation + all requests) for each repeated run. Preparation includes original full rational training, reverse indexing, and anchor evaluation/pruning where used. This avoids treating the pruning index or anchor as free.

| Regime | States | Requests/sequence | Local exact | Pruned exact | Warm exact PI | Warm float VI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 8 | 2 | 7.016 | 6.592 | 5.132 | 41.500 |
| dense | 16 | 4 | 42.836 | 47.962 | 42.633 | 264.035 |
| irrelevant_bridges | 8 | 2 | 4.549 | 3.017 | 4.478 | 11.066 |
| irrelevant_bridges | 16 | 4 | 47.434 | 12.502 | 46.365 | 33.256 |
| modular | 8 | 2 | 2.104 | 2.722 | 4.389 | 9.423 |
| modular | 16 | 4 | 7.895 | 12.420 | 45.730 | 29.700 |

## Affected block sizes

| Regime | States | Structural closure sizes | Pruned closure sizes |
| --- | ---: | --- | --- |
| dense | 8 | [8] | [8] |
| dense | 16 | [16] | [16] |
| irrelevant_bridges | 8 | [8] | [4] |
| irrelevant_bridges | 16 | [16] | [4] |
| modular | 8 | [4] | [4] |
| modular | 16 | [4] | [4] |

## Interpretation

- Dominated bridge actions make structural closure global. Safe action pruning can recover component locality while still allowing the chosen optimal action to change.
- Dense admissible influence still forces full blocks. This is an explicit failure of universal locality, not a hidden excluded setting.
- These are constructed 8/16-state cases with three dataset seeds and three repeats. Request samples within a run are correlated. Medians are descriptive; no significance or broad speedup claim is made.
- Exact rational Gaussian elimination is a correctness-oriented solver. Compare with mature sparse/SCC-aware and action-elimination planners before attributing speed gains to unlearning novelty. Binary64 VI solves a different numerical learner and is included for practical context.
- This benchmark preserves the anchor policy. Tests cover invalidation and global fallback; a timed anchor-invalidation/deletion-shift study is still needed.
- No GPU, learned representation, full-checkpoint deletion, or privacy indistinguishability claim follows from this checkpoint.
